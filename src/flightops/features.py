"""Feature table for the pre-departure models.

Schedule features come from sql/features, holidays are joined here, and high-cardinality keys
(route, carrier x origin, flight number) are target-encoded: out-of-fold on the training window,
training-window statistics for validation/test. Serving lookups go to models/lookups/.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from flightops import config

if TYPE_CHECKING:  # duckdb is only needed to build features, not to serve them
    import duckdb

CAT_FEATURES = ["carrier", "origin", "dest"]
SCHEDULE_FEATURES = [
    "dep_hour", "arr_hour", "crs_dep_min_of_day", "day_of_week", "is_weekend",
    "days_to_holiday", "distance_mi", "crs_elapsed_min", "sched_speed_mpm",
    "origin_sched_deps_day", "origin_sched_deps_hour", "dest_sched_arrs_hour",
    "carrier_sched_day",
]
# Rotation features come from the tail number that actually flew, which isn't known when the
# schedule is published (a missing tail means the flight was cancelled). Day-of-ops comparison only.
ROTATION_FEATURES = ["tail_leg_of_day", "tail_legs_day", "sched_turn_min"]
TE_KEYS = {"route": "route", "carrier_origin": "carrier_origin", "flight": "flight_key"}
TE_TARGETS = {"delay": "arr_del15", "cancel": "cancelled"}
TE_FEATURES = [f"te_{k}_{t}" for k in TE_KEYS for t in TE_TARGETS]
NUM_FEATURES = SCHEDULE_FEATURES + TE_FEATURES
ALL_FEATURES = CAT_FEATURES + NUM_FEATURES          # pre-departure feature set
DAY_OF_OPS_FEATURES = ALL_FEATURES + ROTATION_FEATURES

TE_SMOOTHING = 50.0
N_FOLDS = 5


def holiday_table(start: str = "2024-01-01", end: str = "2027-12-31") -> pd.DataFrame:
    days = pd.date_range(start, end, freq="D")
    hol = USFederalHolidayCalendar().holidays(start, end)
    hol_arr = hol.values.astype("datetime64[D]").astype(np.int64)
    d = days.values.astype("datetime64[D]").astype(np.int64)
    dist = np.abs(d[:, None] - hol_arr[None, :]).min(axis=1)
    return pd.DataFrame({"flight_date": days.date,
                         "days_to_holiday": np.minimum(dist, 30).astype(np.int16),
                         "is_holiday": np.isin(d, hol_arr).astype(np.int8)})


def build(con: duckdb.DuckDBPyConnection | None = None) -> None:
    own = con is None
    import duckdb

    con = con or duckdb.connect(str(config.DB_PATH))
    con.execute("SET memory_limit='2GB'; SET threads=4;")
    con.execute((config.SQL_DIR / "features" / "01_flight_features.sql").read_text())
    cal = holiday_table()
    con.register("cal_df", cal)
    con.execute("CREATE OR REPLACE TABLE features.calendar AS SELECT * FROM cal_df")

    prior_delay = con.execute(
        f"SELECT avg(arr_del15) FROM features.flight_base WHERE flight_date <= '{config.TRAIN_END}'"
    ).fetchone()[0]
    prior_cancel = con.execute(
        f"SELECT avg(cancelled) FROM features.flight_base WHERE flight_date <= '{config.TRAIN_END}'"
    ).fetchone()[0]
    priors = {"delay": prior_delay, "cancel": prior_cancel}
    m = TE_SMOOTHING

    # full-train statistics per key (used for valid/test and for serving)
    for key, col in TE_KEYS.items():
        con.execute(f"""
            CREATE OR REPLACE TABLE features.te_{key} AS
            SELECT {col} AS key,
                   count(arr_del15) AS n_delay, coalesce(sum(arr_del15), 0) AS s_delay,
                   count(*) AS n_cancel, sum(cancelled) AS s_cancel
            FROM features.flight_base
            WHERE flight_date <= '{config.TRAIN_END}'
            GROUP BY 1""")
        con.execute(f"""
            CREATE OR REPLACE TABLE features.te_{key}_fold AS
            SELECT {col} AS key, CAST(hash(flight_id) % {N_FOLDS} AS INTEGER) AS fold,
                   count(arr_del15) AS n_delay, coalesce(sum(arr_del15), 0) AS s_delay,
                   count(*) AS n_cancel, sum(cancelled) AS s_cancel
            FROM features.flight_base
            WHERE flight_date <= '{config.TRAIN_END}'
            GROUP BY 1, 2""")

    te_selects, te_joins = [], []
    for key, col in TE_KEYS.items():
        te_joins.append(f"LEFT JOIN features.te_{key} t_{key} ON t_{key}.key = b.{col}")
        te_joins.append(f"""LEFT JOIN features.te_{key}_fold f_{key}
                            ON f_{key}.key = b.{col} AND f_{key}.fold = b.fold""")
        for tgt in TE_TARGETS:
            p = priors[tgt]
            n = f"(coalesce(t_{key}.n_{tgt}, 0) - coalesce(f_{key}.n_{tgt}, 0))"
            s = f"(coalesce(t_{key}.s_{tgt}, 0) - coalesce(f_{key}.s_{tgt}, 0))"
            te_selects.append(f"CAST(({s} + {m} * {p}) / ({n} + {m}) AS FLOAT) AS te_{key}_{tgt}")

    con.execute(f"""
        CREATE OR REPLACE TABLE features.flight_features AS
        WITH b AS (
            SELECT *, CASE WHEN flight_date <= '{config.TRAIN_END}'
                           THEN CAST(hash(flight_id) % {N_FOLDS} AS INTEGER) ELSE -1 END AS fold
            FROM features.flight_base
        )
        SELECT
            b.* EXCLUDE (sched_turn_raw, fold),
            CASE WHEN b.sched_turn_raw >= 0 THEN b.sched_turn_raw END AS sched_turn_min,
            c.days_to_holiday,
            {", ".join(te_selects)},
            CASE WHEN b.flight_date <= '{config.TRAIN_END}' THEN 'train'
                 WHEN b.flight_date <= '{config.VALID_END}' THEN 'valid'
                 ELSE 'test' END AS split
        FROM b
        JOIN features.calendar c ON c.flight_date = b.flight_date
        {" ".join(te_joins)}""")
    for key in TE_KEYS:
        con.execute(f"DROP TABLE features.te_{key}_fold")
    export_lookups(con, priors)
    print(con.sql("SELECT split, count(*) n, avg(arr_del15) delay_rate, avg(cancelled) cancel_rate "
                  "FROM features.flight_features GROUP BY 1 ORDER BY 1"))
    if own:
        con.close()


def export_lookups(con: duckdb.DuckDBPyConnection, priors: dict) -> None:
    """Artifacts the API needs to rebuild features for a single flight request."""
    out = config.MODELS / "lookups"
    out.mkdir(parents=True, exist_ok=True)
    m = TE_SMOOTHING
    for key in TE_KEYS:
        con.execute(f"""
            COPY (SELECT key,
                         CAST((s_delay + {m} * {priors['delay']}) / (n_delay + {m}) AS FLOAT) AS te_delay,
                         CAST((s_cancel + {m} * {priors['cancel']}) / (n_cancel + {m}) AS FLOAT) AS te_cancel,
                         n_cancel AS n
                  FROM features.te_{key} WHERE n_cancel >= 5)
            TO '{out}/te_{key}.parquet' (FORMAT parquet, COMPRESSION zstd)""")
    # typical schedule intensity for requests that do not pass explicit schedule counts
    con.execute(f"""
        COPY (SELECT origin, day_of_week, dep_hour,
                     avg(origin_sched_deps_day)  AS origin_sched_deps_day,
                     avg(origin_sched_deps_hour) AS origin_sched_deps_hour
              FROM features.flight_base WHERE flight_date <= '{config.TRAIN_END}'
              GROUP BY ALL)
        TO '{out}/origin_schedule_profile.parquet' (FORMAT parquet)""")
    con.execute(f"""
        COPY (SELECT dest, day_of_week, arr_hour, avg(dest_sched_arrs_hour) AS dest_sched_arrs_hour
              FROM features.flight_base WHERE flight_date <= '{config.TRAIN_END}'
              GROUP BY ALL)
        TO '{out}/dest_schedule_profile.parquet' (FORMAT parquet)""")
    con.execute(f"""
        COPY (SELECT carrier, day_of_week, avg(carrier_sched_day) AS carrier_sched_day
              FROM features.flight_base WHERE flight_date <= '{config.TRAIN_END}'
              GROUP BY ALL)
        TO '{out}/carrier_schedule_profile.parquet' (FORMAT parquet)""")
    con.execute(f"""
        COPY (SELECT route, origin, dest, median(distance_mi) AS distance_mi,
                     median(crs_elapsed_min) AS crs_elapsed_min, count(*) AS flights
              FROM features.flight_base WHERE flight_date <= '{config.TRAIN_END}'
              GROUP BY ALL)
        TO '{out}/route_profile.parquet' (FORMAT parquet)""")
    cats = {c: sorted(r[0] for r in con.execute(
        f"SELECT DISTINCT {c} FROM features.flight_base").fetchall()) for c in CAT_FEATURES}
    import json

    (out / "categories.json").write_text(json.dumps(cats))
    (out / "priors.json").write_text(json.dumps(priors))


def load_split(split: str, sample: int | None = None, where: str = "", columns: list[str] | None = None,
               seed: int = config.SEED) -> pd.DataFrame:
    cols = columns or (["flight_id", "flight_date", "split"] + DAY_OF_OPS_FEATURES
                       + ["completed", "cancelled", "arr_del15", "arr_delay_min"])
    import duckdb

    con = duckdb.connect(str(config.DB_PATH), read_only=True)
    con.execute("SET memory_limit='2GB'; SET threads=4;")
    cond = f"split = '{split}'" + (f" AND {where}" if where else "")
    samp = f"USING SAMPLE reservoir({sample} ROWS) REPEATABLE ({seed})" if sample else ""
    # sample AFTER filtering (DuckDB applies USING SAMPLE to the FROM relation)
    df = con.execute(f"SELECT * FROM (SELECT {', '.join(cols)} FROM features.flight_features "
                     f"WHERE {cond}) {samp}").df()
    con.close()
    return prepare(df)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    import json

    cats = json.loads((config.MODELS / "lookups" / "categories.json").read_text())
    for c in CAT_FEATURES:
        if c in df:
            df[c] = pd.Categorical(df[c], categories=cats[c])
    for c in NUM_FEATURES + ROTATION_FEATURES:
        if c in df:
            df[c] = df[c].astype("float32")
    return df


if __name__ == "__main__":
    build()
