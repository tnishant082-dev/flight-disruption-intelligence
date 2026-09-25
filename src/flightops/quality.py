"""Data-quality gate for the staging layer.

Each check returns observed value, threshold and PASS / WARN / FAIL. The pipeline stops on
any FAIL; WARN is reported but not blocking. Results land in reports/data_quality_report.{md,json}.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import duckdb

from flightops import config


@dataclass
class Check:
    name: str
    category: str
    observed: float
    threshold: str
    status: str
    detail: str = ""


def _scalar(con: duckdb.DuckDBPyConnection, sql: str) -> float:
    v = con.execute(sql).fetchone()[0]
    return float(v) if v is not None else 0.0


def _status(ok: bool, warn_only: bool = False) -> str:
    if ok:
        return "PASS"
    return "WARN" if warn_only else "FAIL"


def run_checks(con: duckdb.DuckDBPyConnection, table: str = "staging.stg_flights",
               airports: str | None = "staging.stg_airports",
               carriers: str | None = "staging.stg_carriers",
               raw_table: str | None = "raw.flights",
               expected_days: int | None = None) -> list[Check]:
    t = table
    n = _scalar(con, f"SELECT count(*) FROM {t}")
    checks: list[Check] = []

    def add(name, cat, obs, thr, ok, warn_only=False, detail=""):
        checks.append(Check(name, cat, round(obs, 6), thr, _status(ok, warn_only), detail))

    add("row_count_positive", "completeness", n, "> 0", n > 0)
    if raw_table:
        raw_n = _scalar(con, f"SELECT count(*) FROM {raw_table}")
        add("rows_retained_vs_raw", "completeness", n / raw_n if raw_n else 0, ">= 0.999",
            raw_n > 0 and n / raw_n >= 0.999, detail=f"raw={int(raw_n)} staged={int(n)}")

    dup = _scalar(con, f"SELECT count(*) - count(DISTINCT flight_id) FROM {t}")
    add("duplicate_flight_id", "uniqueness", dup, "== 0", dup == 0)

    for col in ("flight_date", "carrier", "flight_number", "origin", "dest", "crs_dep_hhmm",
                "crs_arr_hhmm", "distance_mi"):
        nulls = _scalar(con, f"SELECT count(*) FILTER (WHERE {col} IS NULL) FROM {t}")
        add(f"not_null_{col}", "completeness", nulls, "== 0", nulls == 0)

    bad_time = _scalar(con, f"""SELECT count(*) FROM {t} WHERE crs_dep_hhmm NOT BETWEEN 0 AND 2400
                                OR crs_dep_hhmm % 100 >= 60""")
    add("valid_scheduled_dep_time", "validity", bad_time, "== 0", bad_time == 0)

    same = _scalar(con, f"SELECT count(*) FROM {t} WHERE origin = dest")
    add("origin_differs_from_dest", "validity", same, "== 0", same == 0)

    bad_flag = _scalar(con, f"""SELECT count(*) FROM {t}
                                WHERE cancelled NOT IN (0, 1) OR diverted NOT IN (0, 1)""")
    add("binary_status_flags", "validity", bad_flag, "== 0", bad_flag == 0)

    canc_no_code = _scalar(con, f"""SELECT count(*) FROM {t}
                                    WHERE cancelled = 1 AND cancellation_code IS NULL""")
    add("cancelled_has_reason_code", "consistency", canc_no_code, "== 0", canc_no_code == 0)

    code_not_canc = _scalar(con, f"""SELECT count(*) FROM {t}
                                     WHERE cancelled = 0 AND cancellation_code IS NOT NULL""")
    add("reason_code_only_when_cancelled", "consistency", code_not_canc, "== 0",
        code_not_canc == 0)

    miss_arr = _scalar(con, f"""SELECT avg((arr_delay_min IS NULL)::INT) FROM {t}
                                WHERE completed = 1""")
    add("completed_has_arrival_delay", "completeness", miss_arr, "<= 0.001", miss_arr <= 0.001)

    flag_mismatch = _scalar(con, f"""SELECT avg(((arr_delay_min >= 15)::INT <> arr_del15)::INT)
                                     FROM {t} WHERE completed = 1 AND arr_delay_min IS NOT NULL""")
    add("arr_del15_matches_delay_minutes", "consistency", flag_mismatch, "<= 0.0005",
        flag_mismatch <= 0.0005)

    cause_mismatch = _scalar(con, f"""
        SELECT avg((abs(coalesce(carrier_delay_min,0) + coalesce(weather_delay_min,0)
                   + coalesce(nas_delay_min,0) + coalesce(security_delay_min,0)
                   + coalesce(late_aircraft_delay_min,0) - arr_delay_min) > 1)::INT)
        FROM {t} WHERE arr_del15 = 1""")
    add("delay_causes_sum_to_arrival_delay", "consistency", cause_mismatch, "<= 0.01",
        cause_mismatch <= 0.01, warn_only=True,
        detail="BTS attributes cause minutes for flights arriving 15+ min late")

    nonpos = _scalar(con, f"SELECT count(*) FROM {t} WHERE distance_mi <= 0 OR crs_elapsed_min <= 0")
    add("positive_distance_and_block_time", "validity", nonpos, "== 0", nonpos == 0)

    taxi_out = _scalar(con, f"""SELECT avg((taxi_out_min NOT BETWEEN 0 AND 240)::INT) FROM {t}
                                WHERE taxi_out_min IS NOT NULL""")
    add("taxi_out_plausible_0_240", "validity", taxi_out, "<= 0.001", taxi_out <= 0.001,
        warn_only=True)

    extreme = _scalar(con, f"SELECT avg((arr_delay_min > 1440)::INT) FROM {t} WHERE completed = 1")
    add("arrival_delay_over_24h_share", "validity", extreme, "<= 0.0005", extreme <= 0.0005,
        warn_only=True)

    if expected_days:
        days = _scalar(con, f"SELECT count(DISTINCT flight_date) FROM {t}")
        add("calendar_coverage_days", "completeness", days, f"== {expected_days}",
            days == expected_days)

    if airports:
        orphan = _scalar(con, f"""SELECT count(*) FROM {t} f
                                  LEFT JOIN {airports} a ON a.airport_code = f.origin
                                  WHERE a.airport_code IS NULL""")
        add("origin_in_airport_dim", "referential", orphan, "== 0", orphan == 0)
        no_geo = _scalar(con, f"SELECT count(*) FROM {airports} WHERE latitude IS NULL")
        add("airport_dim_has_coordinates", "enrichment", no_geo, "== 0", no_geo == 0,
            warn_only=True)
    if carriers:
        unnamed = _scalar(con, f"SELECT count(*) FROM {carriers} WHERE carrier_name = carrier_code")
        add("carrier_dim_named", "enrichment", unnamed, "== 0", unnamed == 0, warn_only=True)
    return checks


def write_report(checks: list[Check], extra: dict | None = None) -> dict:
    config.ensure_dirs()
    summary = {
        "run_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "passed": sum(c.status == "PASS" for c in checks),
        "warned": sum(c.status == "WARN" for c in checks),
        "failed": sum(c.status == "FAIL" for c in checks),
        "gate": "PASS" if not any(c.status == "FAIL" for c in checks) else "FAIL",
        **(extra or {}),
        "checks": [asdict(c) for c in checks],
    }
    (config.REPORTS / "data_quality_report.json").write_text(json.dumps(summary, indent=2))
    lines = [
        "# Data quality report",
        "",
        f"Gate: **{summary['gate']}** - {summary['passed']} passed, {summary['warned']} warnings, "
        f"{summary['failed']} failed (run {summary['run_at_utc']} UTC).",
        "",
        f"Rows quarantined before staging (source errors): {summary.get('quarantined_rows', 0)}.",
        "",
        "| Check | Category | Observed | Threshold | Status | Note |",
        "|---|---|---:|---|---|---|",
    ]
    for c in checks:
        lines.append(f"| `{c.name}` | {c.category} | {c.observed:g} | {c.threshold} | "
                     f"{c.status} | {c.detail} |")
    (config.REPORTS / "data_quality_report.md").write_text("\n".join(lines) + "\n")
    return summary


def run() -> dict:
    con = duckdb.connect(str(config.DB_PATH), read_only=True)
    days = con.execute("SELECT (max(flight_date) - min(flight_date)) + 1 FROM staging.stg_flights"
                       ).fetchone()[0]
    checks = run_checks(con, expected_days=int(days))
    quarantined = con.execute("SELECT count(*) FROM staging.quarantine_flights").fetchone()[0]
    con.close()
    summary = write_report(checks, {"table": "staging.stg_flights",
                                    "quarantined_rows": int(quarantined)})
    for c in checks:
        print(f"{c.status:4}  {c.name:38} observed={c.observed:g}  ({c.threshold})")
    print(f"gate: {summary['gate']}")
    if summary["gate"] == "FAIL":
        raise SystemExit("data-quality gate failed")
    return summary


if __name__ == "__main__":
    run()
