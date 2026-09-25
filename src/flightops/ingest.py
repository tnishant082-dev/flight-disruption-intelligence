"""Ingest BTS monthly zip files -> typed Parquet (one file per month) -> DuckDB raw schema.

Only the columns needed downstream are kept; diversion leg detail (Div1..Div5) is dropped.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

import duckdb

from flightops import config

RAW_COLUMNS: dict[str, str] = {
    "FlightDate": "DATE",
    "Reporting_Airline": "VARCHAR",
    "Tail_Number": "VARCHAR",
    "Flight_Number_Reporting_Airline": "VARCHAR",
    "Origin": "VARCHAR",
    "OriginCityName": "VARCHAR",
    "OriginState": "VARCHAR",
    "Dest": "VARCHAR",
    "DestCityName": "VARCHAR",
    "DestState": "VARCHAR",
    "CRSDepTime": "VARCHAR",
    "DepTime": "VARCHAR",
    "DepDelay": "DOUBLE",
    "DepDel15": "DOUBLE",
    "TaxiOut": "DOUBLE",
    "TaxiIn": "DOUBLE",
    "CRSArrTime": "VARCHAR",
    "ArrTime": "VARCHAR",
    "ArrDelay": "DOUBLE",
    "ArrDel15": "DOUBLE",
    "Cancelled": "DOUBLE",
    "CancellationCode": "VARCHAR",
    "Diverted": "DOUBLE",
    "CRSElapsedTime": "DOUBLE",
    "ActualElapsedTime": "DOUBLE",
    "AirTime": "DOUBLE",
    "Distance": "DOUBLE",
    "CarrierDelay": "DOUBLE",
    "WeatherDelay": "DOUBLE",
    "NASDelay": "DOUBLE",
    "SecurityDelay": "DOUBLE",
    "LateAircraftDelay": "DOUBLE",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def zip_to_parquet(zip_path: Path, out_path: Path, con: duckdb.DuckDBPyConnection) -> int:
    with zipfile.ZipFile(zip_path) as zf:
        csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(zf.extract(csv_name, tmp))
            # BTS files carry a trailing comma (extra unnamed column); read with auto-detect
            # restricted to the typed columns we need.
            con.execute(
                f"""
                COPY (
                    SELECT {", ".join(f'CAST("{c}" AS {t}) AS "{c}"' for c, t in RAW_COLUMNS.items())}
                    FROM read_csv('{csv_path}', header=true, all_varchar=true)
                ) TO '{out_path}' (FORMAT parquet, COMPRESSION zstd)
                """
            )
    return con.execute(f"SELECT count(*) FROM '{out_path}'").fetchone()[0]


def run(months: list[tuple[int, int]] | None = None) -> dict:
    config.ensure_dirs()
    months = months or config.MONTHS
    con = duckdb.connect()
    con.execute("SET memory_limit='2GB'; SET threads=4;")
    manifest = {"source": "US DOT BTS Reporting Carrier On-Time Performance (1987-present)",
                "url_template": config.BTS_URL, "files": []}
    for year, month in months:
        zp = config.RAW / f"bts_ontime_{year}_{month}.zip"
        out = config.INTERIM / f"flights_{year}_{month:02d}.parquet"
        if not zp.exists():
            raise FileNotFoundError(f"{zp} missing - run scripts/download_data.py first")
        if not out.exists():
            n = zip_to_parquet(zp, out, con)
        else:
            n = con.execute(f"SELECT count(*) FROM '{out}'").fetchone()[0]
        manifest["files"].append({"year": year, "month": month, "zip": zp.name,
                                  "zip_bytes": zp.stat().st_size, "zip_sha256": sha256(zp),
                                  "rows": int(n)})
        print(f"{year}-{month:02d}: {n:,} rows")
    manifest["total_rows"] = sum(f["rows"] for f in manifest["files"])
    con.close()

    db = duckdb.connect(str(config.DB_PATH))
    db.execute("SET memory_limit='2GB'; CREATE SCHEMA IF NOT EXISTS raw;")
    db.execute(f"""CREATE OR REPLACE TABLE raw.flights AS
                   SELECT * FROM read_parquet('{config.INTERIM}/flights_*.parquet')""")
    _load_reference(db)
    db.close()
    (config.REPORTS / "ingestion_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"total rows: {manifest['total_rows']:,}")
    return manifest


def _load_reference(db: duckdb.DuckDBPyConnection) -> None:
    ap = config.RAW / "ourairports_airports.csv"
    cr = config.RAW / "L_UNIQUE_CARRIERS.csv"
    db.execute(f"""CREATE OR REPLACE TABLE raw.ourairports AS
                   SELECT * FROM read_csv('{ap}', header=true, all_varchar=true)""")
    db.execute(f"""CREATE OR REPLACE TABLE raw.carriers AS
                   SELECT "Code" AS code, "Description" AS name
                   FROM read_csv('{cr}', header=true, all_varchar=true)""")


if __name__ == "__main__":
    run()
