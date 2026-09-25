"""Run the SQL layer (staging then marts) against the DuckDB warehouse."""

from __future__ import annotations

import time
from pathlib import Path

import duckdb

from flightops import config


def run_folder(con: duckdb.DuckDBPyConnection, folder: Path) -> None:
    for path in sorted(folder.glob("*.sql")):
        t0 = time.time()
        con.execute(path.read_text())
        print(f"  {path.relative_to(config.ROOT)}  ({time.time() - t0:.1f}s)")


def run(layers: tuple[str, ...] = ("staging", "marts")) -> None:
    con = duckdb.connect(str(config.DB_PATH))
    con.execute("SET memory_limit='2GB'; SET threads=4;")
    for layer in layers:
        print(f"[{layer}]")
        run_folder(con, config.SQL_DIR / layer)
    con.close()


if __name__ == "__main__":
    import sys

    run(tuple(sys.argv[1:]) or ("staging", "marts"))
