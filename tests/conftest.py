import sys
from pathlib import Path

import duckdb
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from flightops.ingest import RAW_COLUMNS  # noqa: E402


@pytest.fixture(scope="session")
def sample_db():
    """In-memory warehouse built from the committed 20k-row BTS sample + reference subsets."""
    con = duckdb.connect()
    con.execute("CREATE SCHEMA raw")
    casts = ", ".join(f'CAST("{c}" AS {t}) AS "{c}"' for c, t in RAW_COLUMNS.items())
    con.execute(f"""CREATE TABLE raw.flights AS SELECT {casts}
                    FROM read_csv('{ROOT}/data/sample/bts_ontime_sample.csv', header=true,
                                  all_varchar=true)""")
    con.execute(f"""CREATE TABLE raw.ourairports AS SELECT * FROM
                    read_csv('{ROOT}/data/reference/ourairports_subset.csv', header=true,
                             all_varchar=true)""")
    con.execute(f"""CREATE TABLE raw.carriers AS SELECT "Code" AS code, "Description" AS name
                    FROM read_csv('{ROOT}/data/reference/carriers_subset.csv', header=true,
                                  all_varchar=true)""")
    for layer in ("staging", "marts"):
        for p in sorted((ROOT / "sql" / layer).glob("*.sql")):
            con.execute(p.read_text())
    yield con
    con.close()
