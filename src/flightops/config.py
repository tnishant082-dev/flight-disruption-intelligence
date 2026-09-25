"""Central paths and study-window settings."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("FLIGHTOPS_ROOT", Path(__file__).resolve().parents[2]))

DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"          # per-month parquet (gitignored)
SAMPLE = DATA / "sample"            # small committed sample
MARTS = DATA / "marts"              # small aggregated tables used by the app
POWERBI = DATA / "powerbi"          # star-schema exports for Power BI
REFERENCE = DATA / "reference"      # airport / carrier lookups
DB_PATH = Path(os.environ.get("FLIGHTOPS_DB", DATA / "warehouse" / "flightops.duckdb"))

SQL_DIR = ROOT / "sql"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
METRICS = REPORTS / "metrics"
FIGURES = REPORTS / "figures"
MODEL_CARDS = REPORTS / "model_cards"
MLRUNS = ROOT / "mlruns"

# Study window: 12 most recent months published by BTS when the project was built.
MONTHS: list[tuple[int, int]] = [
    (2025, 8), (2025, 9), (2025, 10), (2025, 11), (2025, 12),
    (2026, 1), (2026, 2), (2026, 3), (2026, 4), (2026, 5), (2026, 6), (2026, 7),
]

# Time-based split used by every supervised model (inclusive date bounds).
TRAIN_END = "2026-03-31"   # Aug 2025 - Mar 2026
VALID_END = "2026-05-31"   # Apr 2026 - May 2026
# test = Jun 2026 - Jul 2026

BTS_URL = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)
OURAIRPORTS_URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"
CARRIER_LOOKUP_URL = "https://www.transtats.bts.gov/Download_Lookup.asp?Y11x72=Y_haVdhR_PNeeVRef"

SEED = 42


def ensure_dirs() -> None:
    for p in (RAW, INTERIM, SAMPLE, MARTS, POWERBI, REFERENCE, DB_PATH.parent, MODELS,
              METRICS, FIGURES, MODEL_CARDS):
        p.mkdir(parents=True, exist_ok=True)
