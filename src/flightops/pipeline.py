"""End-to-end pipeline: ingest -> SQL (staging, marts) -> data-quality gate -> features ->
models -> Power BI / app exports.

    python -m flightops.pipeline            # everything
    python -m flightops.pipeline --skip-ml  # data + SQL + DQ + exports that do not need models
"""

from __future__ import annotations

import argparse
import time

from flightops import features, ingest, powerbi_export, quality, sqlrunner
from flightops.models import (
    anomaly,
    cancellation,
    clustering,
    delay_classifier,
    delay_regression,
    explain,
    forecasting,
)


def step(name, fn):
    t0 = time.time()
    print(f"\n=== {name}")
    out = fn()
    print(f"=== {name} done in {time.time() - t0:.0f}s")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-ml", action="store_true")
    a = ap.parse_args()
    step("ingest", ingest.run)
    step("sql", sqlrunner.run)
    step("data quality", quality.run)
    step("features", features.build)
    if not a.skip_ml:
        step("delay classifier", delay_classifier.run)
        step("delay regression", delay_regression.run)
        step("cancellation model", cancellation.run)
        step("forecasting", forecasting.run)
        step("segmentation", clustering.run)
        step("anomaly detection", anomaly.run)
        step("shap", explain.run)
        step("power bi export", powerbi_export.run)


if __name__ == "__main__":
    main()
