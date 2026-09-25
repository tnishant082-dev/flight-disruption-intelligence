"""Arrival-delay minutes with quantile LightGBM (P10 / P50 / P90) - completed flights only.

Baselines: global training median, route-level training median (fallback: global).
P50 is the point forecast (MAE-optimal); P90 is the planning buffer.
"""

from __future__ import annotations

import time

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

from flightops import config
from flightops.features import ALL_FEATURES, load_split
from flightops.models import common

NAME = "delay_regression"
QUANTILES = (0.1, 0.5, 0.9)
BASE_PARAMS = {"objective": "quantile", "learning_rate": 0.1, "num_leaves": 127,
               "min_child_samples": 100, "feature_fraction": 0.8, "bagging_fraction": 0.8,
               "bagging_freq": 1, "max_cat_to_onehot": 8, "verbosity": -1, "num_threads": 8,
               "seed": config.SEED}


def run(train_rows: int = 400_000, valid_rows: int = 150_000) -> dict:
    t0 = time.time()
    tr = load_split("train", train_rows, where="completed = 1")
    va = load_split("valid", valid_rows, where="completed = 1")
    te = load_split("test", None, where="completed = 1")
    feats = ALL_FEATURES
    ytr, yva, yte = (d["arr_delay_min"].to_numpy(dtype=float) for d in (tr, va, te))
    tracker = common.Tracker("delay_regression")
    results, preds, models = {}, {}, {}

    # baselines (fit on the full training window, not the sample)
    con = duckdb.connect(str(config.DB_PATH), read_only=True)
    g_med = con.execute("SELECT median(arr_delay_min) FROM features.flight_features "
                        "WHERE split='train' AND completed=1").fetchone()[0]
    route_med = con.execute("SELECT route, median(arr_delay_min) AS m FROM features.flight_features "
                            "WHERE split='train' AND completed=1 GROUP BY 1").df()
    test_route = con.execute("SELECT flight_id, route FROM features.flight_features "
                             "WHERE split='test' AND completed=1").df()
    con.close()
    rmap = dict(zip(route_med.route, route_med.m))
    tr_route = te[["flight_id"]].merge(test_route, on="flight_id", how="left")["route"]
    base_route = tr_route.map(rmap).fillna(g_med).to_numpy(dtype=float)
    results["baseline_global_median"] = common.regression_metrics(yte, np.full(len(yte), g_med))
    results["baseline_route_median"] = common.regression_metrics(yte, base_route)
    results["baseline_global_mean"] = common.regression_metrics(
        yte, np.full(len(yte), float(np.mean(ytr))))

    out = common.artifact_dir(NAME)
    for q in QUANTILES:
        with tracker.run(f"lgbm_q{int(q * 100)}"):
            params = {**BASE_PARAMS, "alpha": q, "metric": "quantile"}
            dtr = lgb.Dataset(tr[feats], ytr)
            dva = lgb.Dataset(va[feats], yva, reference=dtr)
            b = lgb.train(params, dtr, num_boost_round=500, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(30, verbose=False)])
            models[q] = b
            preds[q] = b.predict(te[feats], num_iteration=b.best_iteration)
            b.save_model(str(out / f"model_q{int(q * 100)}.txt"), num_iteration=b.best_iteration)
            loss = float(np.mean(np.maximum(q * (yte - preds[q]), (q - 1) * (yte - preds[q]))))
            tracker.log({**params, "best_iteration": b.best_iteration},
                        {"test_pinball": loss})
            results[f"lgbm_q{int(q * 100)}_pinball"] = {"pinball_loss": loss,
                                                       "best_iteration": b.best_iteration}
    p50, p10, p90 = preds[0.5], preds[0.1], preds[0.9]
    results["lgbm_p50"] = common.regression_metrics(yte, p50)
    # pinball loss for the global-quantile baselines, for a like-for-like comparison
    for q in QUANTILES:
        bq = float(np.quantile(ytr, q))
        results[f"baseline_q{int(q * 100)}_pinball"] = {
            "pinball_loss": float(np.mean(np.maximum(q * (yte - bq), (q - 1) * (yte - bq))))}
    results["interval"] = {
        "p90_coverage": float(np.mean(yte <= p90)),
        "p10_p90_coverage": float(np.mean((yte >= p10) & (yte <= p90))),
        "mean_p10_p90_width_min": float(np.mean(p90 - p10)),
        "quantile_crossing_share": float(np.mean(p90 < p50)),
    }
    config.INTERIM.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"flight_id": te["flight_id"].astype("uint64"),
                  "delay_p50": p50.astype("float32"), "delay_p90": p90.astype("float32")}
                 ).to_parquet(config.INTERIM / "scores_delay_minutes_test.parquet", index=False)

    meta = common.write_metadata(NAME, {
        "task": "quantile regression of arrival delay minutes (completed flights)",
        "features": feats, "train_rows": len(tr), "valid_rows": len(va), "test_rows": len(te),
        "sampling": f"uniform sample of {len(tr):,} train / {len(va):,} valid completed flights; "
                    "test = every completed flight in Jun-Jul 2026",
        "params": BASE_PARAMS, "quantiles": list(QUANTILES), "test_metrics": results,
        "baseline_global_median_value": g_med, "runtime_seconds": round(time.time() - t0, 1),
    })
    _card(meta)
    print(pd.DataFrame({k: v for k, v in results.items() if "mae" in v}).T[["mae", "rmse", "median_ae"]])
    print(results["interval"])
    return meta


def _card(meta: dict) -> None:
    r = meta["test_metrics"]
    rows = "\n".join(f"| {k} | {r[k]['mae']:.2f} | {r[k]['rmse']:.2f} | {r[k]['median_ae']:.2f} |"
                     for k in ("baseline_global_median", "baseline_global_mean",
                               "baseline_route_median", "lgbm_p50"))
    pin = "\n".join(f"| P{int(q * 100)} | {r[f'baseline_q{int(q * 100)}_pinball']['pinball_loss']:.3f} | "
                    f"{r[f'lgbm_q{int(q * 100)}_pinball']['pinball_loss']:.3f} |" for q in QUANTILES)
    iv = r["interval"]
    common.write_model_card(NAME, "Arrival-delay minutes (quantile LightGBM)", {
        "Intended use": "Give a typical (P50) and a pessimistic (P90) arrival delay for a scheduled "
                        "flight, e.g. to size connection buffers. Same pre-departure feature set as "
                        "the delay classifier.",
        "Data": f"{meta['sampling']}. Target = BTS ArrDelay (minutes, negative = early).",
        "Model": "Three LightGBM models with quantile loss (alpha 0.1 / 0.5 / 0.9), early stopping "
                 "on the validation window.",
        "Test results - point error (minutes)": "| Model | MAE | RMSE | Median AE |\n|---|---|---|---|\n"
                                                + rows,
        "Test results - pinball loss (lower is better)": "| Quantile | Global-quantile baseline | "
                                                         "LightGBM |\n|---|---|---|\n" + pin,
        "Interval quality": f"Share of test flights at or below P90: {iv['p90_coverage']:.4f} "
                            f"(target 0.90). P10-P90 coverage: {iv['p10_p90_coverage']:.4f} (target "
                            f"0.80). Mean P10-P90 width: {iv['mean_p10_p90_width_min']:.1f} min.",
        "Limitations": "Delay minutes are heavy-tailed (a few flights are hours late), so RMSE is "
                       "dominated by extreme events no schedule-time model can foresee; the P50 "
                       "model improves MAE only modestly over a median baseline. Summer test "
                       "months are more delayed than training, so P90 under-covers.",
    })


if __name__ == "__main__":
    run()
