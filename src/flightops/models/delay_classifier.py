"""Pre-departure arrival-delay classifier (ArrDel15) - completed flights only.

Baselines: prior rate, logistic regression.  Challengers: XGBoost, LightGBM (Optuna-tuned),
plus a day-of-operations LightGBM variant that adds aircraft-rotation features (reported for
context only - those features are not known at scheduling time). The selected model is isotonic-calibrated
on the validation window and its decision threshold is the F1-optimal point on validation.
The test window (Jun-Jul 2026) is used once, for reporting.
"""

from __future__ import annotations

import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

from flightops import config
from flightops.features import ALL_FEATURES, DAY_OF_OPS_FEATURES, load_split
from flightops.models import common, tabular

NAME = "delay_classifier"
TRAIN_ROWS = 500_000
VALID_ROWS = 200_000


def run(train_rows: int = TRAIN_ROWS, valid_rows: int = VALID_ROWS, n_trials: int = 15) -> dict:
    t0 = time.time()
    tr = load_split("train", train_rows, where="completed = 1")
    va = load_split("valid", valid_rows, where="completed = 1")
    te = load_split("test", None, where="completed = 1")
    ytr, yva, yte = (d["arr_del15"].astype(int).to_numpy() for d in (tr, va, te))
    print(f"rows train={len(tr):,} valid={len(va):,} test={len(te):,}  "
          f"delay rate train={ytr.mean():.3f} valid={yva.mean():.3f} test={yte.mean():.3f}")
    feats = ALL_FEATURES
    tracker = common.Tracker("delay_classifier")
    results: dict[str, dict] = {}
    test_probs: dict[str, np.ndarray] = {}

    # --- baseline 0: training prior
    prior = float(ytr.mean())
    results["prior_rate"] = common.classification_metrics(yte, np.full(len(yte), prior), 0.5)

    # --- baseline 1: logistic regression
    with tracker.run("logistic_regression"):
        lr = tabular.logistic_baseline(feats)
        lr.fit(tabular.to_lr_frame(tr, feats), ytr)
        p_va = lr.predict_proba(tabular.to_lr_frame(va, feats))[:, 1]
        thr = common.best_f1_threshold(yva, p_va)
        p = lr.predict_proba(tabular.to_lr_frame(te, feats))[:, 1]
        results["logistic_regression"] = common.classification_metrics(yte, p, thr)
        test_probs["LogReg"] = p
        tracker.log({"model": "logistic_regression", "train_rows": len(tr)},
                    {f"test_{k}": v for k, v in results["logistic_regression"].items()})
    print(f"  logreg done {time.time() - t0:.0f}s  auc={results['logistic_regression']['roc_auc']:.4f}")

    # --- challenger: XGBoost (fixed, sensible params + early stopping)
    with tracker.run("xgboost"):
        params = {"objective": "binary:logistic", "eval_metric": "auc", "tree_method": "hist",
                  "max_depth": 8, "eta": 0.1, "subsample": 0.8, "colsample_bytree": 0.8,
                  "min_child_weight": 20, "max_cat_to_onehot": 8, "nthread": 8,
                  "seed": config.SEED}
        dtr = xgb.DMatrix(tr[feats], ytr, enable_categorical=True)
        dva = xgb.DMatrix(va[feats], yva, enable_categorical=True)
        bst = xgb.train(params, dtr, 600, evals=[(dva, "valid")], early_stopping_rounds=30,
                        verbose_eval=False)
        rng = (0, bst.best_iteration + 1)
        p_va = bst.predict(dva, iteration_range=rng)
        thr = common.best_f1_threshold(yva, p_va)
        p = bst.predict(xgb.DMatrix(te[feats], enable_categorical=True), iteration_range=rng)
        results["xgboost"] = common.classification_metrics(yte, p, thr)
        results["xgboost"]["best_iteration"] = bst.best_iteration
        test_probs["XGBoost"] = p
        tracker.log({**params, "best_iteration": bst.best_iteration},
                    {f"test_{k}": v for k, v in results["xgboost"].items()})
    print(f"  xgb done {time.time() - t0:.0f}s  auc={results['xgboost']['roc_auc']:.4f}")

    # --- challenger: LightGBM + Optuna
    with tracker.run("lightgbm_optuna"):
        best, best_iter, study = tabular.tune_lgbm(tr[feats], ytr, va[feats], yva,
                                                   n_trials=n_trials, timeout=300)
        booster = lgb.train(best, lgb.Dataset(tr[feats], ytr), num_boost_round=best_iter)
        p_va_raw = booster.predict(va[feats])
        p_te_raw = booster.predict(te[feats])
        results["lightgbm_uncalibrated"] = common.classification_metrics(
            yte, p_te_raw, common.best_f1_threshold(yva, p_va_raw))
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p_va_raw, yva)
        p_va = iso.predict(p_va_raw)
        thr = common.best_f1_threshold(yva, p_va)
        p = iso.predict(p_te_raw)
        results["lightgbm_calibrated"] = common.classification_metrics(yte, p, thr)
        results["lightgbm_calibrated"]["valid_roc_auc"] = float(study.best_value)
        test_probs["LightGBM (calibrated)"] = p
        tracker.log({**best, "num_boost_round": best_iter, "optuna_trials": len(study.trials),
                     "train_rows": len(tr)},
                    {f"test_{k}": v for k, v in results["lightgbm_calibrated"].items()})
    print(f"  lgbm done {time.time() - t0:.0f}s  auc={results['lightgbm_calibrated']['roc_auc']:.4f}")

    # --- context only: day-of-operations variant with actual aircraft-rotation features
    b2 = lgb.train(best, lgb.Dataset(tr[DAY_OF_OPS_FEATURES], ytr), num_boost_round=best_iter)
    p2v, p2 = b2.predict(va[DAY_OF_OPS_FEATURES]), b2.predict(te[DAY_OF_OPS_FEATURES])
    results["dayofops_lightgbm_with_rotation"] = common.classification_metrics(
        yte, p2, common.best_f1_threshold(yva, p2v))

    # --- artifacts
    out = common.artifact_dir(NAME)
    booster.save_model(str(out / "model.txt"))
    joblib.dump(iso, out / "calibrator.joblib")
    joblib.dump(lr, out / "logreg_baseline.joblib", compress=3)
    fig = config.FIGURES / "delay_classifier_curves.png"
    tabular.plot_curves(yte, test_probs, fig, "Arrival delay >15 min - held-out test (Jun-Jul 2026)")

    imp = pd.DataFrame({"feature": feats,
                        "gain": booster.feature_importance("gain")}).sort_values("gain", ascending=False)
    imp["gain_share"] = imp["gain"] / imp["gain"].sum()
    imp.to_csv(config.METRICS / "delay_classifier_feature_importance.csv", index=False)

    # monthly test performance (drift view)
    te_m = pd.DataFrame({"month": pd.to_datetime(te["flight_date"]).dt.strftime("%Y-%m"),
                         "y": yte, "p": p})
    by_month = te_m.groupby("month").apply(
        lambda g: pd.Series({"observed_rate": g.y.mean(), "mean_predicted": g.p.mean(),
                             "n": len(g)}), include_groups=False).reset_index()

    # score table for BI / app (all test flights, including the calibrated probability)
    scores = pd.DataFrame({"flight_id": te["flight_id"].astype("uint64"), "delay_prob": p.astype("float32"),
                           "delay_pred": (p >= thr).astype("int8"), "arr_del15": yte.astype("int8")})
    config.INTERIM.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(config.INTERIM / "scores_delay_test.parquet", index=False)

    meta = common.write_metadata(NAME, {
        "task": "binary classification: arrival delay >= 15 min (completed flights)",
        "features": feats, "categorical_features": ["carrier", "origin", "dest"],
        "train_rows": len(tr), "valid_rows": len(va), "test_rows": len(te),
        "sampling": f"uniform reservoir sample of {len(tr):,} train / {len(va):,} valid completed "
                    "flights; test = every completed flight in Jun-Jul 2026",
        "lightgbm_params": best, "num_boost_round": best_iter, "optuna_trials": len(study.trials),
        "decision_threshold": thr, "selected_model": "lightgbm_calibrated",
        "test_metrics": results, "test_by_month": by_month.to_dict(orient="records"),
        "top_features": imp.head(10)[["feature", "gain_share"]].to_dict(orient="records"),
        "runtime_seconds": round(time.time() - t0, 1),
    })
    _card(meta)
    print(pd.DataFrame(results).T[["roc_auc", "pr_auc", "f1", "brier"]].round(4))
    return meta


def _card(meta: dict) -> None:
    r = meta["test_metrics"]
    rows = "\n".join(
        f"| {k} | {v['roc_auc']:.4f} | {v['pr_auc']:.4f} | {v['f1']:.4f} | {v['brier']:.4f} |"
        for k, v in r.items() if not k.startswith("dayofops"))
    d = r["dayofops_lightgbm_with_rotation"]
    top = ", ".join(f"`{t['feature']}`" for t in meta["top_features"][:6])
    common.write_model_card(NAME, "Pre-departure arrival-delay classifier", {
        "Intended use": "Rank scheduled flights by the probability of arriving 15+ minutes late, "
                        "using only information available when the schedule is published. "
                        "Decision support for schedule planners / ops analysts, not an operational "
                        "guarantee.",
        "Data": f"BTS Reporting Carrier On-Time Performance, Aug 2025 - Jul 2026. "
                f"{meta['sampling']}. Train {meta['train_window']}, valid {meta['valid_window']}, "
                f"test {meta['test_window']} (time-based, no shuffling across windows).",
        "Features": "Carrier, origin, destination (native categoricals), scheduled departure hour/minute and arrival hour, "
                    "day of week, weekend flag, distance to nearest US federal holiday, distance, "
                    "scheduled block time, schedule congestion counts at origin/destination "
                    "and out-of-fold "
                    "target encodings of route / carrier-origin / flight number computed on the "
                    "training window only. No actual departure time, taxi time, weather or "
                    "same-day delay information is used.",
        "Model": f"LightGBM (Optuna, {meta['optuna_trials']} trials, {meta['num_boost_round']} "
                 f"rounds) + isotonic calibration fit on validation. Threshold "
                 f"{meta['decision_threshold']:.4f} = F1-optimal on validation.",
        "Test results (Jun-Jul 2026)": "| Model | ROC-AUC | PR-AUC | F1 | Brier |\n|---|---|---|---|---|\n"
                                        + rows,
        "Most influential features": top,
        "Day-of-operations comparison (not schedule-time)": (
            "Adding aircraft-rotation features built from the tail number BTS reports "
            "(leg of day, legs per day, turn time) lifts test ROC-AUC to "
            f"{d['roc_auc']:.4f} / PR-AUC {d['pr_auc']:.4f}. Those tails are the aircraft that "
            "actually flew, so this variant is excluded from the served model and shown only to "
            "quantify what same-day aircraft information would add."),
        "Limitations": "Test months (Jun-Jul) are summer thunderstorm season and have a higher delay "
                       "rate than any training month except August 2025, so calibration drifts "
                       "(see test_by_month in reports/metrics). Month-of-year is deliberately not a "
                       "feature (nor day-of-month) because the test months are never seen in training. "
                       "Discrimination "
                       "is moderate: most delay variance comes from same-day weather and "
                       "knock-on effects that are unknowable at scheduling time.",
    })


if __name__ == "__main__":
    run()
