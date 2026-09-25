"""Pre-departure cancellation-risk model (all scheduled flights, ~2% positive).

Imbalance handling: PR-AUC as the selection metric, Optuna searches `scale_pos_weight`
alongside tree parameters; probabilities are re-calibrated (isotonic, validation window) so the
class weighting does not distort the reported risk. Baselines: prior rate, class-weighted
logistic regression.
"""

from __future__ import annotations

import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn import metrics as skm
from sklearn.isotonic import IsotonicRegression

from flightops import config
from flightops.features import ALL_FEATURES, load_split
from flightops.models import common, tabular

NAME = "cancellation_model"


def run(train_rows: int = 500_000, valid_rows: int = 200_000, n_trials: int = 12) -> dict:
    t0 = time.time()
    tr = load_split("train", train_rows)
    va = load_split("valid", valid_rows)
    te = load_split("test")
    feats = ALL_FEATURES
    ytr, yva, yte = (d["cancelled"].astype(int).to_numpy() for d in (tr, va, te))
    print(f"rows train={len(tr):,} valid={len(va):,} test={len(te):,} cancel rate "
          f"train={ytr.mean():.4f} valid={yva.mean():.4f} test={yte.mean():.4f}")
    tracker = common.Tracker("cancellation_model")
    results, probs = {}, {}
    results["prior_rate"] = common.classification_metrics(yte, np.full(len(yte), ytr.mean()), 0.5)

    with tracker.run("logistic_balanced"):
        lr = tabular.logistic_baseline(feats, class_weight="balanced")
        lr.fit(tabular.to_lr_frame(tr, feats), ytr)
        pv = lr.predict_proba(tabular.to_lr_frame(va, feats))[:, 1]
        p = lr.predict_proba(tabular.to_lr_frame(te, feats))[:, 1]
        results["logistic_regression_balanced"] = common.classification_metrics(
            yte, p, common.best_f1_threshold(yva, pv))
        lr_valid_ap = float(skm.average_precision_score(yva, pv))
        results["logistic_regression_balanced"]["valid_pr_auc"] = lr_valid_ap
        lr_iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(pv, yva)
        p_lr_cal, pv_lr_cal = lr_iso.predict(p), lr_iso.predict(pv)
        results["logistic_regression_calibrated"] = common.classification_metrics(
            yte, p_lr_cal, common.best_f1_threshold(yva, pv_lr_cal))
        probs["LogReg (balanced)"] = p
        tracker.log({"class_weight": "balanced"},
                    {f"test_{k}": v for k, v in results["logistic_regression_balanced"].items()})

    with tracker.run("lightgbm_optuna"):
        best, it, study = tabular.tune_lgbm(
            tr[feats], ytr, va[feats], yva, metric="average_precision", n_trials=n_trials,
            timeout=240, extra=None)
        # second, small search dimension for the imbalance weight on top of the tuned trees
        best_ap, best_w, best_it = -1.0, 1.0, it
        dtr = lgb.Dataset(tr[feats], ytr, params={"feature_pre_filter": False})
        dva = lgb.Dataset(va[feats], yva, reference=dtr)
        for w in (1.0, 5.0, 20.0):
            b = lgb.train({**best, "scale_pos_weight": w}, dtr, 600, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(30, verbose=False)])
            ap = b.best_score["valid_0"]["average_precision"]
            print(f"  scale_pos_weight={w}: valid PR-AUC {ap:.4f}")
            if ap > best_ap:
                best_ap, best_w, best_it = ap, w, b.best_iteration
        best["scale_pos_weight"] = best_w
        booster = lgb.train(best, lgb.Dataset(tr[feats], ytr), num_boost_round=best_it)
        pv_raw, pt_raw = booster.predict(va[feats]), booster.predict(te[feats])
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(pv_raw, yva)
        pv, p = iso.predict(pv_raw), iso.predict(pt_raw)
        thr = common.best_f1_threshold(yva, pv)
        results["lightgbm_calibrated"] = common.classification_metrics(yte, p, thr)
        results["lightgbm_uncalibrated"] = common.classification_metrics(
            yte, pt_raw, common.best_f1_threshold(yva, pv_raw))
        results["lightgbm_calibrated"]["valid_pr_auc"] = float(best_ap)
        probs["LightGBM (calibrated)"] = p
        tracker.log({**best, "num_boost_round": best_it},
                    {f"test_{k}": v for k, v in results["lightgbm_calibrated"].items()})

    # precision in the top-risk slice: what a planner would actually look at
    order = np.argsort(-pt_raw)
    for frac in (0.01, 0.05):
        k = int(len(order) * frac)
        results[f"precision_top_{int(frac * 100)}pct"] = {
            "precision": float(yte[order[:k]].mean()), "lift_vs_prior": float(yte[order[:k]].mean() / yte.mean())}

    # model selection on VALIDATION PR-AUC (the test window is only used for reporting)
    selected = "lightgbm" if best_ap >= lr_valid_ap else "logistic_regression"
    print(f"  valid PR-AUC: lightgbm={best_ap:.4f} logreg={lr_valid_ap:.4f} -> serve {selected}")
    out = common.artifact_dir(NAME)
    booster.save_model(str(out / "model.txt"))
    joblib.dump(iso, out / "calibrator.joblib")
    joblib.dump(lr, out / "logreg.joblib", compress=3)
    joblib.dump(lr_iso, out / "logreg_calibrator.joblib")
    p = p if selected == "lightgbm" else p_lr_cal
    tabular.plot_curves(yte, probs, config.FIGURES / "cancellation_curves.png",
                        "Cancellation risk - held-out test (Jun-Jul 2026)")
    pd.DataFrame({"flight_id": te["flight_id"].astype("uint64"), "cancel_prob": p.astype("float32")}
                 ).to_parquet(config.INTERIM / "scores_cancel_test.parquet", index=False)
    meta = common.write_metadata(NAME, {
        "task": "binary classification: flight cancelled (all scheduled flights)",
        "features": feats, "train_rows": len(tr), "valid_rows": len(va), "test_rows": len(te),
        "sampling": f"uniform sample of {len(tr):,} train / {len(va):,} valid scheduled flights; "
                    "test = every scheduled flight in Jun-Jul 2026",
        "lightgbm_params": best, "num_boost_round": best_it, "optuna_trials": len(study.trials),
        "decision_threshold": thr if selected == "lightgbm" else
        results["logistic_regression_calibrated"]["threshold"],
        "selected_model": selected, "selection_rule": "higher validation PR-AUC",
        "test_metrics": results,
        "runtime_seconds": round(time.time() - t0, 1),
    })
    _card(meta)
    print(pd.DataFrame({k: v for k, v in results.items() if "roc_auc" in v}).T[
        ["roc_auc", "pr_auc", "f1", "brier", "positive_rate"]].round(4))
    return meta


def _card(meta: dict) -> None:
    r = meta["test_metrics"]
    rows = "\n".join(f"| {k} | {r[k]['pr_auc']:.4f} | {r[k]['roc_auc']:.4f} | {r[k]['f1']:.4f} | "
                     f"{r[k]['brier']:.4f} |" for k in r if "pr_auc" in r[k])
    common.write_model_card(NAME, "Pre-departure cancellation risk", {
        "Intended use": "Rank scheduled flights by cancellation risk from schedule-time information "
                        "(route / carrier-origin / flight-number history, schedule congestion, calendar).",
        "Data": f"{meta['sampling']}. Cancellation rate in the test window: "
                f"{r['prior_rate']['positive_rate']:.4f} (validation window was much lower).",
        "Imbalance handling": f"PR-AUC used for tuning and selection; Optuna ({meta['optuna_trials']} "
                              f"trials) on tree parameters, then scale_pos_weight in (1, 5, 20) "
                              f"chosen on validation PR-AUC (selected "
                              f"{meta['lightgbm_params']['scale_pos_weight']}). Isotonic "
                              "calibration on validation restores probability scale.",
        "Test results (Jun-Jul 2026)": "| Model | PR-AUC | ROC-AUC | F1 | Brier |\n|---|---|---|---|---|\n"
                                        + rows,
        "Model selection": f"Served model: **{meta['selected_model']}** (higher validation PR-AUC; "
                           "test never used for selection).",
        "Top-risk slice (LightGBM)": f"Top 1% riskiest flights: precision {r['precision_top_1pct']['precision']:.4f} "
                          f"(lift {r['precision_top_1pct']['lift_vs_prior']:.2f}x). Top 5%: "
                          f"{r['precision_top_5pct']['precision']:.4f} "
                          f"(lift {r['precision_top_5pct']['lift_vs_prior']:.2f}x).",
        "Limitations": "Cancellations are driven mostly by same-day weather, ATC programs and IT / "
                       "crew disruptions, none of which are known at scheduling time, so absolute "
                       "PR-AUC is low. The validation window (Apr-May 2026) had an unusually low "
                       "cancellation rate versus test, so calibrated probabilities under-state "
                       "summer risk. On test the class-weighted logistic baseline ranks better than "
                       "the validation-selected LightGBM, and LightGBM's top-1% slice is below the "
                       "base rate - this model is weak and is kept to show the honest ceiling of "
                       "schedule-only cancellation prediction.",
    })


if __name__ == "__main__":
    run()
