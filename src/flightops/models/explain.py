"""SHAP explanations for the delay classifier (global on a test sample + local examples)."""

from __future__ import annotations

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from flightops import config  # noqa: E402
from flightops.features import ALL_FEATURES, load_split  # noqa: E402
from flightops.models import common  # noqa: E402


def run(n: int = 3000) -> dict:
    booster = lgb.Booster(model_file=str(config.MODELS / "delay_classifier" / "v1" / "model.txt"))
    te = load_split("test", n, where="completed = 1")
    X = te[ALL_FEATURES]
    explainer = shap.TreeExplainer(booster)
    sv = explainer.shap_values(X)
    sv = sv[1] if isinstance(sv, list) else sv
    glob = pd.DataFrame({"feature": ALL_FEATURES, "mean_abs_shap": np.abs(sv).mean(axis=0)}
                        ).sort_values("mean_abs_shap", ascending=False)
    glob.to_csv(config.METRICS / "delay_classifier_shap_global.csv", index=False)

    plt.figure()
    shap.summary_plot(sv, X, show=False, max_display=15, plot_size=(9, 6))
    plt.title("SHAP - delay classifier (log-odds), 3,000 test flights")
    plt.tight_layout()
    plt.savefig(config.FIGURES / "shap_beeswarm_delay.png", dpi=110)
    plt.close("all")
    plt.figure(figsize=(8, 5.5))
    g = glob.head(15)[::-1]
    plt.barh(g["feature"], g["mean_abs_shap"], color="#118DFF")
    plt.xlabel("mean |SHAP| (log-odds)")
    plt.title("Global feature impact - delay classifier")
    plt.tight_layout()
    plt.savefig(config.FIGURES / "shap_bar_delay.png", dpi=110)
    plt.close("all")

    # consistency check: LightGBM's native pred_contrib (used by the API) matches shap
    contrib = booster.predict(X.iloc[:200], pred_contrib=True)[:, :-1]
    max_diff = float(np.abs(contrib - sv[:200]).max())
    res = {"n_explained": int(len(X)), "top_features": glob.head(10).to_dict(orient="records"),
           "max_abs_diff_vs_native_pred_contrib": max_diff}
    common.save_json(res, config.METRICS / "delay_classifier_shap.json")
    print(glob.head(10).to_string(index=False), "\nmax diff native vs shap:", max_diff)
    return res


if __name__ == "__main__":
    run()
