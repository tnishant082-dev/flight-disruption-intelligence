"""Shared tabular-model building blocks (LightGBM tuning, logistic baseline, plots)."""

from __future__ import annotations

import time

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import optuna  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn import metrics as skm  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

from flightops import config  # noqa: E402
from flightops.features import CAT_FEATURES  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)


def logistic_baseline(features: list[str], class_weight=None) -> Pipeline:
    cats = [c for c in CAT_FEATURES if c in features] + ["dep_hour", "day_of_week"]
    nums = [f for f in features if f not in cats]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=200), cats),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), nums),
    ])
    return Pipeline([("pre", pre),
                     ("lr", LogisticRegression(max_iter=300, C=1.0, class_weight=class_weight))])


def to_lr_frame(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    X = df[features].copy()
    for c in CAT_FEATURES:
        if c in X:
            X[c] = X[c].astype(str)
    return X


def tune_lgbm(Xtr, ytr, Xva, yva, objective: str = "binary", metric: str = "auc",
              n_trials: int = 15, timeout: int = 300, extra: dict | None = None,
              maximize: bool = True) -> tuple[dict, int, optuna.Study]:
    dtr = lgb.Dataset(Xtr, ytr, free_raw_data=False, params={"feature_pre_filter": False})
    dva = lgb.Dataset(Xva, yva, reference=dtr, free_raw_data=False)

    def objective_fn(trial: optuna.Trial) -> float:
        params = {
            "objective": objective, "metric": metric, "verbosity": -1, "seed": config.SEED,
            "learning_rate": 0.08, "num_threads": 8,
            "num_leaves": trial.suggest_int("num_leaves", 31, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 400, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": 1,
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 30.0, log=True),
            "cat_smooth": trial.suggest_float("cat_smooth", 5.0, 100.0, log=True),
            "max_cat_to_onehot": 8,
            **(extra or {}),
        }
        booster = lgb.train(params, dtr, num_boost_round=600, valid_sets=[dva],
                            callbacks=[lgb.early_stopping(30, verbose=False)])
        trial.set_user_attr("best_iteration", booster.best_iteration)
        return booster.best_score["valid_0"][metric]

    study = optuna.create_study(direction="maximize" if maximize else "minimize",
                                sampler=optuna.samplers.TPESampler(seed=config.SEED))
    t0 = time.time()
    study.optimize(objective_fn, n_trials=n_trials, timeout=timeout)
    print(f"  optuna: {len(study.trials)} trials in {time.time() - t0:.0f}s, "
          f"best {metric}={study.best_value:.4f}")
    best = {"objective": objective, "metric": metric, "verbosity": -1, "seed": config.SEED,
            "learning_rate": 0.08, "num_threads": 8, "bagging_freq": 1, "max_cat_to_onehot": 8,
            **study.best_params, **(extra or {})}
    return best, int(study.best_trial.user_attrs["best_iteration"]), study


def plot_curves(y: np.ndarray, probs: dict[str, np.ndarray], path, title: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for name, p in probs.items():
        fpr, tpr, _ = skm.roc_curve(y, p)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC {skm.roc_auc_score(y, p):.3f})")
        prec, rec, _ = skm.precision_recall_curve(y, p)
        axes[1].plot(rec, prec, label=f"{name} (AP {skm.average_precision_score(y, p):.3f})")
        frac, mean = calibration_curve(y, p, n_bins=15, strategy="quantile")
        axes[2].plot(mean, frac, marker="o", ms=3, label=name)
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[1].axhline(np.mean(y), color="k", ls="--", lw=0.8, label="prevalence")
    axes[2].plot([0, 1], [0, 1], "k--", lw=0.8)
    for ax, t in zip(axes, ["ROC", "Precision-Recall", "Reliability (test)"]):
        ax.set_title(t)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[2].set_xlabel("Predicted probability")
    axes[2].set_ylabel("Observed frequency")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
