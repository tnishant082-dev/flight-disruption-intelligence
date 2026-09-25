"""Shared helpers: metrics, MLflow logging, versioned artifacts, model cards."""

from __future__ import annotations

import json
import subprocess
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn import metrics as skm

from flightops import config

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")

MODEL_VERSION = "v1"


def classification_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (p >= threshold).astype(int)
    return {
        "roc_auc": float(skm.roc_auc_score(y, p)),
        "pr_auc": float(skm.average_precision_score(y, p)),
        "f1": float(skm.f1_score(y, pred)),
        "precision": float(skm.precision_score(y, pred, zero_division=0)),
        "recall": float(skm.recall_score(y, pred)),
        "brier": float(skm.brier_score_loss(y, p)),
        "log_loss": float(skm.log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
        "threshold": float(threshold),
        "positive_rate": float(np.mean(y)),
        "n": int(len(y)),
    }


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    prec, rec, thr = skm.precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    i = int(np.nanargmax(f1[:-1]))
    return float(thr[i])


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {"mae": float(skm.mean_absolute_error(y, pred)),
            "rmse": float(np.sqrt(skm.mean_squared_error(y, pred))),
            "median_ae": float(skm.median_absolute_error(y, pred)),
            "n": int(len(y))}


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=config.ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "uncommitted"


def artifact_dir(name: str) -> Path:
    d = config.MODELS / name / MODEL_VERSION
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default))


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_metadata(name: str, payload: dict) -> dict:
    meta = {"model": name, "version": MODEL_VERSION,
            "trained_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "code_version": git_sha(),
            "train_window": f"2025-08-01..{config.TRAIN_END}",
            "valid_window": f"2026-04-01..{config.VALID_END}",
            "test_window": "2026-06-01..2026-07-31", **payload}
    save_json(meta, artifact_dir(name) / "metadata.json")
    save_json(meta, config.METRICS / f"{name}.json")
    _update_registry(name, meta)
    return meta


def _update_registry(name: str, meta: dict) -> None:
    reg_path = config.MODELS / "registry.json"
    reg = json.loads(reg_path.read_text()) if reg_path.exists() else {}
    reg[name] = {"version": meta["version"], "path": f"models/{name}/{meta['version']}",
                 "trained_at_utc": meta["trained_at_utc"], "code_version": meta["code_version"]}
    save_json(reg, reg_path)


class Tracker:
    """Thin MLflow wrapper (local file store under ./mlruns). No-ops if MLflow is missing."""

    def __init__(self, experiment: str):
        try:
            import os

            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
            os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
            import mlflow

            mlflow.set_tracking_uri(config.MLRUNS.resolve().as_uri())
            mlflow.set_experiment(experiment)
            self.mlflow = mlflow
        except Exception as exc:  # pragma: no cover
            print(f"mlflow disabled: {exc}")
            self.mlflow = None

    def run(self, name: str):
        if self.mlflow is None:
            from contextlib import nullcontext

            return nullcontext()
        return self.mlflow.start_run(run_name=name)

    def log(self, params: dict | None = None, metrics: dict | None = None,
            artifacts: list[Path] | None = None, tags: dict | None = None) -> None:
        if self.mlflow is None:
            return
        if tags:
            self.mlflow.set_tags(tags)
        if params:
            self.mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
        if metrics:
            self.mlflow.log_metrics({k: float(v) for k, v in metrics.items()
                                     if isinstance(v, (int, float, np.floating, np.integer))})
        for a in artifacts or []:
            if Path(a).exists():
                self.mlflow.log_artifact(str(a))


def fmt(v: float, nd: int = 4) -> str:
    return f"{v:.{nd}f}"


def write_model_card(name: str, title: str, sections: dict[str, str]) -> Path:
    lines = [f"# Model card: {title}", ""]
    for head, body in sections.items():
        lines += [f"## {head}", "", body.strip(), ""]
    path = config.MODEL_CARDS / f"{name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path
