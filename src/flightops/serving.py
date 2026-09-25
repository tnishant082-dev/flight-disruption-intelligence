"""Load versioned artifacts and turn a single scheduled-flight request into model features."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from functools import lru_cache

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from flightops import config
from flightops.features import ALL_FEATURES, CAT_FEATURES, holiday_table, prepare

FEATURE_LABELS = {
    "carrier": "Carrier", "origin": "Origin airport", "dest": "Destination airport",
    "dep_hour": "Scheduled departure hour", "arr_hour": "Scheduled arrival hour",
    "crs_dep_min_of_day": "Departure time of day", "day_of_week": "Day of week",
    "is_weekend": "Weekend", "days_to_holiday": "Days to nearest US holiday",
    "distance_mi": "Distance (mi)", "crs_elapsed_min": "Scheduled block time (min)",
    "sched_speed_mpm": "Schedule tightness (mi per block min)",
    "origin_sched_deps_day": "Origin departures that day",
    "origin_sched_deps_hour": "Origin departures that hour",
    "dest_sched_arrs_hour": "Destination arrivals that hour",
    "carrier_sched_day": "Carrier flights that day",
    "te_route_delay": "Route delay history", "te_route_cancel": "Route cancellation history",
    "te_carrier_origin_delay": "Carrier-at-origin delay history",
    "te_carrier_origin_cancel": "Carrier-at-origin cancellation history",
    "te_flight_delay": "Flight-number delay history",
    "te_flight_cancel": "Flight-number cancellation history",
}


class RequestError(ValueError):
    """Raised for inputs the models cannot score (unknown airport, unknown route length, ...)."""


@dataclass
class FlightInput:
    carrier: str
    origin: str
    dest: str
    flight_date: date
    crs_dep_time: int
    crs_arr_time: int | None = None
    flight_number: str | None = None
    distance_mi: float | None = None
    crs_elapsed_min: float | None = None
    origin_sched_deps_day: float | None = None
    origin_sched_deps_hour: float | None = None
    dest_sched_arrs_hour: float | None = None
    carrier_sched_day: float | None = None


class Artifacts:
    def __init__(self, root=config.MODELS):
        lk = root / "lookups"
        self.categories = json.loads((lk / "categories.json").read_text())
        self.priors = json.loads((lk / "priors.json").read_text())
        self.te = {k: pd.read_parquet(lk / f"te_{k}.parquet").set_index("key")
                   for k in ("route", "carrier_origin", "flight")}
        self.origin_prof = pd.read_parquet(lk / "origin_schedule_profile.parquet")
        self.dest_prof = pd.read_parquet(lk / "dest_schedule_profile.parquet")
        self.carrier_prof = pd.read_parquet(lk / "carrier_schedule_profile.parquet")
        self.route_prof = pd.read_parquet(lk / "route_profile.parquet").set_index("route")
        hol = holiday_table()
        self.holidays = dict(zip(hol["flight_date"], hol["days_to_holiday"]))

        d = root / "delay_classifier" / "v1"
        self.delay_meta = json.loads((d / "metadata.json").read_text())
        self.delay_model = lgb.Booster(model_file=str(d / "model.txt"))
        self.delay_cal = joblib.load(d / "calibrator.joblib")
        self.delay_threshold = float(self.delay_meta["decision_threshold"])

        r = root / "delay_regression" / "v1"
        self.q_models = {q: lgb.Booster(model_file=str(r / f"model_q{q}.txt")) for q in (10, 50, 90)}

        c = root / "cancellation_model" / "v1"
        self.cancel_meta = json.loads((c / "metadata.json").read_text())
        if self.cancel_meta.get("selected_model") == "logistic_regression":
            self.cancel_kind = "logistic_regression"
            self.cancel_model = joblib.load(c / "logreg.joblib")
            self.cancel_cal = joblib.load(c / "logreg_calibrator.joblib")
        else:
            self.cancel_kind = "lightgbm"
            self.cancel_model = lgb.Booster(model_file=str(c / "model.txt"))
            self.cancel_cal = joblib.load(c / "calibrator.joblib")
        self.registry = json.loads((root / "registry.json").read_text())

    # ------------------------------------------------------------------ features
    def build_features(self, f: FlightInput) -> pd.DataFrame:
        carrier, origin, dest = f.carrier.upper(), f.origin.upper(), f.dest.upper()
        for col, val in (("carrier", carrier), ("origin", origin), ("dest", dest)):
            if val not in self.categories[col]:
                raise RequestError(f"unknown {col} '{val}' (not in the BTS study window)")
        if origin == dest:
            raise RequestError("origin and dest must differ")
        hh, mm = divmod(int(f.crs_dep_time), 100)
        if not (0 <= hh <= 24 and 0 <= mm < 60):
            raise RequestError("crs_dep_time must be HHMM")
        dep_min = min(hh * 60 + mm, 1439)
        route = f"{origin}-{dest}"
        rp = self.route_prof.loc[route] if route in self.route_prof.index else None
        distance = f.distance_mi if f.distance_mi is not None else (
            float(rp["distance_mi"]) if rp is not None else None)
        elapsed = f.crs_elapsed_min if f.crs_elapsed_min is not None else (
            float(rp["crs_elapsed_min"]) if rp is not None else None)
        if distance is None or elapsed is None:
            raise RequestError(f"route {route} not flown in training window; pass distance_mi and "
                               "crs_elapsed_min")
        if f.crs_arr_time is not None:
            ah, am = divmod(int(f.crs_arr_time), 100)
            arr_hour = min(ah, 23)
        else:
            # scheduled times are local; this ignores time-zone offset (arr_hour is a coarse feature)
            arr_hour = int(((dep_min + elapsed) // 60) % 24)
        dow = f.flight_date.isoweekday()
        dep_hour = min(hh, 23)

        def prof(df, keys: dict, col: str, override):
            if override is not None:
                return float(override)
            m = df
            for k, v in keys.items():
                m = m[m[k] == v]
            if len(m):
                return float(m[col].mean())
            first = list(keys.items())[0]
            m = df[df[first[0]] == first[1]]
            return float(m[col].median()) if len(m) else float(df[col].median())

        row = {
            "carrier": carrier, "origin": origin, "dest": dest,
            "dep_hour": dep_hour, "arr_hour": arr_hour, "crs_dep_min_of_day": dep_min,
            "day_of_week": dow, "is_weekend": int(dow >= 6),
            "days_to_holiday": int(self.holidays.get(f.flight_date, 30)),
            "distance_mi": distance, "crs_elapsed_min": elapsed,
            "sched_speed_mpm": distance / elapsed if elapsed else np.nan,
            "origin_sched_deps_day": prof(self.origin_prof, {"origin": origin, "day_of_week": dow},
                                          "origin_sched_deps_day", f.origin_sched_deps_day),
            "origin_sched_deps_hour": prof(self.origin_prof, {"origin": origin, "day_of_week": dow,
                                                              "dep_hour": dep_hour},
                                           "origin_sched_deps_hour", f.origin_sched_deps_hour),
            "dest_sched_arrs_hour": prof(self.dest_prof, {"dest": dest, "day_of_week": dow,
                                                          "arr_hour": arr_hour},
                                         "dest_sched_arrs_hour", f.dest_sched_arrs_hour),
            "carrier_sched_day": prof(self.carrier_prof, {"carrier": carrier, "day_of_week": dow},
                                      "carrier_sched_day", f.carrier_sched_day),
        }
        keys = {"route": route, "carrier_origin": f"{carrier}-{origin}",
                "flight": f"{carrier}-{f.flight_number}" if f.flight_number else None}
        for k, key in keys.items():
            t = self.te[k]
            hit = key is not None and key in t.index
            row[f"te_{k}_delay"] = float(t.at[key, "te_delay"]) if hit else self.priors["delay"]
            row[f"te_{k}_cancel"] = float(t.at[key, "te_cancel"]) if hit else self.priors["cancel"]
        return prepare(pd.DataFrame([row])[ALL_FEATURES])

    # ------------------------------------------------------------------ predictions
    def predict_delay(self, X: pd.DataFrame) -> dict:
        raw = float(self.delay_model.predict(X)[0])
        p = float(self.delay_cal.predict([raw])[0])
        q = {k: float(m.predict(X)[0]) for k, m in self.q_models.items()}
        return {"delay_probability": p, "raw_score": raw, "threshold": self.delay_threshold,
                "predicted_delayed": p >= self.delay_threshold, "risk_band": risk_band(p),
                "delay_minutes_p50": q[50], "delay_minutes_p90": max(q[90], q[50])}

    def predict_cancel(self, X: pd.DataFrame) -> dict:
        if self.cancel_kind == "lightgbm":
            raw = float(self.cancel_model.predict(X)[0])
        else:
            Xs = X.copy()
            for c in CAT_FEATURES:
                Xs[c] = Xs[c].astype(str)
            raw = float(self.cancel_model.predict_proba(Xs)[0, 1])
        p = float(self.cancel_cal.predict([raw])[0])
        base = float(self.priors["cancel"])
        return {"cancellation_probability": p, "model": self.cancel_kind,
                "baseline_rate": base, "relative_risk": p / base if base else None}

    def explain_delay(self, X: pd.DataFrame, top_k: int = 8) -> dict:
        contrib = self.delay_model.predict(X, pred_contrib=True)[0]
        base, vals = float(contrib[-1]), contrib[:-1]
        order = np.argsort(-np.abs(vals))[:top_k]
        reasons = []
        for i in order:
            name = ALL_FEATURES[i]
            v = X.iloc[0][name]
            reasons.append({"feature": name, "label": FEATURE_LABELS.get(name, name),
                            "value": v if isinstance(v, str) else (None if pd.isna(v) else float(v)),
                            "shap_log_odds": float(vals[i]),
                            "direction": "increases risk" if vals[i] > 0 else "decreases risk"})
        return {"base_log_odds": base, "reasons": reasons}


def risk_band(p: float) -> str:
    if p < 0.15:
        return "low"
    if p < 0.30:
        return "moderate"
    if p < 0.45:
        return "elevated"
    return "high"


@lru_cache(maxsize=1)
def get_artifacts() -> Artifacts:
    return Artifacts()


def forecast(airport: str, target: str = "arr_delay_rate") -> pd.DataFrame:
    from flightops.models.forecasting import TARGETS, forecast_future

    d = config.MODELS / "forecaster" / "v1"
    cfg = joblib.load(d / "config.joblib")
    if target not in TARGETS:
        raise RequestError(f"target must be one of {list(TARGETS)}")
    if airport not in cfg["airports"]:
        raise RequestError(f"forecasts available for {cfg['airports']}")
    hist = pd.read_parquet(d / "history.parquet")
    model = lgb.Booster(model_file=str(d / f"model_{target}.txt"))
    fut = forecast_future(model, hist, target, cfg["airports"])
    return fut[fut["airport_code"] == airport].reset_index(drop=True), hist[hist["airport_code"] == airport]
