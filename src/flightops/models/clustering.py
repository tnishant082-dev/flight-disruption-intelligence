"""Operational segmentation of airports and routes (KMeans on standardized profiles).

k is chosen by silhouette score within a range that stays interpretable; each segment gets a
readable label derived from its centroid (size tier + most distinctive trait).
"""

from __future__ import annotations

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from flightops import config
from flightops.models import common

AIRPORT_FEATS = ["log_departures", "destinations", "arr_delay_rate", "cancellation_rate",
                 "avg_taxi_out_min", "avg_distance_mi", "evening_share", "weather_cause_share",
                 "nas_cause_share", "late_aircraft_cause_share", "dep_delay_std_min"]
ROUTE_FEATS = ["log_flights", "distance_mi", "carriers", "on_time_pct", "avg_arr_delay_min",
               "p90_arr_delay_min", "cancellation_rate_pct"]
TRAIT_NAMES = {
    "arr_delay_rate": ("delay-prone", "punctual"),
    "cancellation_rate": ("cancellation-prone", "rarely cancels"),
    "avg_taxi_out_min": ("long taxi-out / congested", "quick turn"),
    "avg_distance_mi": ("long-haul mix", "short-haul mix"),
    "evening_share": ("evening-heavy bank", "morning-heavy bank"),
    "weather_cause_share": ("weather-exposed", "low weather exposure"),
    "nas_cause_share": ("ATC/NAS-constrained", "few NAS delays"),
    "late_aircraft_cause_share": ("knock-on (late aircraft) delays", "few knock-on delays"),
    "dep_delay_std_min": ("volatile delays", "stable delays"),
    "on_time_pct": ("reliable", "unreliable"),
    "avg_arr_delay_min": ("high average delay", "low average delay"),
    "p90_arr_delay_min": ("long delay tail", "short delay tail"),
    "cancellation_rate_pct": ("cancellation-prone", "rarely cancels"),
    "distance_mi": ("long-haul", "short-haul"),
    "carriers": ("multi-carrier", "single-carrier"),
}


def _fit(X: pd.DataFrame, k_range=range(4, 8)) -> tuple[KMeans, StandardScaler, dict]:
    sc = StandardScaler().fit(X)
    Z = sc.transform(X)
    scores = {}
    for k in k_range:
        km = KMeans(k, n_init=20, random_state=config.SEED).fit(Z)
        scores[k] = float(silhouette_score(Z, km.labels_))
    best_k = max(scores, key=scores.get)
    km = KMeans(best_k, n_init=20, random_state=config.SEED).fit(Z)
    return km, sc, scores


def _labels(centers_z: pd.DataFrame, size_col: str, size_names: tuple[str, str, str]) -> dict:
    labels = {}
    for c, row in centers_z.iterrows():
        s = row[size_col]
        tier = size_names[0] if s > 0.8 else size_names[1] if s > -0.3 else size_names[2]
        traits = row.drop([size_col, "destinations"], errors="ignore")
        traits = traits[[t for t in traits.index if t in TRAIT_NAMES]]
        top = traits.abs().sort_values(ascending=False).index[:2]
        words = [TRAIT_NAMES[t][0] if traits[t] > 0 else TRAIT_NAMES[t][1] for t in top]
        labels[c] = f"{tier} - {words[0]}, {words[1]}"
    return labels


def run() -> dict:
    con = duckdb.connect(str(config.DB_PATH), read_only=True)
    ap = con.execute("SELECT p.*, a.airport_name, a.state, a.latitude, a.longitude "
                     "FROM marts.airport_profile p JOIN staging.stg_airports a "
                     "USING (airport_code)").df()
    rt = con.execute("SELECT * FROM marts.kpi_route").df()
    con.close()
    ap["log_departures"] = np.log10(ap["scheduled_departures"])
    rt["log_flights"] = np.log10(rt["scheduled_flights"])
    ap[AIRPORT_FEATS] = ap[AIRPORT_FEATS].fillna(ap[AIRPORT_FEATS].median())
    rt[ROUTE_FEATS] = rt[ROUTE_FEATS].fillna(rt[ROUTE_FEATS].median())

    out = common.artifact_dir("segmentation")
    result = {}
    for name, df, feats, size_col, tiers in (
            ("airports", ap, AIRPORT_FEATS, "log_departures", ("Major hub", "Mid-size", "Regional")),
            ("routes", rt, ROUTE_FEATS, "log_flights", ("Trunk route", "Mid-volume route",
                                                        "Thin route"))):
        km, sc, scores = _fit(df[feats])
        Z = sc.transform(df[feats])
        df["segment_id"] = km.labels_
        cz = pd.DataFrame(km.cluster_centers_, columns=feats)
        lab = _labels(cz, size_col, tiers)
        df["segment"] = df["segment_id"].map(lab)
        pcs = PCA(2, random_state=config.SEED).fit_transform(Z)
        df["pc1"], df["pc2"] = pcs[:, 0], pcs[:, 1]
        joblib.dump({"kmeans": km, "scaler": sc, "features": feats, "labels": lab},
                    out / f"kmeans_{name}.joblib")
        prof = df.groupby(["segment_id", "segment"])[feats].mean()
        prof["members"] = df.groupby(["segment_id", "segment"]).size()
        result[name] = {"k": int(km.n_clusters), "silhouette_by_k": scores,
                        "silhouette": scores[km.n_clusters], "n_items": int(len(df)),
                        "segments": {lab[i]: int((df.segment_id == i).sum()) for i in lab}}
        prof.reset_index().to_csv(config.MARTS / f"segment_profile_{name}.csv", index=False)
        print(name, result[name]["k"], round(result[name]["silhouette"], 3), result[name]["segments"])
    ap.to_csv(config.MARTS / "segments_airports.csv", index=False)
    rt.to_csv(config.MARTS / "segments_routes.csv", index=False)
    meta = common.write_metadata("segmentation", {"task": "KMeans segmentation of airports/routes",
                                                  "airport_features": AIRPORT_FEATS,
                                                  "route_features": ROUTE_FEATS,
                                                  "test_metrics": result})
    body = {}
    for name in ("airports", "routes"):
        r = result[name]
        body[f"{name.title()} segments (k={r['k']}, silhouette {r['silhouette']:.3f})"] = "\n".join(
            f"- **{s}**: {n} {name}" for s, n in r["segments"].items())
    common.write_model_card("segmentation", "Airport and route segmentation", {
        "Intended use": "Group airports / routes with similar operating profiles so network planners "
                        "can compare like with like and target interventions by segment.",
        "Data": f"marts.airport_profile ({result['airports']['n_items']} airports with >= 3,000 "
                f"departures) and marts.kpi_route ({result['routes']['n_items']} routes with >= 300 "
                "flights), full study window.",
        "Method": "StandardScaler + KMeans (n_init=20); k in 4-7 picked by silhouette. Labels are "
                  "derived from the centroid: size tier plus the two most distinctive traits "
                  "(largest absolute z-score).",
        **body,
        "Limitations": "Silhouette scores are modest - operating profiles form a continuum, so "
                       "segments are a descriptive lens, not hard categories. Profiles use outcome "
                       "data from the whole window (descriptive, not predictive).",
    })
    return meta


if __name__ == "__main__":
    run()
