"""Disruption-day detection.

1. Network level: STL decomposition (weekly period) of the logit of the daily arrival-delay rate
   and of the daily cancellation rate; days with robust (median/MAD) residual z-score > 3.5 are
   flagged. The logit keeps a jump from 1% to 3% cancellations comparable to 20% -> 40% delays.
2. Airport level: Isolation Forest on airport-day operating features for the 30 busiest
   airports (contamination 2%); features are relative to each airport's own median so large and
   small airports are comparable.
"""

from __future__ import annotations

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import STL

from flightops import config
from flightops.models import common

Z_THRESHOLD = 3.5
IF_FEATS = ["arr_delay_rate", "cancellation_rate", "avg_dep_delay_min", "avg_taxi_out_min",
            "volume_ratio", "weather_share"]


def robust_z(x: pd.Series) -> pd.Series:
    med = x.median()
    mad = (x - med).abs().median() * 1.4826
    return (x - med) / (mad if mad > 0 else 1.0)


def run() -> dict:
    con = duckdb.connect(str(config.DB_PATH), read_only=True)
    net = con.execute("SELECT * FROM marts.daily_network ORDER BY flight_date").df()
    top = [r[0] for r in con.execute("SELECT airport_code FROM marts.daily_airport GROUP BY 1 "
                                     "ORDER BY sum(scheduled_departures) DESC LIMIT 30").fetchall()]
    ad = con.execute("SELECT * FROM marts.daily_airport WHERE airport_code IN "
                     f"({', '.join(repr(a) for a in top)})").df()
    con.close()

    net = net.set_index(pd.to_datetime(net["flight_date"]))
    res = {}
    for col in ("arr_delay_rate", "cancellation_rate"):
        x = np.log(net[col].clip(1e-4, 1 - 1e-4) / (1 - net[col].clip(1e-4, 1 - 1e-4)))
        stl = STL(x, period=7, robust=True).fit()
        net[f"{col}_trend"] = stl.trend
        net[f"{col}_resid"] = stl.resid
        net[f"{col}_z"] = robust_z(stl.resid)
    net["is_anomaly"] = ((net["arr_delay_rate_z"] > Z_THRESHOLD)
                         | (net["cancellation_rate_z"] > Z_THRESHOLD)).astype(int)
    net["anomaly_driver"] = np.where(net["cancellation_rate_z"] > Z_THRESHOLD,
                                     np.where(net["arr_delay_rate_z"] > Z_THRESHOLD,
                                              "delays + cancellations", "cancellations"),
                                     np.where(net["arr_delay_rate_z"] > Z_THRESHOLD, "delays", ""))
    net = net.reset_index(drop=True)
    net.to_csv(config.MARTS / "anomalies_network.csv", index=False)
    res["network_days_flagged"] = int(net["is_anomaly"].sum())
    res["network_days"] = int(len(net))

    ad["volume_ratio"] = ad["scheduled_departures"] / ad.groupby("airport_code")[
        "scheduled_departures"].transform("median")
    ad["weather_share"] = ad["weather_share"].fillna(0)
    rel = ad.copy()
    for c in ("arr_delay_rate", "cancellation_rate", "avg_dep_delay_min", "avg_taxi_out_min"):
        rel[c] = ad[c] - ad.groupby("airport_code")[c].transform("median")
    iso = IsolationForest(n_estimators=300, contamination=0.02, random_state=config.SEED)
    X = rel[IF_FEATS].fillna(0)
    iso.fit(X)
    ad["anomaly_score"] = -iso.score_samples(X)
    ad["is_anomaly"] = (iso.predict(X) == -1).astype(int)
    ad.to_csv(config.MARTS / "anomalies_airport.csv", index=False)
    out = common.artifact_dir("anomaly_detector")
    joblib.dump({"model": iso, "features": IF_FEATS, "airports": top}, out / "isolation_forest.joblib")
    res["airport_days"] = int(len(ad))
    res["airport_days_flagged"] = int(ad["is_anomaly"].sum())
    flagged = ad[ad.is_anomaly == 1]
    res["flagged_airport_days_delay_rate_mean"] = float(flagged["arr_delay_rate"].mean())
    res["normal_airport_days_delay_rate_mean"] = float(ad[ad.is_anomaly == 0]["arr_delay_rate"].mean())
    res["flagged_airport_days_cancel_rate_mean"] = float(flagged["cancellation_rate"].mean())
    res["normal_airport_days_cancel_rate_mean"] = float(ad[ad.is_anomaly == 0]["cancellation_rate"].mean())
    top_net = net[net.is_anomaly == 1].sort_values("cancellation_rate", ascending=False)
    res["top_network_anomalies"] = [
        {"date": str(r.flight_date)[:10], "driver": r.anomaly_driver,
         "arr_delay_rate": round(float(r.arr_delay_rate), 4),
         "cancellation_rate": round(float(r.cancellation_rate), 4)} for r in top_net.itertuples()]
    meta = common.write_metadata("anomaly_detector", {"task": "disruption-day detection",
                                                      "method": "STL robust residual z-score + "
                                                                "Isolation Forest",
                                                      "test_metrics": res})
    tbl = "\n".join(f"| {d['date']} | {d['driver']} | {d['arr_delay_rate']:.4f} | "
                    f"{d['cancellation_rate']:.4f} |" for d in res["top_network_anomalies"][:15])
    common.write_model_card("anomaly_detector", "Disruption-day detector", {
        "Intended use": "Flag network-wide and airport-level disruption days for after-action "
                        "review and to exclude / annotate them in KPI reporting.",
        "Method": f"STL (period 7, robust) on the logit of daily network delay and cancellation rates, robust "
                  f"z-score of residual > {Z_THRESHOLD}. Isolation Forest (300 trees, "
                  "contamination 0.02) on airport-day features relative to the airport's own "
                  "median, for the 30 busiest airports.",
        "Results": f"Network: {res['network_days_flagged']} of {res['network_days']} days flagged. "
                   f"Airport-days: {res['airport_days_flagged']} of {res['airport_days']} flagged; "
                   f"flagged days average a {res['flagged_airport_days_delay_rate_mean']:.3f} delay "
                   f"rate and {res['flagged_airport_days_cancel_rate_mean']:.3f} cancellation rate "
                   f"vs {res['normal_airport_days_delay_rate_mean']:.3f} / "
                   f"{res['normal_airport_days_cancel_rate_mean']:.4f} on other days.",
        "Flagged network days (by cancellation rate)": "| Date | Driver | Delay rate | Cancel rate |"
                                                        "\n|---|---|---|---|\n" + tbl,
        "Limitations": "Unsupervised - there is no labelled ground truth of disruption days, so "
                       "the thresholds (z > 3.5, 2% contamination) are judgement calls. Causes "
                       "(storms, ATC outages, IT failures) are not in the data and must be "
                       "attributed from external sources.",
    })
    print(res["network_days_flagged"], res["airport_days_flagged"])
    print(pd.DataFrame(res["top_network_anomalies"]).head(20).to_string())
    return meta


if __name__ == "__main__":
    run()
