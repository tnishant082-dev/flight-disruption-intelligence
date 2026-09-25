"""Daily forecasting for the busiest airports: departures volume and arrival-delay rate.

Direct 1-7 day-ahead global LightGBM (one model per target across airports). The model learns
the correction to the seasonal-naive value (target = y - y[t-7]); trees cannot extrapolate
levels, so modelling the residual keeps the naive forecast as the fallback. Every feature is
at least 7 days old relative to the target date, so a forecast issued on day T for T+1..T+7 only
uses data up to T. Evaluated with rolling-origin backtesting against:
  * seasonal naive  (same weekday last week)
  * 4-week same-weekday mean
"""

from __future__ import annotations

import time

import duckdb
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from flightops import config
from flightops.features import holiday_table
from flightops.models import common

NAME = "forecaster"
TOP_N = 10
H = 7
TARGETS = {"scheduled_departures": "Departures / day", "arr_delay_rate": "Arrival-delay rate"}
FEATS = ["airport_idx", "dow", "days_to_holiday", "lag7", "lag14", "lag21", "lag28",
         "roll7_lag7", "roll28_lag7", "std28_lag7"]
PARAMS = {"objective": "l2", "learning_rate": 0.05, "num_leaves": 15, "min_child_samples": 10,
          "feature_fraction": 0.9, "verbosity": -1, "seed": config.SEED, "num_threads": 4}
N_ROUNDS = 300


def load_series(top_n: int = TOP_N) -> pd.DataFrame:
    con = duckdb.connect(str(config.DB_PATH), read_only=True)
    top = [r[0] for r in con.execute(
        "SELECT airport_code FROM marts.daily_airport GROUP BY 1 "
        f"ORDER BY sum(scheduled_departures) DESC LIMIT {top_n}").fetchall()]
    df = con.execute("SELECT flight_date, airport_code, scheduled_departures, arr_delay_rate "
                     "FROM marts.daily_airport WHERE airport_code IN "
                     f"({', '.join(repr(a) for a in top)}) ORDER BY 2, 1").df()
    con.close()
    df["flight_date"] = pd.to_datetime(df["flight_date"])
    return df


def make_features(df: pd.DataFrame, target: str, airports: list[str]) -> pd.DataFrame:
    hol = holiday_table()
    hol["flight_date"] = pd.to_datetime(hol["flight_date"])
    out = []
    for a, g in df.groupby("airport_code"):
        g = g.set_index("flight_date").asfreq("D")
        g["airport_code"] = a
        y = g[target]
        f = pd.DataFrame(index=g.index)
        f["y"] = y
        for k in (7, 14, 21, 28):
            f[f"lag{k}"] = y.shift(k)
        s7 = y.shift(7)
        f["roll7_lag7"] = s7.rolling(7).mean()
        f["roll28_lag7"] = s7.rolling(28).mean()
        f["std28_lag7"] = s7.rolling(28).std()
        f["airport_code"] = a
        out.append(f.reset_index())
    X = pd.concat(out, ignore_index=True)
    X["airport_idx"] = X["airport_code"].map({a: i for i, a in enumerate(airports)})
    X["dow"] = X["flight_date"].dt.dayofweek
    X = X.merge(hol[["flight_date", "days_to_holiday"]], on="flight_date", how="left")
    return X


def backtest(df: pd.DataFrame, target: str, airports: list[str], n_folds: int = 8,
             step: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = make_features(df, target, airports).dropna(subset=["lag28", "roll28_lag7"])
    last = X["flight_date"].max()
    origins = [last - pd.Timedelta(days=H + step * i) for i in range(n_folds)][::-1]
    rows = []
    for o in origins:
        train = X[X["flight_date"] <= o].dropna(subset=["y"])
        test = X[(X["flight_date"] > o) & (X["flight_date"] <= o + pd.Timedelta(days=H))]
        m = lgb.train(PARAMS, lgb.Dataset(train[FEATS], train["y"] - train["lag7"],
                                          categorical_feature=["airport_idx"]), N_ROUNDS)
        pred = test["lag7"].values + m.predict(test[FEATS])
        rows.append(pd.DataFrame({
            "origin": o, "flight_date": test["flight_date"].values,
            "airport_code": test["airport_code"].values, "y": test["y"].values,
            "lightgbm": pred, "seasonal_naive": test["lag7"].values,
            "mean_4wk_same_weekday": test[["lag7", "lag14", "lag21", "lag28"]].mean(axis=1).values,
        }))
    bt = pd.concat(rows, ignore_index=True).dropna(subset=["y"])
    summ = []
    for model in ("lightgbm", "seasonal_naive", "mean_4wk_same_weekday"):
        err = bt[model] - bt["y"]
        rec = {"model": model, "mae": float(err.abs().mean()),
               "rmse": float(np.sqrt((err ** 2).mean()))}
        rec["mape_pct"] = float((err.abs() / bt["y"].abs().clip(lower=1e-9)).mean() * 100)
        summ.append(rec)
    return bt, pd.DataFrame(summ)


def run() -> dict:
    t0 = time.time()
    df = load_series()
    airports = sorted(df["airport_code"].unique())
    tracker = common.Tracker("forecasting")
    out = common.artifact_dir(NAME)
    results, backtests, future = {}, [], []
    for target in TARGETS:
        with tracker.run(f"lgbm_{target}"):
            bt, summ = backtest(df, target, airports)
            results[target] = summ.set_index("model").to_dict(orient="index")
            bt["target"] = target
            backtests.append(bt)
            X = make_features(df, target, airports).dropna(subset=["lag28", "roll28_lag7", "y"])
            m = lgb.train(PARAMS, lgb.Dataset(X[FEATS], X["y"] - X["lag7"],
                                              categorical_feature=["airport_idx"]), N_ROUNDS)
            m.save_model(str(out / f"model_{target}.txt"))
            fut = forecast_future(m, df, target, airports)
            fut["target"] = target
            future.append(fut)
            tracker.log({**PARAMS, "target": target, "horizon": H},
                        {f"{k}_{mm}": v for k, d in results[target].items()
                         for mm, v in d.items()})
        print(target)
        print(summ.round(4).to_string(index=False))
    history = df[df["flight_date"] > df["flight_date"].max() - pd.Timedelta(days=40)]
    history.to_parquet(out / "history.parquet", index=False)
    joblib.dump({"airports": airports, "features": FEATS, "horizon": H}, out / "config.joblib")
    config.MARTS.mkdir(parents=True, exist_ok=True)
    pd.concat(backtests).to_csv(config.MARTS / "forecast_backtest.csv", index=False)
    pd.concat(future).to_csv(config.MARTS / "forecast_next7.csv", index=False)
    df.to_csv(config.MARTS / "forecast_history.csv", index=False)
    meta = common.write_metadata(NAME, {
        "task": "direct 1-7 day ahead daily forecast (departures, arrival-delay rate)",
        "airports": airports, "backtest": "8 rolling origins, weekly step, 7-day horizon, "
                                         "expanding training window (last 8 weeks of data)",
        "params": PARAMS, "num_boost_round": N_ROUNDS, "features": FEATS,
        "test_metrics": results, "runtime_seconds": round(time.time() - t0, 1)})
    _card(meta)
    return meta


def forecast_future(model: lgb.Booster, df: pd.DataFrame, target: str,
                    airports: list[str]) -> pd.DataFrame:
    last = df["flight_date"].max()
    ext = []
    for a in airports:
        for d in range(1, H + 1):
            ext.append({"flight_date": last + pd.Timedelta(days=d), "airport_code": a,
                        target: np.nan})
    full = pd.concat([df[["flight_date", "airport_code", target]], pd.DataFrame(ext)])
    X = make_features(full, target, airports)
    X = X[X["flight_date"] > last]
    return pd.DataFrame({"flight_date": X["flight_date"].values,
                         "airport_code": X["airport_code"].values,
                         "forecast": X["lag7"].values + model.predict(X[FEATS])})


def _card(meta: dict) -> None:
    secs = {"Intended use": "Short-range (1-7 day) planning view of departures and arrival-delay "
                            "rate at the 10 busiest origin airports.",
            "Data": "marts.daily_airport (BTS, Aug 2025 - Jul 2026). Airports: "
                    + ", ".join(meta["airports"]) + ".",
            "Model": "One global LightGBM per target predicting the correction to seasonal naive "
                     "(y - y[t-7]), direct multi-horizon: lags 7/14/21/28, "
                     "7- and 28-day rolling mean and 28-day std (all ending 7+ days before the "
                     "target), weekday, holiday distance, airport id.",
            "Backtest": meta["backtest"] + "."}
    for target, label in TARGETS.items():
        r = meta["test_metrics"][target]
        rows = "\n".join(f"| {m} | {v['mae']:.4f} | {v['rmse']:.4f} | {v['mape_pct']:.2f} |"
                         for m, v in r.items())
        secs[f"Backtest results - {label}"] = ("| Model | MAE | RMSE | MAPE % |\n|---|---|---|---|\n"
                                               + rows)
    secs["Limitations"] = ("Daily delay rate is driven by weather that is unknown a week ahead, so "
                           "any model's gain over naive baselines is small there; departures are "
                           "schedule-driven and highly regular. Only ~12 months of history, so no "
                           "yearly seasonality is modelled.")
    common.write_model_card(NAME, "Airport daily forecaster", secs)


if __name__ == "__main__":
    run()
