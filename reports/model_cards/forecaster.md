# Model card: Airport daily forecaster

## Intended use

Short-range (1-7 day) planning view of departures and arrival-delay rate at the 10 busiest origin airports.

## Data

marts.daily_airport (BTS, Aug 2025 - Jul 2026). Airports: ATL, CLT, DEN, DFW, LAS, LAX, MCO, ORD, PHX, SEA.

## Model

One global LightGBM per target predicting the correction to seasonal naive (y - y[t-7]), direct multi-horizon: lags 7/14/21/28, 7- and 28-day rolling mean and 28-day std (all ending 7+ days before the target), weekday, holiday distance, airport id.

## Backtest

8 rolling origins, weekly step, 7-day horizon, expanding training window (last 8 weeks of data).

## Backtest results - Departures / day

| Model | MAE | RMSE | MAPE % |
|---|---|---|---|
| lightgbm | 17.2983 | 38.5819 | 2.66 |
| seasonal_naive | 10.8125 | 30.3050 | 1.78 |
| mean_4wk_same_weekday | 15.5598 | 26.8604 | 2.45 |

## Backtest results - Arrival-delay rate

| Model | MAE | RMSE | MAPE % |
|---|---|---|---|
| lightgbm | 0.0932 | 0.1267 | 29.66 |
| seasonal_naive | 0.1030 | 0.1418 | 35.65 |
| mean_4wk_same_weekday | 0.0860 | 0.1183 | 28.53 |

## Limitations

Daily delay rate is driven by weather that is unknown a week ahead, so any model's gain over naive baselines is small there; departures are schedule-driven and highly regular. Only ~12 months of history, so no yearly seasonality is modelled.
