# Model card: Pre-departure arrival-delay classifier

## Intended use

Rank scheduled flights by the probability of arriving 15+ minutes late, using only information available when the schedule is published. Decision support for schedule planners / ops analysts, not an operational guarantee.

## Data

BTS Reporting Carrier On-Time Performance, Aug 2025 - Jul 2026. uniform reservoir sample of 500,000 train / 200,000 valid completed flights; test = every completed flight in Jun-Jul 2026. Train 2025-08-01..2026-03-31, valid 2026-04-01..2026-05-31, test 2026-06-01..2026-07-31 (time-based, no shuffling across windows).

## Features

Carrier, origin, destination (native categoricals), scheduled departure hour/minute and arrival hour, day of week, weekend flag, distance to nearest US federal holiday, distance, scheduled block time, schedule congestion counts at origin/destination and out-of-fold target encodings of route / carrier-origin / flight number computed on the training window only. No actual departure time, taxi time, weather or same-day delay information is used.

## Model

LightGBM (Optuna, 15 trials, 20 rounds) + isotonic calibration fit on validation. Threshold 0.2103 = F1-optimal on validation.

## Test results (Jun-Jul 2026)

| Model | ROC-AUC | PR-AUC | F1 | Brier |
|---|---|---|---|---|
| prior_rate | 0.5000 | 0.2705 | 0.0000 | 0.2004 |
| logistic_regression | 0.6803 | 0.4124 | 0.4878 | 0.1841 |
| xgboost | 0.6806 | 0.4151 | 0.4873 | 0.1867 |
| lightgbm_uncalibrated | 0.6877 | 0.4223 | 0.4914 | 0.1860 |
| lightgbm_calibrated | 0.6875 | 0.4189 | 0.4914 | 0.1857 |

## Most influential features

`crs_dep_min_of_day`, `te_flight_delay`, `arr_hour`, `dest`, `origin`, `te_route_delay`

## Day-of-operations comparison (not schedule-time)

Adding aircraft-rotation features built from the tail number BTS reports (leg of day, legs per day, turn time) lifts test ROC-AUC to 0.7105 / PR-AUC 0.4902. Those tails are the aircraft that actually flew, so this variant is excluded from the served model and shown only to quantify what same-day aircraft information would add.

## Limitations

Test months (Jun-Jul) are summer thunderstorm season and have a higher delay rate than any training month except August 2025, so calibration drifts (see test_by_month in reports/metrics). Month-of-year is deliberately not a feature (nor day-of-month) because the test months are never seen in training. Discrimination is moderate: most delay variance comes from same-day weather and knock-on effects that are unknowable at scheduling time.
