# Model card: Pre-departure cancellation risk

## Intended use

Rank scheduled flights by cancellation risk from schedule-time information (route / carrier-origin / flight-number history, schedule congestion, calendar).

## Data

uniform sample of 500,000 train / 200,000 valid scheduled flights; test = every scheduled flight in Jun-Jul 2026. Cancellation rate in the test window: 0.0214 (validation window was much lower).

## Imbalance handling

PR-AUC used for tuning and selection; Optuna (12 trials) on tree parameters, then scale_pos_weight in (1, 5, 20) chosen on validation PR-AUC (selected 1.0). Isotonic calibration on validation restores probability scale.

## Test results (Jun-Jul 2026)

| Model | PR-AUC | ROC-AUC | F1 | Brier |
|---|---|---|---|---|
| prior_rate | 0.0214 | 0.5000 | 0.0000 | 0.0210 |
| logistic_regression_balanced | 0.0588 | 0.7311 | 0.1134 | 0.1829 |
| logistic_regression_calibrated | 0.0528 | 0.7291 | 0.1134 | 0.0210 |
| lightgbm_calibrated | 0.0404 | 0.7053 | 0.0574 | 0.0210 |
| lightgbm_uncalibrated | 0.0413 | 0.7078 | 0.0574 | 0.0225 |

## Model selection

Served model: **lightgbm** (higher validation PR-AUC; test never used for selection).

## Top-risk slice (LightGBM)

Top 1% riskiest flights: precision 0.0132 (lift 0.62x). Top 5%: 0.0492 (lift 2.30x).

## Limitations

Cancellations are driven mostly by same-day weather, ATC programs and IT / crew disruptions, none of which are known at scheduling time, so absolute PR-AUC is low. The validation window (Apr-May 2026) had an unusually low cancellation rate versus test, so calibrated probabilities under-state summer risk. On test the class-weighted logistic baseline ranks better than the validation-selected LightGBM, and LightGBM's top-1% slice is below the base rate - this model is weak and is kept to show the honest ceiling of schedule-only cancellation prediction.
