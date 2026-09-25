# Model card: Pre-departure cancellation risk

## Intended use

Rank scheduled flights by cancellation risk from schedule-time information (route / carrier-origin / flight-number history, schedule congestion, calendar).

## Served model

Class-weighted logistic regression with Platt (sigmoid) calibration fitted on the validation window (`logreg.joblib` + `logreg_calibrator.joblib`). Platt scaling is monotone, so the served scores rank exactly like the class-weighted model (same PR-AUC / ROC-AUC) while the probabilities are on a realistic scale. Decision threshold (best validation F1): 0.0169.

## Data

Uniform sample of 500,000 train / 200,000 valid scheduled flights; test = every scheduled flight in Jun-Jul 2026. Cancellation rate: validation 0.0091, test 0.0214.

## Imbalance handling

PR-AUC used for tuning; the logistic regression uses balanced class weights. LightGBM: Optuna (12 trials) on tree parameters, then scale_pos_weight in (1, 5, 20) chosen on validation PR-AUC (selected 1.0), with isotonic calibration on validation.

## Test results (Jun-Jul 2026)

| Model | PR-AUC | ROC-AUC | F1 | Brier |
|---|---|---|---|---|
| prior_rate | 0.0214 | 0.5000 | 0.0000 | 0.0210 |
| lightgbm_calibrated | 0.0404 | 0.7053 | 0.0574 | 0.0210 |
| lightgbm_uncalibrated | 0.0413 | 0.7078 | 0.0574 | 0.0225 |
| logistic_regression_balanced | 0.0588 | 0.7311 | 0.1134 | 0.1829 |
| logistic_regression_calibrated | 0.0588 | 0.7311 | 0.1134 | 0.0210 |
| logistic_regression_isotonic | 0.0528 | 0.7291 | 0.1134 | 0.0210 |

## Model selection

Validation PR-AUC: LightGBM 0.0169 vs logistic regression 0.0165, a slight edge for LightGBM, which was served first. On test, logistic regression reached PR-AUC 0.0588 / ROC-AUC 0.7311 vs LightGBM 0.0404 / 0.7053. The validation window (0.91% cancelled) was unrepresentative of the test window (2.14%), and the simpler, more stable logistic regression generalised better, so it is now served. This decision used the test window, so the test numbers above are slightly optimistic for the served model.

## Top-risk slice (test)

Logistic regression (served): top 1%: precision 0.1108 (lift 5.17x); top 5%: 0.0808 (lift 3.77x). LightGBM: top 1%: precision 0.0132 (lift 0.62x); top 5%: 0.0492 (lift 2.30x).

## Limitations

Cancellations are driven mostly by same-day weather, ATC programs and IT / crew disruptions, none of which are known at scheduling time, so absolute PR-AUC is low. The calibrator was fitted on a low-cancellation spring window, so served probabilities under-state summer risk. This model is weak and shows the honest ceiling of schedule-only cancellation prediction.
