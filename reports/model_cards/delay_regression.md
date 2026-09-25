# Model card: Arrival-delay minutes (quantile LightGBM)

## Intended use

Give a typical (P50) and a pessimistic (P90) arrival delay for a scheduled flight, e.g. to size connection buffers. Same pre-departure feature set as the delay classifier.

## Data

uniform sample of 400,000 train / 150,000 valid completed flights; test = every completed flight in Jun-Jul 2026. Target = BTS ArrDelay (minutes, negative = early).

## Model

Three LightGBM models with quantile loss (alpha 0.1 / 0.5 / 0.9), early stopping on the validation window.

## Test results - point error (minutes)

| Model | MAE | RMSE | Median AE |
|---|---|---|---|
| baseline_global_median | 29.81 | 71.13 | 13.00 |
| baseline_global_mean | 31.79 | 68.64 | 20.43 |
| baseline_route_median | 29.69 | 71.08 | 13.00 |
| lgbm_p50 | 28.84 | 70.13 | 12.53 |

## Test results - pinball loss (lower is better)

| Quantile | Global-quantile baseline | LightGBM |
|---|---|---|
| P10 | 4.252 | 4.055 |
| P50 | 14.904 | 14.422 |
| P90 | 14.149 | 13.227 |

## Interval quality

Share of test flights at or below P90: 0.8674 (target 0.90). P10-P90 coverage: 0.7715 (target 0.80). Mean P10-P90 width: 66.7 min.

## Limitations

Delay minutes are heavy-tailed (a few flights are hours late), so RMSE is dominated by extreme events no schedule-time model can foresee; the P50 model improves MAE only modestly over a median baseline. Summer test months are more delayed than training, so P90 under-covers.
