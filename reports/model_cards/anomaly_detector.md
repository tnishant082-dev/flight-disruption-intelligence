# Model card: Disruption-day detector

## Intended use

Flag network-wide and airport-level disruption days for after-action review and to exclude / annotate them in KPI reporting.

## Method

STL (period 7, robust) on the logit of daily network delay and cancellation rates, robust z-score of residual > 3.5. Isolation Forest (300 trees, contamination 0.02) on airport-day features relative to the airport's own median, for the 30 busiest airports.

## Results

Network: 25 of 365 days flagged. Airport-days: 219 of 10950 flagged; flagged days average a 0.680 delay rate and 0.284 cancellation rate vs 0.227 / 0.0132 on other days.

## Flagged network days (by cancellation rate)

| Date | Driver | Delay rate | Cancel rate |
|---|---|---|---|
| 2026-01-25 | delays + cancellations | 0.4321 | 0.4648 |
| 2026-01-26 | delays + cancellations | 0.4150 | 0.2712 |
| 2026-01-24 | cancellations | 0.2393 | 0.2106 |
| 2026-02-23 | cancellations | 0.1734 | 0.1968 |
| 2026-03-16 | delays | 0.5543 | 0.1957 |
| 2026-01-27 | delays + cancellations | 0.3201 | 0.1386 |
| 2025-11-09 | delays | 0.4514 | 0.1213 |
| 2026-02-22 | cancellations | 0.3256 | 0.1182 |
| 2025-11-10 | delays | 0.3992 | 0.0978 |
| 2026-07-18 | cancellations | 0.3427 | 0.0932 |
| 2026-02-24 | cancellations | 0.2020 | 0.0840 |
| 2025-11-29 | delays + cancellations | 0.3306 | 0.0838 |
| 2025-11-08 | delays | 0.3325 | 0.0749 |
| 2026-03-07 | delays | 0.4125 | 0.0602 |
| 2026-03-17 | delays | 0.3880 | 0.0564 |

## Limitations

Unsupervised - there is no labelled ground truth of disruption days, so the thresholds (z > 3.5, 2% contamination) are judgement calls. Causes (storms, ATC outages, IT failures) are not in the data and must be attributed from external sources.
