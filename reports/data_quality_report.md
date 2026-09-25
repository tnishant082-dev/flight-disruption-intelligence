# Data quality report

Gate: **PASS** - 26 passed, 0 warnings, 0 failed (run 2026-09-25 10:29:39 UTC).

Rows quarantined before staging (source errors): 16.

| Check | Category | Observed | Threshold | Status | Note |
|---|---|---:|---|---|---|
| `row_count_positive` | completeness | 7.04384e+06 | > 0 | PASS |  |
| `rows_retained_vs_raw` | completeness | 0.999998 | >= 0.999 | PASS | raw=7043858 staged=7043842 |
| `duplicate_flight_id` | uniqueness | 0 | == 0 | PASS |  |
| `not_null_flight_date` | completeness | 0 | == 0 | PASS |  |
| `not_null_carrier` | completeness | 0 | == 0 | PASS |  |
| `not_null_flight_number` | completeness | 0 | == 0 | PASS |  |
| `not_null_origin` | completeness | 0 | == 0 | PASS |  |
| `not_null_dest` | completeness | 0 | == 0 | PASS |  |
| `not_null_crs_dep_hhmm` | completeness | 0 | == 0 | PASS |  |
| `not_null_crs_arr_hhmm` | completeness | 0 | == 0 | PASS |  |
| `not_null_distance_mi` | completeness | 0 | == 0 | PASS |  |
| `valid_scheduled_dep_time` | validity | 0 | == 0 | PASS |  |
| `origin_differs_from_dest` | validity | 0 | == 0 | PASS |  |
| `binary_status_flags` | validity | 0 | == 0 | PASS |  |
| `cancelled_has_reason_code` | consistency | 0 | == 0 | PASS |  |
| `reason_code_only_when_cancelled` | consistency | 0 | == 0 | PASS |  |
| `completed_has_arrival_delay` | completeness | 0 | <= 0.001 | PASS |  |
| `arr_del15_matches_delay_minutes` | consistency | 0 | <= 0.0005 | PASS |  |
| `delay_causes_sum_to_arrival_delay` | consistency | 7.7e-05 | <= 0.01 | PASS | BTS attributes cause minutes for flights arriving 15+ min late |
| `positive_distance_and_block_time` | validity | 0 | == 0 | PASS |  |
| `taxi_out_plausible_0_240` | validity | 1e-06 | <= 0.001 | PASS |  |
| `arrival_delay_over_24h_share` | validity | 0.000106 | <= 0.0005 | PASS |  |
| `calendar_coverage_days` | completeness | 365 | == 365 | PASS |  |
| `origin_in_airport_dim` | referential | 0 | == 0 | PASS |  |
| `airport_dim_has_coordinates` | enrichment | 0 | == 0 | PASS |  |
| `carrier_dim_named` | enrichment | 0 | == 0 | PASS |  |
