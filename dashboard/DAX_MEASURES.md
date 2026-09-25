# DAX measures

All 40 measures live in the `_Measures` table ([`_Measures.tmdl`](./FlightDisruption.SemanticModel/definition/tables/_Measures.tmdl)). Rates follow the SQL marts: on-time % is computed over completed (non-cancelled, non-diverted) flights; cancellation rate over all scheduled flights.

## Volume

### Flights

Scheduled flights

```dax
Flights = COUNTROWS ( fact_flights )
```
Format: `#,0`

### Completed Flights

Not cancelled or diverted

```dax
Completed Flights = SUM ( fact_flights[completed] )
```
Format: `#,0`

## Punctuality

### Delayed Flights

Arrived 15+ min late

```dax
Delayed Flights = SUM ( fact_flights[arr_del15] )
```
Format: `#,0`

### On-Time %

Completed flights arriving < 15 min late

```dax
On-Time % = DIVIDE ( [Completed Flights] - [Delayed Flights], [Completed Flights] )
```
Format: `0.0%`

### Delay Rate %

Completed flights arriving 15+ min late

```dax
Delay Rate % = DIVIDE ( [Delayed Flights], [Completed Flights] )
```
Format: `0.0%`

### Avg Arrival Delay (min)

Mean arrival delay, early arrivals negative

```dax
Avg Arrival Delay (min) = CALCULATE ( AVERAGE ( fact_flights[arr_delay_min] ), fact_flights[completed] = 1 )
```
Format: `0.0`

### Avg Delay When Late (min)

```dax
Avg Delay When Late (min) = CALCULATE ( AVERAGE ( fact_flights[arr_delay_min] ), fact_flights[arr_del15] = 1 )
```
Format: `0.0`

### Dep Delay Rate %

```dax
Dep Delay Rate % = DIVIDE ( SUM ( fact_flights[dep_del15] ), [Flights] - [Cancelled Flights] )
```
Format: `0.0%`

### Avg Taxi-Out (min)

```dax
Avg Taxi-Out (min) = AVERAGE ( fact_flights[taxi_out_min] )
```
Format: `0.0`

## Disruption

### Cancelled Flights

```dax
Cancelled Flights = SUM ( fact_flights[cancelled] )
```
Format: `#,0`

### Cancellation Rate %

```dax
Cancellation Rate % = DIVIDE ( [Cancelled Flights], [Flights] )
```
Format: `0.00%`

### Diversion Rate %

```dax
Diversion Rate % = DIVIDE ( SUM ( fact_flights[diverted] ), [Flights] )
```
Format: `0.00%`

## Delay causes

### Carrier Delay Min

```dax
Carrier Delay Min = SUM ( fact_flights[carrier_delay_min] )
```
Format: `#,0`

### Weather Delay Min

```dax
Weather Delay Min = SUM ( fact_flights[weather_delay_min] )
```
Format: `#,0`

### NAS Delay Min

```dax
NAS Delay Min = SUM ( fact_flights[nas_delay_min] )
```
Format: `#,0`

### Security Delay Min

```dax
Security Delay Min = SUM ( fact_flights[security_delay_min] )
```
Format: `#,0`

### Late Aircraft Delay Min

```dax
Late Aircraft Delay Min = SUM ( fact_flights[late_aircraft_delay_min] )
```
Format: `#,0`

### Total Cause Min

```dax
Total Cause Min = [Carrier Delay Min] + [Weather Delay Min] + [NAS Delay Min] + [Security Delay Min] + [Late Aircraft Delay Min]
```
Format: `#,0`

### Late Aircraft Share %

Knock-on delay share

```dax
Late Aircraft Share % = DIVIDE ( [Late Aircraft Delay Min], [Total Cause Min] )
```
Format: `0.0%`

### Weather Share %

```dax
Weather Share % = DIVIDE ( [Weather Delay Min], [Total Cause Min] )
```
Format: `0.0%`

### Carrier Share %

```dax
Carrier Share % = DIVIDE ( [Carrier Delay Min], [Total Cause Min] )
```
Format: `0.0%`

### NAS Share %

```dax
NAS Share % = DIVIDE ( [NAS Delay Min], [Total Cause Min] )
```
Format: `0.0%`

## Time intelligence

### On-Time % PM

Previous month

```dax
On-Time % PM = CALCULATE ( [On-Time %], DATEADD ( dim_date[date], -1, MONTH ) )
```
Format: `0.0%`

### On-Time % MoM (pp)

```dax
On-Time % MoM (pp) = IF ( NOT ISBLANK ( [On-Time % PM] ), ( [On-Time %] - [On-Time % PM] ) * 100 )
```
Format: `+0.0;-0.0;0.0`

### Flights MoM %

```dax
Flights MoM % = VAR pm = CALCULATE ( [Flights], DATEADD ( dim_date[date], -1, MONTH ) ) RETURN DIVIDE ( [Flights] - pm, pm )
```
Format: `+0.0%;-0.0%`

## Ranking

### Carrier OTP Rank

```dax
Carrier OTP Rank = IF ( HASONEVALUE ( dim_carrier[carrier_name] ), RANKX ( ALL ( dim_carrier[carrier_name] ), [On-Time %], , DESC ) )
```
Format: `0`

### Route OTP vs Network (pp)

```dax
Route OTP vs Network (pp) = ( [On-Time %] - CALCULATE ( [On-Time %], ALL ( dim_route ), ALL ( dim_airport ) ) ) * 100
```
Format: `+0.0;-0.0`

## ML scores

### Scored Flights

Test window flights with model scores

```dax
Scored Flights = COUNTROWS ( fact_flight_scores )
```
Format: `#,0`

### Avg Predicted Delay Risk

Mean calibrated P(arrival 15+ min late)

```dax
Avg Predicted Delay Risk = AVERAGE ( fact_flight_scores[delay_prob] )
```
Format: `0.0%`

### Observed Delay Rate (Scored)

```dax
Observed Delay Rate (Scored) = AVERAGE ( fact_flight_scores[arr_del15] )
```
Format: `0.0%`

### Calibration Gap (pp)

Negative = model under-predicts

```dax
Calibration Gap (pp) = ( [Avg Predicted Delay Risk] - [Observed Delay Rate (Scored)] ) * 100
```
Format: `+0.0;-0.0`

### High-Risk Flights

Calibrated risk >= 45%

```dax
High-Risk Flights = CALCULATE ( COUNTROWS ( fact_flight_scores ), fact_flight_scores[delay_prob] >= 0.45 )
```
Format: `#,0`

### High-Risk Hit Rate

Share of high-risk flights that were delayed

```dax
High-Risk Hit Rate = CALCULATE ( AVERAGE ( fact_flight_scores[arr_del15] ), fact_flight_scores[delay_prob] >= 0.45 )
```
Format: `0.0%`

### Flagged Precision

Precision at the F1-optimal threshold

```dax
Flagged Precision = CALCULATE ( AVERAGE ( fact_flight_scores[arr_del15] ), fact_flight_scores[delay_pred] = 1 )
```
Format: `0.0%`

### Avg P90 Delay (min)

```dax
Avg P90 Delay (min) = AVERAGE ( fact_flight_scores[delay_p90] )
```
Format: `0.0`

### Avg Cancel Risk

```dax
Avg Cancel Risk = AVERAGE ( fact_flight_scores[cancel_prob] )
```
Format: `0.00%`

## Disruption days

### Anomaly Days

STL-flagged network days

```dax
Anomaly Days = CALCULATE ( COUNTROWS ( anomalies_network ), anomalies_network[is_anomaly] = 1 )
```
Format: `0`

## Forecast

### Backtest MAE LightGBM

Mean absolute backtest error of the LightGBM forecaster. Delay-rate errors are shown in percentage points so the table matches the metrics in the README.

```dax
Backtest MAE LightGBM =
AVERAGEX ( forecast_backtest, ABS ( forecast_backtest[lightgbm] - forecast_backtest[y] ) )
    * IF ( SELECTEDVALUE ( forecast_backtest[target] ) = "arr_delay_rate", 100, 1 )
```
Format: `0.00`

### Backtest MAE Seasonal Naive

Same, for the same-weekday-last-week baseline.

```dax
Backtest MAE Seasonal Naive =
AVERAGEX ( forecast_backtest, ABS ( forecast_backtest[seasonal_naive] - forecast_backtest[y] ) )
    * IF ( SELECTEDVALUE ( forecast_backtest[target] ) = "arr_delay_rate", 100, 1 )
```
Format: `0.00`

### Backtest MAE 4-Week Mean

Same, for the 4-week same-weekday mean baseline.

```dax
Backtest MAE 4-Week Mean =
AVERAGEX ( forecast_backtest, ABS ( forecast_backtest[mean_4wk_same_weekday] - forecast_backtest[y] ) )
    * IF ( SELECTEDVALUE ( forecast_backtest[target] ) = "arr_delay_rate", 100, 1 )
```
Format: `0.00`

## Relationships

| From | To | Active |
|---|---|---|
| `fact_flights[date_key]` | `dim_date[date_key]` | yes |
| `fact_flights[carrier_code]` | `dim_carrier[carrier_code]` | yes |
| `fact_flights[origin_airport]` | `dim_airport[airport_code]` | yes |
| `fact_flights[dest_airport]` | `dim_airport[airport_code]` | no (use USERELATIONSHIP) |
| `fact_flights[route]` | `dim_route[route]` | yes |
| `fact_flight_scores[date_key]` | `dim_date[date_key]` | yes |
| `fact_flight_scores[carrier_code]` | `dim_carrier[carrier_code]` | yes |
| `fact_flight_scores[origin_airport]` | `dim_airport[airport_code]` | yes |
| `fact_flight_scores[route]` | `dim_route[route]` | yes |
| `score_carrier_month[carrier_code]` | `dim_carrier[carrier_code]` | yes |
| `score_route_risk[route]` | `dim_route[route]` | yes |
