-- Mart: origin airport x month departures performance.
CREATE OR REPLACE TABLE marts.kpi_airport_month AS
SELECT
    f.year_month,
    f.origin                                                  AS airport_code,
    a.airport_name,
    a.state,
    count(*)                                                  AS scheduled_departures,
    round(100 * (1 - avg(f.arr_del15)), 2)                    AS on_time_arrival_pct,
    round(100 * avg(f.dep_del15) FILTER (WHERE f.cancelled = 0), 2) AS dep_delay_rate_pct,
    round(avg(f.dep_delay_min) FILTER (WHERE f.cancelled = 0), 2)   AS avg_dep_delay_min,
    round(avg(f.taxi_out_min), 2)                             AS avg_taxi_out_min,
    round(100 * avg(f.cancelled), 3)                          AS cancellation_rate_pct
FROM staging.stg_flights f
JOIN staging.stg_airports a ON a.airport_code = f.origin
GROUP BY ALL;
