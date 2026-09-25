-- Mart: scheduled departure hour x ISO day of week (1 = Monday).
CREATE OR REPLACE TABLE marts.kpi_hour_dow AS
SELECT
    dep_hour,
    day_of_week,
    count(*)                                                  AS scheduled_flights,
    round(100 * avg(arr_del15), 2)                            AS arr_delay_rate_pct,
    round(avg(arr_delay_min) FILTER (WHERE completed = 1), 2) AS avg_arr_delay_min,
    round(100 * avg(cancelled), 3)                            AS cancellation_rate_pct
FROM staging.stg_flights
GROUP BY ALL;
