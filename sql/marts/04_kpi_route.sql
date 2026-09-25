-- Mart: route reliability (routes with at least 300 scheduled flights in the window).
CREATE OR REPLACE TABLE marts.kpi_route AS
SELECT
    route,
    origin,
    dest,
    any_value(distance_mi)                                    AS distance_mi,
    count(*)                                                  AS scheduled_flights,
    count(DISTINCT carrier)                                   AS carriers,
    round(100 * (1 - avg(arr_del15)), 2)                      AS on_time_pct,
    round(avg(arr_delay_min) FILTER (WHERE completed = 1), 2) AS avg_arr_delay_min,
    round(quantile_cont(arr_delay_min, 0.9) FILTER (WHERE completed = 1), 1) AS p90_arr_delay_min,
    round(100 * avg(cancelled), 3)                            AS cancellation_rate_pct
FROM staging.stg_flights
GROUP BY route, origin, dest
HAVING count(*) >= 300;
