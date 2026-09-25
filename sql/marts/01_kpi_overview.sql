-- Mart: headline KPIs for the whole study window.
-- on_time_pct = completed flights arriving < 15 min late / completed flights.
CREATE SCHEMA IF NOT EXISTS marts;

CREATE OR REPLACE TABLE marts.kpi_overview AS
SELECT
    min(flight_date)                                        AS first_date,
    max(flight_date)                                        AS last_date,
    count(*)                                                AS scheduled_flights,
    sum(completed)                                          AS completed_flights,
    count(DISTINCT carrier)                                 AS carriers,
    count(DISTINCT origin)                                  AS origin_airports,
    count(DISTINCT route)                                   AS routes,
    round(100 * (1 - avg(arr_del15)), 2)                    AS on_time_pct,
    round(avg(arr_delay_min) FILTER (WHERE completed = 1), 2) AS avg_arr_delay_min,
    round(avg(arr_delay_min) FILTER (WHERE arr_del15 = 1), 2) AS avg_delay_when_late_min,
    round(100 * avg(cancelled), 3)                          AS cancellation_rate_pct,
    round(100 * avg(diverted), 3)                           AS diversion_rate_pct
FROM staging.stg_flights;
