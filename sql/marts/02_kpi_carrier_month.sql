-- Mart: carrier x month performance scorecard.
CREATE OR REPLACE TABLE marts.kpi_carrier_month AS
SELECT
    f.year_month,
    f.carrier,
    c.carrier_name,
    count(*)                                                  AS scheduled_flights,
    sum(f.completed)                                          AS completed_flights,
    round(100 * (1 - avg(f.arr_del15)), 2)                    AS on_time_pct,
    round(avg(f.arr_delay_min) FILTER (WHERE f.completed = 1), 2) AS avg_arr_delay_min,
    round(100 * avg(f.cancelled), 3)                          AS cancellation_rate_pct,
    round(100 * avg(f.diverted), 3)                           AS diversion_rate_pct,
    sum(f.carrier_delay_min)                                  AS carrier_delay_min,
    sum(f.weather_delay_min)                                  AS weather_delay_min,
    sum(f.nas_delay_min)                                      AS nas_delay_min,
    sum(f.security_delay_min)                                 AS security_delay_min,
    sum(f.late_aircraft_delay_min)                            AS late_aircraft_delay_min
FROM staging.stg_flights f
JOIN staging.stg_carriers c ON c.carrier_code = f.carrier
GROUP BY ALL;
