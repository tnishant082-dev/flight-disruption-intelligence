-- Mart: operational profile per origin airport (input to the segmentation model).
-- Restricted to airports with >= 3,000 scheduled departures in the window.
CREATE OR REPLACE TABLE marts.airport_profile AS
SELECT
    origin                                                    AS airport_code,
    count(*)                                                  AS scheduled_departures,
    count(DISTINCT dest)                                      AS destinations,
    count(DISTINCT carrier)                                   AS carriers,
    avg(arr_del15)                                            AS arr_delay_rate,
    avg(dep_del15) FILTER (WHERE cancelled = 0)               AS dep_delay_rate,
    avg(cancelled)                                            AS cancellation_rate,
    avg(dep_delay_min) FILTER (WHERE cancelled = 0)           AS avg_dep_delay_min,
    avg(taxi_out_min)                                         AS avg_taxi_out_min,
    avg(distance_mi)                                          AS avg_distance_mi,
    avg(CASE WHEN dep_hour >= 17 THEN 1 ELSE 0 END)           AS evening_share,
    sum(weather_delay_min) / nullif(sum(carrier_delay_min + weather_delay_min + nas_delay_min
        + security_delay_min + late_aircraft_delay_min), 0)   AS weather_cause_share,
    sum(nas_delay_min) / nullif(sum(carrier_delay_min + weather_delay_min + nas_delay_min
        + security_delay_min + late_aircraft_delay_min), 0)   AS nas_cause_share,
    sum(late_aircraft_delay_min) / nullif(sum(carrier_delay_min + weather_delay_min + nas_delay_min
        + security_delay_min + late_aircraft_delay_min), 0)   AS late_aircraft_cause_share,
    stddev_samp(dep_delay_min) FILTER (WHERE cancelled = 0)   AS dep_delay_std_min
FROM staging.stg_flights
GROUP BY origin
HAVING count(*) >= 3000;
