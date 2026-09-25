-- Mart: daily series used by forecasting and anomaly detection.
CREATE OR REPLACE TABLE marts.daily_network AS
SELECT
    flight_date,
    count(*)                                                  AS scheduled_flights,
    round(avg(arr_del15), 5)                                  AS arr_delay_rate,
    round(avg(cancelled), 5)                                  AS cancellation_rate,
    round(avg(arr_delay_min) FILTER (WHERE completed = 1), 3) AS avg_arr_delay_min,
    sum(weather_delay_min)                                    AS weather_delay_min,
    sum(nas_delay_min)                                        AS nas_delay_min,
    sum(carrier_delay_min)                                    AS carrier_delay_min,
    sum(late_aircraft_delay_min)                              AS late_aircraft_delay_min
FROM staging.stg_flights
GROUP BY flight_date
ORDER BY flight_date;

CREATE OR REPLACE TABLE marts.daily_airport AS
SELECT
    flight_date,
    origin                                                    AS airport_code,
    count(*)                                                  AS scheduled_departures,
    round(avg(arr_del15), 5)                                  AS arr_delay_rate,
    round(avg(dep_del15) FILTER (WHERE cancelled = 0), 5)     AS dep_delay_rate,
    round(avg(cancelled), 5)                                  AS cancellation_rate,
    round(avg(dep_delay_min) FILTER (WHERE cancelled = 0), 3) AS avg_dep_delay_min,
    round(avg(taxi_out_min), 3)                               AS avg_taxi_out_min,
    round(sum(weather_delay_min) / nullif(sum(carrier_delay_min + weather_delay_min + nas_delay_min
          + security_delay_min + late_aircraft_delay_min), 0), 5) AS weather_share
FROM staging.stg_flights
GROUP BY ALL
ORDER BY flight_date, airport_code;
