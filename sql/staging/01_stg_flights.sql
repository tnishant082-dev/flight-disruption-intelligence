-- Staging: typed, cleaned, one row per scheduled flight leg.
-- Source: raw.flights (BTS Reporting Carrier On-Time Performance).
CREATE SCHEMA IF NOT EXISTS staging;

-- Rows with a missing / non-positive scheduled block time are source errors (e.g. scheduled
-- arrival before departure) and are moved to staging.quarantine_flights instead of being used.
CREATE OR REPLACE TABLE staging.quarantine_flights AS
SELECT *, 'non_positive_or_missing_crs_elapsed' AS quarantine_reason
FROM raw.flights
WHERE CRSElapsedTime IS NULL OR CRSElapsedTime <= 0 OR Distance IS NULL OR Distance <= 0;

CREATE OR REPLACE TABLE staging.stg_flights AS
WITH typed AS (
    SELECT
        FlightDate                                         AS flight_date,
        upper(trim(Reporting_Airline))                     AS carrier,
        nullif(upper(trim(Tail_Number)), '')               AS tail_number,
        trim(Flight_Number_Reporting_Airline)              AS flight_number,
        upper(trim(Origin))                                AS origin,
        OriginCityName                                     AS origin_city,
        OriginState                                        AS origin_state,
        upper(trim(Dest))                                  AS dest,
        DestCityName                                       AS dest_city,
        DestState                                          AS dest_state,
        TRY_CAST(CRSDepTime AS INTEGER)                    AS crs_dep_hhmm,
        TRY_CAST(CRSArrTime AS INTEGER)                    AS crs_arr_hhmm,
        TRY_CAST(DepTime AS INTEGER)                       AS dep_hhmm,
        TRY_CAST(ArrTime AS INTEGER)                       AS arr_hhmm,
        DepDelay                                           AS dep_delay_min,
        ArrDelay                                           AS arr_delay_min,
        CAST(coalesce(DepDel15, 0) AS TINYINT)             AS dep_del15,
        CAST(ArrDel15 AS TINYINT)                          AS arr_del15_raw,
        CAST(coalesce(Cancelled, 0) AS TINYINT)            AS cancelled,
        nullif(trim(CancellationCode), '')                 AS cancellation_code,
        CAST(coalesce(Diverted, 0) AS TINYINT)             AS diverted,
        TaxiOut                                            AS taxi_out_min,
        TaxiIn                                             AS taxi_in_min,
        CRSElapsedTime                                     AS crs_elapsed_min,
        ActualElapsedTime                                  AS actual_elapsed_min,
        AirTime                                            AS air_time_min,
        Distance                                           AS distance_mi,
        CarrierDelay                                       AS carrier_delay_min,
        WeatherDelay                                       AS weather_delay_min,
        NASDelay                                           AS nas_delay_min,
        SecurityDelay                                      AS security_delay_min,
        LateAircraftDelay                                  AS late_aircraft_delay_min
    FROM raw.flights
    WHERE FlightDate IS NOT NULL
      AND Origin IS NOT NULL AND Dest IS NOT NULL
      AND Reporting_Airline IS NOT NULL
      AND CRSElapsedTime > 0 AND Distance > 0
)
SELECT
    hash(carrier, flight_number, origin, dest, flight_date, crs_dep_hhmm) AS flight_id,
    *,
    -- 2400 is used by BTS for midnight at the end of the day
    least((crs_dep_hhmm // 100) * 60 + crs_dep_hhmm % 100, 1439)        AS crs_dep_min_of_day,
    least((crs_arr_hhmm // 100) * 60 + crs_arr_hhmm % 100, 1439)        AS crs_arr_min_of_day,
    least(crs_dep_hhmm // 100, 23)                                       AS dep_hour,
    least(crs_arr_hhmm // 100, 23)                                       AS arr_hour,
    flight_date + to_minutes(least((crs_dep_hhmm // 100) * 60 + crs_dep_hhmm % 100, 1439))
                                                                         AS sched_dep_ts,
    origin || '-' || dest                                                AS route,
    CAST(strftime(flight_date, '%Y-%m') AS VARCHAR)                      AS year_month,
    isodow(flight_date)                                                  AS day_of_week,
    CAST(cancelled = 0 AND diverted = 0 AS TINYINT)                      AS completed,
    -- arrival-delay flag is only defined for completed flights
    CASE WHEN cancelled = 0 AND diverted = 0 THEN coalesce(arr_del15_raw, 0) END AS arr_del15,
    CASE cancellation_code
        WHEN 'A' THEN 'Carrier' WHEN 'B' THEN 'Weather'
        WHEN 'C' THEN 'National Air System' WHEN 'D' THEN 'Security'
    END                                                                  AS cancellation_reason
FROM typed;
