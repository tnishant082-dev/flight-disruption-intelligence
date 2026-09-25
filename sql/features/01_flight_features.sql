-- Feature layer: one row per scheduled flight with features known at scheduling time.
-- Nothing here uses actual departure/arrival times, delays, or cancellations of the flight
-- itself or of other flights on the same day. Schedule counts include flights that were later
-- cancelled, because they were on the published schedule.
CREATE SCHEMA IF NOT EXISTS features;

CREATE OR REPLACE TABLE features.flight_base AS
SELECT
    f.flight_id,
    f.flight_date,
    f.carrier,
    f.flight_number,
    f.carrier || '-' || f.flight_number                                   AS flight_key,
    f.origin,
    f.dest,
    f.route,
    f.carrier || '-' || f.origin                                          AS carrier_origin,
    f.tail_number,
    f.dep_hour,
    f.arr_hour,
    f.crs_dep_min_of_day,
    f.day_of_week,
    dayofmonth(f.flight_date)                                             AS day_of_month,
    CAST(f.day_of_week >= 6 AS TINYINT)                                   AS is_weekend,
    f.distance_mi,
    f.crs_elapsed_min,
    f.distance_mi / f.crs_elapsed_min                                     AS sched_speed_mpm,
    count(*) OVER (PARTITION BY f.origin, f.flight_date)                  AS origin_sched_deps_day,
    count(*) OVER (PARTITION BY f.origin, f.flight_date, f.dep_hour)      AS origin_sched_deps_hour,
    count(*) OVER (PARTITION BY f.dest, f.flight_date, f.arr_hour)        AS dest_sched_arrs_hour,
    count(*) OVER (PARTITION BY f.carrier, f.flight_date)                 AS carrier_sched_day,
    CASE WHEN f.tail_number IS NOT NULL THEN
        row_number() OVER (PARTITION BY f.tail_number, f.flight_date ORDER BY f.crs_dep_min_of_day)
    END                                                                   AS tail_leg_of_day,
    CASE WHEN f.tail_number IS NOT NULL THEN
        count(*) OVER (PARTITION BY f.tail_number, f.flight_date)
    END                                                                   AS tail_legs_day,
    CASE WHEN f.tail_number IS NOT NULL THEN
        f.crs_dep_min_of_day - lag(f.crs_arr_min_of_day) OVER (
            PARTITION BY f.tail_number, f.flight_date ORDER BY f.crs_dep_min_of_day)
    END                                                                   AS sched_turn_raw,
    -- targets / outcomes (never used as features)
    f.completed,
    f.cancelled,
    f.arr_del15,
    f.arr_delay_min
FROM staging.stg_flights f;
