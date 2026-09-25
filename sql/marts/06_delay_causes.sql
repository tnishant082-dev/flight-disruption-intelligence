-- Mart: delay-cause minutes and shares by month and carrier.
-- BTS attributes cause minutes only for flights arriving 15+ minutes late.
CREATE OR REPLACE TABLE marts.delay_causes_month AS
WITH m AS (
    SELECT
        year_month,
        carrier,
        sum(carrier_delay_min)       AS carrier_min,
        sum(weather_delay_min)       AS weather_min,
        sum(nas_delay_min)           AS nas_min,
        sum(security_delay_min)      AS security_min,
        sum(late_aircraft_delay_min) AS late_aircraft_min,
        count(*) FILTER (WHERE arr_del15 = 1) AS delayed_flights
    FROM staging.stg_flights
    GROUP BY ALL
)
SELECT
    *,
    carrier_min + weather_min + nas_min + security_min + late_aircraft_min AS total_cause_min,
    round(100 * carrier_min / nullif(carrier_min + weather_min + nas_min + security_min + late_aircraft_min, 0), 2) AS carrier_share_pct,
    round(100 * weather_min / nullif(carrier_min + weather_min + nas_min + security_min + late_aircraft_min, 0), 2) AS weather_share_pct,
    round(100 * nas_min / nullif(carrier_min + weather_min + nas_min + security_min + late_aircraft_min, 0), 2) AS nas_share_pct,
    round(100 * security_min / nullif(carrier_min + weather_min + nas_min + security_min + late_aircraft_min, 0), 2) AS security_share_pct,
    round(100 * late_aircraft_min / nullif(carrier_min + weather_min + nas_min + security_min + late_aircraft_min, 0), 2) AS late_aircraft_share_pct
FROM m;

CREATE OR REPLACE TABLE marts.cancellation_reasons_month AS
SELECT year_month, cancellation_reason, count(*) AS cancelled_flights
FROM staging.stg_flights
WHERE cancelled = 1
GROUP BY ALL;
