-- Staging: airport dimension = every BTS airport code seen in the study window,
-- enriched with OurAirports metadata (name, coordinates, type) where the IATA code matches.
CREATE OR REPLACE TABLE staging.stg_airports AS
WITH seen AS (
    SELECT origin AS iata, any_value(origin_city) AS city, any_value(origin_state) AS state
    FROM staging.stg_flights GROUP BY 1
    UNION
    SELECT dest, any_value(dest_city), any_value(dest_state)
    FROM staging.stg_flights GROUP BY 1
),
seen_dedup AS (
    SELECT iata, any_value(city) AS city, any_value(state) AS state FROM seen GROUP BY 1
),
oa_keys AS (
    -- primary key: IATA code; fallback: ICAO ident 'K' + code (covers renamed IATA codes,
    -- e.g. BTS still reports PBI while OurAirports now lists DJT for the same airport)
    SELECT iata_code AS iata, 1 AS priority, * FROM raw.ourairports
    WHERE iata_code IS NOT NULL AND iata_code <> ''
    UNION ALL
    SELECT substr(ident, 2) AS iata, 2 AS priority, * FROM raw.ourairports
    WHERE ident LIKE 'K%' AND length(ident) = 4
),
oa AS (
    SELECT
        iata,
        name,
        TRY_CAST(latitude_deg AS DOUBLE)  AS latitude,
        TRY_CAST(longitude_deg AS DOUBLE) AS longitude,
        TRY_CAST(elevation_ft AS DOUBLE)  AS elevation_ft,
        type AS airport_type,
        row_number() OVER (
            PARTITION BY iata
            ORDER BY priority,
                     (scheduled_service = 'yes') DESC,
                     CASE type WHEN 'large_airport' THEN 1 WHEN 'medium_airport' THEN 2
                               WHEN 'small_airport' THEN 3 ELSE 4 END
        ) AS rn
    FROM oa_keys
)
SELECT
    s.iata                                   AS airport_code,
    coalesce(oa.name, s.city)                AS airport_name,
    s.city, s.state,
    oa.latitude, oa.longitude, oa.elevation_ft,
    coalesce(oa.airport_type, 'unknown')     AS airport_type,
    oa.iata IS NOT NULL                      AS has_ourairports_match
FROM seen_dedup s
LEFT JOIN oa ON oa.iata = s.iata AND oa.rn = 1;
