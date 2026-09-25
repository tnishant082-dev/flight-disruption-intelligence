-- Staging: reporting carriers present in the study window with BTS lookup names.
CREATE OR REPLACE TABLE staging.stg_carriers AS
SELECT
    f.carrier                          AS carrier_code,
    coalesce(c.name, f.carrier)        AS carrier_name,
    count(*)                           AS flights_in_window
FROM staging.stg_flights f
LEFT JOIN raw.carriers c ON c.code = f.carrier
GROUP BY 1, 2;
