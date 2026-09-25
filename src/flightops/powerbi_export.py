"""Export the star schema + model score tables consumed by the Power BI project and the app.

Committed (small): dimensions, model score aggregates, forecast / anomaly / segment tables,
fact_flights_sample.csv (50k-row random sample).
Built locally only (large, gitignored): fact_flights.parquet (every flight in the window) and
fact_flight_scores.parquet (every test-window flight with model scores).
"""

from __future__ import annotations

import duckdb

from flightops import config

APP_MARTS = ["kpi_overview", "kpi_carrier_month", "kpi_airport_month", "kpi_route",
             "kpi_hour_dow", "delay_causes_month", "cancellation_reasons_month", "daily_network"]


def run() -> None:
    config.ensure_dirs()
    pb, mk, it = config.POWERBI, config.MARTS, config.INTERIM
    con = duckdb.connect(str(config.DB_PATH))
    con.execute("SET memory_limit='2GB'; SET threads=4;")

    for t in APP_MARTS:
        con.execute(f"COPY marts.{t} TO '{mk}/{t}.csv' (HEADER)")

    con.execute(f"""COPY (
        SELECT CAST(strftime(c.flight_date, '%Y%m%d') AS INTEGER) AS date_key,
               c.flight_date AS date, year(c.flight_date) AS year, month(c.flight_date) AS month,
               strftime(c.flight_date, '%b %Y') AS month_label, strftime(c.flight_date, '%Y-%m') AS year_month,
               quarter(c.flight_date) AS quarter, isodow(c.flight_date) AS iso_dow,
               strftime(c.flight_date, '%a') AS dow_name, isodow(c.flight_date) >= 6 AS is_weekend,
               c.is_holiday = 1 AS is_federal_holiday, c.days_to_holiday,
               CASE WHEN c.flight_date <= DATE '{config.TRAIN_END}' THEN 'train'
                    WHEN c.flight_date <= DATE '{config.VALID_END}' THEN 'valid' ELSE 'test' END AS model_split
        FROM features.calendar c
        WHERE c.flight_date BETWEEN (SELECT min(flight_date) FROM staging.stg_flights)
                                AND (SELECT max(flight_date) FROM staging.stg_flights)
        ORDER BY 1) TO '{pb}/dim_date.csv' (HEADER)""")
    con.execute(f"COPY (SELECT carrier_code, carrier_name, flights_in_window FROM staging.stg_carriers "
                f"ORDER BY 3 DESC) TO '{pb}/dim_carrier.csv' (HEADER)")
    con.execute(f"""COPY (
        SELECT a.*, s.segment AS ops_segment
        FROM staging.stg_airports a
        LEFT JOIN read_csv_auto('{mk}/segments_airports.csv') s USING (airport_code)
        ORDER BY 1) TO '{pb}/dim_airport.csv' (HEADER)""")
    con.execute(f"""COPY (
        SELECT r.route, r.origin, r.dest, r.distance_mi, r.scheduled_flights,
               s.segment AS route_segment
        FROM (SELECT route, any_value(origin) origin, any_value(dest) dest,
                     median(distance_mi) distance_mi, count(*) scheduled_flights
              FROM staging.stg_flights GROUP BY route) r
        LEFT JOIN read_csv_auto('{mk}/segments_routes.csv') s USING (route)
        ORDER BY 1) TO '{pb}/dim_route.csv' (HEADER)""")

    fact_sql = """
        SELECT CAST(flight_id AS VARCHAR) AS flight_id,
               CAST(strftime(flight_date, '%Y%m%d') AS INTEGER) AS date_key,
               carrier AS carrier_code, origin AS origin_airport, dest AS dest_airport, route,
               flight_number, dep_hour, crs_dep_hhmm, distance_mi, crs_elapsed_min,
               completed, cancelled, diverted, arr_del15, dep_del15,
               arr_delay_min, dep_delay_min, taxi_out_min, cancellation_reason,
               carrier_delay_min, weather_delay_min, nas_delay_min, security_delay_min,
               late_aircraft_delay_min
        FROM staging.stg_flights"""
    con.execute(f"COPY ({fact_sql}) TO '{pb}/fact_flights.parquet' (FORMAT parquet, COMPRESSION zstd)")
    con.execute(f"""COPY (SELECT * FROM ({fact_sql}) USING SAMPLE reservoir(50000 ROWS) REPEATABLE (42)
                    ORDER BY date_key) TO '{pb}/fact_flights_sample.csv' (HEADER)""")

    # model scores (test window, Jun-Jul 2026)
    con.execute(f"""CREATE OR REPLACE TEMP TABLE scores AS
        SELECT f.flight_id, f.flight_date, f.carrier, f.origin, f.dest, f.route, f.dep_hour,
               f.year_month, f.completed, f.cancelled, f.arr_del15, f.arr_delay_min,
               d.delay_prob, d.delay_pred, m.delay_p50, m.delay_p90, c.cancel_prob
        FROM staging.stg_flights f
        JOIN read_parquet('{it}/scores_cancel_test.parquet') c USING (flight_id)
        LEFT JOIN read_parquet('{it}/scores_delay_test.parquet') d USING (flight_id)
        LEFT JOIN read_parquet('{it}/scores_delay_minutes_test.parquet') m USING (flight_id)""")
    con.execute(f"""COPY (SELECT CAST(flight_id AS VARCHAR) AS flight_id,
                    CAST(strftime(flight_date, '%Y%m%d') AS INTEGER) AS date_key,
                    carrier AS carrier_code, origin AS origin_airport, route, delay_prob, delay_pred,
                    delay_p50, delay_p90, cancel_prob, arr_del15, cancelled FROM scores)
                    TO '{pb}/fact_flight_scores.parquet' (FORMAT parquet, COMPRESSION zstd)""")
    con.execute(f"""COPY (
        SELECT CASE WHEN delay_prob < 0.1 THEN '0-10%' WHEN delay_prob < 0.2 THEN '10-20%'
                    WHEN delay_prob < 0.3 THEN '20-30%' WHEN delay_prob < 0.4 THEN '30-40%'
                    WHEN delay_prob < 0.5 THEN '40-50%' ELSE '50%+' END AS risk_band,
               count(*) AS flights, avg(delay_prob) AS mean_predicted, avg(arr_del15) AS observed_rate
        FROM scores WHERE completed = 1 GROUP BY 1 ORDER BY 1) TO '{pb}/score_risk_bands.csv' (HEADER)""")
    con.execute(f"""COPY (
        SELECT year_month, carrier AS carrier_code, count(*) FILTER (WHERE completed = 1) AS completed_flights,
               avg(delay_prob) AS mean_delay_prob, avg(arr_del15) AS observed_delay_rate,
               avg(delay_p50) AS mean_p50_min, avg(delay_p90) AS mean_p90_min,
               avg(cancel_prob) AS mean_cancel_prob, avg(cancelled) AS observed_cancel_rate
        FROM scores GROUP BY ALL ORDER BY ALL) TO '{pb}/score_carrier_month.csv' (HEADER)""")
    con.execute(f"""COPY (
        SELECT route, origin AS origin_airport, dest AS dest_airport,
               count(*) FILTER (WHERE completed = 1) AS completed_flights,
               avg(delay_prob) AS mean_delay_prob, avg(arr_del15) AS observed_delay_rate,
               avg(delay_p90) AS mean_p90_min, avg(cancel_prob) AS mean_cancel_prob
        FROM scores GROUP BY ALL HAVING count(*) >= 100 ORDER BY mean_delay_prob DESC)
        TO '{pb}/score_route_risk.csv' (HEADER)""")
    con.execute(f"""COPY (
        SELECT dep_hour, count(*) AS flights, avg(delay_prob) AS mean_delay_prob,
               avg(arr_del15) AS observed_delay_rate
        FROM scores WHERE completed = 1 GROUP BY 1 ORDER BY 1) TO '{pb}/score_hour.csv' (HEADER)""")
    for f in ("forecast_backtest", "forecast_next7", "anomalies_network", "anomalies_airport",
              "segments_airports", "segments_routes"):
        con.execute(f"COPY (SELECT * FROM read_csv_auto('{mk}/{f}.csv')) TO '{pb}/{f}.csv' (HEADER)")
    n = con.execute(f"SELECT count(*) FROM '{pb}/fact_flights.parquet'").fetchone()[0]
    con.close()
    print(f"power bi export done: fact_flights rows = {n:,}")


if __name__ == "__main__":
    run()
