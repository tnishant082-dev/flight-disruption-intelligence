"""SQL layer + data-quality gate on the committed BTS sample."""

from flightops.quality import run_checks


def test_staging_keeps_all_valid_rows(sample_db):
    raw = sample_db.execute("SELECT count(*) FROM raw.flights").fetchone()[0]
    stg = sample_db.execute("SELECT count(*) FROM staging.stg_flights").fetchone()[0]
    q = sample_db.execute("SELECT count(*) FROM staging.quarantine_flights").fetchone()[0]
    assert raw == 20000
    assert stg + q == raw


def test_derived_columns(sample_db):
    bad = sample_db.execute("""SELECT count(*) FROM staging.stg_flights
                               WHERE dep_hour NOT BETWEEN 0 AND 23
                                  OR crs_dep_min_of_day NOT BETWEEN 0 AND 1439
                                  OR route <> origin || '-' || dest
                                  OR (completed = 1 AND arr_del15 IS NULL)
                                  OR (completed = 0 AND arr_del15 IS NOT NULL)""").fetchone()[0]
    assert bad == 0


def test_quality_gate_passes_on_sample(sample_db):
    checks = run_checks(sample_db)
    failed = [c.name for c in checks if c.status == "FAIL"]
    assert not failed, failed


def test_quality_gate_catches_duplicates_and_bad_times(sample_db):
    sample_db.execute("""CREATE OR REPLACE TEMP TABLE broken AS
                         SELECT * FROM staging.stg_flights UNION ALL
                         (SELECT * FROM staging.stg_flights LIMIT 5)""")
    sample_db.execute("UPDATE broken SET crs_dep_hhmm = 2575 WHERE rowid < 3")
    status = {c.name: c.status for c in run_checks(sample_db, table="broken", raw_table=None)}
    assert status["duplicate_flight_id"] == "FAIL"
    assert status["valid_scheduled_dep_time"] == "FAIL"


def test_marts_are_consistent(sample_db):
    total = sample_db.execute("SELECT scheduled_flights FROM marts.kpi_overview").fetchone()[0]
    by_carrier = sample_db.execute("SELECT sum(scheduled_flights) FROM marts.kpi_carrier_month"
                                   ).fetchone()[0]
    daily = sample_db.execute("SELECT sum(scheduled_flights) FROM marts.daily_network").fetchone()[0]
    assert total == by_carrier == daily
    otp = sample_db.execute("SELECT on_time_pct FROM marts.kpi_overview").fetchone()[0]
    assert 50 < otp < 95


def test_airports_enriched(sample_db):
    miss = sample_db.execute("SELECT count(*) FROM staging.stg_airports WHERE latitude IS NULL"
                             ).fetchone()[0]
    assert miss == 0
