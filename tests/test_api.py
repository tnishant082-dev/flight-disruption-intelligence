import math

import pytest
from fastapi.testclient import TestClient

from api.main import app
from flightops.serving import FlightInput, get_artifacts

client = TestClient(app)
REQ = {"carrier": "AA", "origin": "ORD", "dest": "LGA", "flight_date": "2026-07-17",
       "crs_dep_time": 1730, "flight_number": "1234"}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert {"delay_classifier", "cancellation_model", "delay_regression"} <= set(body["models"])


def test_predict_delay():
    r = client.post("/predict/delay", json=REQ)
    assert r.status_code == 200
    b = r.json()
    assert 0 <= b["delay_probability"] <= 1
    assert b["delay_minutes_p90"] >= b["delay_minutes_p50"]
    assert b["risk_band"] in {"low", "moderate", "elevated", "high"}
    assert b["predicted_delayed"] == (b["delay_probability"] >= b["threshold"])


def test_evening_riskier_than_early_morning():
    early = client.post("/predict/delay", json={**REQ, "crs_dep_time": 600}).json()
    late = client.post("/predict/delay", json={**REQ, "crs_dep_time": 1900}).json()
    assert late["delay_probability"] > early["delay_probability"]


def test_predict_cancellation():
    r = client.post("/predict/cancellation", json=REQ)
    assert r.status_code == 200
    assert 0 <= r.json()["cancellation_probability"] <= 1


def test_explain_contributions_add_up():
    r = client.post("/explain", params={"top_k": 22}, json=REQ)
    assert r.status_code == 200
    b = r.json()
    assert len(b["reasons"]) == 22
    art = get_artifacts()
    X = art.build_features(FlightInput(**{**REQ, "flight_date": __import__("datetime").date(2026, 7, 17)}))
    raw = art.delay_model.predict(X)[0]
    logit = b["base_log_odds"] + sum(x["shap_log_odds"] for x in b["reasons"])
    assert math.isclose(1 / (1 + math.exp(-logit)), raw, rel_tol=1e-6)


def test_forecast():
    r = client.get("/forecast", params={"airport": "ATL", "target": "scheduled_departures"})
    assert r.status_code == 200
    b = r.json()
    assert b["horizon_days"] == 7
    assert all(p["forecast"] > 0 for p in b["points"])


@pytest.mark.parametrize("patch", [{"origin": "XXX"}, {"crs_dep_time": 1275}, {"dest": "ORD"},
                                   {"carrier": "ZZ"}])
def test_bad_requests_rejected(patch):
    assert client.post("/predict/delay", json={**REQ, **patch}).status_code == 422


def test_forecast_unknown_airport():
    assert client.get("/forecast", params={"airport": "BOI"}).status_code == 422
