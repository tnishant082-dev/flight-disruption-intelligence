"""FastAPI service for the flight disruption models.

Run: uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query

from api.schemas import (
    CancellationPrediction,
    DelayPrediction,
    Explanation,
    FlightRequest,
    ForecastPoint,
    ForecastResponse,
    Health,
)
from flightops import __version__
from flightops.serving import FlightInput, RequestError, forecast, get_artifacts

app = FastAPI(title="Flight Disruption Intelligence API", version=__version__,
              description="Pre-departure delay / cancellation risk, explanations and airport "
                          "forecasts trained on US DOT BTS on-time data (Aug 2025 - Jul 2026).")


def _features(req: FlightRequest):
    art = get_artifacts()
    try:
        return art, art.build_features(FlightInput(**req.model_dump()))
    except RequestError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/health", response_model=Health)
def health() -> Health:
    art = get_artifacts()
    return Health(status="ok", models=art.registry)


@app.post("/predict/delay", response_model=DelayPrediction)
def predict_delay(req: FlightRequest) -> DelayPrediction:
    art, X = _features(req)
    out = art.predict_delay(X)
    return DelayPrediction(**{k: out[k] for k in DelayPrediction.model_fields if k in out},
                           model_version=art.registry["delay_classifier"]["version"])


@app.post("/predict/cancellation", response_model=CancellationPrediction)
def predict_cancellation(req: FlightRequest) -> CancellationPrediction:
    art, X = _features(req)
    out = art.predict_cancel(X)
    return CancellationPrediction(**out,
                                  model_version=art.registry["cancellation_model"]["version"])


@app.post("/explain", response_model=Explanation)
def explain(req: FlightRequest, top_k: int = Query(8, ge=1, le=22)) -> Explanation:
    art, X = _features(req)
    p = art.predict_delay(X)["delay_probability"]
    return Explanation(delay_probability=p, **art.explain_delay(X, top_k))


@app.get("/forecast", response_model=ForecastResponse)
def get_forecast(airport: str = Query(..., min_length=3, max_length=3, examples=["ATL"]),
                 target: str = Query("arr_delay_rate",
                                     pattern="^(arr_delay_rate|scheduled_departures)$")
                 ) -> ForecastResponse:
    try:
        fut, hist = forecast(airport.upper(), target)
    except RequestError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return ForecastResponse(
        airport=airport.upper(), target=target, horizon_days=len(fut),
        last_observed_date=hist["flight_date"].max().date(),
        points=[ForecastPoint(date=d.date(), forecast=float(v))
                for d, v in zip(fut["flight_date"], fut["forecast"])])
