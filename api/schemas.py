"""Pydantic request / response models for the prediction API."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class FlightRequest(BaseModel):
    carrier: str = Field(..., min_length=2, max_length=3, examples=["AA"],
                         description="BTS reporting carrier code")
    origin: str = Field(..., min_length=3, max_length=3, examples=["ORD"])
    dest: str = Field(..., min_length=3, max_length=3, examples=["LGA"])
    flight_date: date = Field(..., examples=["2026-07-17"])
    crs_dep_time: int = Field(..., ge=0, le=2400, examples=[1730],
                              description="Scheduled departure, local time HHMM")
    crs_arr_time: int | None = Field(None, ge=0, le=2400, description="Scheduled arrival HHMM")
    flight_number: str | None = Field(None, max_length=6, examples=["1234"])
    distance_mi: float | None = Field(None, gt=0)
    crs_elapsed_min: float | None = Field(None, gt=0)
    origin_sched_deps_day: float | None = Field(None, gt=0, description="Optional schedule count; "
                                                "defaults to the typical value for origin/weekday")
    origin_sched_deps_hour: float | None = Field(None, gt=0)
    dest_sched_arrs_hour: float | None = Field(None, gt=0)
    carrier_sched_day: float | None = Field(None, gt=0)


class DelayPrediction(BaseModel):
    delay_probability: float
    risk_band: str
    threshold: float
    predicted_delayed: bool
    delay_minutes_p50: float
    delay_minutes_p90: float
    model_version: str


class CancellationPrediction(BaseModel):
    cancellation_probability: float
    baseline_rate: float
    relative_risk: float | None
    model: str
    model_version: str


class Reason(BaseModel):
    feature: str
    label: str
    value: float | str | None
    shap_log_odds: float
    direction: str


class Explanation(BaseModel):
    delay_probability: float
    base_log_odds: float
    reasons: list[Reason]


class ForecastPoint(BaseModel):
    date: date
    forecast: float


class ForecastResponse(BaseModel):
    airport: str
    target: str
    horizon_days: int
    last_observed_date: date
    points: list[ForecastPoint]


class Health(BaseModel):
    status: str
    models: dict[str, dict]
