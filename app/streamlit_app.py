"""Flight Disruption Intelligence - Streamlit front end.

Run: streamlit run app/streamlit_app.py   (set API_URL to use the FastAPI service)
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from app import views  # noqa: E402

st.set_page_config(page_title="Flight Disruption Intelligence", page_icon="✈",
                   layout="wide", initial_sidebar_state="expanded")
views.inject_css()

pages = [
    st.Page(views.overview, title="Executive overview", icon=":material/dashboard:",
            url_path="overview", default=True),
    st.Page(views.predictor, title="Delay risk predictor", icon=":material/flight_takeoff:",
            url_path="predictor"),
    st.Page(views.forecasts, title="Forecasts", icon=":material/trending_up:", url_path="forecasts"),
    st.Page(views.segments, title="Segments", icon=":material/hub:", url_path="segments"),
    st.Page(views.anomalies, title="Disruption days", icon=":material/warning:", url_path="anomalies"),
    st.Page(views.performance, title="Model performance", icon=":material/insights:",
            url_path="models"),
]
st.navigation(pages).run()
