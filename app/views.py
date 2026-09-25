"""Pages of the Streamlit app."""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MARTS = ROOT / "data" / "marts"
PBI = ROOT / "data" / "powerbi"
METRICS = ROOT / "reports" / "metrics"
FIG = ROOT / "reports" / "figures"
API_URL = os.environ.get("API_URL", "")

BLUE, NAVY, TEAL, ORANGE, RED, GREY = "#1664D9", "#0B2545", "#12A4A4", "#F28C28", "#D64545", "#8A94A6"
PALETTE = [BLUE, TEAL, ORANGE, "#7B61FF", RED, "#3FA34D", "#C9A227", GREY]


def inject_css() -> None:
    st.markdown("""
    <style>
      .block-container {padding-top: 1.6rem; padding-bottom: 1rem; max-width: 1400px;}
      h1 {font-size: 1.65rem !important; margin-bottom: 0 !important;}
      .sub {color:#5B6475; font-size:0.92rem; margin-bottom:0.9rem;}
      .kpi {background:#fff; border:1px solid #E3E8EF; border-radius:10px; padding:12px 14px;
            box-shadow:0 1px 2px rgba(16,24,40,.04);}
      .kpi .l {font-size:0.72rem; letter-spacing:.04em; text-transform:uppercase; color:#5B6475;}
      .kpi .v {font-size:1.55rem; font-weight:700; color:#0B2545; line-height:1.3;}
      .kpi .d {font-size:0.76rem; color:#5B6475;}
      .pill {display:inline-block; padding:2px 10px; border-radius:12px; font-weight:600;
             font-size:0.85rem;}
      [data-testid="stSidebar"] {background:#0B2545;}
      [data-testid="stSidebar"] * {color:#E8EEF7 !important;}
      [data-testid="stSidebarNav"] a[aria-current="page"] {background:rgba(255,255,255,.12);}
    </style>""", unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("### ✈ Flight Disruption Intelligence")
        st.caption("US DOT BTS on-time data · Aug 2025 – Jul 2026 · 7.04M flights")


def header(title: str, sub: str) -> None:
    st.markdown(f"# {title}")
    st.markdown(f"<div class='sub'>{sub}</div>", unsafe_allow_html=True)


def kpi(col, label: str, value: str, detail: str = "") -> None:
    col.markdown(f"<div class='kpi'><div class='l'>{label}</div><div class='v'>{value}</div>"
                 f"<div class='d'>{detail}</div></div>", unsafe_allow_html=True)


def style(fig: go.Figure, h: int = 330) -> go.Figure:
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=40, b=10), template="plotly_white",
                      font=dict(family="Segoe UI, Inter, sans-serif", size=12),
                      legend=dict(orientation="h", y=-0.18), title_font_size=14,
                      colorway=PALETTE)
    return fig


@st.cache_data
def mart(name: str, folder: str = "marts") -> pd.DataFrame:
    return pd.read_csv((MARTS if folder == "marts" else PBI) / f"{name}.csv")


@st.cache_data
def metrics(name: str) -> dict:
    return json.loads((METRICS / f"{name}.json").read_text())


# --------------------------------------------------------------------------- pages
def overview() -> None:
    header("Executive overview", "How reliable was the US domestic network over the last 12 months, "
           "and where does disruption concentrate?")
    o = mart("kpi_overview").iloc[0]
    c = st.columns(5)
    kpi(c[0], "Scheduled flights", f"{int(o.scheduled_flights):,}",
        f"{o.first_date[:10]} → {o.last_date[:10]}")
    kpi(c[1], "On-time arrival", f"{o.on_time_pct:.1f}%", "completed flights, <15 min late")
    kpi(c[2], "Avg arrival delay", f"{o.avg_arr_delay_min:.1f} min",
        f"{o.avg_delay_when_late_min:.0f} min when late")
    kpi(c[3], "Cancellation rate", f"{o.cancellation_rate_pct:.2f}%",
        f"diversions {o.diversion_rate_pct:.2f}%")
    kpi(c[4], "Network", f"{int(o.carriers)} carriers", f"{int(o.origin_airports)} airports · "
        f"{int(o.routes):,} routes")

    cm = mart("kpi_carrier_month")
    monthly = cm.groupby("year_month").apply(lambda g: pd.Series({
        "on_time_pct": (g.on_time_pct * g.completed_flights).sum() / g.completed_flights.sum(),
        "cancellation_rate_pct": (g.cancellation_rate_pct * g.scheduled_flights).sum()
        / g.scheduled_flights.sum(), "flights": g.scheduled_flights.sum()}),
        include_groups=False).reset_index()
    left, r = st.columns([1.35, 1])
    fig = go.Figure()
    fig.add_bar(x=monthly.year_month, y=monthly.flights, name="Flights", marker_color="#C9D8F0",
                yaxis="y2")
    fig.add_scatter(x=monthly.year_month, y=monthly.on_time_pct, name="On-time %",
                    line=dict(color=BLUE, width=3), mode="lines+markers")
    fig.update_layout(title="Monthly on-time % and volume",
                      yaxis=dict(title="On-time %", range=[60, 90]),
                      yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Flights"))
    left.plotly_chart(style(fig), use_container_width=True)

    car = cm.groupby(["carrier", "carrier_name"]).apply(lambda g: pd.Series({
        "on_time_pct": (g.on_time_pct * g.completed_flights).sum() / g.completed_flights.sum(),
        "flights": g.scheduled_flights.sum()}), include_groups=False).reset_index()
    car = car.sort_values("on_time_pct")
    fig = px.bar(car, x="on_time_pct", y="carrier_name", orientation="h", text_auto=".1f",
                 title="On-time % by carrier", color_discrete_sequence=[BLUE])
    fig.update_xaxes(range=[55, 90], title="")
    fig.update_yaxes(title="")
    r.plotly_chart(style(fig), use_container_width=True)

    left, r = st.columns([1, 1])
    dc = mart("delay_causes_month")
    tot = dc[["carrier_min", "weather_min", "nas_min", "security_min", "late_aircraft_min"]].sum()
    tot.index = ["Carrier", "Weather", "National Air System", "Security", "Late aircraft"]
    fig = px.pie(values=tot.values, names=tot.index, hole=0.55, title="Delay minutes by cause",
                 color_discrete_sequence=PALETTE)
    left.plotly_chart(style(fig), use_container_width=True)
    hd = mart("kpi_hour_dow")
    hh = hd.groupby("dep_hour").apply(lambda g: pd.Series({
        "rate": (g.arr_delay_rate_pct * g.scheduled_flights).sum() / g.scheduled_flights.sum(),
        "flights": g.scheduled_flights.sum()}), include_groups=False).reset_index()
    hh = hh[hh.flights > 5000]
    fig = px.bar(hh, x="dep_hour", y="rate", title="Arrival-delay rate by scheduled departure hour",
                 color="rate", color_continuous_scale=["#CFE0FA", BLUE, NAVY])
    fig.update_layout(coloraxis_showscale=False, xaxis_title="Scheduled departure hour",
                      yaxis_title="% delayed 15+ min")
    r.plotly_chart(style(fig), use_container_width=True)

    st.markdown("##### Least reliable routes (≥300 flights)")
    rt = mart("kpi_route").sort_values("on_time_pct").head(12)
    st.dataframe(rt[["route", "scheduled_flights", "carriers", "on_time_pct", "avg_arr_delay_min",
                     "p90_arr_delay_min", "cancellation_rate_pct"]], hide_index=True,
                 use_container_width=True)


def _predict(payload: dict) -> tuple[dict, dict, dict, str]:
    if API_URL:
        try:
            import httpx

            with httpx.Client(base_url=API_URL, timeout=10) as c:
                d = c.post("/predict/delay", json=payload)
                d.raise_for_status()
                cx = c.post("/predict/cancellation", json=payload).json()
                ex = c.post("/explain", json=payload).json()
            return d.json(), cx, ex, f"FastAPI {API_URL}"
        except Exception as exc:  # fall back to in-process models
            st.toast(f"API unavailable ({exc.__class__.__name__}); using local models")
    from flightops.serving import FlightInput, get_artifacts

    art = get_artifacts()
    X = art.build_features(FlightInput(**{**payload, "flight_date": date.fromisoformat(
        payload["flight_date"])}))
    d = art.predict_delay(X)
    return d, art.predict_cancel(X), {"delay_probability": d["delay_probability"],
                                      **art.explain_delay(X)}, "in-process models"


def predictor() -> None:
    header("Delay risk predictor", "Score a scheduled flight with the pre-departure models: "
           "probability of arriving 15+ min late, P50/P90 delay, cancellation risk and the reasons.")
    from flightops.serving import get_artifacts

    art = get_artifacts()
    cats = art.categories
    carriers = pd.read_csv(PBI / "dim_carrier.csv")
    names = dict(zip(carriers.carrier_code, carriers.carrier_name))
    with st.form("flight"):
        c = st.columns(4)
        carrier = c[0].selectbox("Carrier", cats["carrier"], index=cats["carrier"].index("AA"),
                                 format_func=lambda x: f"{x} · {names.get(x, x)}")
        origin = c[1].selectbox("Origin", cats["origin"], index=cats["origin"].index("ORD"))
        dest = c[2].selectbox("Destination", cats["dest"], index=cats["dest"].index("LGA"))
        fdate = c[3].date_input("Flight date", value=date(2026, 7, 17),
                                min_value=date(2025, 8, 1), max_value=date(2027, 12, 31))
        c = st.columns(4)
        dep = c[0].time_input("Scheduled departure (local)", value=pd.Timestamp("17:30").time())
        fnum = c[1].text_input("Flight number (optional)", value="")
        dist = c[2].number_input("Distance mi (optional, else route median)", min_value=0.0,
                                 value=0.0, step=10.0)
        block = c[3].number_input("Block minutes (optional)", min_value=0.0, value=0.0, step=5.0)
        go_btn = st.form_submit_button("Score flight", type="primary")
    if not go_btn and "last_pred" not in st.session_state:
        st.info("Pick a flight and press **Score flight**. Defaults: AA ORD→LGA, Fri 17 Jul 2026, 17:30.")
        return
    payload = {"carrier": carrier, "origin": origin, "dest": dest, "flight_date": fdate.isoformat(),
               "crs_dep_time": dep.hour * 100 + dep.minute,
               "flight_number": fnum or None, "distance_mi": dist or None,
               "crs_elapsed_min": block or None}
    try:
        d, cx, ex, src = _predict(payload)
        st.session_state["last_pred"] = True
    except Exception as exc:
        st.error(str(exc))
        return
    p = d["delay_probability"]
    color = {"low": "#3FA34D", "moderate": "#C9A227", "elevated": ORANGE, "high": RED}[d["risk_band"]]
    c = st.columns([1.1, 1, 1, 1])
    gauge = go.Figure(go.Indicator(mode="gauge+number", value=p * 100,
                                   number={"suffix": "%", "valueformat": ".1f"},
                                   gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
                                          "threshold": {"line": {"color": NAVY, "width": 3},
                                                        "value": d["threshold"] * 100}},
                                   title={"text": "P(arrival 15+ min late)"}))
    c[0].plotly_chart(style(gauge, 230), use_container_width=True)
    kpi(c[1], "Risk band", f"<span class='pill' style='background:{color}22;color:{color}'>"
        f"{d['risk_band'].upper()}</span>", f"decision threshold {d['threshold']:.2f} "
        f"({'flagged' if d['predicted_delayed'] else 'not flagged'})")
    kpi(c[2], "Arrival delay P50 / P90", f"{d['delay_minutes_p50']:.0f} / {d['delay_minutes_p90']:.0f} min",
        "quantile LightGBM (negative = early)")
    kpi(c[3], "Cancellation risk", f"{cx['cancellation_probability'] * 100:.2f}%",
        f"network prior {cx['baseline_rate'] * 100:.2f}% · {cx['model']}")
    st.caption(f"Scored by {src}. Network average delay rate in training: "
               f"{art.priors['delay'] * 100:.1f}%.")
    reasons = pd.DataFrame(ex["reasons"])
    def _fmt(row):
        v = row["value"]
        if isinstance(v, str):
            return v
        if v is None or pd.isna(v):
            return "—"
        if row["feature"] == "crs_dep_min_of_day":
            return f"{int(v) // 60:02d}:{int(v) % 60:02d}"
        return f"{v:,.0f}" if abs(v) >= 100 else f"{v:.3g}"

    reasons["value_txt"] = reasons.apply(_fmt, axis=1)
    reasons["label_full"] = reasons["label"] + " = " + reasons["value_txt"].astype(str)
    reasons = reasons.iloc[::-1]
    fig = go.Figure(go.Bar(x=reasons.shap_log_odds, y=reasons.label_full, orientation="h",
                           marker_color=[RED if v > 0 else TEAL for v in reasons.shap_log_odds]))
    fig.update_layout(title="Why: SHAP contributions (log-odds) - red pushes risk up, teal down",
                      xaxis_title="SHAP value (log-odds)")
    st.plotly_chart(style(fig, 360), use_container_width=True)


def forecasts() -> None:
    header("Airport forecasts", "7-day-ahead departures and arrival-delay rate for the 10 busiest "
           "airports, with rolling-origin backtests against naive baselines.")
    hist = mart("forecast_history")
    bt = mart("forecast_backtest")
    fut = mart("forecast_next7")
    m = metrics("forecaster")["test_metrics"]
    c = st.columns([1, 1, 2])
    airport = c[0].selectbox("Airport", sorted(hist.airport_code.unique()), index=0)
    target = c[1].radio("Target", ["arr_delay_rate", "scheduled_departures"], horizontal=True,
                        format_func=lambda x: "Arrival-delay rate" if x == "arr_delay_rate"
                        else "Departures / day")
    r = m[target]
    fmt = (lambda v: f"{v * 100:.2f} pp") if target == "arr_delay_rate" else (lambda v: f"{v:.1f}")
    c[2].markdown(
        f"**Backtest MAE (8 origins × 7 days, 10 airports)** — LightGBM {fmt(r['lightgbm']['mae'])} · "
        f"seasonal naive {fmt(r['seasonal_naive']['mae'])} · 4-week weekday mean "
        f"{fmt(r['mean_4wk_same_weekday']['mae'])}")
    h = hist[hist.airport_code == airport].copy()
    h["flight_date"] = pd.to_datetime(h.flight_date)
    h = h[h.flight_date > h.flight_date.max() - pd.Timedelta(days=90)]
    b = bt[(bt.airport_code == airport) & (bt.target == target)].copy()
    f = fut[(fut.airport_code == airport) & (fut.target == target)]
    fig = go.Figure()
    fig.add_scatter(x=h.flight_date, y=h[target], name="Actual", line=dict(color=NAVY, width=2))
    fig.add_scatter(x=b.flight_date, y=b.lightgbm, name="LightGBM backtest", mode="markers",
                    marker=dict(color=BLUE, size=6))
    fig.add_scatter(x=b.flight_date, y=b.seasonal_naive, name="Seasonal naive backtest",
                    mode="markers", marker=dict(color=GREY, size=5, symbol="x"))
    fig.add_scatter(x=f.flight_date, y=f.forecast, name="Next 7 days", mode="lines+markers",
                    line=dict(color=ORANGE, width=3, dash="dot"))
    fig.update_layout(title=f"{airport} · last 90 days, backtests and forward forecast")
    st.plotly_chart(style(fig, 420), use_container_width=True)
    tbl = pd.DataFrame(r).T.reset_index().rename(columns={"index": "model"})
    st.dataframe(tbl, hide_index=True, use_container_width=True)
    if r["lightgbm"]["mae"] > r["seasonal_naive"]["mae"]:
        st.warning("For this target LightGBM does **not** beat the seasonal-naive baseline in the "
                   "backtest; the naive forecast is the better choice here.")


def segments() -> None:
    header("Operational segments", "KMeans segmentation of airports and routes on operating "
           "profile (volume, delay/cancellation behaviour, delay-cause mix, schedule shape).")
    kind = st.radio("Segment", ["airports", "routes"], horizontal=True)
    df = mart(f"segments_{kind}")
    prof = mart(f"segment_profile_{kind}")
    m = metrics("segmentation")["test_metrics"][kind]
    st.caption(f"k = {m['k']} (silhouette {m['silhouette']:.3f}; chosen from k = 4-7). "
               "Labels come from each centroid's size tier and two most distinctive traits.")
    left, r = st.columns([1.3, 1])
    hover = "airport_code" if kind == "airports" else "route"
    fig = px.scatter(df, x="pc1", y="pc2", color="segment", hover_name=hover,
                     title="Segments in PCA space", color_discrete_sequence=PALETTE, opacity=0.8)
    left.plotly_chart(style(fig, 440), use_container_width=True)
    if kind == "airports":
        fig = px.scatter_geo(df, lat="latitude", lon="longitude", color="segment", scope="usa",
                             size="scheduled_departures", hover_name="airport_code",
                             title="Airports by segment", color_discrete_sequence=PALETTE)
        fig.update_layout(showlegend=False)
        r.plotly_chart(style(fig, 440), use_container_width=True)
    else:
        fig = px.box(df, x="segment", y="on_time_pct", color="segment", title="On-time % by segment",
                     color_discrete_sequence=PALETTE)
        fig.update_layout(showlegend=False, xaxis_title="")
        fig.update_xaxes(showticklabels=False)
        r.plotly_chart(style(fig, 440), use_container_width=True)
    st.markdown("##### Segment profiles (centroid means)")
    st.dataframe(prof.round(3), hide_index=True, use_container_width=True)


def anomalies() -> None:
    header("Disruption days", "Network days flagged by STL residuals and airport-days flagged by "
           "Isolation Forest.")
    net = mart("anomalies_network")
    net["flight_date"] = pd.to_datetime(net.flight_date)
    m = metrics("anomaly_detector")["test_metrics"]
    c = st.columns(4)
    kpi(c[0], "Network days flagged", f"{m['network_days_flagged']} / {m['network_days']}",
        "robust z > 3.5 on STL residual")
    kpi(c[1], "Airport-days flagged", f"{m['airport_days_flagged']:,} / {m['airport_days']:,}",
        "Isolation Forest, top-30 airports")
    kpi(c[2], "Delay rate on flagged airport-days",
        f"{m['flagged_airport_days_delay_rate_mean'] * 100:.1f}%",
        f"vs {m['normal_airport_days_delay_rate_mean'] * 100:.1f}% otherwise")
    kpi(c[3], "Cancel rate on flagged airport-days",
        f"{m['flagged_airport_days_cancel_rate_mean'] * 100:.1f}%",
        f"vs {m['normal_airport_days_cancel_rate_mean'] * 100:.2f}% otherwise")
    fig = go.Figure()
    fig.add_scatter(x=net.flight_date, y=net.arr_delay_rate * 100, name="Delay rate %",
                    line=dict(color=BLUE))
    fig.add_scatter(x=net.flight_date, y=net.cancellation_rate * 100, name="Cancellation rate %",
                    line=dict(color=ORANGE), yaxis="y2")
    a = net[net.is_anomaly == 1]
    fig.add_scatter(x=a.flight_date, y=a.arr_delay_rate * 100, mode="markers", name="Flagged day",
                    marker=dict(color=RED, size=10, symbol="diamond"))
    fig.update_layout(title="Daily network delay and cancellation rate",
                      yaxis=dict(title="Delay rate %"),
                      yaxis2=dict(title="Cancel rate %", overlaying="y", side="right",
                                  showgrid=False))
    st.plotly_chart(style(fig, 380), use_container_width=True)
    left, r = st.columns([1, 1.2])
    left.markdown("##### Flagged network days")
    left.dataframe(pd.DataFrame(m["top_network_anomalies"]), hide_index=True, use_container_width=True,
                height=320)
    ad = mart("anomalies_airport")
    heat = ad[ad.is_anomaly == 1].copy()
    heat["month"] = heat.flight_date.str[:7]
    piv = heat.pivot_table(index="airport_code", columns="month", values="is_anomaly",
                           aggfunc="sum", fill_value=0)
    piv = piv.loc[piv.sum(axis=1).sort_values(ascending=False).index[:15]]
    fig = px.imshow(piv, color_continuous_scale=["#F4F7FB", ORANGE, RED], aspect="auto",
                    title="Flagged airport-days by month (top 15 airports)")
    r.plotly_chart(style(fig, 360), use_container_width=True)


def performance() -> None:
    header("Model performance", "Held-out test window Jun–Jul 2026 (time-based split; train "
           "Aug 2025–Mar 2026, validation Apr–May 2026). All numbers are from the saved run.")
    dc = metrics("delay_classifier")
    rows = []
    for k, v in dc["test_metrics"].items():
        rows.append({"model": k, "ROC-AUC": v["roc_auc"], "PR-AUC": v["pr_auc"], "F1": v["f1"],
                     "Brier": v["brier"]})
    st.markdown("##### 1 · Arrival delay >15 min (classification)")
    st.caption(f"Train sample {dc['train_rows']:,} · validation {dc['valid_rows']:,} · test "
               f"{dc['test_rows']:,} completed flights. Test delay rate "
               f"{dc['test_metrics']['prior_rate']['positive_rate']:.3f}. "
               "`dayofops_*` uses actual aircraft rotation - not available at scheduling time, "
               "shown for context only.")
    st.dataframe(pd.DataFrame(rows).round(4), hide_index=True, use_container_width=True)
    st.image(str(FIG / "delay_classifier_curves.png"))
    left, r = st.columns(2)
    left.image(str(FIG / "shap_bar_delay.png"), caption="Global SHAP importance (3,000 test flights)")
    r.image(str(FIG / "shap_beeswarm_delay.png"))

    st.markdown("##### 2 · Delay minutes (quantile regression)")
    rg = metrics("delay_regression")["test_metrics"]
    st.dataframe(pd.DataFrame([{"model": k, "MAE": v["mae"], "RMSE": v["rmse"],
                                "Median AE": v["median_ae"]} for k, v in rg.items() if "mae" in v]
                              ).round(2), hide_index=True, use_container_width=True)
    iv = rg["interval"]
    st.caption(f"P90 coverage {iv['p90_coverage']:.3f} (target 0.90) · P10–P90 coverage "
               f"{iv['p10_p90_coverage']:.3f} (target 0.80) · mean width "
               f"{iv['mean_p10_p90_width_min']:.1f} min")

    st.markdown("##### 3 · Cancellation risk (imbalanced)")
    cm = metrics("cancellation_model")
    st.dataframe(pd.DataFrame([{"model": k, "PR-AUC": v["pr_auc"], "ROC-AUC": v["roc_auc"],
                                "F1": v["f1"], "Brier": v["brier"]}
                               for k, v in cm["test_metrics"].items() if "pr_auc" in v]).round(4),
                 hide_index=True, use_container_width=True)
    vp, tm = cm["validation_pr_auc"], cm["test_metrics"]
    st.caption(f"Served model: calibrated logistic regression (class-weighted, Platt scaling). "
               f"Validation PR-AUC slightly favoured LightGBM ({vp['lightgbm_calibrated']:.4f} vs "
               f"{vp['logistic_regression_balanced']:.4f}), but on the test window, where "
               f"{tm['prior_rate']['positive_rate']:.2%} of flights were cancelled vs "
               f"{cm['valid_positive_rate']:.2%} in validation, logistic regression generalised "
               f"better (PR-AUC {tm['logistic_regression_calibrated']['pr_auc']:.4f} vs "
               f"{tm['lightgbm_calibrated']['pr_auc']:.4f}).")
    st.image(str(FIG / "cancellation_curves.png"))

    st.markdown("##### 4 · Forecasting backtest (MAE)")
    fc = metrics("forecaster")["test_metrics"]
    st.dataframe(pd.DataFrame({t: {k: v["mae"] for k, v in d.items()} for t, d in fc.items()}
                              ).round(4), use_container_width=True)
