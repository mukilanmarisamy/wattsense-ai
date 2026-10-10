import importlib.util
import inspect
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analysis import (APPLIANCES, UNUSUAL, EvidenceModel, fault_report, learn_hour_samples,
                      reconstruct_runs, tn_bill, usage_table, usual_hours)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

SIM_PATH = DATA_DIR / "sim_day.csv"
EVENTS_PATH = DATA_DIR / "events.csv"
SUMMARY_PATH = DATA_DIR / "model_summary.json"
INFO_PATH = DATA_DIR / "run_info.json"

# One colour per appliance, reused in every chart
COLORS = {
    "baseline": "#6B7A8C",
    "fridge": "#4CC9F0",
    "ac": "#9D8DF1",
    "microwave": "#F5B83D",
    "kettle": "#F25C54",
    "unknown": "#9AA5B1",
    UNUSUAL: "#6B7A8C",
}
NAMES = {"ac": "AC", "fridge": "Fridge", "microwave": "Microwave", "kettle": "Kettle",
         "baseline": "Baseline", UNUSUAL: "Unusual event", "unknown": "Unknown"}
FAULT_OPTIONS = {
    "Healthy home": None,
    "Fridge compressor wearing out": "fridge",
    "AC stuck running": "ac",
}
EVENT_COLUMNS_FOR_CSV = {
    "timestamp": "time",
    "event_type": "event_type",
    "delta_p": "power_change_W",
    "delta_q": "reactive_power_change_VAR",
    "inrush_ratio": "startup_spike_x",
    "ramp_speed": "ramp_speed_W_per_s",
    "hour_of_day": "hour_of_day_h",
    "appliance_label": "appliance",
}


def wide(fn):
    """Full-width argument that works on both older and newer Streamlit versions."""
    try:
        default = inspect.signature(fn).parameters["width"].default
    except (KeyError, ValueError, TypeError):
        default = None
    if isinstance(default, str):
        return {} if default == "stretch" else {"width": "stretch"}
    return {"use_container_width": True}


WIDE_PLOT = wide(st.plotly_chart)
WIDE_TABLE = wide(st.dataframe)
WIDE_BUTTON = wide(st.button)


def nice(name):
    return NAMES.get(name, str(name).capitalize())


# ----------------------------------------------------------------------------
# Pipeline (same calls as before)
# ----------------------------------------------------------------------------
def load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@st.cache_data
def load_simulation_data():
    if not SIM_PATH.exists():
        return None
    return pd.read_csv(SIM_PATH, parse_dates=[0], index_col=0)


@st.cache_data
def load_event_data():
    if not EVENTS_PATH.exists():
        return None
    return pd.read_csv(EVENTS_PATH, parse_dates=["timestamp"])


@st.cache_data
def load_model_summary():
    if not SUMMARY_PATH.exists():
        return None
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_run_info():
    if not INFO_PATH.exists():
        return {"seed": 42, "fault": None}
    return json.loads(INFO_PATH.read_text(encoding="utf-8"))


@st.cache_data
def get_runs(events_df, day_end):
    return reconstruct_runs(events_df, day_end)


@st.cache_resource
def get_hour_samples():
    simulator = load_module("simulator_hours", ROOT / "simulator.py")
    return learn_hour_samples(simulator.generate_day)


@st.cache_resource
def get_evidence_model(events_df):
    on_events = events_df[events_df["event_type"] == "ON"]
    return EvidenceModel(on_events, get_hour_samples())


def generate_simulation(seed=42, fault=None):
    simulator = load_module("simulator", ROOT / "simulator.py")
    if "fault" in inspect.signature(simulator.generate_day).parameters:
        df = simulator.generate_day(seed=seed, fault=fault)
    else:
        st.warning("simulator.py is the old version, so the problem setting was ignored.")
        fault = None
        df = simulator.generate_day(seed=seed)
    df.to_csv(SIM_PATH)
    INFO_PATH.write_text(json.dumps({"seed": int(seed), "fault": fault}), encoding="utf-8")
    st.cache_data.clear()
    return df


def extract_events():
    extractor = load_module("event_extractor", ROOT / "event-extractor.py")
    events_df = extractor.extract_events(SIM_PATH, EVENTS_PATH)
    st.cache_data.clear()
    return events_df


def train_classifier():
    trainer = load_module("ml_train", ROOT / "ml_train.py")
    results = trainer.train_model(EVENTS_PATH, SUMMARY_PATH)
    st.cache_data.clear()
    return results


def run_pipeline(seed=42, fault=None):
    generate_simulation(seed, fault)
    extract_events()
    train_classifier()


# ----------------------------------------------------------------------------
# Page setup and styling
# ----------------------------------------------------------------------------
st.set_page_config(page_title="WattSense AI", page_icon="⚡", layout="wide")

st.markdown(
    """
<style>
.block-container { padding-top: 1.6rem; max-width: 1200px; }

.ws-hero {
    border-left: 4px solid #F5B83D;
    padding: 0.2rem 0 0.2rem 1rem;
    margin-bottom: 1.2rem;
}
.ws-hero h1 { margin: 0; font-size: 2.1rem; line-height: 1.15; }
.ws-hero p { margin: 0.35rem 0 0 0; color: #A9B6C4; font-size: 1.02rem; max-width: 46rem; }

div[data-testid="stMetric"] {
    background: #172230;
    border: 1px solid #25364A;
    border-radius: 10px;
    padding: 14px 16px;
}
div[data-testid="stMetricLabel"] p { color: #A9B6C4; }

button[data-baseweb="tab"] { font-size: 1rem; }
div[data-testid="stDataFrame"] { border-radius: 10px; }

@media (max-width: 640px) {
    .ws-hero h1 { font-size: 1.6rem; }
    .block-container { padding-left: 1rem; padding-right: 1rem; }
}
</style>
""",
    unsafe_allow_html=True,
)


def style_fig(fig, height=380, y_suffix=None, y_prefix=None, x_suffix=None):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=1.12, x=0),
        font=dict(size=13),
    )
    fig.update_xaxes(gridcolor="#25364A")
    fig.update_yaxes(gridcolor="#25364A")
    if y_suffix:
        fig.update_yaxes(ticksuffix=y_suffix)
    if y_prefix:
        fig.update_yaxes(tickprefix=y_prefix)
    if x_suffix:
        fig.update_xaxes(ticksuffix=x_suffix)
    return fig


def show_table(df):
    st.dataframe(df, **WIDE_TABLE, hide_index=True)


def clock(hour):
    return f"{int(hour):02d}:00"


# ----------------------------------------------------------------------------
# First load: build the data if it is missing (needed on a fresh deploy)
# ----------------------------------------------------------------------------
if "booted" not in st.session_state:
    st.session_state.booted = True
    if not (SIM_PATH.exists() and EVENTS_PATH.exists() and SUMMARY_PATH.exists()):
        try:
            with st.spinner("Setting up: simulating a day, finding events, training the model..."):
                run_pipeline(42)
        except Exception as exc:
            st.error(f"Automatic setup failed: {exc}")

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Controls")
    seed = st.number_input("Simulated day (seed number)", min_value=0, max_value=9999, value=42, step=1)
    fault_label = st.selectbox("Simulate a problem", list(FAULT_OPTIONS))
    fault = FAULT_OPTIONS[fault_label]
    period_days = st.number_input("Billing period (days)", min_value=1, max_value=120, value=60, step=1)

    if st.button("Run full pipeline", type="primary", **WIDE_BUTTON):
        try:
            with st.spinner("Simulating, extracting events and training..."):
                run_pipeline(int(seed), fault)
            st.success("Pipeline finished.")
        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")

    with st.expander("Run one step at a time"):
        if st.button("1. Generate simulation data", **WIDE_BUTTON):
            with st.spinner("Generating a smart-meter day..."):
                generate_simulation(int(seed), fault)
            st.success("Saved data/sim_day.csv")

        if st.button("2. Extract power events", **WIDE_BUTTON):
            if not SIM_PATH.exists():
                st.warning("Generate simulation data first.")
            else:
                with st.spinner("Scanning for appliance events..."):
                    extract_events()
                st.success("Saved data/events.csv")

        if st.button("3. Train classifier", **WIDE_BUTTON):
            if not EVENTS_PATH.exists():
                st.warning("Extract events first.")
            else:
                with st.spinner("Training the model..."):
                    train_classifier()
                st.success("Saved data/model_summary.json")

    st.caption("The same seed number gives the same simulated day. The bill repeats that day for the whole billing period.")

simulation_df = load_simulation_data()
events_df = load_event_data()
model_summary = load_model_summary()

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    """
<div class="ws-hero">
  <h1>WattSense AI</h1>
  <p>One smart meter sees the whole house's power. WattSense finds the moments an appliance
  switches on or off and works out which appliance it was.</p>
</div>
""",
    unsafe_allow_html=True,
)

(tab_sim, tab_events, tab_model,
 tab_bill, tab_fault, tab_live) = st.tabs(["Meter data", "Detected events", "Appliance classifier",
                                           "Appliance bill", "Fault alerts", "Try the model"])

# ----------------------------------------------------------------------------
# Tab 1: meter data
# ----------------------------------------------------------------------------
with tab_sim:
    if simulation_df is None:
        st.info("No meter data yet. Click Run full pipeline in the sidebar.")
    else:
        total_kwh = simulation_df["total_p"].sum() / 3600 / 1000
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Energy used in the day", f"{total_kwh:.1f} kWh")
        c2.metric("Peak load", f"{simulation_df['total_p'].max():,.0f} W")
        c3.metric("Average load", f"{simulation_df['total_p'].mean():,.0f} W")
        c4.metric("Meter readings", f"{len(simulation_df):,} readings")

        st.subheader("What the meter sees over the day")
        per_minute = simulation_df["total_p"].resample("1min").mean()
        fig_day = go.Figure(
            go.Scatter(x=per_minute.index, y=per_minute.values, mode="lines",
                       line=dict(color="#F5B83D", width=1.5), name="Total power",
                       hovertemplate="%{x|%H:%M}<br>%{y:,.0f} W<extra></extra>")
        )
        fig_day.update_yaxes(title="Power (W)")
        fig_day.update_xaxes(title="Time of day (hh:mm)", tickformat="%H:%M")
        st.plotly_chart(style_fig(fig_day, 300, y_suffix=" W"), **WIDE_PLOT)
        st.caption("Averaged over each 1 minute so the chart loads quickly.")

        st.subheader("Who used the power: 12:30 to 14:30")
        st.caption("The line is the total the meter reports. The coloured areas are the real appliance contributions from the simulator.")
        window = simulation_df.loc["2026-01-01 12:30":"2026-01-01 14:30"]
        fig_zoom = go.Figure()
        for name in ["baseline"] + APPLIANCES:
            col = f"{name}_p"
            if col in window.columns:
                fig_zoom.add_trace(
                    go.Scatter(x=window.index, y=window[col], name=nice(name), stackgroup="one",
                               mode="lines", line=dict(width=0.5, color=COLORS[name]),
                               fillcolor=COLORS[name],
                               hovertemplate=f"{nice(name)}: " + "%{y:,.0f} W<extra></extra>")
                )
        fig_zoom.add_trace(
            go.Scatter(x=window.index, y=window["total_p"], name="Meter total",
                       mode="lines", line=dict(color="#FFFFFF", width=1),
                       hovertemplate="Meter total: %{y:,.0f} W<extra></extra>")
        )
        fig_zoom.update_yaxes(title="Power (W)")
        fig_zoom.update_xaxes(title="Time of day (hh:mm)", tickformat="%H:%M")
        st.plotly_chart(style_fig(fig_zoom, 420, y_suffix=" W"), **WIDE_PLOT)

        st.subheader("Energy by appliance")
        rows = []
        for name in ["baseline"] + APPLIANCES:
            col = f"{name}_p"
            if col in simulation_df.columns:
                rows.append({"appliance": nice(name), "key": name,
                             "kWh": simulation_df[col].sum() / 3600 / 1000})
        kwh_df = pd.DataFrame(rows)
        fig_kwh = px.bar(kwh_df, x="appliance", y="kWh", color="key", color_discrete_map=COLORS)
        fig_kwh.update_traces(texttemplate="%{y:.2f} kWh", textposition="outside",
                              hovertemplate="%{x}: %{y:.2f} kWh<extra></extra>")
        fig_kwh.update_layout(showlegend=False)
        fig_kwh.update_yaxes(title="Energy in the day (kWh)")
        fig_kwh.update_xaxes(title=None)
        st.plotly_chart(style_fig(fig_kwh, 320, y_suffix=" kWh"), **WIDE_PLOT)

# ----------------------------------------------------------------------------
# Tab 2: events
# ----------------------------------------------------------------------------
with tab_events:
    if events_df is None:
        st.info("No events yet. Click Run full pipeline in the sidebar.")
    else:
        n_on = int((events_df["event_type"] == "ON").sum())
        n_off = int((events_df["event_type"] == "OFF").sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Events found", f"{len(events_df):,} events")
        c2.metric("Switch-on events", f"{n_on:,} events")
        c3.metric("Switch-off events", f"{n_off:,} events")

        st.subheader("Each appliance has its own signature")
        st.caption(
            "Every dot is a switch-on event. Microwave and kettle draw similar power (W), "
            "but they separate on reactive power (VAR)."
        )
        on_df = events_df[events_df["event_type"] == "ON"].copy()
        on_df["appliance"] = on_df["appliance_label"].map(nice)
        color_map = {nice(k): v for k, v in COLORS.items()}
        fig_sig = px.scatter(
            on_df, x="delta_p", y="delta_q", color="appliance", color_discrete_map=color_map,
            custom_data=["timestamp", "inrush_ratio", "ramp_speed"],
            labels={"delta_p": "Change in power (W)", "delta_q": "Change in reactive power (VAR)",
                    "appliance": "Appliance"},
        )
        fig_sig.update_traces(
            marker=dict(size=10, line=dict(width=0.5, color="#0F1720")),
            hovertemplate=("Power change: %{x:,.0f} W<br>Reactive power change: %{y:,.0f} VAR<br>"
                           "Startup spike: %{customdata[1]:.2f}×<br>Ramp speed: %{customdata[2]:.1f} W/s<br>"
                           "Time: %{customdata[0]|%H:%M:%S}<extra></extra>"),
        )
        st.plotly_chart(style_fig(fig_sig, 420, y_suffix=" VAR", x_suffix=" W"), **WIDE_PLOT)

        st.subheader("Event table")
        event_view = pd.DataFrame({
            "Time": events_df["timestamp"].dt.strftime("%H:%M:%S"),
            "Type": events_df["event_type"].map({"ON": "Switch on", "OFF": "Switch off"}),
            "Power change": events_df["delta_p"].map(lambda v: f"{v:,.0f} W"),
            "Reactive power change": events_df["delta_q"].map(lambda v: f"{v:,.0f} VAR"),
            "Startup spike": events_df["inrush_ratio"].map(lambda v: f"{v:.2f}×"),
            "Ramp speed": events_df["ramp_speed"].map(lambda v: f"{v:.1f} W/s"),
            "Hour of day": events_df["hour_of_day"].map(clock),
            "Appliance": events_df["appliance_label"].map(nice),
        })
        st.dataframe(event_view, **WIDE_TABLE, hide_index=True, height=320)
        csv_df = events_df.rename(columns=EVENT_COLUMNS_FOR_CSV)
        st.download_button(
            "Download events as CSV",
            data=csv_df.to_csv(index=False).encode("utf-8"),
            file_name="events.csv",
            mime="text/csv",
        )

# ----------------------------------------------------------------------------
# Tab 3: classifier
# ----------------------------------------------------------------------------
with tab_model:
    if model_summary is None:
        st.info("No model results yet. Click Run full pipeline in the sidebar.")
    else:
        accuracy = float(model_summary.get("accuracy", 0.0))
        report = {k: v for k, v in model_summary.get("report", {}).items() if isinstance(v, dict)}
        report_df = pd.DataFrame(report).T

        macro_f1 = report_df["f1-score"].get("macro avg") if "f1-score" in report_df else None
        c1, c2 = st.columns(2)
        c1.metric("Accuracy on test events", f"{accuracy * 100:.1f} %")
        if macro_f1 is not None:
            c2.metric("Average F1 score across appliances", f"{macro_f1 * 100:.0f} %")

        class_df = report_df.drop(index=[i for i in ("macro avg", "weighted avg") if i in report_df.index])
        if not class_df.empty:
            st.subheader("Score per appliance")
            long_df = (
                class_df[["precision", "recall", "f1-score"]]
                .rename_axis("appliance").reset_index()
                .melt(id_vars="appliance", var_name="metric", value_name="score")
            )
            long_df["appliance"] = long_df["appliance"].map(nice)
            long_df["metric"] = long_df["metric"].map({"precision": "Precision", "recall": "Recall",
                                                       "f1-score": "F1 score"})
            long_df["score"] = long_df["score"] * 100
            fig_scores = px.bar(long_df, x="appliance", y="score", color="metric", barmode="group",
                                color_discrete_sequence=["#4CC9F0", "#9D8DF1", "#F5B83D"])
            fig_scores.update_traces(texttemplate="%{y:.0f} %", textposition="outside",
                                     hovertemplate="%{x}: %{y:.1f} %<extra></extra>")
            fig_scores.update_yaxes(range=[0, 115], title="Score (%)")
            fig_scores.update_xaxes(title=None)
            st.plotly_chart(style_fig(fig_scores, 340, y_suffix=" %"), **WIDE_PLOT)

        cm = model_summary.get("confusion_matrix")
        labels = model_summary.get("labels")
        if cm and labels:
            st.subheader("Confusion matrix")
            nice_labels = [nice(x) for x in labels]
            fig_cm = px.imshow(cm, x=nice_labels, y=nice_labels, text_auto=True,
                               color_continuous_scale=["#172230", "#F5B83D"],
                               labels=dict(x="Predicted appliance", y="Actual appliance",
                                           color="Events (count)"))
            fig_cm.update_traces(texttemplate="%{z} events",
                                 hovertemplate="Actual %{y}, predicted %{x}: %{z} events<extra></extra>")
            st.plotly_chart(style_fig(fig_cm, 400), **WIDE_PLOT)

        st.subheader("Full report")
        report_view = pd.DataFrame({
            "Appliance": [{"macro avg": "Average (each appliance equal)",
                           "weighted avg": "Average (weighted by events)"}.get(i, nice(i))
                          for i in report_df.index],
            "Precision": report_df["precision"].map(lambda v: f"{v * 100:.1f} %").values,
            "Recall": report_df["recall"].map(lambda v: f"{v * 100:.1f} %").values,
            "F1 score": report_df["f1-score"].map(lambda v: f"{v * 100:.1f} %").values,
            "Test events": report_df["support"].map(lambda v: f"{int(v):,} events").values,
        })
        show_table(report_view)

# ----------------------------------------------------------------------------
# Tab 4: appliance bill (Tamil Nadu slabs)
# ----------------------------------------------------------------------------
have_both = simulation_df is not None and events_df is not None
runs = get_runs(events_df, simulation_df.index[-1]) if have_both else None

with tab_bill:
    if not have_both:
        st.info("Run the pipeline first. The bill needs meter data and detected events.")
    else:
        st.write(
            "The model names each switch-on and switch-off. Matching them gives how long each "
            "appliance ran, and its power gives the energy. The bill then follows the Tamil Nadu "
            "slab rates. No sensor on any appliance is used."
        )

        usage = usage_table(runs, simulation_df)
        day_kwh = simulation_df["total_p"].sum() / 3600 / 1000
        period_kwh = day_kwh * period_days
        scheme, slab_df, bill_total = tn_bill(period_kwh)
        billed_units = int(round(period_kwh))

        c1, c2, c3 = st.columns(3)
        c1.metric(f"Projected use over {period_days} days", f"{billed_units:,} units (kWh)")
        c2.metric("Projected bill", f"₹{bill_total:,.0f}")
        c3.metric("Average price paid", f"₹{bill_total / max(billed_units, 1):.2f} per unit")
        st.caption(
            f"Slab used: {scheme}. 1 unit = 1 kWh. The projection repeats this simulated day "
            f"for {period_days} days. Energy charges from the slab rates only."
        )

        # What the 500-unit line means for this household
        at_500 = tn_bill(500)[2]
        just_over = tn_bill(501)[2]
        if billed_units > 500:
            cut_units = billed_units - 500
            st.info(
                f"Above 500 units, the bill uses the higher-rate scheme. Staying at 500 units "
                f"would cost ₹{at_500:,.0f}, which is ₹{bill_total - at_500:,.0f} less. "
                f"That needs {cut_units:,} units less over {period_days} days, "
                f"about {cut_units / period_days:.1f} kWh less per day."
            )
        else:
            st.info(
                f"At 500 units the bill is ₹{at_500:,.0f}. At 501 units it becomes ₹{just_over:,.0f}, "
                f"because the scheme changes above 500 units. You are at {billed_units:,} units."
            )

        st.subheader("How the bill is calculated")
        slab_view = pd.DataFrame({
            "Slab": slab_df["slab"],
            "Units used in slab": slab_df["units"].map(lambda v: f"{int(v):,} units"),
            "Price": slab_df["rate"].map(lambda r: "Free (₹0.00 per unit)" if r == 0 else f"₹{r:.2f} per unit"),
            "Amount": slab_df["amount"].map(lambda v: f"₹{v:,.2f}"),
        })
        show_table(slab_view)

        st.subheader("Cost by appliance")
        tracked = usage.copy()
        tracked["period_kwh"] = tracked["estimated_kwh"] * period_days
        scale = bill_total / period_kwh if period_kwh > 0 else 0.0
        tracked["cost"] = tracked["period_kwh"] * scale
        other_kwh = max(period_kwh - tracked["period_kwh"].sum(), 0.0)
        cost_rows = [{"key": r.appliance, "appliance": nice(r.appliance), "kwh_day": r.estimated_kwh,
                      "kwh_period": r.period_kwh, "cost": r.cost} for r in tracked.itertuples()]
        cost_rows.append({"key": "baseline", "appliance": "Always-on and small loads",
                          "kwh_day": other_kwh / period_days, "kwh_period": other_kwh,
                          "cost": other_kwh * scale})
        cost_df = pd.DataFrame(cost_rows)

        fig_cost = px.bar(cost_df, x="appliance", y="cost", color="key", color_discrete_map=COLORS)
        fig_cost.update_traces(texttemplate="₹%{y:,.0f}", textposition="outside",
                               hovertemplate="%{x}: ₹%{y:,.0f}<extra></extra>")
        fig_cost.update_layout(showlegend=False)
        fig_cost.update_yaxes(title=f"Cost over {period_days} days (₹)")
        fig_cost.update_xaxes(title=None)
        st.plotly_chart(style_fig(fig_cost, 340, y_prefix="₹"), **WIDE_PLOT)

        cost_view = pd.DataFrame({
            "Appliance": cost_df["appliance"],
            "Energy per day": cost_df["kwh_day"].map(lambda v: f"{v:.2f} kWh"),
            f"Energy over {period_days} days": cost_df["kwh_period"].map(lambda v: f"{v:,.0f} units (kWh)"),
            "Share of use": (cost_df["kwh_period"] / max(period_kwh, 1e-9) * 100).map(lambda v: f"{v:.1f} %"),
            "Cost": cost_df["cost"].map(lambda v: f"₹{v:,.0f}"),
            "Cost per day": (cost_df["cost"] / period_days).map(lambda v: f"₹{v:,.1f} per day"),
        })
        show_table(cost_view)
        st.caption(
            "Slab prices rise as total use grows, so one appliance's cost is not a fixed price per unit. "
            "Each appliance gets its share of the units, multiplied by the average price paid. "
            "Always-on and small loads are the part of the meter total that no appliance run explains."
        )

        st.subheader("How close are the estimates?")
        cmp_df = usage.melt(id_vars="appliance", value_vars=["estimated_kwh", "true_kwh"],
                            var_name="source", value_name="kWh")
        cmp_df["source"] = cmp_df["source"].map({"estimated_kwh": "Estimated from events",
                                                 "true_kwh": "True (simulator)"})
        cmp_df["appliance"] = cmp_df["appliance"].map(nice)
        fig_cmp = px.bar(cmp_df, x="appliance", y="kWh", color="source", barmode="group",
                         color_discrete_sequence=["#F5B83D", "#6B7A8C"])
        fig_cmp.update_traces(texttemplate="%{y:.2f} kWh", textposition="outside",
                              hovertemplate="%{x}: %{y:.2f} kWh<extra></extra>")
        fig_cmp.update_yaxes(title="Energy in the day (kWh)")
        fig_cmp.update_xaxes(title=None)
        st.plotly_chart(style_fig(fig_cmp, 340, y_suffix=" kWh"), **WIDE_PLOT)

        acc_view = pd.DataFrame({
            "Appliance": usage["appliance"].map(nice),
            "Runs found": usage["runs"].map(lambda v: f"{int(v)} runs"),
            "Estimated energy": usage["estimated_kwh"].map(lambda v: f"{v:.2f} kWh"),
            "True energy": usage["true_kwh"].map(lambda v: f"{v:.2f} kWh"),
            "Difference": usage["error_pct"].map(lambda v: "n/a" if pd.isna(v) else f"{v:+.1f} %"),
        })
        show_table(acc_view)
        st.caption(
            "True values come from the simulator's answer key, which a real meter would not have. "
            "They are here to check how close the method gets. Appliances that run rarely, like the "
            "microwave, show larger percentage differences because a few seconds of error matter more."
        )

        if not runs.empty:
            st.subheader("When each appliance ran")
            runs_plot = runs.copy()
            runs_plot["appliance"] = runs_plot["appliance"].map(nice)
            runs_plot["key"] = runs["appliance"]
            fig_tl = px.timeline(runs_plot, x_start="start", x_end="end", y="appliance",
                                 color="key", color_discrete_map=COLORS,
                                 category_orders={"appliance": [nice(a) for a in APPLIANCES]},
                                 custom_data=["minutes"])
            fig_tl.update_traces(hovertemplate="%{y}: %{customdata[0]:.0f} min<extra></extra>")
            fig_tl.update_layout(showlegend=False)
            fig_tl.update_yaxes(title=None)
            fig_tl.update_xaxes(title="Time of day (hh:mm)", tickformat="%H:%M")
            st.plotly_chart(style_fig(fig_tl, 300), **WIDE_PLOT)

# ----------------------------------------------------------------------------
# Tab 5: fault alerts
# ----------------------------------------------------------------------------
with tab_fault:
    if not have_both:
        st.info("Run the pipeline first. Fault alerts need detected events.")
    else:
        st.write(
            "Fridges and air conditioners switch on and off in cycles. When a cycle gets much "
            "longer than normal, the appliance is working harder than it should."
        )
        report = fault_report(runs)
        if report.empty:
            st.info("Not enough cycles were found to judge.")
        else:
            for _, r in report.iterrows():
                name = nice(r["appliance"])
                if r["alert"]:
                    st.error(
                        f"**{name}:** each cycle averages {r['avg_run_min']:.0f} min, "
                        f"{r['ratio']:.1f}× the normal {r['normal_run_min']:.0f} min "
                        f"({int(r['cycles'])} cycles). The longest ran {r['longest_run_min']:.0f} min. "
                        "Worth a service check."
                    )
                else:
                    st.success(
                        f"**{name}:** cycles average {r['avg_run_min']:.0f} min, "
                        f"in line with the normal {r['normal_run_min']:.0f} min ({int(r['cycles'])} cycles)."
                    )

            plot_df = report.melt(id_vars="appliance", value_vars=["avg_run_min", "normal_run_min"],
                                  var_name="kind", value_name="minutes")
            plot_df["kind"] = plot_df["kind"].map({"avg_run_min": "Measured average",
                                                   "normal_run_min": "Normal"})
            plot_df["appliance"] = plot_df["appliance"].map(nice)
            fig_f = px.bar(plot_df, x="appliance", y="minutes", color="kind", barmode="group",
                           color_discrete_sequence=["#F25C54", "#6B7A8C"])
            fig_f.update_traces(texttemplate="%{y:.0f} min", textposition="outside",
                                hovertemplate="%{x}: %{y:.0f} min<extra></extra>")
            fig_f.update_yaxes(title="Length of one cycle (min)")
            fig_f.update_xaxes(title=None)
            st.plotly_chart(style_fig(fig_f, 320, y_suffix=" min"), **WIDE_PLOT)

            fault_view = pd.DataFrame({
                "Appliance": report["appliance"].map(nice),
                "Cycles found": report["cycles"].map(lambda v: f"{int(v)} cycles"),
                "Average cycle": report["avg_run_min"].map(lambda v: f"{v:.0f} min"),
                "Longest cycle": report["longest_run_min"].map(lambda v: f"{v:.0f} min"),
                "Normal cycle": report["normal_run_min"].map(lambda v: f"{v:.0f} min"),
                "Compared with normal": report["ratio"].map(lambda v: f"{v:.1f}×"),
            })
            show_table(fault_view)

        info = load_run_info()
        injected = {v: k for k, v in FAULT_OPTIONS.items()}.get(info.get("fault"), "Healthy home")
        st.caption(
            f"Setting used for this run: {injected}. Change it in the sidebar and run the pipeline "
            "again to see the alerts respond. Normal cycle lengths come from the healthy simulator "
            "settings, so a real home would need its own baseline."
        )

# ----------------------------------------------------------------------------
# Tab 6: try the model
# ----------------------------------------------------------------------------
LIVE_INPUTS = [
    # key, label, min, max, step, format
    ("delta_p", "Power change (W)", 0, 2500, 5, "%d W"),
    ("delta_q", "Reactive power change (VAR)", -100, 1500, 5, "%d VAR"),
    ("inrush_ratio", "Startup spike (× steady power)", 1.0, 5.0, 0.1, "%.1f×"),
    ("ramp_speed", "Ramp speed (W per second)", -100, 100, 1, "%d W/s"),
]
LIVE_BOUNDS = {k: (lo, hi) for k, _, lo, hi, _, _ in LIVE_INPUTS}


def preset_values(app, events, hour_samples):
    on = events[(events["event_type"] == "ON") & (events["appliance_label"] == app)]
    values = {}
    for key, _, lo, hi, step, _ in LIVE_INPUTS:
        v = min(max(float(on[key].median()), lo), hi)
        if key == "inrush_ratio":
            values[f"lv_{key}"] = round(round(v / step) * step, 1)
        else:
            values[f"lv_{key}"] = int(round(v / step) * step)
    hours = np.floor(np.asarray(hour_samples[app])).astype(int) % 24
    values["lv_hour"] = int(np.bincount(hours, minlength=24).argmax())
    return values


def apply_preset(events, hour_samples):
    choice = st.session_state.get("lv_preset")
    if choice:
        for k, v in preset_values(choice, events, hour_samples).items():
            st.session_state[k] = v


with tab_live:
    if events_df is None:
        st.info("Run the pipeline first. The model learns from the detected events.")
    else:
        hour_samples = get_hour_samples()
        model = get_evidence_model(events_df)

        if len(model.classes) < 2:
            st.info("There are not enough detected events of each appliance yet. Run the pipeline again.")
        else:
            st.write(
                "Describe a switch-on event and the model says which appliance it most likely is. "
                "Every input counts, including the time of day: kettles and microwaves are used at "
                "certain hours, and the model has learned those hours."
            )

            presets = list(model.classes)
            if st.session_state.get("lv_preset") not in presets:
                st.session_state["lv_preset"] = "kettle" if "kettle" in presets else presets[0]
                for k, v in preset_values(st.session_state["lv_preset"], events_df, hour_samples).items():
                    st.session_state[k] = v
            st.selectbox("Start from a typical event of", presets, key="lv_preset",
                         format_func=nice, on_change=apply_preset, args=(events_df, hour_samples))

            row1 = st.columns(3)
            row2 = st.columns(2)
            for col, spec in zip(row1, LIVE_INPUTS[:3]):
                k, label, lo, hi, step, fmt = spec
                col.slider(label, min_value=lo, max_value=hi, step=step, format=fmt, key=f"lv_{k}")
            k, label, lo, hi, step, fmt = LIVE_INPUTS[3]
            row2[0].slider(label, min_value=lo, max_value=hi, step=step, format=fmt, key=f"lv_{k}")
            row2[1].slider("Time of day (hh:mm)", min_value=0, max_value=23, step=1,
                           format="%02d:00", key="lv_hour")

            x = {spec[0]: st.session_state[f"lv_{spec[0]}"] for spec in LIVE_INPUTS}
            x["hour_of_day"] = st.session_state["lv_hour"]
            probs, match_df, fit = model.predict(x)

            best_app = max(model.classes, key=lambda c: probs[c])
            winner = max(probs, key=probs.get)
            m1, m2, m3 = st.columns(3)
            m1.metric("Most likely", nice(winner))
            m2.metric("Probability", f"{probs[winner] * 100:.0f} %")
            m3.metric(f"Fit with the usual {nice(best_app)} pattern", f"{fit[best_app]:.0f} %")

            if winner == UNUSUAL:
                weakest = match_df.loc[best_app].idxmin()
                weak_names = {"delta_p": "power change", "delta_q": "reactive power change",
                              "inrush_ratio": "startup spike", "ramp_speed": "ramp speed",
                              "hour_of_day": "time of day"}
                st.warning(
                    f"This does not match any appliance well. The closest is "
                    f"{nice(best_app)}, but the {weak_names[weakest]} fits it only "
                    f"{match_df.loc[best_app, weakest]:.0f} %."
                )

            prob_df = pd.DataFrame({"key": list(probs), "appliance": [nice(k) for k in probs],
                                    "probability": [v * 100 for v in probs.values()]})
            prob_df = prob_df.sort_values("probability")
            fig_p = px.bar(prob_df, x="probability", y="appliance", orientation="h",
                           color="key", color_discrete_map=COLORS)
            fig_p.update_traces(texttemplate="%{x:.0f} %", textposition="outside",
                                hovertemplate="%{y}: %{x:.1f} %<extra></extra>")
            fig_p.update_layout(showlegend=False)
            fig_p.update_xaxes(range=[0, 115], title="Probability (%)", ticksuffix=" %")
            fig_p.update_yaxes(title=None)
            st.plotly_chart(style_fig(fig_p, 280), **WIDE_PLOT)

            st.subheader("How well each input fits each appliance")
            st.caption(
                "100 % means the value is common for that appliance. Near 0 % means it almost never "
                "happens. Change any slider and these bars move."
            )
            fit_names = {"delta_p": "Power change", "delta_q": "Reactive power change",
                         "inrush_ratio": "Startup spike", "ramp_speed": "Ramp speed",
                         "hour_of_day": "Time of day"}
            fit_long = match_df.rename(columns=fit_names).rename_axis("appliance").reset_index() \
                .melt(id_vars="appliance", var_name="input", value_name="fit")
            fit_long["key"] = fit_long["appliance"]
            fit_long["appliance"] = fit_long["appliance"].map(nice)
            fig_m = px.bar(fit_long, x="input", y="fit", color="appliance", barmode="group",
                           color_discrete_map={nice(k): v for k, v in COLORS.items()})
            fig_m.update_traces(hovertemplate="%{fullData.name}, %{x}: %{y:.0f} %<extra></extra>")
            fig_m.update_yaxes(range=[0, 105], title="Fit (%)")
            fig_m.update_xaxes(title=None)
            st.plotly_chart(style_fig(fig_m, 360, y_suffix=" %"), **WIDE_PLOT)

            st.subheader("When each appliance usually switches on")
            rows = []
            for app in model.classes:
                ranges = usual_hours(hour_samples[app])
                if ranges == [(0, 24)]:
                    text = "All day"
                else:
                    text = ", ".join(f"{clock(a)} to {clock(b % 24)}" for a, b in ranges)
                rows.append({"Appliance": nice(app), "Usual hours": text})
            show_table(pd.DataFrame(rows))
            st.caption(
                "Learned from 30 simulated days. The Random Forest results are in the Appliance "
                "classifier tab; this page uses a simpler scoring model so every input visibly "
                "changes the answer."
            )
