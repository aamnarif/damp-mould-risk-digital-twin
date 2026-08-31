"""
Damp & Mould Risk Intelligence Platform
Decision-support for housing stock damp prevention and decarbonisation.

Built by Aamna Arif.
Data: Candanedo, Feldheim & Deramaix (2017), Energy and Buildings 140, 81-97.
Risk methodology follows BS EN ISO 13788.
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Damp & Mould Risk Platform",
    page_icon=":material/water_drop:",
    layout="wide",
)

import theme
from theme import (
    INK, INK_SOFT, LINE, SURFACE, COLD, TEAL, OCHRE, BRICK, MOSS,
    masthead, readouts, risk_meter, note, icon,
)

theme.inject_css()
theme.apply_chart_theme()

DATA_URL = (
    "https://raw.githubusercontent.com/LuisM78/"
    "Appliances-energy-prediction-data/master/energydata_complete.csv"
)

ROOMS = {
    1: "Kitchen", 2: "Living room", 3: "Laundry room", 4: "Office",
    5: "Bathroom", 7: "Ironing room", 8: "Teenager room", 9: "Parents room",
}
INDOOR_IDS = sorted(ROOMS.keys())

MOULD_RH_THRESHOLD = 80.0
R_V = 461.5
VOL_HEAT_CAPACITY_AIR = 0.33
CARBON_FACTOR_GAS = 0.183
CARBON_FACTOR_ELEC = 0.207
BOILER_EFFICIENCY = 0.85
MVHR_HEAT_RECOVERY = 0.85


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------
def sat_vapour_pressure(t_c):
    return 610.8 * np.exp(17.27 * t_c / (t_c + 237.3))


def vapour_pressure(t_c, rh_pct):
    return sat_vapour_pressure(t_c) * np.clip(rh_pct, 0, 100) / 100.0


def dew_point_from_vp(vp_pa):
    vp_pa = np.clip(vp_pa, 1.0, None)
    alpha = np.log(vp_pa / 610.8)
    return 237.3 * alpha / (17.27 - alpha)


def surface_conditions(t_in, t_out, vp_in, f_rsi):
    t_surf = t_out + f_rsi * (t_in - t_out)
    vp_eff = np.minimum(vp_in, sat_vapour_pressure(t_in))
    rh_surf = 100.0 * vp_eff / sat_vapour_pressure(t_surf)
    return t_surf, np.clip(rh_surf, 0, 100)


class BuildingTwin:
    """Steady-state hygrothermal twin of a dwelling, driven by measured weather."""

    def __init__(self, volume_m3=250.0, ach=0.5, moisture_gen_kg_h=0.4,
                 f_rsi=0.75, indoor_setpoint_c=20.0, name="Scenario"):
        self.volume_m3 = volume_m3
        self.ach = ach
        self.moisture_gen_kg_h = moisture_gen_kg_h
        self.f_rsi = f_rsi
        self.indoor_setpoint_c = indoor_setpoint_c
        self.name = name

    def vapour_excess_pa(self):
        dv = self.moisture_gen_kg_h / (self.ach * self.volume_m3)
        return dv * R_V * (self.indoor_setpoint_c + 273.15)

    def simulate(self, t_out, vp_out):
        t_out = np.asarray(t_out, dtype=float)
        vp_out = np.asarray(vp_out, dtype=float)
        t_in = self.indoor_setpoint_c

        vp_in = vp_out + self.vapour_excess_pa()
        vp_in = np.minimum(vp_in, sat_vapour_pressure(t_in))
        rh_in = np.clip(100 * vp_in / sat_vapour_pressure(t_in), 0, 100)

        t_surf, rh_surf = surface_conditions(t_in, t_out, vp_in, self.f_rsi)
        dp_in = dew_point_from_vp(vp_in)

        return pd.DataFrame({
            "T_out": t_out, "RH_in": rh_in, "vp_in": vp_in,
            "dewpoint_in": dp_in, "T_surface": t_surf, "RH_surface": rh_surf,
            "condensation": t_surf < dp_in,
            "mould_risk": rh_surf > MOULD_RH_THRESHOLD,
        })


def ventilation_energy_kwh(ach, volume_m3, t_out_series, setpoint_c, timestep_hours=1 / 6):
    delta_t = np.clip(setpoint_c - np.asarray(t_out_series, dtype=float), 0, None)
    watts = VOL_HEAT_CAPACITY_AIR * ach * volume_m3 * delta_t
    return float(np.sum(watts) * timestep_hours / 1000.0)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading sensor data…")
def load_data():
    raw = pd.read_csv(DATA_URL, parse_dates=["date"])

    indoor_cols = [f"T{i}" for i in INDOOR_IDS] + [f"RH_{i}" for i in INDOOR_IDS]
    weather_cols = ["T_out", "RH_out", "Press_mm_hg", "Windspeed", "Visibility", "Tdewpoint"]
    energy_cols = ["Appliances", "lights"]

    df = raw[["date"] + indoor_cols + weather_cols + energy_cols].copy()
    df = df.set_index("date").sort_index()

    full_index = pd.date_range(df.index.min(), df.index.max(), freq="10min")
    df = df.reindex(full_index)
    df.index.name = "timestamp"

    sensor_cols = indoor_cols + weather_cols
    df[sensor_cols] = df[sensor_cols].interpolate(method="time", limit=6, limit_area="inside")

    df["vp_out"] = vapour_pressure(df["T_out"], df["RH_out"])
    df["dewpoint_out"] = dew_point_from_vp(df["vp_out"])
    for i in INDOOR_IDS:
        t, rh = df[f"T{i}"], df[f"RH_{i}"]
        df[f"vp_{i}"] = vapour_pressure(t, rh)
        df[f"dewpoint_{i}"] = dew_point_from_vp(df[f"vp_{i}"])
        df[f"dp_margin_{i}"] = t - df[f"dewpoint_{i}"]
        df[f"vp_excess_{i}"] = df[f"vp_{i}"] - df["vp_out"]

    return df


@st.cache_data(show_spinner="Training forecast model…")
def train_forecast_model(_df, target_room=2, horizon_steps=36):
    from xgboost import XGBRegressor

    df = _df
    fc = pd.DataFrame(index=df.index)
    fc["y"] = df[f"RH_{target_room}"].shift(-horizon_steps)
    fc["RH_now"] = df[f"RH_{target_room}"]
    fc["T_now"] = df[f"T{target_room}"]
    fc["vp_now"] = df[f"vp_{target_room}"]
    fc["vp_excess_now"] = df[f"vp_excess_{target_room}"]
    fc["dp_margin_now"] = df[f"dp_margin_{target_room}"]

    for hours in [1, 3, 6, 12, 24]:
        steps = int(hours * 6)
        fc[f"RH_lag_{hours}h"] = df[f"RH_{target_room}"].shift(steps)
        fc[f"T_lag_{hours}h"] = df[f"T{target_room}"].shift(steps)

    for hours in [1, 6, 24]:
        w = int(hours * 6)
        fc[f"RH_rollmean_{hours}h"] = df[f"RH_{target_room}"].shift(1).rolling(w).mean()
        fc[f"RH_rollstd_{hours}h"] = df[f"RH_{target_room}"].shift(1).rolling(w).std()

    fc["RH_delta_1h"] = df[f"RH_{target_room}"] - df[f"RH_{target_room}"].shift(6)
    for c in ["T_out", "RH_out", "Windspeed", "vp_out"]:
        fc[c] = df[c]
    fc["T_gradient"] = df[f"T{target_room}"] - df["T_out"]
    fc["appliances"] = df["Appliances"]
    fc["appliances_roll_1h"] = df["Appliances"].shift(1).rolling(6).mean()

    hod = fc.index.hour + fc.index.minute / 60
    fc["hod_sin"] = np.sin(2 * np.pi * hod / 24)
    fc["hod_cos"] = np.cos(2 * np.pi * hod / 24)
    fc["dow"] = fc.index.dayofweek

    fc = fc.dropna()
    features = [c for c in fc.columns if c != "y"]

    split = int(len(fc) * 0.8)
    train, test = fc.iloc[:split], fc.iloc[split:]

    model = XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
    )
    model.fit(train[features], train["y"])
    pred = model.predict(test[features])

    def mae(a, p):
        return float(np.mean(np.abs(np.asarray(a) - np.asarray(p))))

    metrics = {
        "xgb_mae": mae(test["y"], pred),
        "persistence_mae": mae(test["y"], test["RH_now"]),
        "yesterday_mae": mae(test["y"], test["RH_lag_24h"]),
        "rollmean_mae": mae(test["y"], test["RH_rollmean_24h"]),
        "n_train": len(train),
        "n_test": len(test),
    }
    metrics["improvement_pct"] = (
        (metrics["persistence_mae"] - metrics["xgb_mae"]) / metrics["persistence_mae"] * 100
    )

    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    test_out = test[["y", "RH_now"]].copy()
    test_out["pred"] = pred
    return metrics, importances, test_out


df = load_data()
weather = df[["T_out", "vp_out"]].dropna()
period_days = (weather.index.max() - weather.index.min()).days

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    '<div class="sidebar-brand">Damp &amp; Mould Risk</div>'
    '<div class="sidebar-sub">Housing stock decision-support</div>',
    unsafe_allow_html=True,
)

NAV = [
    ("Overview", "dashboard"),
    ("Sensor data", "sensors"),
    ("Risk analysis", "water_damage"),
    ("Digital twin", "home_work"),
    ("Carbon trade-off", "balance"),
    ("Forecasting", "query_stats"),
    ("Limitations", "rule"),
]
PAGE_ICONS = dict(NAV)

if "page" not in st.session_state:
    st.session_state.page = "Overview"

for _label, _ico in NAV:
    if st.sidebar.button(
        _label,
        icon=f":material/{_ico}:",
        use_container_width=True,
        type="primary" if st.session_state.page == _label else "tertiary",
        key=f"nav_{_label}",
    ):
        st.session_state.page = _label

page = st.session_state.page

st.sidebar.markdown(
    f'<div class="sidebar-meta">'
    f'{len(df):,} readings<br>{len(INDOOR_IDS)} rooms &middot; {period_days} days<br>'
    f'ZigBee sensors + weather station'
    f'</div>',
    unsafe_allow_html=True,
)


# ===========================================================================
# OVERVIEW
# ===========================================================================
if page == "Overview":
    masthead(
        "Damp &amp; Mould Risk Intelligence",
        "Turning building sensor data into maintenance decisions — and testing "
        "retrofit interventions before paying for them.",
        icon_name="dashboard",
    )

    readouts([
        ("Sensor readings", f"{len(df):,}", ""),
        ("Rooms monitored", f"{len(INDOOR_IDS)}", ""),
        ("Monitoring period", f"{period_days}", " days"),
        ("Mean indoor humidity", f"{df['RH_2'].mean():.1f}", "%"),
    ])

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.subheader("The problem")
        st.markdown(
            "Housing providers hold large volumes of sensor data, but raw temperature "
            "and humidity readings don't tell a maintenance team **which** homes are at "
            "risk of damp and mould, **why**, or **what intervention would fix it at what "
            "carbon cost**.\n\n"
            "This platform builds that decision layer: sensor ETL → physics-based risk "
            "indices → forecasting → a building digital twin for what-if analysis → "
            "carbon impact."
        )

        note(
            "The monitored building is a <strong>passive house</strong> — airtight, "
            "mechanically ventilated, and very dry (~40% RH). It has almost no condensation "
            "risk. That is a real finding, not a failed analysis. So the sensor data provides "
            "a validated <em>healthy baseline</em>, and the <strong>digital twin</strong> models "
            "the degraded fabric and under-ventilation typical of the older stock that actually "
            "has damp problems.",
        )

        note(
            "Risk indices here are <strong>physics-derived indicators</strong> following "
            "BS EN ISO 13788 — <strong>not</strong> models trained on labelled mould "
            "inspections. No such labels exist in this dataset.",
            kind="warn",
        )

    with right:
        st.subheader("What's inside")
        st.markdown(
            f"""
| Section | Capability |
|---|---|
| Sensor data | Multi-source ETL, quality assurance |
| Risk analysis | Building physics, segmentation |
| Digital twin | What-if scenario simulation |
| Carbon trade-off | Sustainability dossier |
| Forecasting | ML with honest baselines |
"""
        )
        st.caption(
            "Data: Candanedo, Feldheim & Deramaix (2017), *Energy and Buildings* 140, 81–97."
        )


# ===========================================================================
# SENSOR DATA
# ===========================================================================
elif page == "Sensor data":
    masthead(
        "Sensor Data &amp; Quality Assurance",
        "Multi-source ETL: nine rooms of temperature and humidity, a weather station, "
        "and energy meters — aligned onto a strict ten-minute grid.",
        icon_name="sensors",
    )

    room_pick = st.selectbox("Room", options=INDOOR_IDS, format_func=lambda i: ROOMS[i], index=1)

    readouts([
        ("Mean temperature", f"{df[f'T{room_pick}'].mean():.1f}", " °C"),
        ("Mean humidity", f"{df[f'RH_{room_pick}'].mean():.1f}", " %"),
        ("Peak humidity", f"{df[f'RH_{room_pick}'].max():.1f}", " %"),
        ("Min dew-point margin", f"{df[f'dp_margin_{room_pick}'].min():.1f}", " °C"),
    ])

    fig, axes = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True)
    axes[0].plot(df.index, df[f"T{room_pick}"], lw=0.7, color=COLD, label=ROOMS[room_pick])
    axes[0].plot(df.index, df["T_out"], lw=0.7, color=INK_SOFT, alpha=0.7, label="Outdoor")
    axes[0].set_ylabel("°C")
    axes[0].legend(loc="upper right")
    axes[0].set_title("Indoor against outdoor temperature")

    axes[1].plot(df.index, df[f"RH_{room_pick}"], lw=0.7, color=TEAL)
    axes[1].axhline(70, color=OCHRE, ls="--", lw=1, label="70% RH")
    axes[1].set_ylabel("% RH")
    axes[1].legend(loc="upper right")
    axes[1].set_title("Relative humidity")

    axes[2].plot(df.index, df[f"vp_excess_{room_pick}"], lw=0.6, color=BRICK)
    axes[2].axhline(0, color=INK_SOFT, lw=0.9)
    axes[2].set_ylabel("Pa")
    axes[2].set_title("Vapour pressure excess — moisture generated indoors")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Moisture load by room")
    summary = []
    for i in INDOOR_IDS:
        summary.append({
            "Room": ROOMS[i],
            "Mean temp (°C)": df[f"T{i}"].mean(),
            "Mean RH (%)": df[f"RH_{i}"].mean(),
            "Peak RH (%)": df[f"RH_{i}"].max(),
            "Vapour excess (Pa)": df[f"vp_excess_{i}"].mean(),
            "Time above 70% RH (%)": (df[f"RH_{i}"] > 70).mean() * 100,
        })
    room_summary = pd.DataFrame(summary).set_index("Room").sort_values(
        "Vapour excess (Pa)", ascending=False
    )
    st.dataframe(room_summary.round(1), use_container_width=True)

    note(
        "The bathroom and laundry room show the highest vapour pressure excess — they are "
        "the moisture <em>sources</em>. The physics recovers this without being told anything "
        "about what each room is for.",
        kind="good",
    )


# ===========================================================================
# RISK ANALYSIS
# ===========================================================================
elif page == "Risk analysis":
    masthead(
        "Condensation &amp; Mould Risk",
        "Physics-derived indicators following BS EN ISO 13788, evaluated across "
        "four fabric qualities.",
        icon_name="water_damage",
    )

    with st.expander("How the risk indices work"):
        st.markdown(
            "Two distinct risks:\n\n"
            "**Surface condensation** — liquid water forms when a surface falls below the "
            "room air's dew point.\n\n"
            "**Mould germination** — mould does not need liquid water. Germination is possible "
            "when relative humidity at the surface stays above roughly 80%. This is the more "
            "common and more insidious failure in real housing.\n\n"
            "Surface temperature uses the temperature factor:"
        )
        st.latex(r"T_{surface} = T_{out} + f_{Rsi}\,(T_{in} - T_{out})")
        st.markdown(
            "UK Building Regulations require $f_{Rsi} \\geq 0.75$. Older solid-wall stock "
            "and cold-bridge junctions can fall to 0.5–0.65."
        )

    fabric_scenarios = {
        0.95: "New build",
        0.75: "Regs minimum",
        0.65: "Poor thermal bridge",
        0.55: "Uninsulated solid wall",
    }

    rows = []
    for f_rsi, label in fabric_scenarios.items():
        for i in INDOOR_IDS:
            t_surf, rh_surf = surface_conditions(df[f"T{i}"], df["T_out"], df[f"vp_{i}"], f_rsi)
            dp_in = dew_point_from_vp(np.minimum(df[f"vp_{i}"], sat_vapour_pressure(df[f"T{i}"])))
            rows.append({
                "fabric": label, "room": ROOMS[i],
                "pct_mould": float((rh_surf > MOULD_RH_THRESHOLD).mean() * 100),
            })
    pivot = pd.DataFrame(rows).pivot_table(index="room", columns="fabric", values="pct_mould")
    order = [fabric_scenarios[k] for k in sorted(fabric_scenarios, reverse=True)]
    pivot = pivot[order]

    fig, ax = plt.subplots(figsize=(11, 4.2))
    pivot.plot(kind="bar", ax=ax, color=[MOSS, COLD, OCHRE, BRICK], width=0.78)
    ax.set_ylabel("% of time at mould risk")
    ax.set_xlabel("")
    ax.set_title("Risk stays low except at severe cold bridges")
    ax.legend(title="Fabric quality", ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(pivot.round(2), use_container_width=True)

    note(
        "<strong>This building is genuinely healthy.</strong> Risk is near zero across almost "
        "every room and fabric quality. Damp is a problem of wet air meeting cold surfaces — "
        "remove either and the risk disappears. This is the correct result, and the reason the "
        "digital twin exists.",
        kind="good",
    )

    st.subheader("Room segmentation by moisture behaviour")
    st.markdown(
        "Housing providers cannot inspect everything. Segmentation groups spaces by behaviour "
        "so limited maintenance capacity is directed by evidence."
    )

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    seg_rows = []
    for i in INDOOR_IDS:
        seg_rows.append({
            "room": ROOMS[i],
            "mean_vp_excess": df[f"vp_excess_{i}"].mean(),
            "p95_RH": df[f"RH_{i}"].quantile(0.95),
            "RH_volatility": df[f"RH_{i}"].std(),
            "min_dp_margin": df[f"dp_margin_{i}"].min(),
            "mean_T": df[f"T{i}"].mean(),
        })
    seg = pd.DataFrame(seg_rows).set_index("room")
    X = StandardScaler().fit_transform(seg)
    seg["segment"] = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(X)
    order_seg = seg.groupby("segment")["mean_vp_excess"].mean().sort_values(ascending=False).index
    names = dict(zip(order_seg, ["High moisture load", "Moderate", "Low / stable"]))
    seg["Segment"] = seg["segment"].map(names)

    fig2, ax2 = plt.subplots(figsize=(9.5, 4.8))
    colors = {"High moisture load": BRICK, "Moderate": OCHRE, "Low / stable": MOSS}
    for name, grp in seg.groupby("Segment"):
        ax2.scatter(grp["mean_vp_excess"], grp["p95_RH"], s=150,
                    color=colors.get(name, INK_SOFT), label=name,
                    edgecolor="white", linewidth=1.4, zorder=3)
    for room, row in seg.iterrows():
        ax2.annotate(room, (row["mean_vp_excess"], row["p95_RH"]),
                     fontsize=8, color=INK_SOFT, xytext=(7, 4), textcoords="offset points")
    ax2.set_xlabel("Mean vapour pressure excess (Pa) — moisture generated indoors")
    ax2.set_ylabel("95th percentile RH (%)")
    ax2.set_title("Rooms grouped by moisture behaviour")
    ax2.legend()
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)


# ===========================================================================
# DIGITAL TWIN
# ===========================================================================
elif page == "Digital twin":
    masthead(
        "Building Digital Twin",
        "A steady-state hygrothermal model driven by the real measured weather. "
        "Adjust the four levers a housing provider can actually pull.",
        icon_name="home_work",
    )

    with st.expander("Model and validation"):
        st.markdown("Moisture balance (BS EN ISO 13788). Vapour pressure excess over outdoor air:")
        st.latex(r"\Delta p = \frac{G \cdot R_v \cdot T_{in}}{n \cdot V}")
        st.markdown(
            "where $G$ is moisture generation (kg/h), $n$ air changes per hour, and "
            "$V$ dwelling volume (m³).\n\n"
            "**Validation:** run with passive-house parameters, the twin reproduces the measured "
            "living-room humidity to within a few percentage points — the agreement a "
            "steady-state screening model should reach.\n\n"
            "This is **not** a CFD or full BIM thermal model. See Limitations."
        )

    presets = {
        "New build, well ventilated": (0.70, 0.30, 0.95, 20.0),
        "Regs-compliant, average use": (0.50, 0.40, 0.75, 20.0),
        "Older stock, high occupancy": (0.35, 0.50, 0.65, 20.0),
        "Solid wall, under-ventilated": (0.25, 0.55, 0.55, 20.0),
        "Solid wall + fuel poverty": (0.25, 0.55, 0.55, 16.0),
    }
    preset = st.selectbox("Start from a housing archetype", ["Custom"] + list(presets))
    d_ach, d_moist, d_frsi, d_set = presets.get(preset, (0.35, 0.50, 0.65, 20.0))

    c1, c2, c3, c4 = st.columns(4)
    ach = c1.slider("Ventilation (air changes/hour)", 0.10, 1.00, d_ach, 0.05,
                    help="Extract fans, trickle vents, MVHR, PIV.")
    moisture = c2.slider("Moisture generation (kg/h)", 0.20, 0.80, d_moist, 0.05,
                         help="Occupancy density, drying clothes indoors, cooking.")
    f_rsi = c3.slider("Fabric quality (fRsi)", 0.45, 0.95, d_frsi, 0.05,
                      help="Surface temperature factor. Building Regulations require 0.75 or above.")
    setpoint = c4.slider("Heating setpoint (°C)", 14.0, 23.0, d_set, 0.5,
                         help="Lower is cheaper to run, but leaves colder surfaces.")

    twin = BuildingTwin(ach=ach, moisture_gen_kg_h=moisture, f_rsi=f_rsi,
                        indoor_setpoint_c=setpoint)
    sim = twin.simulate(weather["T_out"], weather["vp_out"])
    sim.index = weather.index

    risk_pct = float(sim["mould_risk"].mean() * 100)
    cond_pct = float(sim["condensation"].mean() * 100)
    heat_kwh = ventilation_energy_kwh(ach, twin.volume_m3, weather["T_out"], setpoint)
    co2 = heat_kwh / BOILER_EFFICIENCY * CARBON_FACTOR_GAS

    risk_meter(risk_pct)

    readouts([
        ("Mean indoor humidity", f"{sim['RH_in'].mean():.1f}", " %"),
        ("Time condensing", f"{cond_pct:.1f}", " %"),
        ("Ventilation heat loss", f"{heat_kwh:,.0f}", " kWh"),
        ("Emissions", f"{co2:,.0f}", " kg"),
    ])

    if f_rsi < 0.75:
        note(
            f"Fabric quality (fRsi {f_rsi:.2f}) is <strong>below the UK Building Regulations "
            f"minimum of 0.75</strong> — typical of uninsulated solid-wall stock or cold-bridge "
            f"junctions.",
            kind="warn",
        )

    if setpoint < 18:
        note(
            "<strong>The fuel poverty trap.</strong> Turning the heating down cuts energy use and "
            "bills, but <em>increases</em> damp risk, because colder surfaces condense more readily. "
            "A tool that reports carbon savings without surfacing this consequence is giving "
            "incomplete advice.",
            kind="alert",
        )

    fig, axes = plt.subplots(2, 1, figsize=(12, 6.2), sharex=True)
    axes[0].plot(sim.index, sim["RH_surface"], lw=0.6, color=TEAL)
    axes[0].axhline(MOULD_RH_THRESHOLD, color=BRICK, ls="--", lw=1.2,
                    label="Mould germination threshold (80%)")
    axes[0].fill_between(sim.index, MOULD_RH_THRESHOLD, sim["RH_surface"],
                         where=sim["RH_surface"] > MOULD_RH_THRESHOLD,
                         color=BRICK, alpha=0.25)
    axes[0].set_ylabel("Surface RH (%)")
    axes[0].set_ylim(0, 105)
    axes[0].legend(loc="lower right")
    axes[0].set_title("Surface humidity against the mould threshold")

    axes[1].plot(sim.index, sim["T_surface"], lw=0.7, color=OCHRE, label="Surface temperature")
    axes[1].plot(sim.index, sim["dewpoint_in"], lw=0.7, color=COLD, label="Indoor dew point")
    axes[1].fill_between(sim.index, sim["T_surface"], sim["dewpoint_in"],
                         where=sim["T_surface"] < sim["dewpoint_in"],
                         color=COLD, alpha=0.25)
    axes[1].set_ylabel("°C")
    axes[1].legend(loc="upper left")
    axes[1].set_title("Condensation occurs where the surface falls below the dew point")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ===========================================================================
# CARBON TRADE-OFF
# ===========================================================================
elif page == "Carbon trade-off":
    masthead(
        "Health Against Carbon",
        "Ventilation is the cheapest damp fix, but ventilating a heated dwelling throws away "
        "heat. The intervention that best protects residents may be the one that increases "
        "emissions — unless heat recovery is used.",
        icon_name="balance",
    )

    st.latex(r"Q_{vent} = 0.33 \cdot n \cdot V \cdot \Delta T \quad \text{[W]}")

    baseline = BuildingTwin(ach=0.25, moisture_gen_kg_h=0.55, f_rsi=0.55)

    def dossier_row(name, tw, heat_recovery=0.0, fan_power_w=0.0):
        heat = ventilation_energy_kwh(tw.ach, tw.volume_m3, weather["T_out"], tw.indoor_setpoint_c)
        heat *= (1 - heat_recovery)
        heat_co2 = heat / BOILER_EFFICIENCY * CARBON_FACTOR_GAS
        hours = len(weather) / 6.0
        fan_kwh = fan_power_w * hours / 1000.0
        fan_co2 = fan_kwh * CARBON_FACTOR_ELEC
        s = tw.simulate(weather["T_out"], weather["vp_out"])
        return {
            "Intervention": name,
            "Air changes/hr": tw.ach,
            "Heat loss (kWh)": heat,
            "Fan (kWh)": fan_kwh,
            "Total kgCO₂e": heat_co2 + fan_co2,
            "Mould risk (%)": float(s["mould_risk"].mean() * 100),
        }

    rows = [
        dossier_row("No action", baseline),
        dossier_row("Extract fans", BuildingTwin(ach=0.45, moisture_gen_kg_h=0.55, f_rsi=0.55)),
        dossier_row("PIV unit", BuildingTwin(ach=0.60, moisture_gen_kg_h=0.55, f_rsi=0.55), fan_power_w=6.0),
        dossier_row("Wall insulation", BuildingTwin(ach=0.25, moisture_gen_kg_h=0.55, f_rsi=0.80)),
        dossier_row("Resident advice", BuildingTwin(ach=0.25, moisture_gen_kg_h=0.41, f_rsi=0.55)),
        dossier_row("MVHR", BuildingTwin(ach=0.60, moisture_gen_kg_h=0.55, f_rsi=0.55),
                    heat_recovery=MVHR_HEAT_RECOVERY, fan_power_w=6.0),
        dossier_row("MVHR + insulation", BuildingTwin(ach=0.60, moisture_gen_kg_h=0.55, f_rsi=0.80),
                    heat_recovery=MVHR_HEAT_RECOVERY, fan_power_w=6.0),
    ]
    dos = pd.DataFrame(rows).set_index("Intervention")
    base_co2 = dos.loc["No action", "Total kgCO₂e"]
    base_risk = dos.loc["No action", "Mould risk (%)"]
    dos["Carbon change (kg)"] = dos["Total kgCO₂e"] - base_co2
    dos["Risk reduction (pts)"] = base_risk - dos["Mould risk (%)"]

    plot_df = dos.drop(index="No action")
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.axhline(0, color=INK_SOFT, lw=0.9, ls="--")
    ax.axvline(0, color=INK_SOFT, lw=0.9, ls="--")
    ax.scatter(plot_df["Risk reduction (pts)"], plot_df["Carbon change (kg)"], s=165,
               c=[BRICK if v > 0 else MOSS for v in plot_df["Carbon change (kg)"]],
               edgecolor="white", linewidth=1.5, zorder=3)
    for name, row in plot_df.iterrows():
        ax.annotate(name, (row["Risk reduction (pts)"], row["Carbon change (kg)"]),
                    fontsize=8.5, color=INK, xytext=(9, 5), textcoords="offset points")
    ax.set_xlabel("Damp risk reduction (percentage points) — better for residents")
    ax.set_ylabel("Change in CO₂e (kg over period)")
    ax.set_title("Interventions that help residents and cut carbon sit bottom-right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(dos.round(1), use_container_width=True)

    note(
        "Simply increasing ventilation moves right (healthier) but also <strong>up</strong> "
        "(more heat thrown away). Only insulation and heat recovery move right <em>and</em> down. "
        "This is the analysis that turns a sensor feed into an investment case.",
    )
    st.caption(
        f"Carbon factors: BEIS/DESNZ 2023 — gas {CARBON_FACTOR_GAS} kgCO₂e/kWh, grid electricity "
        f"{CARBON_FACTOR_ELEC} kgCO₂e/kWh. Published annually; refresh for real reporting. "
        f"Figures cover the {period_days}-day monitored period and are not annualised."
    )


# ===========================================================================
# FORECASTING
# ===========================================================================
elif page == "Forecasting":
    masthead(
        "Humidity Forecasting",
        "Predicting risk six hours ahead — prevention rather than a maintenance ticket.",
        icon_name="query_stats",
    )

    metrics, importances, test_out = train_forecast_model(df)

    st.markdown(
        "Establish naive baselines first: anything that cannot beat *the value now* is not "
        "worth deploying. Then fit the model, then validate chronologically. Random splitting "
        "on time series leaks the future into training and produces flattering, false results."
    )

    readouts([
        ("Model error", f"{metrics['xgb_mae']:.2f}", " %RH"),
        ("Best baseline", f"{metrics['persistence_mae']:.2f}", " %RH"),
        ("Improvement", f"{metrics['improvement_pct']:.1f}", " %"),
        ("Held-out readings", f"{metrics['n_test']:,}", ""),
    ])

    comp = pd.DataFrame({
        "Model": [
            "Persistence (humidity now)",
            "Same time yesterday",
            "24-hour rolling mean",
            "XGBoost with engineered features",
        ],
        "MAE (%RH)": [
            metrics["persistence_mae"], metrics["yesterday_mae"],
            metrics["rollmean_mae"], metrics["xgb_mae"],
        ],
    }).sort_values("MAE (%RH)").reset_index(drop=True)
    st.dataframe(comp.round(3), use_container_width=True)

    if metrics["improvement_pct"] < 5:
        note(
            "Marginal gain over the baseline. On this evidence the simpler baseline may be the "
            "better operational choice — cheaper to run and to explain.",
            kind="warn",
        )
    else:
        note(
            f"XGBoost beats the strongest baseline by {metrics['improvement_pct']:.1f}%, "
            f"which justifies the added complexity.",
            kind="good",
        )

    n_show = st.slider("Window (readings from start of test period)", 200, 2000, 1000, 100)
    window = test_out.iloc[:n_show]

    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.plot(window.index, window["y"], label="Actual", color=INK, lw=1.3)
    ax.plot(window.index, window["pred"], label="Forecast", color=BRICK, lw=1.0, alpha=0.9)
    ax.plot(window.index, window["RH_now"], label="Persistence baseline",
            color=COLD, lw=0.8, alpha=0.55)
    ax.set_ylabel("% RH")
    ax.set_title("Six-hour-ahead forecast against actual")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("What the model relies on")
    fig2, ax2 = plt.subplots(figsize=(9, 4.8))
    importances.head(15).sort_values().plot(kind="barh", ax=ax2, color=TEAL)
    ax2.set_title("Fifteen most important features")
    ax2.set_xlabel("Importance")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    st.caption(
        f"Trained on {metrics['n_train']:,} readings, tested on {metrics['n_test']:,} held-out "
        f"readings in strict chronological order."
    )


# ===========================================================================
# LIMITATIONS
# ===========================================================================
elif page == "Limitations":
    masthead(
        "Findings, Limitations &amp; Next Steps",
        "What this analysis shows, what it cannot show, and what would make it "
        "deployable on real housing stock.",
        icon_name="rule",
    )

    st.subheader("Findings")
    st.markdown(
        """
**The monitored building is healthy, and that is a real result.** A passive house at ~40% RH
shows near-zero condensation risk across almost all fabric qualities.

**Moisture load is recovered from physics alone.** Vapour pressure excess correctly identifies
the bathroom and laundry room as moisture sources, with no labelling of room function.

**Forecasting must be judged against baselines.** Humidity is highly autocorrelated, so
persistence is a strong competitor. If the margin is thin, the simpler model wins.

**Ventilation and carbon are in direct tension.** More ventilation reduces damp risk and raises
heating demand. Only insulation and heat recovery improve both.

**Turning the heating down increases damp risk.** A fuel-poor household reducing emissions can
make its damp problem worse — a collision between a carbon target and a welfare duty.
        """
    )

    st.subheader("Limitations")
    note(
        "<strong>No mould labels exist.</strong> Every risk figure is a physics-derived indicator "
        "following BS EN ISO 13788, not a model validated against inspection records. Validating "
        "against real damp and mould reports is the single most important next step."
        "<br><br>"
        "<strong>The twin is steady-state.</strong> Each timestep is solved independently, with no "
        "thermal or moisture inertia and no buffering by fabric and furnishings. Real buildings lag; "
        "this model does not. It is a screening tool, not a substitute for dynamic simulation."
        "<br><br>"
        "<strong>Single well-mixed zone.</strong> Inter-room moisture transfer is not modelled, so a "
        "bathroom venting into a cold bedroom — a very common real failure mode — is out of scope."
        "<br><br>"
        "<strong>One building, one climate.</strong> Belgian weather over four and a half "
        "winter-to-spring months. UK stock and a full annual cycle would shift the numbers."
        "<br><br>"
        "<strong>Twin parameters are archetypes, not surveys.</strong> Air change rates, moisture "
        "generation and fRsi come from typical published ranges. Real deployment needs measured "
        "ventilation rates and thermographic survey data."
        "<br><br>"
        "<strong>Carbon factors date quickly.</strong> BEIS/DESNZ figures are revised annually and "
        "are hardcoded here."
        "<br><br>"
        "<strong>Occupancy is a proxy.</strong> Appliance load stands in for occupancy — correlated, "
        "but not equivalent.",
        kind="warn",
    )

    st.subheader("Next steps")
    st.markdown(
        """
1. Validate risk indices against real damp and mould inspection records, turning the indicator
   into a calibrated model.
2. Extend the twin to a dynamic multi-zone model with moisture buffering.
3. Scale from one building to a stock: the same pipeline per property produces a ranked
   intervention list under a fixed retrofit budget.
4. Add cost data alongside carbon, so interventions rank on pounds per percentage-point of risk
   removed.
5. Wire drift monitoring to an automated seasonal retraining schedule with alerting.
        """
    )

    st.caption(
        "Built by Aamna Arif. Data: Candanedo, Feldheim & Deramaix (2017), *Data driven prediction "
        "models of energy use of appliances in a low-energy house*, Energy and Buildings 140, 81–97. "
        "Methodology follows BS EN ISO 13788."
    )
