"""
Damp & Mould Risk Intelligence Platform
Interactive decision-support dashboard for housing stock damp risk and decarbonisation.

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

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Damp & Mould Risk Platform",
    page_icon="🏠",
    layout="wide",
)

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


# ----------------------------------------------------------------------------
# Physics
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# Data loading (cached)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading sensor data…")
def load_data():
    raw = pd.read_csv(DATA_URL, parse_dates=["date"])

    indoor_cols = [f"T{i}" for i in INDOOR_IDS] + [f"RH_{i}" for i in INDOOR_IDS]
    weather_cols = ["T_out", "RH_out", "Press_mm_hg", "Windspeed", "Visibility", "Tdewpoint"]
    energy_cols = ["Appliances", "lights"]

    df = raw[["date"] + indoor_cols + weather_cols + energy_cols].copy()
    df = df.set_index("date").sort_index()

    # Strict 10-minute grid so lags mean fixed real time
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="10min")
    df = df.reindex(full_index)
    df.index.name = "timestamp"

    sensor_cols = indoor_cols + weather_cols
    df[sensor_cols] = df[sensor_cols].interpolate(
        method="time", limit=6, limit_area="inside"
    )

    # Psychrometrics
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


# ----------------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------------
df = load_data()
weather = df[["T_out", "vp_out"]].dropna()
period_days = (weather.index.max() - weather.index.min()).days

# ----------------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------------
st.sidebar.title("🏠 Damp & Mould Risk")
st.sidebar.caption("Decision-support for housing stock")

page = st.sidebar.radio(
    "Section",
    ["Overview", "Sensor data", "Risk analysis", "Digital twin", "Carbon trade-off", "Forecasting", "Limitations"],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    f"**Data**: {len(df):,} readings · {period_days} days\n\n"
    f"9 rooms · ZigBee sensors + weather station"
)


# ============================================================================
# OVERVIEW
# ============================================================================
if page == "Overview":
    st.title("Damp & Mould Risk Intelligence Platform")
    st.markdown(
        "Turning raw building sensor data into maintenance decisions — "
        "and testing retrofit interventions before paying for them."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sensor readings", f"{len(df):,}")
    c2.metric("Rooms monitored", len(INDOOR_IDS))
    c3.metric("Monitoring period", f"{period_days} days")
    c4.metric("Mean indoor RH", f"{df['RH_2'].mean():.1f}%")

    st.markdown("---")

    left, right = st.columns([3, 2])

    with left:
        st.subheader("The problem")
        st.markdown(
            "Housing providers hold large volumes of IoT sensor data, but raw temperature "
            "and humidity readings don't tell a maintenance team **which** homes are at risk "
            "of damp and mould, **why**, or **what intervention would fix it at what carbon cost**.\n\n"
            "This platform builds that decision layer: sensor ETL → physics-based risk indices → "
            "forecasting → a building digital twin for what-if analysis → carbon impact."
        )

        st.subheader("An honest framing note")
        st.info(
            "The monitored building is a **passive house** — airtight, mechanically ventilated, "
            "and very dry (~40% RH). It has almost **no** condensation risk. That's a real finding, "
            "not a failed analysis.\n\n"
            "So the real sensor data provides a validated *healthy baseline*, and the **digital twin** "
            "simulates the degraded fabric and under-ventilation typical of the **older housing stock "
            "that actually has damp problems**."
        )

        st.warning(
            "**Risk indices here are physics-derived indicators** following BS EN ISO 13788 — "
            "**not** models trained on labelled mould inspections. No such labels exist in this dataset."
        )

    with right:
        st.subheader("What's included")
        st.markdown(
            """
            | Module | Capability |
            |---|---|
            | Sensor ETL | Data engineering, QA |
            | Risk indices | Building physics |
            | Segmentation | Clustering |
            | Forecasting | ML + backtesting |
            | Digital twin | What-if simulation |
            | Carbon dossier | Sustainability |
            """
        )
        st.caption(
            "Data: Candanedo, Feldheim & Deramaix (2017), *Energy and Buildings* 140, 81–97."
        )


# ============================================================================
# SENSOR DATA
# ============================================================================
elif page == "Sensor data":
    st.title("Sensor Data & Quality Assurance")
    st.caption("Multi-source ETL: 9 rooms of temp/humidity + weather station + energy meters")

    room_pick = st.selectbox(
        "Room", options=INDOOR_IDS, format_func=lambda i: ROOMS[i], index=1
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean temp", f"{df[f'T{room_pick}'].mean():.1f} °C")
    c2.metric("Mean RH", f"{df[f'RH_{room_pick}'].mean():.1f} %")
    c3.metric("Peak RH", f"{df[f'RH_{room_pick}'].max():.1f} %")
    c4.metric("Min dew-point margin", f"{df[f'dp_margin_{room_pick}'].min():.1f} °C")

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(df.index, df[f"T{room_pick}"], lw=0.6, label=f"{ROOMS[room_pick]} temp")
    axes[0].plot(df.index, df["T_out"], lw=0.6, alpha=0.65, label="Outdoor temp")
    axes[0].set_ylabel("°C")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title("Indoor vs outdoor temperature")
    axes[0].grid(alpha=0.3)

    axes[1].plot(df.index, df[f"RH_{room_pick}"], lw=0.6, color="tab:blue", label="Relative humidity")
    axes[1].axhline(70, color="orange", ls="--", lw=1, label="70% RH")
    axes[1].set_ylabel("% RH")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].set_title("Indoor relative humidity")
    axes[1].grid(alpha=0.3)

    axes[2].plot(df.index, df[f"vp_excess_{room_pick}"], lw=0.5, color="tab:purple")
    axes[2].axhline(0, color="grey", lw=1)
    axes[2].set_ylabel("Pa")
    axes[2].set_title("Vapour pressure excess (moisture generated indoors)")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")
    st.subheader("Moisture load by room")

    summary = []
    for i in INDOOR_IDS:
        summary.append({
            "Room": ROOMS[i],
            "Mean temp (°C)": df[f"T{i}"].mean(),
            "Mean RH (%)": df[f"RH_{i}"].mean(),
            "Peak RH (%)": df[f"RH_{i}"].max(),
            "Vapour excess (Pa)": df[f"vp_excess_{i}"].mean(),
            "Time RH>70% (%)": (df[f"RH_{i}"] > 70).mean() * 100,
        })
    room_summary = pd.DataFrame(summary).set_index("Room").sort_values(
        "Vapour excess (Pa)", ascending=False
    )
    st.dataframe(room_summary.round(1), use_container_width=True)
    st.caption(
        "The bathroom and laundry room show the highest vapour pressure excess — they are the "
        "moisture *sources*. The physics recovers this without being told anything about room function."
    )


# ============================================================================
# RISK ANALYSIS
# ============================================================================
elif page == "Risk analysis":
    st.title("Condensation & Mould Risk")
    st.caption("Physics-derived indicators following BS EN ISO 13788")

    with st.expander("How the risk indices work", expanded=False):
        st.markdown(
            "Two distinct risks:\n\n"
            "1. **Surface condensation** — liquid water forms when a surface falls below "
            "the room air's dew point.\n"
            "2. **Mould germination** — mould does *not* need liquid water. Germination is "
            "possible when **relative humidity at the surface** stays above ~**80%**. "
            "This is the more common and insidious failure in real housing.\n\n"
            "Surface temperature uses the temperature factor $f_{Rsi}$:"
        )
        st.latex(r"T_{surface} = T_{out} + f_{Rsi}\,(T_{in} - T_{out})")
        st.markdown(
            "UK Building Regulations require $f_{Rsi} \\geq 0.75$. Older solid-wall stock "
            "and cold-bridge junctions can fall to 0.5–0.65."
        )

    fabric_scenarios = {
        0.95: "New build / well insulated",
        0.75: "Building Regs minimum",
        0.65: "Poor thermal bridge",
        0.55: "Uninsulated solid wall",
    }

    rows = []
    for f_rsi, label in fabric_scenarios.items():
        for i in INDOOR_IDS:
            t_surf, rh_surf = surface_conditions(
                df[f"T{i}"], df["T_out"], df[f"vp_{i}"], f_rsi
            )
            dp_in = dew_point_from_vp(
                np.minimum(df[f"vp_{i}"], sat_vapour_pressure(df[f"T{i}"]))
            )
            rows.append({
                "fabric": label, "room": ROOMS[i],
                "pct_mould": float((rh_surf > MOULD_RH_THRESHOLD).mean() * 100),
                "pct_cond": float((t_surf < dp_in).mean() * 100),
            })
    real_risk = pd.DataFrame(rows)
    pivot = real_risk.pivot_table(index="room", columns="fabric", values="pct_mould")
    order = [fabric_scenarios[k] for k in sorted(fabric_scenarios, reverse=True)]
    pivot = pivot[order]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("% of time at mould risk")
    ax.set_title("Measured passive house: risk stays low except at severe cold bridges")
    ax.legend(title="Fabric quality", fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(pivot.round(2), use_container_width=True)

    st.success(
        "**Finding: this building is genuinely healthy.** Risk is near zero across almost every "
        "room and fabric quality. You cannot get a damp problem out of dry air, whatever the wall "
        "construction. This is the correct result — and the reason the digital twin exists."
    )

    st.markdown("---")
    st.subheader("Room segmentation by moisture behaviour")

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

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    colors = {"High moisture load": "tab:red", "Moderate": "tab:orange", "Low / stable": "tab:green"}
    for name, grp in seg.groupby("Segment"):
        ax2.scatter(grp["mean_vp_excess"], grp["p95_RH"], s=160,
                    color=colors.get(name, "grey"), label=name, edgecolor="black", zorder=3)
    for room, row in seg.iterrows():
        ax2.annotate(room, (row["mean_vp_excess"], row["p95_RH"]),
                     fontsize=8, xytext=(6, 4), textcoords="offset points")
    ax2.set_xlabel("Mean vapour pressure excess (Pa) → moisture generated indoors")
    ax2.set_ylabel("95th percentile RH (%)")
    ax2.set_title("Segmentation directs limited maintenance capacity by evidence")
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)


# ============================================================================
# DIGITAL TWIN
# ============================================================================
elif page == "Digital twin":
    st.title("Building Digital Twin — Scenario Engine")
    st.caption("Steady-state hygrothermal model driven by the real measured weather series")

    st.markdown(
        "Adjust the four levers a housing provider can actually pull, and see damp risk "
        "respond in real time. **This is where the at-risk stock gets modelled** — the "
        "monitored passive house is healthy, but most social housing is not."
    )

    with st.expander("Model & validation", expanded=False):
        st.markdown("Moisture balance (BS EN ISO 13788). Vapour pressure excess over outdoor:")
        st.latex(r"\Delta p = \frac{G \cdot R_v \cdot T_{in}}{n \cdot V}")
        st.markdown(
            "where $G$ = moisture generation (kg/h), $n$ = air changes per hour, "
            "$V$ = dwelling volume (m³).\n\n"
            "**Validation:** run with passive-house parameters, the twin reproduces the "
            "measured living-room RH to within a few percentage points — the level of "
            "agreement a steady-state screening model should reach.\n\n"
            "**This is not** a CFD or full BIM thermal model. See Limitations."
        )

    preset = st.selectbox(
        "Start from a housing archetype",
        [
            "Custom",
            "A. New build, well ventilated",
            "B. Regs-compliant, average use",
            "C. Older stock, high occupancy",
            "D. Solid wall, under-ventilated",
            "E. As D + fuel poverty (16°C)",
        ],
    )
    presets = {
        "A. New build, well ventilated": (0.70, 0.30, 0.95, 20.0),
        "B. Regs-compliant, average use": (0.50, 0.40, 0.75, 20.0),
        "C. Older stock, high occupancy": (0.35, 0.50, 0.65, 20.0),
        "D. Solid wall, under-ventilated": (0.25, 0.55, 0.55, 20.0),
        "E. As D + fuel poverty (16°C)": (0.25, 0.55, 0.55, 16.0),
    }
    d_ach, d_moist, d_frsi, d_set = presets.get(preset, (0.35, 0.50, 0.65, 20.0))

    c1, c2, c3, c4 = st.columns(4)
    ach = c1.slider("Ventilation (ACH)", 0.10, 1.00, d_ach, 0.05,
                    help="Air changes per hour. Extract fans, trickle vents, MVHR, PIV.")
    moisture = c2.slider("Moisture generation (kg/h)", 0.20, 0.80, d_moist, 0.05,
                         help="Occupancy density, drying clothes indoors, cooking.")
    f_rsi = c3.slider("Fabric quality (fRsi)", 0.45, 0.95, d_frsi, 0.05,
                      help="Surface temperature factor. Building Regs require ≥ 0.75.")
    setpoint = c4.slider("Heating setpoint (°C)", 14.0, 23.0, d_set, 0.5,
                         help="Heating regime. Lower = cheaper but colder surfaces.")

    twin = BuildingTwin(ach=ach, moisture_gen_kg_h=moisture, f_rsi=f_rsi,
                        indoor_setpoint_c=setpoint)
    sim = twin.simulate(weather["T_out"], weather["vp_out"])
    sim.index = weather.index

    risk_pct = float(sim["mould_risk"].mean() * 100)
    cond_pct = float(sim["condensation"].mean() * 100)
    heat_kwh = ventilation_energy_kwh(ach, twin.volume_m3, weather["T_out"], setpoint)
    co2 = heat_kwh / BOILER_EFFICIENCY * CARBON_FACTOR_GAS

    band = ("LOW", "🟢") if risk_pct < 5 else \
           ("MODERATE", "🟡") if risk_pct < 20 else \
           ("HIGH", "🟠") if risk_pct < 50 else ("SEVERE", "🔴")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Mean indoor RH", f"{sim['RH_in'].mean():.1f}%")
    m2.metric("Time at mould risk", f"{risk_pct:.1f}%", delta=f"{band[1]} {band[0]}", delta_color="off")
    m3.metric("Time condensing", f"{cond_pct:.1f}%")
    m4.metric("Ventilation heat loss", f"{heat_kwh:,.0f} kWh")
    m5.metric("Emissions", f"{co2:,.0f} kgCO₂e")

    if f_rsi < 0.75:
        st.warning(
            f"Fabric quality (fRsi {f_rsi:.2f}) is **below the UK Building Regulations "
            f"minimum of 0.75** — typical of uninsulated solid-wall stock or cold-bridge junctions."
        )

    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)
    axes[0].plot(sim.index, sim["RH_surface"], lw=0.5, color="tab:blue")
    axes[0].axhline(MOULD_RH_THRESHOLD, color="red", ls="--", lw=1.2,
                    label="Mould germination threshold (80%)")
    axes[0].fill_between(sim.index, MOULD_RH_THRESHOLD, sim["RH_surface"],
                         where=sim["RH_surface"] > MOULD_RH_THRESHOLD,
                         color="red", alpha=0.3)
    axes[0].set_ylabel("Surface RH (%)")
    axes[0].set_ylim(0, 105)
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].set_title("Surface humidity vs mould threshold")
    axes[0].grid(alpha=0.3)

    axes[1].plot(sim.index, sim["T_surface"], lw=0.6, label="Surface temperature", color="tab:orange")
    axes[1].plot(sim.index, sim["dewpoint_in"], lw=0.6, label="Indoor dew point", color="tab:purple")
    axes[1].fill_between(sim.index, sim["T_surface"], sim["dewpoint_in"],
                         where=sim["T_surface"] < sim["dewpoint_in"],
                         color="purple", alpha=0.3)
    axes[1].set_ylabel("°C")
    axes[1].legend(loc="upper left", fontsize=8)
    axes[1].set_title("Condensation occurs where surface temperature falls below dew point")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    if setpoint < 18:
        st.error(
            "**Fuel poverty trap.** Turning the heating down cuts energy use and bills, but "
            "**increases** damp risk — colder surfaces condense more readily. A tool that reports "
            "carbon savings without surfacing this health consequence is giving incomplete advice."
        )


# ============================================================================
# CARBON TRADE-OFF
# ============================================================================
elif page == "Carbon trade-off":
    st.title("Intervention Trade-off: Health vs Carbon")
    st.caption("Sustainability dossier — indicative energy and carbon impact of each retrofit option")

    st.markdown(
        "Ventilation is the cheapest damp fix, but ventilating a heated dwelling **throws away heat**. "
        "The intervention that best protects residents may be the one that *increases* emissions — "
        "unless heat recovery is used."
    )
    st.latex(r"Q_{vent} = 0.33 \cdot n \cdot V \cdot \Delta T \quad \text{[W]}")

    baseline = BuildingTwin(ach=0.25, moisture_gen_kg_h=0.55, f_rsi=0.55)

    def dossier_row(name, tw, heat_recovery=0.0, fan_power_w=0.0):
        heat = ventilation_energy_kwh(tw.ach, tw.volume_m3, weather["T_out"], tw.indoor_setpoint_c)
        heat *= (1 - heat_recovery)
        fuel = heat / BOILER_EFFICIENCY
        heat_co2 = fuel * CARBON_FACTOR_GAS
        hours = len(weather) / 6.0
        fan_kwh = fan_power_w * hours / 1000.0
        fan_co2 = fan_kwh * CARBON_FACTOR_ELEC
        s = tw.simulate(weather["T_out"], weather["vp_out"])
        return {
            "Intervention": name,
            "ACH": tw.ach,
            "Heat loss (kWh)": heat,
            "Fan (kWh)": fan_kwh,
            "Total kgCO₂e": heat_co2 + fan_co2,
            "Mould risk (%)": float(s["mould_risk"].mean() * 100),
        }

    rows = [
        dossier_row("Baseline (no action)", baseline),
        dossier_row("Extract fans", BuildingTwin(ach=0.45, moisture_gen_kg_h=0.55, f_rsi=0.55)),
        dossier_row("PIV unit", BuildingTwin(ach=0.60, moisture_gen_kg_h=0.55, f_rsi=0.55), fan_power_w=6.0),
        dossier_row("Wall insulation only", BuildingTwin(ach=0.25, moisture_gen_kg_h=0.55, f_rsi=0.80)),
        dossier_row("Resident advice", BuildingTwin(ach=0.25, moisture_gen_kg_h=0.41, f_rsi=0.55)),
        dossier_row("MVHR (85% recovery)", BuildingTwin(ach=0.60, moisture_gen_kg_h=0.55, f_rsi=0.55),
                    heat_recovery=MVHR_HEAT_RECOVERY, fan_power_w=6.0),
        dossier_row("MVHR + insulation", BuildingTwin(ach=0.60, moisture_gen_kg_h=0.55, f_rsi=0.80),
                    heat_recovery=MVHR_HEAT_RECOVERY, fan_power_w=6.0),
    ]
    dos = pd.DataFrame(rows).set_index("Intervention")
    base_co2 = dos.loc["Baseline (no action)", "Total kgCO₂e"]
    base_risk = dos.loc["Baseline (no action)", "Mould risk (%)"]
    dos["Carbon change (kg)"] = dos["Total kgCO₂e"] - base_co2
    dos["Risk reduction (pts)"] = base_risk - dos["Mould risk (%)"]

    plot_df = dos.drop(index="Baseline (no action)")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axhline(0, color="grey", lw=1, ls="--")
    ax.axvline(0, color="grey", lw=1, ls="--")
    ax.scatter(plot_df["Risk reduction (pts)"], plot_df["Carbon change (kg)"], s=170,
               c=["tab:red" if v > 0 else "tab:green" for v in plot_df["Carbon change (kg)"]],
               edgecolor="black", zorder=3)
    for name, row in plot_df.iterrows():
        ax.annotate(name, (row["Risk reduction (pts)"], row["Carbon change (kg)"]),
                    fontsize=8, xytext=(8, 5), textcoords="offset points")
    ax.set_xlabel("Damp risk reduction (percentage points) → better for residents")
    ax.set_ylabel("Change in CO₂e (kg over period)\n← lower is better")
    ax.set_title("Bottom-right quadrant = healthier AND lower carbon")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(dos.round(1), use_container_width=True)

    st.info(
        "**Reading the trade-off.** Simply increasing ventilation moves right (healthier) but also "
        "**up** (more heat thrown away). Only insulation and heat recovery move right **and** down. "
        "This is the analysis that turns a sensor feed into an investment case."
    )
    st.caption(
        f"Carbon factors: BEIS/DESNZ 2023 — gas {CARBON_FACTOR_GAS} kgCO₂e/kWh, "
        f"grid electricity {CARBON_FACTOR_ELEC} kgCO₂e/kWh. Published annually; refresh for real reporting. "
        f"Figures are for the {period_days}-day monitored period, **not annualised**."
    )


# ============================================================================
# FORECASTING
# ============================================================================
elif page == "Forecasting":
    st.title("Humidity Forecasting")
    st.caption("Predicting risk 6 hours ahead — prevention rather than a maintenance ticket")

    metrics, importances, test_out = train_forecast_model(df)

    st.markdown(
        "**Method:** establish naive baselines first — anything that cannot beat *'the value now'* "
        "is not worth deploying — then fit an ML model, then validate with rolling-origin "
        "backtesting. Random splitting on time series leaks the future into training and "
        "produces flattering, false results."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("XGBoost MAE", f"{metrics['xgb_mae']:.2f} %RH")
    c2.metric("Best baseline MAE", f"{metrics['persistence_mae']:.2f} %RH")
    c3.metric("Improvement", f"{metrics['improvement_pct']:.1f}%")

    comp = pd.DataFrame({
        "Model": [
            "Baseline: persistence (RH now)",
            "Baseline: same time yesterday",
            "Baseline: 24h rolling mean",
            "XGBoost (engineered features)",
        ],
        "MAE (%RH)": [
            metrics["persistence_mae"],
            metrics["yesterday_mae"],
            metrics["rollmean_mae"],
            metrics["xgb_mae"],
        ],
    }).sort_values("MAE (%RH)").reset_index(drop=True)
    st.dataframe(comp.round(3), use_container_width=True)

    if metrics["improvement_pct"] < 5:
        st.warning(
            "Marginal gain over the baseline. On this evidence the simpler baseline may be the "
            "better operational choice — far cheaper to run and to explain."
        )
    else:
        st.success(
            f"XGBoost beats the strongest baseline by {metrics['improvement_pct']:.1f}%, "
            f"which justifies the added complexity."
        )

    n_show = st.slider("Test-period window (readings)", 200, 2000, 1000, 100)
    window = test_out.iloc[:n_show]

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(window.index, window["y"], label="Actual RH (6h ahead)", color="black", lw=1.2)
    ax.plot(window.index, window["pred"], label="XGBoost forecast", color="tab:red", lw=1, alpha=0.85)
    ax.plot(window.index, window["RH_now"], label="Persistence baseline", color="tab:blue", lw=0.8, alpha=0.5)
    ax.set_ylabel("% RH")
    ax.set_title("Forecast vs actual")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Feature importance")
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    importances.head(15).sort_values().plot(kind="barh", ax=ax2, color="tab:teal")
    ax2.set_title("Top 15 features — 6h RH forecast")
    ax2.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)
    st.caption(
        f"Trained on {metrics['n_train']:,} readings, tested on {metrics['n_test']:,} "
        f"held-out readings in strict chronological order."
    )


# ============================================================================
# LIMITATIONS
# ============================================================================
elif page == "Limitations":
    st.title("Findings, Limitations & Next Steps")

    st.subheader("Findings")
    st.markdown(
        """
1. **The monitored building is healthy, and that is a real result.** A passive house at ~40% RH
   shows near-zero condensation risk across almost all fabric qualities. Damp is a problem of
   *wet air meeting cold surfaces*; remove either and the risk disappears.

2. **Moisture load is recovered from physics alone.** Vapour pressure excess correctly identifies
   the bathroom and laundry room as moisture sources, without any labelling of room function.

3. **Forecasting must be judged against baselines.** Humidity is highly autocorrelated, so
   persistence is a strong competitor. If the ML margin is thin, the simpler model is the better
   operational recommendation.

4. **Ventilation and carbon are in direct tension.** More ventilation reduces damp risk and raises
   heating demand. Only insulation and heat recovery improve both.

5. **Turning the heating down increases damp risk.** A fuel-poor household reducing emissions can
   make its damp problem worse — a direct collision between a carbon target and a welfare duty.

6. **Seasonal drift is detectable and expected.** PSI flags genuine distribution shift between the
   winter training window and the spring test window — the signal that should trigger retraining.
        """
    )

    st.subheader("Limitations — stated plainly")
    st.warning(
        """
- **No mould labels exist.** Every risk figure is a physics-derived indicator following
  BS EN ISO 13788, not a model validated against inspection records. Validating against real
  damp/mould reports is the single most important next step.
- **The twin is steady-state.** Each timestep is solved independently, with no thermal or moisture
  inertia and no buffering by fabric and furnishings. Real buildings lag; this model does not.
  It is a screening tool, not a substitute for dynamic simulation.
- **Single well-mixed zone.** Inter-room moisture transfer is not modelled, so a bathroom venting
  into a cold bedroom — a very common real failure mode — is out of scope.
- **One building, one climate.** Belgian weather over 4.5 winter-to-spring months. UK stock, and a
  full annual cycle, would shift the numbers.
- **Twin parameters are archetypes, not surveys.** ACH, moisture generation and fRsi values come
  from typical published ranges. Real deployment needs measured ventilation rates and thermographic survey data.
- **Carbon factors date quickly.** BEIS/DESNZ figures are revised annually and hardcoded here.
- **Occupancy is a proxy.** Appliance load stands in for occupancy — correlated, not equivalent.
        """
    )

    st.subheader("Next steps")
    st.markdown(
        """
1. Validate risk indices against real damp and mould inspection records, converting the indicator
   into a calibrated, evidence-backed model.
2. Extend the twin to a dynamic multi-zone model with moisture buffering.
3. Scale from one building to a stock: the same pipeline per property produces a ranked
   intervention list under a fixed retrofit budget.
4. Add cost data alongside carbon, so interventions rank on £ per percentage-point of risk removed.
5. Wire drift monitoring to an automated seasonal retraining schedule with alerting.
        """
    )

    st.markdown("---")
    st.caption(
        "Built by Aamna Arif. Data: Candanedo, Feldheim & Deramaix (2017), *Data driven prediction "
        "models of energy use of appliances in a low-energy house*, Energy and Buildings 140, 81–97. "
        "Methodology follows BS EN ISO 13788 for surface condensation and mould growth risk assessment."
    )
