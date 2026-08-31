# Damp & Mould Risk Intelligence Platform

Turning building sensor data into maintenance decisions — and testing retrofit
interventions before paying for them.

**Live app:** https://aamna-damp-risk.streamlit.app/

Housing providers hold large volumes of IoT sensor data, but raw temperature and humidity
readings don't tell a maintenance team **which** homes are at risk of damp and mould,
**why**, or **what intervention would fix it at what carbon cost**. This project builds
that decision layer end to end.

---

## What it does

| Section | Capability |
|---|---|
| **Sensor data** | Multi-source ETL across 9 rooms plus a weather station and energy meters. Strict 10-minute time grid, stuck-sensor detection, bounded gap-filling |
| **Risk analysis** | Condensation and mould risk indices via surface temperature factor (BS EN ISO 13788), plus room segmentation by moisture behaviour (KMeans) |
| **Digital twin** | Steady-state hygrothermal simulator with four intervention levers, validated against the measured building |
| **Carbon trade-off** | Energy and kgCO₂e per intervention, surfacing the tension between resident health and emissions |
| **Forecasting** | 6-hour-ahead humidity forecast — naive baselines first, then XGBoost, validated chronologically |

## Key results

- **XGBoost beats the strongest naive baseline by ~30%** (MAE 2.64 vs 3.81 %RH) on
  held-out data in strict chronological order.
- **The twin reproduces the real building to within 4.2 percentage points** of measured
  humidity when run with passive-house parameters.
- **Vapour pressure excess correctly identifies the bathroom and laundry room** as the
  moisture sources — recovered from physics alone, with no labelling of room function.

## Honest framing

The monitored building is a **passive house** — airtight, mechanically ventilated, and
very dry (~40% RH). It has almost no condensation risk. **That is a real finding, not a
failed analysis:** damp is a problem of wet air meeting cold surfaces, and this building
has neither.

So the design follows from it. The real sensor data provides a validated *healthy
baseline* and the full ETL and forecasting pipeline. The **digital twin** then models the
degraded fabric and under-ventilation typical of the older housing stock that actually has
damp problems.

Risk indices here are **physics-derived indicators**, not models trained on labelled mould
inspections — no such labels exist in this dataset. Validating them against real damp and
mould inspection records is the single most important next step, and the Limitations page
in the app sets out the rest.

One finding worth calling out: turning the heating down to save carbon **increases** damp
risk, because colder surfaces condense more readily. A tool that reports carbon savings
without surfacing that consequence is giving incomplete advice.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```
├── app.py              # Streamlit application
├── theme.py            # Design system: palette, CSS, chart theme, components
├── requirements.txt
└── .streamlit/
    └── config.toml     # Base theme
```

## Built with

Python · Streamlit · XGBoost · scikit-learn · pandas · Matplotlib

## Data

Candanedo, L., Feldheim, V. & Deramaix, D. (2017). *Data driven prediction models of
energy use of appliances in a low-energy house.* **Energy and Buildings**, 140, 81–97.

19,735 readings at 10-minute intervals over 4.5 months, from a ZigBee wireless sensor
network in an occupied low-energy house, merged with weather-station and m-bus energy
meter data. Loaded directly from the authors' public repository — no download needed.

Risk methodology follows **BS EN ISO 13788** for surface condensation and mould growth
risk assessment. Carbon factors are BEIS/DESNZ 2023; these are revised annually and should
be refreshed for any real reporting.

---

Built by Aamna Arif

---
Built by Aamna Arif.
