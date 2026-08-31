# Damp & Mould Risk Intelligence Platform

Interactive decision-support dashboard turning building sensor data into maintenance
decisions and retrofit investment cases.

**Live demo:** _(add your Streamlit Cloud URL here after deploying)_

## What it does

| Section | Capability |
|---|---|
| Sensor data | Multi-source ETL — 9 rooms of temp/humidity + weather station + energy meters, strict 10-min time grid, bounded gap-filling |
| Risk analysis | Condensation & mould risk indices (BS EN ISO 13788), room segmentation via KMeans |
| Digital twin | Steady-state hygrothermal simulator with 4 intervention levers, validated against the measured building |
| Carbon trade-off | Energy + kgCO₂e per intervention, surfacing the health-vs-carbon tension |
| Forecasting | 6-hour-ahead humidity forecast, baselines → XGBoost, chronological validation |

## Honest framing

The monitored building is a **passive house** — it has almost no damp risk. That is a real
finding, not a failed analysis. The real sensor data provides a validated healthy baseline;
the **digital twin** models the degraded fabric and under-ventilation typical of the older
housing stock that actually has damp problems.

Risk indices are **physics-derived indicators**, not models trained on labelled mould
inspections — no such labels exist in this dataset. See the Limitations page.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

Push to GitHub, then at [share.streamlit.io](https://share.streamlit.io) point a new app at
this repo with `app.py` as the entry point. Free, no server admin.

## Data

Candanedo, Feldheim & Deramaix (2017), *Data driven prediction models of energy use of
appliances in a low-energy house*, **Energy and Buildings** 140, 81–97.
Loaded directly from the authors' public repository — no download needed.

---
Built by Aamna Arif.
