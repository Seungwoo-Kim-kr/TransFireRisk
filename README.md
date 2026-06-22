# TransFireRisk IMS

**Transformer Fire Risk Integrated Management System**  
변압기 화재 위험 통합관리 시스템 · 2026 Korea Weather Big Data Contest

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Overview

TransFireRisk IMS predicts the monthly probability of transformer fire occurrence across 17 Korean provinces by combining machine learning with a physics-based composite risk index. The system is designed as an operational decision-support tool for power utility engineers and risk managers.

| Item | Details |
|---|---|
| **Data** | Korea Fire Agency (NFDS) 2020–2024 · Open-Meteo weather API |
| **Granularity** | Province (시도) × Month — 1,020 records |
| **Target** | Binary: fire occurrence (yes/no) → probability 0–100% |
| **Model** | Ensemble v3: XGBoost + RandomForest + LogisticRegression |
| **Composite Index** | TFRI = WHI (45%) + IDA (35%) + HRI (20%) |
| **Weather Forecast** | Open-Meteo (14-day, real-time) |
| **Language** | Korean / English toggle |

---

## Quick Start

```bash
git clone https://github.com/Seungwoo-Kim-kr/TransFireRisk.git
cd TransFireRisk

pip install -r requirements.txt

# Optional: configure API key for AI briefings
cp .env.example .env
# Edit .env and set OPENAI_API_KEY

streamlit run app.py --server.port 8503
# → http://localhost:8503
```

---

## Dashboard Tabs

| # | Tab | Purpose |
|---|---|---|
| 1 | **Dashboard** | National risk overview, AI operational briefing, province cards, KPI metrics |
| 2 | **Regional Detail** | Province drill-down, 12-month trend, TFRI radar, AI inspection guide |
| 3 | **Weather Forecast** | 14-day Open-Meteo forecast, daily ML + TFRI risk timeline |
| 4 | **Scenario Simulation** | Custom weather input, sensitivity heatmap (temp × humidity), AI analysis |
| 5 | **Risk Analysis** | TFRI component breakdown (WHI / IDA / HRI), national stack comparison |
| 6 | **Inspection Mgmt** | P1/P2/P3 priority list, status tracking, CSV export |
| 7 | **History & Model** | Fire heatmaps, ROC curve, feature importance, model performance table |

---

## Project Structure

```
TransFireRisk/
├── app.py                          # Streamlit dashboard (v7.1)
├── requirements.txt
├── .env.example                    # API key template
├── .streamlit/config.toml
├── data/
│   ├── weather_sido_2020_2024.csv  # Province-level daily weather (2020–2024)
│   └── ...
├── model/
│   ├── ensemble_v3.pkl             # Production model bundle
│   ├── full_predictions_final.csv  # Pre-computed fire probabilities
│   ├── model_input.csv             # Feature-engineered training data
│   └── model_comparison_v3.csv     # Experiment results
├── output/                         # Saved figures
└── src/
    ├── 01_preprocess.py            # Data preprocessing pipeline
    ├── 02_model.py                 # Baseline XGBoost regressor (v1)
    ├── 02b_model_improved.py       # Experiment B
    └── 02c_model_advanced.py       # Production ensemble v3
```

---

## Model — Ensemble v3

### Problem Framing

The dataset has severe class imbalance (7.8% fire occurrence rate across 1,020 province × month records). A regression approach on fire count systematically underestimates actual events. The solution reframes the task as **binary classification**: predict the probability that at least one transformer fire occurs in a given province and month.

### Feature Engineering (16 → 28 features)

| Feature | Formula | Physical Rationale |
|---|---|---|
| `열지수` | T_max × RH / 100 | Combined heat-moisture stress |
| `열습도스트레스` | max(0,T−30) × max(0,(RH−60)/40) | Nonlinear overload zone |
| `강수고온지수` | Rain × max(0,T_avg−15) / 100 | Post-rain heat → tracking fault risk |
| `폭염습도` | HeatWaveDays × RH / 100 | Heat wave × humidity interaction |
| `과부하스트레스` | max(0, T_max−33)² | Quadratic thermal overload |
| `강수습도` | RainDays × RH / 100 | Moisture ingress composite |
| `기온/습도/강수 편차` | Value − climatological mean | Anomaly from historical baseline |
| `지역발화율` | Province historical fire rate | Regional baseline vulnerability |
| `월_sin / 월_cos` | sin/cos(2π·month/12) | Cyclic seasonality encoding |

### Performance

| Metric | v1 Baseline | v3 Ensemble | Change |
|---|---|---|---|
| ROC-AUC | 0.612 | **0.640** | +0.028 |
| PR-AUC | ~0.08 | **0.122** | +53% |
| F2 β=2 (thr=0.25) | — | **0.324** | New |
| Recall @ thr=0.20 | 0.219 | **0.562** | +0.343 |
| MCC | — | **0.134** | New |
| Fire events caught | 13/32 | **18/32** | +5 |

**Why F2 and MCC instead of F1?** For 7.8% class imbalance, F1 equally penalises missed fires and false alarms. F2 (β=2) weights recall 2×, reflecting the asymmetric cost structure where missing a fire is more costly than a false alarm. MCC is the most robust single metric for skewed class distributions (range −1 to +1, unaffected by imbalance).

---

## TFRI — Physics-Based Composite Index

```
TFRI = 0.45 × WHI + 0.35 × IDA + 0.20 × HRI
```

| Component | Description | Reference Standard |
|---|---|---|
| **WHI** — Weather Hazard Index | Thermal stress (0.32) + moisture (0.28) + precipitation (0.25) + thermal fatigue (0.15) | CIGRE WG A2.49 · Hwang et al. (2020) KIEE |
| **IDA** — Insulation Degradation Accelerator | Hot-spot temperature model (IEC 60076-7) + Montsinger aging rule + moisture correction | IEC 60076-7:2005 Annex A · Emsley & Stevens (1994) |
| **HRI** — Historical Risk Index | Province-month fire incidence relative to national average | NFDS statistics · Kim & Park (2019) JKIIEE |

The **combined score** = `(ML probability + TFRI) / 2` balances data-driven and physics-driven signals, compensating for the weaknesses of each approach in isolation.

---

## AI Features

All AI-generated content uses structured prompts with system/user role separation. Output follows a fixed four-section format (Risk Mechanism / Inspection Checklist / Maintenance Plan / Weather Response). Emoji usage is explicitly prohibited in prompts; emphasis is conveyed through **bold** text only.

| Feature | Tab | Trigger |
|---|---|---|
| Operational Briefing | Dashboard | Auto on load |
| Inspection Guide | Regional Detail | Auto on region/month select |
| Forecast Analysis | Weather Forecast | Auto on forecast load |
| Scenario Analysis | Scenario Simulation | Auto on slider change |

A rule-based fallback (no API key required) follows the same four-section structure with IEC-referenced acceptance criteria.

---

## Known Limitations

The fundamental constraint is the **small positive sample** (81 fire events in 1,020 records). The practical performance ceiling at province × month granularity is approximately PR-AUC ≈ 0.15, ROC-AUC ≈ 0.65.

**Recommended use cases:**
- Risk ranking and prioritisation across provinces and months
- Complementing the physics-based TFRI index
- Seasonal and regional trend detection

**Not recommended for:** standalone fire-count prediction.

### Granularity Constraint

All predictions operate at the **province × month** level. When the model flags a province as high-risk, the AI inspection guide applies uniformly to all transformers in that province. Identifying which specific substation or transformer unit is at elevated risk is not possible with the current dataset.

To enable facility-level risk prediction, the following data would need to be integrated:

| Data Type | Description | Expected Impact |
|---|---|---|
| Individual transformer registry | Equipment ID, location, voltage class, capacity, year of manufacture | Enable pinpointing specific high-risk units |
| Maintenance and inspection history | Last inspection date, insulation test results, oil sample records, fault history | Improve IDA accuracy at unit level |
| Real-time load and temperature | Per-transformer load factor, winding temperature | Replace monthly weather aggregates with live data |
| Substation topology | Grid connectivity, transformer roles | Model cascading risk across interconnected facilities |

With facility-level data, the system could evolve from a province-wide alert platform into a **unit-specific predictive maintenance system**.

---

## Roadmap

- [ ] **Facility-level granularity** — integrate individual transformer registry and location data
- [ ] **Real-time operational data** — per-unit load factor and winding temperature feeds
- [ ] Extended training history (2010–2024, 3× more fire events)
- [ ] SHAP explainability per prediction
- [ ] Email / Slack alerts for P1 provinces
- [ ] Auto-generated monthly PDF report
- [ ] Long-range climate scenario projections (SSP2-4.5, SSP5-8.5)

---

## Requirements

```
streamlit>=1.28.0
xgboost>=2.0.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=2.0.0
plotly>=5.18.0
joblib>=1.3.0
requests>=2.31.0
openai>=1.30.0
python-dotenv>=1.0.0
```

---

## License

MIT License — see [LICENSE](LICENSE)

---

*2026 Korea Weather Big Data Contest*
