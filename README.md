# ⚡ TransFireRisk IMS

> **Transformer Fire Risk Integrated Management System**  
> 변압기 화재 위험 통합관리 시스템 · 2026 Weather Big Data Contest

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🗂️ Overview

TransFireRisk IMS predicts **monthly transformer fire probability** across 17 Korean provinces using weather data, delivering an operational dashboard for field engineers and risk managers.

| Item | Details |
|---|---|
| **Data** | Korea Fire Agency (NFDS) 2020–2024 · Open-Meteo weather API |
| **Granularity** | Province (시도) × Month — 1,020 records |
| **Target** | Binary: fire occurrence probability (0–100%) |
| **Model** | Ensemble v3: XGBoost + RandomForest + LogisticRegression |
| **Composite Index** | TFRI = WHI(45%) + IDA(35%) + HRI(20%) |
| **Language** | Korean / English (toggle in sidebar) |

---

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/Seungwoo-Kim-kr/TransFireRisk.git
cd TransFireRisk

# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run app.py --server.port 8503
# → http://localhost:8503
```

**Optional API keys** (enter in sidebar at runtime):
- **KMA API** (기상청): 3-day official weather forecast — [data.go.kr](https://data.go.kr)
- **OpenAI API**: AI operational briefings and inspection guides (GPT-4o-mini)

---

## 🏗️ Project Structure

```
TransFireRisk/
├── app.py                          # Streamlit IMS dashboard (v7.0)
├── requirements.txt
├── .streamlit/config.toml
├── data/
│   ├── fire_data_20241231.csv      # Korea Fire Agency raw data
│   ├── weather_sido_2020_2024.csv  # Province-level daily weather
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
    ├── 02b_model_improved.py       # Experiment B: Poisson / Hurdle / SMOTE
    └── 02c_model_advanced.py       # Experiment C → production ensemble v3
```

---

## 📊 Model — Ensemble v3

### Problem Reframing

The dataset has **severe class imbalance** (7.8% fire occurrence, 81/1,020 rows). A single XGBoost regressor on fire *count* severely underestimates real events (predicted 0.17 fires/month vs actual 1.06).

**Solution:** Binary classification (fire probability) + feature engineering + soft-voting ensemble.

### Feature Engineering (+12 features, 16 → 28 total)

| Feature | Formula | Physical Rationale |
|---|---|---|
| `열지수` | T_max × RH / 100 | Combined heat-moisture stress |
| `열습도스트레스` | max(0,T−30) × max(0,(RH−60)/40) | Nonlinear overload zone |
| `강수고온지수` | Rain × max(0,T_avg−15) / 100 | Post-rain heat → tracking fault risk |
| `폭염습도` | HeatWaveDays × RH / 100 | Heat wave × humidity interaction |
| `과부하스트레스` | max(0, T_max−33)² | Quadratic overload above 33 °C |
| `강수습도` | RainDays × RH / 100 | Moisture ingress composite |
| `기온편차` | T_avg − climatological mean | Anomaly relative to historical normal |
| `습도편차` | RH − climatological mean | |
| `강수편차` | Rain − climatological mean | |
| `지역발화율` | Province historical fire rate | Regional baseline vulnerability |
| `월_sin / 월_cos` | sin/cos(2π·month/12) | Cyclic seasonality encoding |

### Performance Comparison

| Metric | v1 Baseline | v3 Ensemble | Δ |
|---|---|---|---|
| ROC-AUC | 0.612 | **0.640** | +0.028 |
| PR-AUC | ~0.08 | **0.122** | +53% |
| F2 β=2 (thr=0.25) | — | **0.324** | New |
| Recall @ thr=0.20 | 0.219 | **0.562** | +0.343 |
| MCC | — | **0.134** | New |
| Fire events caught | 13/32 | **18/32** | +5 |

### Why F2 and MCC — Not F1?

For **heavily imbalanced** binary classification (7.8% positive rate):

| Metric | Why it matters here |
|---|---|
| **F1** | Equal weight to precision & recall — *not ideal* when costs are asymmetric |
| **F2 (β=2)** | Recall weighted 2× — missing a fire costs more than a false alarm ✓ |
| **MCC** | −1 to +1 range; robust to any class distribution; most informative single score ✓ |
| **PR-AUC** | Summarises precision–recall without threshold selection; better than ROC-AUC for imbalanced data ✓ |

---

## 🔬 TFRI — Physics-Based Composite Index

```
TFRI = 0.45 × WHI + 0.35 × IDA + 0.20 × HRI
```

| Component | Description | Reference |
|---|---|---|
| **WHI** — Weather Hazard Index | Thermal stress (0.32) + moisture (0.28) + precipitation (0.25) + thermal fatigue (0.15) | CIGRE WG A2.49 · Hwang et al. (2020) KIEE |
| **IDA** — Insulation Degradation Accelerator | Hot-spot temperature (IEC 60076-7) + Montsinger aging rule + moisture correction | **IEC 60076-7:2005 Annex A** (Dakin equation, B=15000 K) · Emsley & Stevens (1994) |
| **HRI** — Historical Risk Index | Province-month fire incidence relative to national baseline | NFDS statistics · Kim & Park (2019) JKIIEE |

**Combined score** = `(ML probability + TFRI) / 2` — complements data-driven and physics-driven signals.

---

## 🖥️ Dashboard Tabs

| Tab | Content |
|---|---|
| 🏠 **Dashboard** | 17-province risk cards, **AI operational briefing**, alert banners, KPI metrics |
| 🗺️ **Regional Detail** | Province drill-down, 12-month trend, TFRI radar chart, **AI inspection guide** |
| 📡 **Weather Forecast** | KMA API (3 days) + Open-Meteo (14 days), daily ML+TFRI risk timeline |
| 🔬 **Risk Analysis** | TFRI component breakdown, radar chart, national stacked comparison |
| 📋 **Inspection Mgmt** | P1/P2/P3 priority list, status tracking (대기/점검중/완료), CSV export |
| 📊 **History & Model** | Fire heatmaps, ROC curve, feature importance, v1→v3 performance table |

**Language**: Korean ↔ English toggle in sidebar.

---

## 🤖 AI Features

| Feature | Location | Description |
|---|---|---|
| **AI Operational Briefing** | Dashboard tab | GPT-4o-mini: national risk summary, priority regions, top 2 actions |
| **AI Inspection Guide** | Regional / Forecast tabs | Region-specific maintenance plan based on risk scores and TFRI |
| **Rule-based fallback** | All tabs | Structured output without OpenAI key |

---

## ⚠️ Known Limitations

The fundamental constraint is the **small positive sample** (81 fire events in 1,020 records at province×month granularity). The practical performance ceiling at this granularity is approximately **PR-AUC ≈ 0.15, ROC-AUC ≈ 0.65**.

**Recommended use cases:**
- ✅ Risk **ranking and prioritisation** across provinces/months
- ✅ **Complementing** the physics-based TFRI index
- ✅ **Seasonal/regional trend** detection
- ❌ Not a standalone fire-count prediction system

### Granularity Constraint

All predictions and inspection guidance operate at the **province (시도) × month** level — the finest granularity available in the current dataset. As a result, when the model identifies a high-risk region such as Seoul, the AI-generated inspection guide applies uniformly to all transformers within that province. It is not possible to identify which specific substation or individual transformer unit is at elevated risk.

To enable facility-level risk prediction, the following data would need to be integrated:

| Data Type | Description | Expected Impact |
|---|---|---|
| Individual transformer registry | Equipment ID, location (lat/lon), voltage class, rated capacity, year of manufacture | Enable pinpointing specific high-risk units |
| Maintenance and inspection history | Last inspection date, insulation test results, oil sample records, fault history | Improve IDA component accuracy at unit level |
| Real-time load and temperature data | Per-transformer load factor, winding temperature measurements | Replace monthly weather aggregates with live operational data |
| Substation topology | Grid connectivity, transformer roles (primary/secondary/tertiary) | Model cascading risk across interconnected facilities |

With facility-level data, the risk model could evolve from a province-wide alert system into a **unit-specific predictive maintenance platform**, enabling prioritised dispatch of inspection teams to individual transformers rather than entire provinces.

---

## 🗺️ Roadmap

- [ ] **Facility-level granularity** — integrate individual transformer registry and location data (see Granularity Constraint above)
- [ ] **Real-time operational data** — per-unit load factor and winding temperature feeds
- [ ] Extended history (2010–2024, 3× more fire events for model training)
- [ ] SHAP explainability per prediction
- [ ] Email / Slack alerts for P1 regions
- [ ] Auto-generated monthly PDF report
- [ ] Long-range climate scenario projections (SSP2-4.5, SSP5-8.5)

---

## 📋 Requirements

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
imbalanced-learn>=0.12.0
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

*2026 날씨 빅데이터 콘테스트 (Korea Weather Big Data Contest 2026)*
