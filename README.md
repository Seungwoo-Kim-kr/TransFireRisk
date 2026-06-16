# ⚡ TransFireRisk
### Transformer Fire Risk Prediction Using Weather Data
**2026 Weather Big Data Contest**

---

## Overview

**TransFireRisk** is a machine learning project that predicts the risk of electrical transformer fires across South Korean regions using weather data. Transformers are critical infrastructure components of the power grid, and their failure due to weather conditions causes significant property damage and power outages.

This project answers the question:
> *"Given today's weather in a region, how likely is a transformer fire this month?"*

---

## Key Findings

- **~55 transformer fires** occur nationwide every year in South Korea (2020–2024)
- **29%** of annual transformer fires are concentrated in July and August
- **55%** occur on days with average humidity above 80%
- **64%** occur on days with precipitation

### Weather Attacks Transformers in 3 Ways

| Mechanism | Weather Variables | Related Cause |
|-----------|-------------------|---------------|
| 🌡️ **Heat** | Max temperature, Heat wave days | Overload / Overcurrent |
| 💧 **Moisture** | Avg humidity, Total precipitation | Tracking arc, Leakage |
| 📊 **Temperature Swing** | Daily temperature range | Insulation degradation |

---

## Dataset

| Source | Description | Size |
|--------|-------------|------|
| [National Fire Agency (소방청)](https://www.data.go.kr/data/15044003/fileData.do) | Fire incident records 2020–2024 | 191,510 rows |
| [Open-Meteo Historical API](https://archive-api.open-meteo.com) | Daily weather data for 17 regions | 31,059 rows |
| [National Fire Data System (nfds.go.kr)](https://www.nfds.go.kr) | Transformer fire statistics by region/year | Aggregated |

**Analysis Unit:** 17 regions (시도) × 5 years × 12 months = **1,020 rows**

---

## Model

### Architecture
- **Algorithm:** XGBoost Regressor
- **Target (Y):** Number of transformer fires per region per month
- **Train:** 2020–2022 (612 rows) | **Test:** 2023–2024 (408 rows)

### Feature Groups

**① Current Weather (8 features)**
```
월최고기온    Monthly max temperature (℃)
월평균기온    Monthly avg temperature (℃)
월최저기온    Monthly min temperature (℃)
월평균습도    Monthly avg humidity (%)
월강수합계    Monthly total precipitation (mm)
월최대풍속    Monthly max wind speed (km/h)
월평균일교차  Monthly avg daily temperature range (℃)
강수일수      Number of rainy days
```

**② Cumulative Weather (3 features)**
```
월전3일평균기온   3-day rolling avg temperature
월전7일강수합계   7-day rolling total precipitation
월연속고온일수    Consecutive days above 33℃
```

**③ Temporal (2 features)**
```
월        Month (1–12)
계절코드   Season code (Spring/Summer/Fall/Winter)
```

**④ Regional History (3 features)**
```
시도코드             Region code (0–16)
전년전기화재건수     Previous year's electrical fire count in region
전년변압기화재건수   Previous year's transformer fire count in region
```

### Performance

| Metric | Train | Test |
|--------|-------|------|
| RMSE | 0.1052 | 0.3169 |
| MAE  | 0.0539 | 0.1788 |
| R²   | 0.8735 | -0.1653 |
| 5-Fold CV RMSE | **0.3060 ± 0.0547** | — |

> **Note:** The negative test R² reflects the challenge of predicting rare events (81 fire-months out of 1,020). The model captures seasonal and regional patterns but requires more historical data for stronger generalization.

### Top Feature Importances
```
1. 월전3일평균기온   (3-day avg temp)        0.0890
2. 전년전기화재건수  (regional fire history)  0.0807
3. 월전7일강수합계   (7-day rain total)       0.0760
4. 시도코드          (region)                 0.0716
5. 월평균습도        (avg humidity)           0.0699
```

---

## Risk Grade System

| Grade | Predicted Count | Color |
|-------|----------------|-------|
| 🟢 Low       | < 0.3  | Safe conditions |
| 🟡 Moderate  | 0.3 – 0.7 | Elevated awareness |
| 🟠 High      | 0.7 – 1.2 | Pre-emptive inspection recommended |
| 🔴 Very High | ≥ 1.2  | Immediate response preparation |

---

## Dashboard

3-page Streamlit dashboard:

| Page | Description |
|------|-------------|
| 📊 Current Status | Annual trends, regional distribution, season × month heatmap |
| 🌦️ Weather Correlation | Feature importance, cause × weather conditions, scatter plots |
| 🔮 Risk Prediction | Real-time risk score from weather slider inputs |

---

## Project Structure

```
TransFireRisk/
├── app.py                      # Streamlit dashboard
├── src/
│   ├── 01_preprocess.py        # Data preprocessing & feature engineering
│   ├── 02_model.py             # XGBoost training & evaluation
│   └── 03_visualize.py         # Static chart generation
├── data/
│   ├── weather_sido_2020_2024.csv
│   ├── transformer_fire_yearly.csv
│   ├── transformer_fire_sido.csv
│   └── transformer_fire_weather_merged.csv
├── model/
│   ├── xgb_transformer_fire.pkl
│   ├── model_input.csv
│   ├── full_predictions.csv
│   └── feature_importance.csv
└── output/
    ├── fig1_feature_importance.png
    ├── fig2_yearly_comparison.png
    ├── fig3_heatmap_sido_month.png
    ├── fig4_weather_scatter.png
    └── fig5_sido_risk_2024.png
```

---

## Setup & Run

```bash
# Clone repository
git clone https://github.com/Seungwoo-Kim-kr/TransFireRisk.git
cd TransFireRisk

# Install dependencies
pip install -r requirements.txt

# Run preprocessing & model training
python src/01_preprocess.py
python src/02_model.py
python src/03_visualize.py

# Launch dashboard
streamlit run app.py
```

---

## Requirements

```
streamlit>=1.28.0
xgboost>=2.0.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=2.0.0
plotly>=5.18.0
xarray>=2024.1.0
joblib>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
requests>=2.31.0
```

---

## Limitations & Future Work

| Current Limitation | Future Improvement |
|-------------------|-------------------|
| Region-level spatial resolution (시도) | Grid-level precision with KEPCO transformer location data |
| Monthly time resolution | Daily prediction with sufficient incident data |
| No transformer age/condition data | Integration with KEPCO installation records |
| 5-year training window | Improved generalization with longer history |

---

## Data Sources

- **소방청 화재발생정보** — [data.go.kr](https://www.data.go.kr/data/15044003/fileData.do)
- **국가화재정보시스템 (nfds.go.kr)** — [nfds.go.kr](https://www.nfds.go.kr/stat/general.do)
- **Open-Meteo Historical Weather API** — [open-meteo.com](https://open-meteo.com)

---

## License

This project is for academic and competition purposes.
Data sourced from public Korean government open data portals.

---

*2026 날씨 빅데이터 콘테스트 (Weather Big Data Contest 2026)*
