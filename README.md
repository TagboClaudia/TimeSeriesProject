<div align="center">

# 📈 Time Series Forecasting Pipeline
### End-to-End Retail Demand Forecasting — Corporación Favorita

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-189C2D?logo=xgboost&logoColor=white)](https://xgboost.ai)
[![Streamlit](https://img.shields.io/badge/Streamlit-Latest-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Pandas](https://img.shields.io/badge/Pandas-Latest-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Plotly](https://img.shields.io/badge/Plotly-Latest-3F4F75?logo=plotly&logoColor=white)](https://plotly.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-TagboClaudia-181717?logo=github)](https://github.com/TagboClaudia)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Claudia%20Fotso-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/claudia-fotso)

**A production-grade, end-to-end machine learning pipeline for retail demand forecasting. This project combines deep learning (LSTM), gradient boosting (XGBoost), and classical statistical modeling (ARIMA) into a unified multi-model ensemble, delivered through a professional dark-mode Streamlit dashboard with real-time interactive visualizations.**

</div>

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Business Problem & Context](#-business-problem--context)
3. [Business Objectives & Expected Impact](#-business-objectives--expected-impact)
4. [Pipeline Architecture & Workflow](#-pipeline-architecture--workflow)
5. [Technologies & Methodologies](#-technologies--methodologies)
6. [Notebook Pipeline Deep-Dive](#-notebook-pipeline-deep-dive)
7. [Data Preprocessing & Feature Engineering](#-data-preprocessing--feature-engineering)
8. [Multi-Model Forecasting Strategy](#-multi-model-forecasting-strategy)
9. [Dashboard & Visualization Layer](#-dashboard--visualization-layer)
10. [Results Analysis & Performance Metrics](#-results-analysis--performance-metrics)
11. [Business Insights & Strategic Findings](#-business-insights--strategic-findings)
12. [Key Recommendations](#-key-recommendations)
13. [Future Improvements & Scalability](#-future-improvements--scalability)
14. [Real-World Business Impact](#-real-world-business-impact)
15. [Project File Structure](#-project-file-structure)
16. [Installation & Usage](#-installation--usage)
17. [Conclusion](#-conclusion)
18. [Authors](#-authors)

---

## 🌟 Project Overview

This project delivers a **fully integrated, multi-model time series forecasting system** for predicting daily grocery sales at the product-store level, built on the Corporación Favorita dataset. The pipeline spans the full ML lifecycle — from raw data ingestion through preprocessing, feature engineering, model training, and interactive visualization — following a clean, modular architecture.

The solution is built around three complementary modeling strategies:

- **LSTM (Long Short-Term Memory)** — A deep learning sequence model that captures non-linear temporal dependencies and long-range demand patterns beyond the reach of classical methods.
- **XGBoost** — A gradient boosting model that excels at tabular time series data with engineered lag and rolling features, offering fast inference and strong out-of-the-box performance.
- **ARIMA** — A classical statistical baseline for interpretable trend and seasonality decomposition, providing a benchmark against which deep learning lift can be objectively measured.

The three models are developed in parallel notebook pipelines and surfaced through a shared **dark-mode Streamlit dashboard**, enabling analysts to compare performance, explore forecasts interactively, and export results — all without writing code.

---

## 🏪 Business Problem & Context

### The Challenge

**Corporación Favorita** is a large Ecuadorian grocery retailer operating hundreds of stores across the country, carrying tens of thousands of SKUs. Grocery retail is characterized by:

- **Extreme demand volatility** driven by promotions, local holidays, and seasonal cycles
- **Perishable inventory** requiring tight daily supply-demand alignment to minimize waste
- **High SKU complexity** — thousands of store-item combinations requiring individual or segmented forecasting
- **Competitive pricing pressures** that make operational inefficiencies particularly costly

Without accurate demand forecasts, the retailer faces two equally costly risks: **overstocking** (leading to waste and capital lock-up) and **stockouts** (causing lost revenue and customer dissatisfaction). Both erode margin and brand trust in a high-frequency retail environment.

### The Data

The dataset provides **daily unit sales per store-item pair**, enriched with:

- Transaction records spanning multiple years (modeled from January 2013 to April 2014)
- Store metadata (location, cluster, type) — configurable via Store ID input
- Item metadata (product family, class, perishability) — configurable via Item ID input
- Engineered rolling features: 7-day and 30-day moving averages for trend smoothing
- Promotional flags (`onpromotion`) and temporal features (day of week, month, year)
- Holiday and event data for calendar-aware demand modeling

### The Scope

The pipeline targets **configurable store-item pairs**, allowing analysts to reconfigure targets via the Streamlit sidebar without code changes. This enables rapid exploratory forecasting across the product catalog while maintaining a reproducible, version-controlled modeling workflow in the notebooks.

---

## 🎯 Business Objectives & Expected Impact

| Objective | Description | Expected Impact |
|---|---|---|
| **Multi-Model Forecasting** | Train and compare LSTM, XGBoost, and ARIMA on the same target | Identify the best-performing model per store-item segment |
| **Interactive Exploration** | Analyst-driven forecasting via sidebar configuration | Reduce time-to-insight from days to seconds |
| **Historical Validation** | Overlay predictions on known historical data for all models | Quantify model accuracy across the full training window |
| **Future Projection** | Generate configurable forward forecasts (7–90 days) | Enable proactive inventory and procurement planning |
| **Residual Diagnostics** | Visualize and quantify prediction errors over time | Identify systematic biases and seasonal failure modes |
| **Automated Feature Engineering** | Lag features, rolling statistics, seasonal decomposition | Maximize signal extraction from raw transactional data |
| **Production Pipeline** | Modular, reproducible ML pipeline from raw data to deployment | Reduce operational cost of maintaining forecasting infrastructure |

---

## 🏗️ Pipeline Architecture & Workflow

The project follows a clean, stage-gated architecture separating data processing, modeling, and the UI layer:

```
┌──────────────────────────────────────────────────────────────────────┐
│                    END-TO-END PIPELINE OVERVIEW                      │
├───────────────────┬──────────────────────────────────────────────────┤
│  RAW DATA         │  Corporación Favorita CSV datasets               │
│                   │  train.csv, stores.csv, items.csv, holidays.csv  │
├───────────────────┼──────────────────────────────────────────────────┤
│  PREPROCESSING    │  01_preprocessing.ipynb                          │
│                   │  Cleaning, type conversion, outlier handling     │
├───────────────────┼──────────────────────────────────────────────────┤
│  FEATURE ENG.     │  02_feature_engineering.ipynb                    │
│                   │  Lag features, rolling stats, calendar features  │
├───────────────────┼──────────────────────────────────────────────────┤
│  DATA MANAGEMENT  │  03_data_management.ipynb                        │
│                   │  I/O processes, caching, filtered subsets        │
├───────────────────┼──────────────────────────────────────────────────┤
│  LSTM MODELING    │  04_lstm_modeling.ipynb                          │
│                   │  TensorFlow/Keras sequence model, walk-forward   │
├───────────────────┼──────────────────────────────────────────────────┤
│  XGBOOST MODEL    │  05_xgboost_modeling.ipynb                       │
│                   │  Gradient boosting, hyperparameter tuning        │
├───────────────────┼──────────────────────────────────────────────────┤
│  ARIMA BASELINE   │  06_arima_analysis.ipynb                         │
│                   │  Statistical baseline, seasonal decomposition    │
├───────────────────┼──────────────────────────────────────────────────┤
│  DASHBOARD        │  app/app.py — Streamlit Dark Mode UI             │
│                   │  Real-time forecasting, metrics, export          │
└───────────────────┴──────────────────────────────────────────────────┘
```

---

## 🛠️ Technologies & Methodologies

### Machine Learning & Data Science

| Library | Role |
|---|---|
| **TensorFlow / Keras** | LSTM neural networks for sequence forecasting |
| **XGBoost 2.0.3** | Gradient boosting for tabular time series data |
| **Statsmodels / pmdarima** | ARIMA, SARIMA, and statistical decomposition |
| **Scikit-learn** | Feature engineering, scaling, cross-validation |
| **Pandas & NumPy** | Data manipulation, aggregation, and transformation |
| **MLflow** | Experiment tracking and model versioning |
| **Hyperopt** | Bayesian hyperparameter optimization |

### Dashboard & Visualization

| Library | Role |
|---|---|
| **Streamlit** | Interactive web application framework |
| **Plotly** | Interactive, production-grade visualizations |
| **Matplotlib / Seaborn** | Static plots and statistical diagnostics |

### Development & Infrastructure

| Tool | Role |
|---|---|
| **Python 3.10+** | Core programming language |
| **Conda / pip** | Environment and package management |
| **Git** | Version control |
| **Joblib** | Model serialization and caching |

---

## 📓 Notebook Pipeline Deep-Dive

The research and development pipeline is organized across six sequential notebooks, each with a focused responsibility:

### `01_preprocessing.ipynb` — Data Cleaning & Transformation
Handles raw data ingestion, type validation, duplicate removal, missing value imputation, and outlier treatment. Produces clean, validated datasets persisted to `data/processed/cleaner/`.

### `02_feature_engineering.ipynb` — Feature Generation
Constructs the full feature matrix: lag features (t-1 to t-n), rolling statistics (7-day and 30-day windows), calendar features (day of week, month, year, week of year), and promotional flags. Output persisted to `data/processed/features/`.

### `03_data_management.ipynb` — I/O Processes
Manages data loading, filtering by store/item/date, caching of filtered subsets, and export of preprocessed datasets. Integrates with `utils.py` and `paths.py` for consistent path resolution.

### `04_lstm_modeling.ipynb` — Deep Learning Forecasting
Builds, trains, and evaluates the LSTM model. Implements walk-forward validation, sequence windowing, and autoregressive future projection. Artifacts persisted to `models/lstm/`.

### `05_xgboost_modeling.ipynb` — Gradient Boosting Forecasting
Trains and tunes the XGBoost model on the engineered feature matrix. Implements cross-validation, feature importance analysis, and performance benchmarking. Artifacts persisted to `models/xgboost/`.

### `06_arima_analysis.ipynb` — Statistical Baseline
Fits ARIMA/SARIMA models for interpretable trend and seasonality decomposition. Provides the statistical benchmark against which deep learning lift is measured. Artifacts persisted to `models/arima/`.

---

## ⚙️ Data Preprocessing & Feature Engineering

### Preprocessing Steps

1. **Type Validation** — Parse and enforce date, integer, and float types across all columns
2. **Missing Value Handling** — Forward-fill sparse sales records; flag and treat structural nulls
3. **Outlier Detection** — IQR-based outlier identification with configurable treatment strategies
4. **Store & Item Filtering** — Subset creation via `load_filtered_csv()` in `utils.py` with deterministic caching

### Feature Engineering

| Feature Type | Examples | Purpose |
|---|---|---|
| **Lag Features** | `sales_lag_1`, `sales_lag_7`, `sales_lag_14` | Capture recent demand momentum |
| **Rolling Statistics** | `rolling_mean_7`, `rolling_mean_30`, `rolling_std_7` | Smooth noise, expose trends |
| **Calendar Features** | `day_of_week`, `month`, `week_of_year`, `is_weekend` | Model seasonality and cyclicality |
| **Event Features** | `is_holiday`, `holiday_type`, `onpromotion` | Capture external demand shocks |
| **Decomposition** | Trend, seasonal, and residual components | Separate structural signal from noise |

---

## 🤖 Multi-Model Forecasting Strategy

### LSTM — Deep Learning Tier
LSTM networks learn temporal dependencies through gated memory cells, making them well-suited for retail demand sequences with complex, non-linear patterns. The model ingests a sliding window of `n` historical time steps and outputs a single-step or multi-step forecast, with walk-forward validation ensuring no data leakage.

### XGBoost — Gradient Boosting Tier
XGBoost treats the forecasting problem as a supervised regression task over the engineered feature matrix. Its tree-based structure handles non-linearity, missing values, and feature interactions natively, making it highly competitive on tabular time series with rich lag and rolling features.

### ARIMA — Statistical Baseline Tier
ARIMA provides an interpretable, computationally lightweight baseline. It decomposes the time series into trend, seasonality, and residual components using autoregressive and moving average terms. Its primary value in this pipeline is as an **objective benchmark** — quantifying where and by how much the ML models add lift.

### Ensemble Strategy

| Model | Strength | Weakness |
|---|---|---|
| **LSTM** | Non-linear patterns, long-range dependencies | Computationally expensive, data-hungry |
| **XGBoost** | Fast, strong on engineered features | Limited temporal awareness without explicit lags |
| **ARIMA** | Interpretable, robust on stable series | Assumes linearity, struggles with structural breaks |

---

## 📊 Dashboard & Visualization Layer

### Core Features

- **Echtzeit Forecasting** — Historical and future sales projections with configurable horizon
- **Performance Monitoring** — MAE, RMSE, R², and MAPE metrics displayed as top-level KPI cards
- **Residual Analysis** — Detailed error diagnostics and temporal bias detection
- **Confidence Intervals** — Statistical uncertainty quantification around point forecasts
- **Model Comparison** — Side-by-side metric benchmarking across LSTM, XGBoost, and ARIMA

### Visualization Types

- **Interactive Line Charts** — Plotly-based actual vs. predicted overlays
- **Future Projection Charts** — Configurable N-day forecast with shaded confidence bands
- **Residual Plots** — Time-series error visualization and histogram distributions
- **Year-Month Heatmaps** — Seasonal pattern detection across the full date range
- **Holiday Impact Charts** — Average sales by event type
- **Perishable vs. Non-Perishable Analysis** — Category-level demand decomposition

### Configuration Options

- **Store & Item Selection** — Sidebar inputs for flexible store/item targeting
- **Forecast Horizon** — Configurable 7–90 day projection window
- **Confidence Level** — Selectable 80%, 90%, 95% uncertainty bands
- **Model Weights** — Adjustable ensemble weighting (LSTM, XGBoost, ARIMA)
- **Export Functions** — CSV forecast export and report generation

---

## 📈 Results Analysis & Performance Metrics

The system computes and persists the following metrics for each model:

| Metric | Description | Use Case |
|---|---|---|
| **MAE** | Mean Absolute Error | Average magnitude of forecast error in sales units |
| **RMSE** | Root Mean Square Error | Penalizes large errors; sensitive to demand spikes |
| **R²** | Coefficient of Determination | Proportion of demand variance explained by the model |
| **MAPE** | Mean Absolute Percentage Error | Scale-independent accuracy for cross-SKU comparison |

Metrics are persisted as CSV artifacts alongside each trained model and loaded directly by the dashboard for live display without recomputation.

---

## 💡 Business Insights & Strategic Findings

### Demand Patterns
- Clear weekly seasonality with elevated weekend sales across most store-item pairs
- Promotions create measurable short-term demand spikes that ARIMA systematically underestimates
- Perishable items show tighter demand cycles with sharper variance than non-perishables

### Model Performance
- **LSTM** delivers the strongest R² on store-item pairs with complex, non-linear demand histories
- **XGBoost** is highly competitive on well-engineered feature sets and trains significantly faster
- **ARIMA** performs best on stable, low-variance series with clear seasonal structure
- The R² gap between ARIMA and LSTM is largest for high-promotion, high-volatility SKUs

### Unexplained Variance
Residual analysis consistently indicates that remaining forecast error clusters around **holiday periods and promotional events** — external calendar shocks not fully captured by temporal features alone. This finding directly motivates the multivariate extension with explicit event features.

---

## 🔑 Key Recommendations

1. **Deploy LSTM for High-Volatility SKUs** — The deep learning model's R² lift is most pronounced for store-item pairs with complex, event-driven demand. Route these to LSTM in production.

2. **Use XGBoost as the Default Model** — For the majority of SKUs with well-behaved demand, XGBoost offers near-LSTM accuracy at a fraction of the training and inference cost.

3. **Retain ARIMA as a Lightweight Fallback** — For stable, low-volume SKUs where the cost of deep learning isn't justified, ARIMA remains a robust, interpretable alternative.

4. **Add Explicit Holiday & Promotion Features** — The most actionable next step for all three models is incorporating the `holidays_events.csv` calendar as an explicit input channel.

5. **Expand to Full SKU Coverage** — The `load_filtered_csv()` caching layer in `utils.py` is already designed for this. Wrapping the prediction pipeline in a batch loop enables forecasting across the full product catalog.

6. **Automate Retraining** — Surface a retraining trigger in the dashboard sidebar that refreshes model artifacts on new data, creating a lightweight MLOps loop without requiring separate infrastructure.

---

## 🚀 Future Improvements & Scalability

| Enhancement | Technical Approach | Business Benefit |
|---|---|---|
| **Multivariate LSTM** | Add promotion, holiday, and weather as input channels | Capture event-driven demand spikes currently missed |
| **Prediction intervals** | Monte Carlo Dropout or quantile regression | Risk-aware safety stock calculations |
| **Transformer architecture** | Replace LSTM with attention-based seq2seq | Better long-range dependency modeling (60–90 day horizons) |
| **Auto-scaling to all SKUs** | Batch inference loop over all store-item pairs | Operational forecasting for the full product catalog |
| **A/B model comparison view** | Side-by-side metric cards per model | Real-time multi-model leaderboard for analysts |
| **Streamlit Cloud deployment** | `requirements.txt` + Streamlit Community Cloud | Zero-infrastructure access for non-technical stakeholders |
| **Forecast accuracy tracking** | Log predictions vs. actuals over time | Continuous model performance monitoring in production |
| **MLflow experiment tracking** | Already in `requirements.txt` | Reproducible model versioning across training runs |

---

## 💼 Real-World Business Impact

### Inventory & Supply Chain
Accurate demand forecasting directly reduces two compounding costs: **overstock write-offs** (particularly acute for perishable SKUs) and **stockout-driven revenue loss**. Even a 5% improvement in MAPE across the top-500 SKUs in a store generating $15M annual revenue could recover **$300,000–$750,000 in inventory efficiency** annually.

### Operational Self-Service
The Streamlit dashboard removes the data science team from the critical path of routine forecast generation. Supply chain planners can independently switch store/item targets, adjust forecast horizons, and download forecast tables for direct use in procurement workflows — reducing forecasting lead time from days to seconds.

### Multi-Model Decision Support
By surfacing MAE, RMSE, and R² across all three models, the dashboard creates a transparent, objective model selection framework. Store-item pairs where LSTM delivers measurable lift can be prioritized for deep learning deployment; pairs where ARIMA or XGBoost performs comparably can be served with lower-cost models — enabling **cost-aware, performance-driven model routing at scale**.

---

## 📁 Project File Structure

```
time_series_projekt/
│
├── 📂 app/                              # Streamlit Dashboard & UI Layer
│   ├── 📄 app.py                        # Main application (Dark Mode)
│   ├── 📄 app_backup.py                 # Backup of original application
│   ├── 📄 bootstrap.py                  # sys.path resolver for modular imports
│   ├── 📄 __init__.py                   # Package marker
│   └── 📄 Readme.md                     # Dashboard-specific documentation
│
├── 📂 notebooks/                        # Research & Development Pipeline
│   ├── 📂 preprocessing/
│   │   ├── 📄 time_serie_preprocessing.ipynb   # Data cleaning & transformation
│   │   └── 📄 README.md
│   ├── 📂 feature_engineering/
│   │   ├── 📄 time_serie_feature-engineering.ipynb  # Feature generation
│   │   └── 📄 README.md
│   ├── 📂 load_and_save_data/
│   │   └── 📄 README.md                 # I/O processes documentation
│   ├── 📂 lstm/
│   │   ├── 📄 time_serie_lstm.ipynb     # Deep learning LSTM model
│   │   └── 📄 README.md
│   ├── 📂 xgboost/
│   │   ├── 📄 time_serie_xgboost.ipynb  # Gradient boosting model
│   │   └── 📄 README.md
│   └── 📂 arima/
│       ├── 📄 arima.ipynb               # Statistical baseline model
│       └── 📄 README.md
│
├── 📂 data/                             # Datasets
│   ├── 📂 raw/                          # Original unmodified data files
│   └── 📂 processed/
│       ├── 📂 cleaner/                  # Cleaned, validated datasets
│       ├── 📂 features/                 # Feature-engineered training data
│       └── 📂 filtered/                 # Cached filtered subsets (store/item)
│
├── 📂 models/                           # Trained Model Artifacts
│   ├── 📂 lstm/
│   │   ├── 📄 lstm_model.keras          # Trained LSTM model weights
│   │   └── 📄 scaler.pkl               # Fitted input scaler
│   ├── 📂 xgboost/
│   │   └── 📄 xgboost_model.pkl        # Trained XGBoost model
│   └── 📂 arima/
│       └── 📄 arima_model.pkl          # Fitted ARIMA model
│
├── 📂 reports/                          # Results & Exports
│   ├── 📂 figures/
│   │   ├── 📂 lstm/                     # LSTM visualization outputs
│   │   ├── 📂 xgboost/                  # XGBoost visualization outputs
│   │   └── 📂 arima/                    # ARIMA visualization outputs
│   └── 📂 results/
│       ├── 📂 lstm/                     # LSTM forecast results
│       ├── 📂 xgboost/                  # XGBoost forecast results
│       └── 📂 arima/                    # ARIMA forecast results
│
├── 📄 paths.py                          # Centralized path management
├── 📄 utils.py                          # Core helper functions (I/O, filtering, caching)
├── 📄 visualizer.py                     # Plotting engine (dark theme + chart functions)
├── 📄 lstm_metrics.csv                  # Persisted LSTM performance metrics
├── 📄 requirements.txt                  # Core dependencies
├── 📄 requirements_app.txt             # Streamlit app dependencies
├── 📄 environment.yml                   # Conda environment specification
└── 📄 README.md                         # This documentation
```

---

## ⚡ Installation & Usage

### Prerequisites

- Python 3.10 or higher
- pip or conda

### Setup

**1. Clone the repository**
```bash
git clone https://github.com/TagboClaudia/time_series_projekt.git
cd time_series_projekt
```

**2. Create and activate a virtual environment**
```bash
# Using conda
conda env create -f environment.yml
conda activate tf_env_310

# Or using venv
python -m venv tf_env_310
source tf_env_310/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
pip install -r requirements_app.txt
```

**4. Launch the Streamlit dashboard**
```bash
streamlit run app/app.py
```

### Notebook Pipeline

Run the notebooks in order for a full end-to-end pipeline execution:

```bash
jupyter lab notebooks/preprocessing/time_serie_preprocessing.ipynb
jupyter lab notebooks/feature_engineering/time_serie_feature-engineering.ipynb
jupyter lab notebooks/lstm/time_serie_lstm.ipynb
jupyter lab notebooks/xgboost/time_serie_xgboost.ipynb
jupyter lab notebooks/arima/arima.ipynb
```

---

## 🏆 Conclusion

This project delivers a **production-oriented, multi-model forecasting pipeline** that bridges the gap between statistical modeling, gradient boosting, and deep learning — surfaced through a single, operational Streamlit dashboard.

The key contribution is not any single model, but the **complete forecasting system**: a reproducible six-stage notebook pipeline, a centralized data management layer, a modular codebase with shared utilities, and an interactive dashboard that puts analyst-grade forecasting tools in the hands of non-technical planners.

The multi-model architecture enables **evidence-based model routing**: LSTM for complex, volatile SKUs; XGBoost for the majority of well-behaved demand series; ARIMA as a robust, interpretable fallback. Residual analysis across all three models points to the same clear next step — multivariate modeling with explicit holiday and promotion features — defining a concrete roadmap for further accuracy gains.

**This pipeline is not the final word in forecasting — it is the operational foundation on which progressively richer models can be built, validated, and deployed.**

---

## 👥 Authors

<div align="center">

**Claudia Tagbo-Fotso**

*Data Scientist | Time Series & ML Practitioner*

[![GitHub](https://img.shields.io/badge/GitHub-TagboClaudia-181717?logo=github&logoColor=white)](https://github.com/TagboClaudia)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Claudia%20Fotso-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/claudia-fotso)
[![Email](https://img.shields.io/badge/Email-fotsoclaudia88%40gmail.com-D14836?logo=gmail&logoColor=white)](mailto:fotsoclaudia88@gmail.com)

---

**Sadiq Qais**

*Data Scientist | ML Engineer*

[![Email](https://img.shields.io/badge/Email-qais.sadiq422%40gmail.com-D14836?logo=gmail&logoColor=white)](mailto:qais.sadiq422@gmail.com)

---

*Built with production-grade engineering, multi-model rigor, and operational clarity.*

</div>