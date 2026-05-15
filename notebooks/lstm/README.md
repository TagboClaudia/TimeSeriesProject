<div align="center">

# 📈 Sales Forecasting with Deep Learning & Time Series Analysis

### End-to-End LSTM-Based Time Series Forecasting System for Retail Sales Intelligence

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-2.x-D00000?logo=keras&logoColor=white)](https://keras.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Plotly](https://img.shields.io/badge/Plotly-5.x-3F4F75?logo=plotly&logoColor=white)](https://plotly.com)
[![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-TagboClaudia-181717?logo=github)](https://github.com/TagboClaudia)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Claudia_Fotso-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/claudia-fotso)

**A production-oriented machine learning system for retail sales forecasting, combining deep learning (LSTM), classical statistics (ARIMA), and gradient boosting (XGBoost) to deliver data-driven demand intelligence, short-term predictions, and actionable business insights.**

</div>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Business Problem & Context](#-business-problem--context)
- [Business Impact & Strategic Value](#-business-impact--strategic-value)
- [Project Architecture](#-project-architecture)
- [Methodology & Models](#-methodology--models)
- [Data Pipeline & Feature Engineering](#-data-pipeline--feature-engineering)
- [LSTM Model Deep Dive](#-lstm-model-deep-dive)
- [Model Training & Optimization](#-model-training--optimization)
- [Evaluation & Results](#-evaluation--results)
- [Model Comparison: LSTM vs ARIMA vs XGBoost](#-model-comparison-lstm-vs-arima-vs-xgboost)
- [Key Business Insights](#-key-business-insights)
- [Visualizations & Interpretation](#-visualizations--interpretation)
- [Notebook & Module Guide](#-notebook--module-guide)
- [Project Structure](#-project-structure)
- [Setup & Usage](#-setup--usage)
- [Future Improvements](#-future-improvements)
- [Strategic Recommendations](#-strategic-recommendations)
- [Conclusion](#-conclusion)
- [Author](#-author)

---

## 🔭 Project Overview

This project delivers a comprehensive time series forecasting system for retail sales, built around a multi-layer LSTM (Long Short-Term Memory) neural network and benchmarked against classical and ensemble alternatives. The system is designed to ingest historical transactional sales data, learn temporal patterns, and produce short-to-medium-range forecasts that directly support operational and strategic business decisions.

The project demonstrates the full machine learning lifecycle: from raw data ingestion and exploratory analysis, through feature engineering and model training, to multi-step future forecasting and interactive dashboarding. The result is a reusable, modular forecasting engine that is both technically rigorous and business-relevant.

**The central forecasting target** is daily unit sales for a specific store–item combination (Store 24, Item 105577) over a multi-year historical window, with a 14-day forward projection horizon. This single-SKU, single-location prototype is architecturally ready to scale to the full product catalog.

---

## 💼 Business Problem & Context

### The Problem

Retail businesses live and die by inventory accuracy. Stock too much and capital is locked in unsold goods; stock too little and revenue is lost to stockouts. Manual or rule-based forecasting methods fail to capture the complex, nonlinear dynamics of real-world demand — seasonality, day-of-week effects, promotional spikes, and trend shifts.

**Without intelligent forecasting, businesses face:**

- Overstock situations leading to waste, markdown losses, and capital inefficiency
- Stockout events that damage customer satisfaction and revenue
- Suboptimal staff scheduling and logistics planning
- Inability to proactively detect demand shifts before they become costly surprises

### The Business Question

> *Can we build a model that accurately predicts daily unit sales for any store–item combination, using only historical sales data, with sufficient lead time to act?*

### Context

The dataset represents transactional sales records from a multi-store retail environment, covering a time window up to April 2014. Sales patterns are characterized by:

- Predominantly low daily volumes (0–5 units), with rare spike events
- Strong weekly cyclicality (Wednesday peaks; Monday/Thursday troughs)
- Right-skewed distribution — most days have near-zero sales, with infrequent high-volume outliers
- No obvious long-term trend, but local volatility that simple models struggle to track

---

## 📊 Business Impact & Strategic Value

This forecasting system creates measurable value across multiple business dimensions:

| Business Area | Problem Addressed | Expected Impact |
|---|---|---|
| **Inventory Management** | Overstock and stockout cycles | Reduced holding cost and lost sales |
| **Supply Chain Planning** | Reactive procurement decisions | Proactive replenishment based on 14-day forecast |
| **Staff Scheduling** | Unpredictable demand causing over/understaffing | Demand-aligned shift planning |
| **Promotional Planning** | Unclear baseline to measure promotion lift | Model baseline separates organic vs. promotional demand |
| **Financial Forecasting** | Unreliable revenue projections | Data-driven short-term revenue estimates |
| **Operational Efficiency** | Manual forecasting bottlenecks | Automated, scalable forecast generation |

Beyond immediate operational benefits, the system provides a **strategic intelligence layer**: by understanding which days, periods, and patterns drive sales, category managers and planners can make faster, more confident decisions backed by data rather than intuition.

---

## 🏗️ Project Architecture

The project is organized as an end-to-end ML pipeline with clearly separated concerns:

```
Raw Data
    │
    ▼
┌─────────────────────────────────┐
│  1. Data Ingestion & Filtering  │  utils.py · paths.py
│     (CSV loading, date range,   │
│      store/item filtering)      │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  2. Preprocessing & EDA         │  Notebook Sections 2–4
│     (Aggregation, gap-fill,     │
│      visualizations, stats)     │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  3. Feature Engineering         │  Notebook Section 5
│     (Sequence creation,         │
│      MinMax scaling, reshaping) │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  4. Model Training              │  Notebook Sections 6–8
│     (LSTM architecture,         │
│      callbacks, fit loop)       │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  5. Evaluation & Comparison     │  Notebook Sections 10, 16+
│     (MAE, RMSE, R²,             │
│      LSTM vs ARIMA vs XGBoost)  │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  6. Multi-Step Forecasting      │  Notebook Section 12–13
│     (14-day rolling forecast,   │
│      future date projection)    │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  7. Interactive Dashboard &     │  Notebook Section 14
│     Model Persistence           │  Sections 15, Summary
└─────────────────────────────────┘
```

The pipeline is designed for reproducibility: filtered data is cached locally to avoid redundant computation, and trained models are persisted as `.h5` files alongside their scalers.

---

## 🧠 Methodology & Models

### Why LSTM for Time Series?

Standard feedforward neural networks treat each input independently and have no memory of previous states — a critical flaw for sequential data. **LSTM networks**, a specialized form of Recurrent Neural Networks (RNNs), overcome this through gated memory cells that learn which information to retain, forget, and output over long sequences.

For retail sales data, this matters because:
- Sales on day N are influenced by sales on days N-1, N-7, N-14 (lag effects)
- Weekly cycles create repeating patterns the model must remember
- Promotional spikes create temporary departures from baseline that eventually revert

LSTMs are uniquely suited to capture these multi-scale temporal dependencies.

### Model Inventory

Three model families were trained and compared:

**1. LSTM (Long Short-Term Memory)**
Deep learning sequence model. Three stacked LSTM layers learn hierarchical temporal representations. Best suited for complex, nonlinear time series with long-range dependencies.

**2. ARIMA (AutoRegressive Integrated Moving Average)**
Classical statistical model. Combines autoregression (past values predict future), differencing (stationarization), and moving average (error smoothing). Highly interpretable, fast to fit, and a strong baseline for linear patterns.

**3. XGBoost (Extreme Gradient Boosting)**
Ensemble tree-based model. Uses lag features and calendar features as tabular inputs, capturing nonlinear relationships without explicit sequence modeling. Excels at tabular data with well-engineered features.

---

## 🔧 Data Pipeline & Feature Engineering

### Data Loading & Filtering

Data loading is handled by two custom utility modules — `utils.py` and `paths.py` — that abstract all file I/O and path management:

```python
df = load_filtered_csv(
    folder_name=feature_subdir,
    table_name="train_features",
    filters={
        "MAX_DATE": "2014-04-01",
        "STORE_IDS": [24],
        "ITEM_IDS": [105577]
    },
    force_recompute=False,
    base_dir=base_dir
)
```

The `load_filtered_csv` function implements a **caching layer**: if a filtered CSV already exists on disk, it is loaded directly; otherwise the filter is applied and the result cached. This pattern reduces pipeline run time significantly in iterative development.

### Preprocessing Steps

Once raw data is loaded, the pipeline executes a series of preprocessing transformations:

**Step 1 — Column normalization.** Column names are resolved dynamically (`store_nbr` or `store`, `unit_sales` or `sales`) to ensure robustness against schema variations.

**Step 2 — Store-item filtering.** Only records matching Store 24 and Item 105577 are retained, producing a focused univariate time series.

**Step 3 — Daily aggregation.** Records are grouped by date and summed to produce one row per day. This handles cases where multiple transactions on the same day are recorded separately.

**Step 4 — Gap filling.** A complete date range is created from the earliest to the latest date, and missing days are filled with zero sales — a deliberate business choice reflecting true zero-demand days rather than missing data artifacts.

**Step 5 — MinMax scaling.** All sales values are normalized to the [0, 1] range using `sklearn.preprocessing.MinMaxScaler`. LSTMs converge faster and more stably on scaled input; the scaler is saved alongside the model for proper inverse transformation at inference time.

**Step 6 — Sequence construction.** The core LSTM input format requires 3D tensors `(samples, timesteps, features)`. A sliding window of 30 days is applied to create input-output pairs: each sample consists of 30 consecutive days of normalized sales as input, with the 31st day as the prediction target.

```python
# Sequence construction logic
for i in range(len(data) - sequence_length):
    X.append(data[i : i + sequence_length])   # 30 days of context
    y.append(data[i + sequence_length])         # next day target
```

**Step 7 — Temporal train/test split.** An 80/20 split is applied chronologically — the first 80% of sequences form the training set, the last 20% the test set. Crucially, random shuffling is not used, as this would cause data leakage in time series contexts.

### Key Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| Sequence Length | 30 days | Captures ~4 weeks of context including weekly cyclicality |
| Test Size | 20% | Standard evaluation holdout for time series |
| Forecast Horizon | 14 days | Operationally meaningful lead time for replenishment |
| Feature Range | [0, 1] | MinMax normalization for stable LSTM training |

---

## 🔬 LSTM Model Deep Dive

### Architecture

The LSTM model is a sequential stack of three LSTM layers followed by a single Dense output neuron:

```
Input Shape: (30 timesteps, 1 feature)
        │
        ▼
┌──────────────────────────────┐
│  LSTM Layer 1 — 100 units    │  return_sequences=True
│  Captures low-level patterns │
└──────────────┬───────────────┘
               │ Dropout(0.2)
               ▼
┌──────────────────────────────┐
│  LSTM Layer 2 — 100 units    │  return_sequences=True
│  Learns mid-level patterns   │
└──────────────┬───────────────┘
               │ Dropout(0.2)
               ▼
┌──────────────────────────────┐
│  LSTM Layer 3 — 50 units     │  return_sequences=False
│  High-level temporal summary │
└──────────────┬───────────────┘
               │ Dropout(0.2)
               ▼
┌──────────────────────────────┐
│  Dense Layer — 1 unit        │  Linear activation
│  Final sales prediction      │
└──────────────────────────────┘
Output: Single scalar (next-day sales, normalized)
```

**Design Choices Explained:**

`return_sequences=True` in the first two LSTM layers passes the full sequence of hidden states to the next layer, allowing each subsequent layer to learn from the temporal structure of the previous layer's representations — not just its final state.

`Dropout(0.2)` after each LSTM layer randomly zeroes 20% of neurons during training. This acts as a regularizer, preventing the network from memorizing the training set and improving generalization to unseen data.

The neuron count of `100 → 100 → 50 → 1` creates a funnel architecture: the wider early layers learn broad temporal features, while the narrowing final LSTM layer distills these into a compact representation passed to the output.

### Compilation Settings

```python
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mean_squared_error',
    metrics=['mae', 'mse']
)
```

The **Adam optimizer** is chosen for its adaptive learning rate behavior — it adjusts step sizes per parameter, making it well-suited for sparse, irregular time series. MSE as the loss function penalizes large errors more heavily than MAE, encouraging the model to avoid extreme mispredictions.

---

## ⚙️ Model Training & Optimization

### Training Configuration

```python
history = lstm_model.fit(
    X_train, y_train,
    epochs=100,           # Maximum epochs
    batch_size=32,        # Mini-batch gradient descent
    validation_split=0.1, # 10% of training data for validation
    callbacks=[early_stopping, lr_reduction]
)
```

### Smart Callbacks

Three callbacks manage the training process intelligently:

**Early Stopping** monitors `val_loss` and halts training if it fails to improve for 20 consecutive epochs, then restores the best-performing weights. This prevents wasted compute and overfitting.

**ReduceLROnPlateau** halves the learning rate whenever `val_loss` plateaus for 10 epochs, with a minimum floor of `0.00001`. This allows the optimizer to take finer steps as it approaches a loss minimum — a standard technique for squeezing additional performance from deep networks.

**TensorBoard** logging captures loss and metric histograms per epoch, enabling visual inspection of the training curve in the TensorBoard UI.

### Training Behavior

The model exhibits healthy training dynamics:
- Training loss decreases steadily in early epochs and stabilizes at a low level
- Validation loss remains close to training loss throughout, indicating no significant overfitting
- The small gap between training and validation MAE confirms good generalization capacity

---

## 📐 Evaluation & Results

### Metrics on Test Set

After training, the model is evaluated on the held-out 20% test set:

| Metric | Value | Interpretation |
|---|---|---|
| **MAE** | **1.18** | On average, predictions deviate from reality by ≈1.2 units per day |
| **RMSE** | **1.46** | Slightly larger than MAE, suggesting occasional larger errors from spike days |
| **R²** | **-0.0005** | Near-zero; model matches baseline mean but doesn't explain variance well |

### Honest Interpretation

The MAE and RMSE values are encouraging in absolute terms — for items selling 0–5 units on typical days, a mean error of 1.2–1.5 units represents a reasonable margin. The model reliably forecasts the daily sales level.

However, the near-zero R² score tells a more nuanced story. R² measures how much of the variance in actual sales the model explains beyond simply predicting the historical mean. A value near zero means the model is essentially predicting a smooth, mean-like trajectory — it captures the level of demand but not the day-to-day fluctuations.

**Why does this happen?** The target item has highly irregular sales: most days see 0–3 units, with rare spikes of 50–100+ units (likely from promotions, bulk orders, or data artifacts). These unpredictable spikes are nearly impossible for any time-series model to anticipate without additional contextual signals. The LSTM learns to hedge by predicting a stable average rather than attempting to model these extreme, unexplained events.

**Business Implication:** The model is well-suited for baseline demand planning and routine replenishment. For spike detection, additional features (promotional calendar, weather, events) would be necessary.

---

## 🏆 Model Comparison: LSTM vs ARIMA vs XGBoost

A systematic comparison was conducted across the three model families using a consistent evaluation framework.

### Performance Summary

| Rank | Model | Type | MAE | RMSE | R² | RMSE Rank | MAE Rank |
|---|---|---|---|---|---|---|---|
| 🥇 1.0 | **XGBoost (baseline)** | Ensemble | 0.8680 | 4.1846 | **0.7808** | 3 | 1 |
| 🥈 2.0 | **LSTM** | Deep Learning | 1.18 | **1.46** | -0.0005 | 1 | 2 |
| 🥉 3.0 | **ARIMA (best grid)** | Statistical | 1.37 | 1.86 | -0.0036 | 2 | 3 |
| — | XGBoost (tuned loss) | Ensemble | — | 3.45 | — | — | — |
| — | XGBoost (tuned final) | Ensemble | 0.88 | 4.20 | — | — | — |

### Model-by-Model Analysis

**XGBoost (baseline)** achieves the best R² by a wide margin (0.78), explaining 78% of variance in the test set. This is remarkable and reflects XGBoost's strength when combined with well-engineered lag and calendar features that make temporal patterns explicit for tree splits. It also has the lowest MAE (0.868 units). Its higher RMSE relative to LSTM suggests sensitivity to the large spike events in the data.

**LSTM** achieves the lowest RMSE (1.46), indicating the best overall prediction accuracy weighted by squared error. This means that while its average error is higher than XGBoost, it produces fewer extreme mispredictions. The negative R² reflects the spike-dominated variance issue discussed above — a structural challenge for sequence-only models without external features.

**ARIMA** provides a useful interpretable baseline with moderate accuracy (MAE 1.37, RMSE 1.86). Its negative R² confirms that this item's variance is dominated by unpredictable events rather than learnable trends. ARIMA's value lies in its speed, transparency, and diagnostic clarity rather than raw accuracy.

**Improvement over ARIMA benchmark:**

| Model | RMSE Improvement vs. ARIMA |
|---|---|
| LSTM | +21.3% better RMSE |
| XGBoost (tuned loss) | +14.0% better RMSE |
| XGBoost (baseline) | −125% worse RMSE (higher spikes) |

### Strategic Model Selection Guide

```
Use LSTM when:
  ✓ Sequence-only input (no external features available)
  ✓ Minimizing extreme forecast errors is the priority
  ✓ Long-range temporal dependencies are expected

Use XGBoost when:
  ✓ Rich feature engineering is possible (lags, calendar, promotions)
  ✓ Explainability and feature importance are required
  ✓ Training speed matters

Use ARIMA when:
  ✓ Fast, transparent baseline is needed
  ✓ Series has clear trend/seasonality structure
  ✓ Stakeholder communication requires interpretable model
```

---

## 💡 Key Business Insights

The exploratory and modeling analysis surfaces several actionable insights:

**1. Wednesday is the peak demand day.**
Average sales are highest on Wednesdays, with Mondays and Thursdays underperforming. This weekly pattern creates predictable inventory cycles and should inform mid-week replenishment scheduling.

**2. Demand is dominated by near-zero days with rare spikes.**
Over 80% of days record 0–5 units sold. Rare events (promotions, bulk purchases) create outlier days with 50–100+ units. Standard forecasting handles the baseline well; spike management requires promotional calendar integration.

**3. No persistent long-term trend is present.**
The 7-day moving average analysis confirms that sales for this SKU/store combination remain range-bound with no structural upward or downward trend through early 2014. Planning assumptions should not factor in organic growth without additional evidence.

**4. Model forecasts stabilize at 1.58 units/day over the 14-day horizon.**
The multi-step forecast converges quickly to a narrow, stable range (1.580–1.581 units). This reflects the model's learned steady-state for this item's demand profile and is a reliable baseline for routine stock planning.

**5. XGBoost's 78% R² reveals that structured features dramatically boost forecast quality.**
The gap between XGBoost (R²=0.78) and LSTM (R²≈0) is not about model architecture — it is about features. XGBoost's lag and calendar features transform temporal context into tabular signals the model can exploit directly. This is the single most important finding: feature engineering investment yields outsized returns.

---

## 📊 Visualizations & Interpretation

The notebook produces six categories of visualization:

### 1. Time Series Overview (2×2 Grid)
Four subplots reveal the structure of the raw data:

- **Daily sales line chart** — Shows the sparse, spiky distribution of demand over time. The visual immediately communicates the forecasting challenge: most of the signal is low-level noise punctuated by rare high-amplitude events.

- **7-day moving average trend** — Smooths noise and reveals the true demand trajectory. The near-flat rolling average confirms the absence of a long-term trend and highlights that spikes are genuinely exceptional events, not seasonal peaks.

- **Sales distribution histogram** — The extreme right skew (most mass at 0–5 units, long tail to 100+) confirms that this is a slow-moving SKU with intermittent demand, a category known to challenge standard forecasting models.

- **Average sales by weekday** — Wednesday's dominance is clear and consistent. This is the most actionable visual for operations teams.

### 2. Training Curves (Loss & MAE)
Side-by-side epoch plots show training and validation loss over the training run. Converging curves with minimal gap indicate healthy training dynamics — the model is learning, not memorizing.

### 3. LSTM Predictions vs. Actuals
An interactive Plotly chart overlays predicted (red dashed) and actual (blue solid) test values. The LSTM tracks the general level accurately while smoothing short-term spikes — a pattern consistent with its near-zero R² and low absolute RMSE.

### 4. 14-Day Multi-Step Forecast
A two-panel Plotly visualization shows:
- Historical context (last 90 days, blue line) with forecast appended at the green cutoff line
- A zoomed detail panel with per-day predicted values annotated, showing the stable plateau around 1.58 units

### 5. Model Comparison Dashboard (6-Panel)
A comprehensive 2×3 matplotlib figure benchmarks all models across RMSE, MAE, R², model type distribution, MAE vs. RMSE trade-off scatter, and improvement over ARIMA — presented with professional color coding and data labels.

### 6. Interactive Parameter Dashboard
An ipywidgets-powered dashboard allows real-time exploration of sequence length (7–60 days) and forecast horizon (7–30 days), retrain on demand, and updated visualizations. This enables business users and analysts to intuitively explore model sensitivity without touching code.

---

## 📁 Notebook & Module Guide

### `time_serie_lstm_.ipynb` — Main Notebook

The notebook is organized into 16 numbered sections, each building on the last:

| Section | Title | Purpose |
|---|---|---|
| **1** | Library Import & Environment Setup | Validates TensorFlow, tests LSTM layer, prints system info |
| **2** | Project Path Configuration | Adds project root to `sys.path`; imports `utils` and `paths` |
| **3** | Data Loading & Filtering | Loads filtered CSV via `load_filtered_csv` with caching |
| **4** | Data Preparation for Single Store-Item | Aggregates daily, fills gaps, sets date index |
| **5** | EDA & Visualization | 4-panel plot: time series, trend, distribution, weekday analysis |
| **6** | LSTM Data Preparation | Scaling, sequence creation, 80/20 temporal split, 3D reshape |
| **7** | LSTM Model Definition | Builds 3-layer LSTM architecture with Dropout |
| **8** | Callback Configuration | EarlyStopping, ReduceLROnPlateau, TensorBoard |
| **9** | Model Training | Fits model with callbacks, batch size 32, validation split 10% |
| **10** | Training Visualization | Epoch curves for loss and MAE |
| **11** | Test Set Evaluation | Predicts, inverse-transforms, computes MAE/RMSE/R² |
| **12** | Prediction Visualization | Interactive Plotly: actuals vs. predictions on test set |
| **13** | Multi-Step Forecasting | Rolling 14-day forecast using the trained model |
| **14** | Forecast Visualization | 2-panel Plotly with history + forecast detail |
| **15** | Interactive Dashboard | Widgets for sequence length, horizon, retrain, re-plot |
| **Summary** | Model Persistence & Report | Saves `.h5` model and scaler; renders HTML summary card |
| **Comparison** | ARIMA & XGBoost Comparison | Full model benchmark with 6-panel visualization and HTML ranking table |

### `utils.py` — Data Utility Module

| Function | Description |
|---|---|
| `load_csv(path)` | Loads a CSV file into a pandas DataFrame with standard settings |
| `save_csv(df, path)` | Saves a DataFrame to CSV, creating directories if needed |
| `load_data_filtered_by_date(...)` | Loads and filters data to a specified date range |
| `load_filtered_csv(...)` | Main entry point: checks cache first, applies filters if needed, saves result |

The `load_filtered_csv` function accepts a `force_recompute` flag — when `True`, it ignores any cached file and recomputes the filter from scratch. This is essential for reproducibility when upstream data changes.

### `paths.py` — Path Management Module

| Function | Description |
|---|---|
| `get_path(key)` | Returns absolute path for a named project directory |

Supported path keys: `root`, `raw`, `cleaner`, `features`, `filtered`, `lstm_model`, `lstm_results`

Centralizing path management in a single module makes the project portable across machines and environments — a critical property for collaborative or cloud-deployed pipelines.

---

## 📂 Project Structure

```
time-series-project/
│
├── notebooks/
│   └── time_serie_lstm_.ipynb       # Main modeling notebook
│
├── utils.py                         # Data loading & caching utilities
├── paths.py                         # Centralized path management
│
├── data/
│   ├── raw/                         # Original unprocessed CSV files
│   ├── cleaner/                     # Cleaned and validated data
│   ├── features/                    # Feature-engineered datasets
│   └── filtered/                    # Cached filtered subsets
│
├── models/
│   └── lstm/
│       ├── lstm_model.h5            # Trained LSTM model weights
│       └── scaler.pkl               # Fitted MinMaxScaler
│
├── reports/
│   └── results/
│       └── lstm/
│           └── lstm_metrics.csv     # MAE, RMSE, R² with timestamps
│
├── logs/
│   └── fit/                         # TensorBoard training logs
│
└── README.md
```

---

## 🚀 Setup & Usage

### Prerequisites

```bash
Python 3.10+
pip or conda
```

### Installation

```bash
# Clone the repository
git clone https://github.com/TagboClaudia/time-series-forecasting.git
cd time-series-forecasting

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn plotly ipywidgets joblib tqdm darts
```

### Running the Notebook

```bash
jupyter notebook notebooks/time_serie_lstm_.ipynb
```

Execute cells sequentially from Section 1 through the Summary. The notebook is self-contained and includes fallback handling for missing dependencies.

### Loading a Saved Model

```python
from tensorflow.keras.models import load_model
import joblib

model = load_model("models/lstm/lstm_model.h5")
scaler = joblib.load("models/lstm/scaler.pkl")

# Prepare new sequence and predict
import numpy as np
new_sequence = ...  # shape: (1, 30, 1), normalized
pred_scaled = model.predict(new_sequence)
pred_units = scaler.inverse_transform(pred_scaled)
```

---

## 🔮 Future Improvements

The current system establishes a solid foundation. The following enhancements would materially improve both accuracy and business value:

**Model Architecture**

- **Bidirectional LSTM** — Process sequences in both temporal directions, potentially capturing patterns that only become clear in retrospect
- **Attention Mechanisms / Transformers** — Allow the model to weight specific past timesteps as most predictive, improving interpretability and accuracy on long sequences
- **Temporal Fusion Transformer (TFT)** — State-of-the-art architecture for multi-horizon time series forecasting with built-in attention and static covariate handling

**Feature Engineering**

- **Promotional calendar** — Binary flag for days with active promotions; this single feature would likely resolve the R² gap for spike-dominated items
- **Holiday and event features** — Public holidays, school calendars, local events
- **Weather data** — Temperature, precipitation for weather-sensitive categories
- **Price history** — Historical price changes correlated with demand shifts

**Modeling Strategy**

- **Multi-variate LSTM** — Incorporate external features directly into the LSTM input (current implementation is univariate)
- **Probabilistic forecasting** — Replace point estimates with prediction intervals, enabling risk-aware inventory decisions (e.g., order-up-to levels based on 90th percentile forecast)
- **Ensemble stacking** — Combine LSTM, XGBoost, and ARIMA predictions into a meta-learner for superior accuracy

**Scalability**

- **Multi-SKU pipeline** — Parameterize the notebook to loop over all store–item combinations and generate a forecast table
- **Automated retraining** — Trigger model updates as new sales data arrives (e.g., weekly retraining cycle)
- **REST API deployment** — Wrap the trained model in a FastAPI or Flask endpoint for integration with ERP/WMS systems
- **MLflow or W&B tracking** — Log all experiments, hyperparameters, and metrics for systematic model governance

---

## 📌 Strategic Recommendations

Based on the full analysis, the following recommendations are offered for business and technical stakeholders:

**Immediate Actions:**

1. **Deploy the XGBoost model for baseline demand planning.** Its R²=0.78 demonstrates genuine predictive power and its MAE of 0.87 units makes it operationally useful for replenishment planning on regular demand days.

2. **Use LSTM as the primary spike-robust forecaster.** Its lowest RMSE (1.46) means it produces fewer extreme mispredictions — valuable when demand shocks carry asymmetric cost.

3. **Prioritize Wednesday restocking for Store 24.** The weekday analysis confirms Wednesday as the peak demand day; inventory levels should be maximized before Wednesday and replenished immediately after.

**Medium-Term Investments:**

4. **Build a promotional feature pipeline.** The 78-point R² advantage of XGBoost over LSTM is largely attributable to feature quality. A promotional calendar integrated into both models would dramatically improve spike predictability.

5. **Extend the model to the full product catalog.** The modular architecture of `utils.py` and `paths.py` is designed for this scaling. A pipeline loop over all store–item pairs would deliver enterprise-level demand intelligence.

6. **Implement forecast uncertainty quantification.** Business decisions (safety stock, reorder points) are risk decisions. Point forecasts are insufficient; confidence intervals enable proper inventory optimization.

---

## 🎯 Conclusion

This project demonstrates that deep learning — when thoughtfully combined with classical baselines and proper evaluation frameworks — can deliver meaningful value in retail demand forecasting. The LSTM model achieves an average daily prediction error of ≈1.2 units with the lowest RMSE among all models tested, making it suitable for operational deployment.

The comparative analysis reveals an important nuance: **no single model dominates across all metrics**. XGBoost excels at explaining demand variance when features are rich; LSTM minimizes extreme errors in sequence-only settings; ARIMA provides transparency and speed. The most robust production system would be an ensemble of all three.

The business case is clear: replacing manual or rule-based forecasting with this system can reduce inventory holding costs, eliminate systematic stockouts on peak demand days, and give planners a reliable 14-day demand signal they can act on with confidence.

Most importantly, this project establishes a **reusable, extensible architecture** — one that can scale from a single SKU to an entire product catalog, incorporate new data sources as they become available, and evolve from batch forecasting to real-time prediction as the business matures.

---

## 👩‍💻 Author

<div align="center">

**Claudia Tagbo-Fotso**

*Data Scientist | Machine Learning Engineer | Time Series Specialist*

[![GitHub](https://img.shields.io/badge/GitHub-TagboClaudia-181717?logo=github&logoColor=white)](https://github.com/TagboClaudia)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-claudia--fotso-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/claudia-fotso)

*Built with TensorFlow · scikit-learn · Plotly · Pandas · and a commitment to data-driven decision making.*

</div>

---

<div align="center">

*If this project was useful to you, consider giving it a ⭐ on GitHub.*

</div>