import bootstrap 
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
import plotly.express as px
from datetime import datetime, timedelta
from typing import Optional, Tuple, Any
from paths import get_path

# ---------------------------------------------------------
# BASIC CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="LSTM Forecast Dashboard", layout="wide")

TARGET_COL = "unit_sales"
TIME_STEPS = 30  # Dein Modell wurde mit 30 Zeitpunkten trainiert!

# Features used to fit the scaler/model
TRAINED_FEATURES = ["unit_sales"]

# Fixed default file path
FEATURE_FILE = (
    "/Users/claudiatagbo/data/processed/filtered/train_features__MAXDATE-2014-04-01__STORE-24__ITEM-105577.csv"
)

# Helper function to convert hex to rgb
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# ---------------------------------------------------------
# LOAD MODEL, SCALER, METRICS
# ---------------------------------------------------------
model_dir = get_path("lstm_model")
metrics_dir = get_path("lstm_results")

lstm_model = tf.keras.models.load_model(os.path.join(model_dir, "lstm_model.h5"))
scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
metrics_df = pd.read_csv(os.path.join(metrics_dir, "lstm_metrics.csv"))



# Custom color palette
COLORS = {
    "primary": "#1f77b4",    # Blue
    "secondary": "#ff7f0e",  # Orange
    "success": "#2ca02c",    # Green
    "danger": "#d62728",     # Red
    "warning": "#9467bd",     # Purple
    "info": "#8c564b",        # Brown
    "light": "#e377c2",      # Pink
    "dark": "#7f7f7f",       # Gray
    "background": "#f7f7f7"  # Light gray background
}

# Set Plotly color sequence
px.defaults.color_discrete_sequence = [COLORS["primary"], COLORS["secondary"]]

# ---------------------------------------------------------
# BASIC CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="LSTM Forecast Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown(f"""
<style>
    .reportview-container .main .block-container{{
        background-color: {COLORS['background']};
        padding-top: 2rem;
        padding-bottom: 2rem;
    }}
    .st-bf{{
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }}
    .st-df{{
        background-color: white;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }}
    .metric-card{{
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }}
    .success-box{{
        background-color: rgba({hex_to_rgb(COLORS['success'])}, 0.1);
        border-left: 5px solid {COLORS['success']};
        padding: 10px;
        border-radius: 5px;
    }}
    .error-box{{
        background-color: rgba({hex_to_rgb(COLORS['danger'])}, 0.1);
        border-left: 5px solid {COLORS['danger']};
        padding: 10px;
        border-radius: 5px;
    }}
    .info-box{{
        background-color: rgba({hex_to_rgb(COLORS['info'])}, 0.1);
        border-left: 5px solid {COLORS['info']};
        padding: 10px;
        border-radius: 5px;
    }}
</style>
""", unsafe_allow_html=True)




# Debug: Show model information
print("=" * 80)
print("DEBUG: Model Information")
print("=" * 80)
print(f"Model Input Shape: {lstm_model.input_shape}")
print(f"Model Output Shape: {lstm_model.output_shape}")
print(f"Model Layers:")
for i, layer in enumerate(lstm_model.layers):
    print(f"  Layer {i}: {layer.name} - {layer.output_shape}")

# ---------------------------------------------------------
# PREPARE STORE-ITEM DATA
# ---------------------------------------------------------
def prepare_store_item_data(
    df: pd.DataFrame,
    selected_store: int,
    selected_item: int,
    store_col: str = 'store_nbr',
    item_col: str = 'item_nbr',
    date_col: str = 'date',
    sales_col: str = 'unit_sales'
) -> Tuple[pd.DataFrame, str]:
    print("🔍 Preparing data...")

    store_col = store_col if store_col in df.columns else ('store_nbr' if 'store_nbr' in df.columns else 'store')
    item_col = item_col if item_col in df.columns else ('item_nbr' if 'item_nbr' in df.columns else 'item')
    date_col = date_col if date_col in df.columns else 'date'
    sales_col = sales_col if sales_col in df.columns else ('unit_sales' if 'unit_sales' in df.columns else 'sales')

    df_filtered = df[
        (df[store_col] == selected_store) &
        (df[item_col] == selected_item)
    ].copy()

    if df_filtered.empty:
        raise ValueError(f"No data found for store {selected_store} and item {selected_item}")

    df_filtered[date_col] = pd.to_datetime(df_filtered[date_col])
    df_filtered = df_filtered.sort_values(date_col)

    df_daily = df_filtered.groupby(date_col)[sales_col].sum().reset_index()

    date_range = pd.date_range(
        start=df_daily[date_col].min(),
        end=df_daily[date_col].max(),
        freq='D'
    )

    df_complete = pd.DataFrame({'date': date_range})
    df_complete = df_complete.merge(df_daily, on='date', how='left')
    df_complete[sales_col] = df_complete[sales_col].fillna(0)

    summary_message = (
        f"Prepared data:\n"
        f"   Time range: {df_complete['date'].min().date()} to {df_complete['date'].max().date()}\n"
        f"   Total days: {len(df_complete)}\n"
        f"   Average daily sales: {df_complete[sales_col].mean():.2f}"
    )
    print(summary_message)

    return df_complete, summary_message

# ---------------------------------------------------------
# CREATE SEQUENCES FOR LSTM
# ---------------------------------------------------------
def create_sequences(data, time_steps=TIME_STEPS):
    """
    Creates sequences for LSTM model
    data: 1D array of sales figures
    time_steps: Length of the sequence (for you: 30)
    """
    X = []
    y = []

    for i in range(time_steps, len(data)):
        X.append(data[i-time_steps:i])  # Sequence of the last 'time_steps' days
        y.append(data[i])  # Target: next day

    return np.array(X), np.array(y)

# ---------------------------------------------------------
# PREDICT WITH LSTM (correct shape)
# ---------------------------------------------------------
def predict_lstm(model, df, scaler):
    """
    Makes predictions with the LSTM model
    """
    # 1. Scale the data
    sales_data = df[TARGET_COL].values.reshape(-1, 1)
    sales_scaled = scaler.transform(sales_data)

    # 2. Create sequences
    X_sequences, y_true = create_sequences(sales_scaled.flatten())

    # 3. Reshape for LSTM: (samples, time_steps, features)
    X_sequences = X_sequences.reshape((X_sequences.shape[0], X_sequences.shape[1], 1))

    print(f"X_sequences Shape: {X_sequences.shape}")  # Should be: (n_samples, 30, 1)

    # 4. Make predictions
    y_pred_scaled = model.predict(X_sequences, verbose=0)

    # 5. Inverse transform
    y_pred = scaler.inverse_transform(y_pred_scaled).flatten()

    # 6. For the first TIME_STEPS days no prediction is possible
    #    We fill with NaN
    predictions_full = np.full(len(df), np.nan)
    predictions_full[TIME_STEPS:] = y_pred

    return predictions_full

# ---------------------------------------------------------
# FUTURE FORECASTING WITH SEQUENCES
# ---------------------------------------------------------
def forecast_future(df, scaler, model, days=30):
    """
    Future forecast with recursive prediction
    """
    # Scaled sales data
    sales_data = df[TARGET_COL].values.reshape(-1, 1)
    sales_scaled = scaler.transform(sales_data)

    # Last TIME_STEPS days as starting sequence
    last_sequence = sales_scaled[-TIME_STEPS:].flatten()

    future_predictions = []

    for _ in range(days):
        # Reshape for model: (1, time_steps, 1)
        input_seq = last_sequence.reshape((1, TIME_STEPS, 1))

        # Prediction
        pred_scaled = model.predict(input_seq, verbose=0)[0, 0]
        future_predictions.append(pred_scaled)

        # Update sequence: remove oldest value, append new prediction
        last_sequence = np.append(last_sequence[1:], pred_scaled)

    # Inverse transform
    future_predictions = scaler.inverse_transform(
        np.array(future_predictions).reshape(-1, 1)
    ).flatten()

    return future_predictions

# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------
st.title("📈 LSTM Forecast Dashboard")
st.markdown("<div class='success-box'>Forecast retail demand using your trained LSTM model.</div>", unsafe_allow_html=True)

# Info-Box about model structure
with st.expander("ℹ️ Model Information"):
    st.markdown(f"""
    <div class='info-box'>
    <p><strong>Input Shape:</strong> {lstm_model.input_shape}</p>
    <p><strong>Output Shape:</strong> {lstm_model.output_shape}</p>
    <p><strong>Time Steps (Sequence Length):</strong> {TIME_STEPS}</p>
    <p><strong>Features:</strong> {TRAINED_FEATURES}</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# KPI METRICS
# ---------------------------------------------------------
st.subheader("Model Performance KPIs")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
    <div class='metric-card'>
    <h4>MAE</h4>
    <p style='font-size: 24px; color: {COLORS['primary']};'>{metrics_df.loc[metrics_df.Metric=='MAE','Value'].values[0]:.3f}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='metric-card'>
    <h4>RMSE</h4>
    <p style='font-size: 24px; color: {COLORS['secondary']};'>{metrics_df.loc[metrics_df.Metric=='RMSE','Value'].values[0]:.3f}</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='metric-card'>
    <h4>R²</h4>
    <p style='font-size: 24px; color: {COLORS['success']};'>{metrics_df.loc[metrics_df.Metric=='R2','Value'].values[0]:.3f}</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
uploaded_file = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
selected_store = 24
selected_item = 105577

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    try:
        df, summary = prepare_store_item_data(df, selected_store, selected_item)
        st.markdown(f"<div class='success-box'>{summary}</div>", unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f"<div class='error-box'>Error preparing data: {e}</div>", unsafe_allow_html=True)
        df = None
else:
    try:
        df = pd.read_csv(FEATURE_FILE)
        df, summary = prepare_store_item_data(df, selected_store, selected_item)
        st.markdown(f"<div class='success-box'>{summary}</div>", unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f"<div class='error-box'>Failed to load data from default path: {e}</div>", unsafe_allow_html=True)
        df = None

# Check if we have enough data for sequences
if df is not None:
    df = df.sort_values("date").reset_index(drop=True)

    # Check minimum data length
    if len(df) < TIME_STEPS:
        st.markdown(f"<div class='error-box'>❌ Not enough data! Need at least {TIME_STEPS} days, but only have {len(df)}.</div>", unsafe_allow_html=True)
        df = None
    else:
        st.markdown(f"<div class='success-box'>✅ Enough data: {len(df)} days (minimum {TIME_STEPS} required)</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB: HISTORY FORECAST
# ---------------------------------------------------------
st.subheader("Historical Forecast")

if df is not None:
    st.write("### Data Preview")
    st.dataframe(df.head())

    # Check required features
    missing = set(TRAINED_FEATURES) - set(df.columns)
    if missing:
        st.markdown(f"<div class='error-box'>❌ Missing required features: {missing}</div>", unsafe_allow_html=True)
        df = None
    else:
        # Make predictions
        try:
            predictions = predict_lstm(lstm_model, df, scaler)
            df["forecast"] = predictions

            # Calculate residuals where we have predictions
            mask = ~np.isnan(df["forecast"])
            df.loc[mask, "residual"] = df.loc[mask, TARGET_COL] - df.loc[mask, "forecast"]

            st.markdown("<div class='success-box'>✅ Forecast successful!</div>", unsafe_allow_html=True)

            # Show prediction coverage
            n_predictions = mask.sum()
            st.write(f"**Predictions made:** {n_predictions} days (from day {TIME_STEPS+1} to {len(df)})")

            # Plots with custom colors
            fig = px.line(
                df,
                x="date",
                y=[TARGET_COL, "forecast"],
                labels={"value": "Unit Sales", "date": "Date"},
                title="Actual vs Forecast (Historical)",
                color_discrete_map={
                    TARGET_COL: COLORS["primary"],
                    "forecast": COLORS["secondary"]
                }
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Residual plot (only where we have predictions)
            if mask.any():
                fig_res = px.line(
                    df[mask],
                    x="date",
                    y="residual",
                    title="Residuals (Actual - Forecast)",
                    color_discrete_sequence=[COLORS["danger"]]
                )
                fig_res.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_res, use_container_width=True)

                fig_hist = px.histogram(
                    df[mask],
                    x="residual",
                    nbins=30,
                    title="Distribution of Forecast Errors",
                    color_discrete_sequence=[COLORS["warning"]]
                )
                fig_hist.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    bargap=0.1
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            # Calculate metrics on available predictions
            if mask.any():
                mae = np.abs(df.loc[mask, "residual"]).mean()
                rmse = np.sqrt((df.loc[mask, "residual"]**2).mean())

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    <div class='metric-card'>
                    <h4>Current MAE</h4>
                    <p style='font-size: 24px; color: {COLORS['primary']};'>{mae:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class='metric-card'>
                    <h4>Current RMSE</h4>
                    <p style='font-size: 24px; color: {COLORS['secondary']};'>{rmse:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)

            st.subheader("Forecast Summary")
            st.dataframe(df[["date", TARGET_COL, "forecast", "residual"]].tail(20))

        except Exception as e:
            st.markdown(f"<div class='error-box'>❌ Prediction error: {str(e)}</div>", unsafe_allow_html=True)
            with st.expander("🔍 Technical Details"):
                st.write(f"Error: {repr(e)}")
                st.write(f"Data shape: {df.shape}")
                st.write(f"Target column: {TARGET_COL}")
                st.write(f"Features: {TRAINED_FEATURES}")
else:
    st.markdown("<div class='info-box'>Data could not be loaded. Check file path and columns.</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB: FUTURE FORECAST
# ---------------------------------------------------------
st.subheader("Future Forecast")

if df is not None:
    days = st.slider("Days to forecast", 7, 90, 30)

    if st.button("Generate Future Forecast"):
        with st.spinner("Generating forecast..."):
            try:
                future_predictions = forecast_future(df, scaler, lstm_model, days=days)

                # Create future dates
                last_date = df["date"].iloc[-1]
                future_dates = [last_date + timedelta(days=i+1) for i in range(days)]

                # Create future dataframe
                future_df = pd.DataFrame({
                    "date": future_dates,
                    TARGET_COL: np.nan,
                    "forecast": future_predictions
                })

                # Combine with historical data
                combined_df = pd.concat([df, future_df], ignore_index=True)

                # Plot with custom colors
                fig_future = px.line(
                    combined_df,
                    x="date",
                    y=[TARGET_COL, "forecast"],
                    title=f"Historical + Next {days} Days Forecast",
                    color_discrete_map={
                        TARGET_COL: COLORS["primary"],
                        "forecast": COLORS["success"]
                    }
                )
                fig_future.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    hovermode="x unified"
                )
                fig_future.add_vrect(
                    x0=last_date,
                    x1=future_dates[-1],
                    fillcolor=COLORS["light"],
                    opacity=0.2,
                    line_width=0,
                    annotation_text="Forecast",
                    annotation_position="top left"
                )
                st.plotly_chart(fig_future, use_container_width=True)

                st.subheader("Future Forecast Values")
                st.dataframe(future_df.style.format({TARGET_COL: "{:.2f}", "forecast": "{:.2f}"}))

            except Exception as e:
                st.markdown(f"<div class='error-box'>❌ Future forecast error: {str(e)}</div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='info-box'>Data could not be loaded. Check file path and columns.</div>", unsafe_allow_html=True)
