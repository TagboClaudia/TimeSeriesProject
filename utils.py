# retails/utils.py
# ============================================================
# Core Libraries
# ============================================================
import os
import io
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from IPython.display import display
import joblib

from paths import get_path



# ============================================================
# Load CSV (Base Loader)
# ============================================================
def load_csv(folder: str, filename: str) -> pd.DataFrame:
    """
    Loads a CSV file from a fully resolved folder path.

    Parameters
    ----------
    folder : str
        Absolute folder path (e.g. get_path("raw"), get_path("processed")).
    filename : str
        CSV filename.

    Returns
    -------
    pd.DataFrame
    """

    # Build full path
    file_path = os.path.join(folder, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Datei nicht gefunden: {file_path}")

    print(f"📂 Geladen: {file_path}")

    try:
        df = pd.read_csv(file_path)
        print(
            f"📊 Dataset geladen: '{filename}' | "
            f"Zeilen: {len(df):,} | Spalten: {df.shape[1]}\n"
        )
        display(df.sample(n=min(10, len(df))))

    except pd.errors.EmptyDataError:
        print(f"⚠️ Warnung: '{filename}' ist leer oder konnte nicht geparst werden.")
        df = pd.DataFrame()

    return df



# ============================================================
# Save Model
# ============================================================


def save_model(model, model_name: str, model_type: str, model_dir: str) -> str:
    """
    Saves a trained model using joblib into the given model directory.

    Parameters
    ----------
    model : object
        The trained model to save.
    model_name : str
        Base name of the model (e.g. "best_arima").
    model_type : str
        Additional identifier (e.g. "p3_d1_q2").
    model_dir : Path
        Directory where the model should be saved. Must be a Path object.

    Returns
    -------
    Path
        Full path to the saved model file.
    """

    # Ensure directory exists
    #model_dir.mkdir(parents=True, exist_ok=True)

    # Build filename
    filename = f"{model_name}_{model_type}.pkl"
    filepath = os.path.join(model_dir, filename)

    # Save model
    joblib.dump(model, filepath)

    print(f"💾 Model saved: {filepath}")

    return filepath

# ============================================================
# Save CSV
# ============================================================
def save_csv(
    df: pd.DataFrame,
    folder: str,
    filename: str,

) -> None:
    """
    Saves a DataFrame as CSV.
    """

    #save_dir = os.path.join(str(base_dir), folder)
    #os.makedirs(save_dir, exist_ok=True)

    file_path = os.path.join(folder, filename)
    df.to_csv(file_path, index=False)

    print(f"💾 Gespeichert: {file_path}")


# ============================================================
# Helper: Build deterministic filtered filename
# ============================================================
def build_filtered_filename(table_name: str, filters: dict) -> str:
    """
    Build deterministic filename based on filters.
    """
    base = table_name.replace(".csv", "")
    parts = [base]

    if "MAX_DATE" in filters:
        clean_date = str(filters["MAX_DATE"]).split(" ")[0]
        parts.append(f"MAXDATE-{clean_date}")

    if "STORE_IDS" in filters:
        parts.append("STORE-" + "-".join(map(str, filters["STORE_IDS"])))

    if "ITEM_IDS" in filters:
        parts.append("ITEM-" + "-".join(map(str, filters["ITEM_IDS"])))

    return "__".join(parts) + ".csv"


# ============================================================
# Load filtered CSV with caching
# ============================================================
def load_filtered_csv(
    folder_name: str,
    table_name: str,
    filters: dict,
    filter_folder: str = "filtered",
    force_recompute: bool = False,
    **kwargs
) -> pd.DataFrame:
    """
    Loads a CSV, applies filters, and caches the filtered result.
    Uses only folder_name + filename + filter_folder.
    """

    # --- 1. Resolve filter output directory ---
    if not isinstance(filter_folder, str):
        raise TypeError("filter_folder must be a string")

    filtered_dir = get_path(filter_folder)  # already ensures directory exists

    # Build deterministic filename
    filtered_filename = build_filtered_filename(table_name, filters)
    filtered_path = os.path.join(filtered_dir, filtered_filename)

    # --- 2. Load cached file if available ---
    if os.path.exists(filtered_path) and not force_recompute:
        print(f"⚡ Loading existing filtered dataset: {filtered_filename}")
        df = pd.read_csv(filtered_path)

        if not df.empty:
            print("\n📅 Date Range:")
            print(f"   Start: {df['date'].min()}")
            print(f"   End:   {df['date'].max()}")
            print(f"   Days:  {len(df['date'].unique())}")

        return df

    print("🔍 No cached file found. Computing filtering...")

    # --- 3. Load raw CSV ---
    df = load_csv(folder=folder_name, filename=table_name)

    if not df.empty:
        print("\n📅 Date Range:")
        print(f"   Start: {df['date'].min()}")
        print(f"   End:   {df['date'].max()}")
        print(f"   Days:  {len(df['date'].unique())}")

    # --- 4. Apply filters ---
    print("🔎 Applying filters...")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if "MAX_DATE" in filters:
        df = df[df["date"] <= pd.to_datetime(filters["MAX_DATE"])]

    if "STORE_IDS" in filters:
        df = df[df["store_nbr"].isin(filters["STORE_IDS"])]

    if "ITEM_IDS" in filters:
        df = df[df["item_nbr"].isin(filters["ITEM_IDS"])]

    print(f"✅ Filtered shape: {df.shape}")

    # --- 5. Save filtered dataset ---
    df.to_csv(filtered_path, index=False)
    print(f"💾 Saved filtered dataset to: {filtered_path}")

    return df



# ============================================================
# Load filtered CSV by Max Date only
# ============================================================
def load_data_filtered_by_date(
    folder_name: str,
    table_name: str,
    max_date: str,
    filter_folder: str = "filtered",
    force_recompute: bool = False,
    **kwargs
) -> pd.DataFrame:
    """
    Loads a CSV and filters it by a max date.
    """
    # 0. Ensure processed/<filter_folder> exists 
    processed_folder = get_path("processed") 
    filtered_dir = os.path.join(processed_folder, filter_folder) 
    os.makedirs(filtered_dir, exist_ok=True)
    
    clean_date = str(max_date).split(" ")[0]
    base_name = table_name.replace(".csv", "")
    filtered_filename = f"{base_name}__MAXDATE-{clean_date}.csv"

    filtered_path = os.path.join(filtered_dir, filtered_filename)

    # 1. Load cached
    if os.path.exists(filtered_path) and not force_recompute:
        print(f"⚡ Loading existing date-filtered dataset: {filtered_filename}")
        df = pd.read_csv(filtered_path)

        if not df.empty:
            print("\n📅 Date Range:")
            print(f"   Start: {df['date'].min()}")
            print(f"   End:   {df['date'].max()}")
            print(f"   Days:  {len(df['date'].unique())}")

        return df

    # 2. Load raw
    print(f"🔍 Filtering {table_name} by Max Date: {clean_date}...")
    df = load_csv(
        folder=folder_name,
        filename=table_name,
    )

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[df["date"] <= pd.to_datetime(max_date)]
    else:
        print(f"⚠️ Warning: 'date' column not found in {table_name}. Returning unfiltered.")

    # 3. Save
    os.makedirs(os.path.dirname(filtered_path), exist_ok=True)
    df.to_csv(filtered_path, index=False)
    print(f"💾 Saved date-filtered dataset (Shape: {df.shape})")

    return df

