# retails/paths.py

# retails/paths.py
import os

# Absolute path to this file
current_file = os.path.abspath(__file__)

# Project root = 3 levels above
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))

# External data root (mounted disk)
#external_data_root = "/Volumes/Intenso/my_work_spaces/retail_data/corporación_favorita_grocery_dataset"

# Core directories
notebook_dir = os.path.join(project_root, "notebooks")
data_dir = os.path.join(project_root, "data")
models_dir = os.path.join(project_root, "models")
reports_dir = os.path.join(project_root, "reports")
figures_dir = os.path.join(reports_dir, "figures")
results_dir = os.path.join(reports_dir, "results")

# Data subdirectories
raw_dir = os.path.join(data_dir, "raw")
processed_dir = os.path.join(data_dir, "processed")
cleaner_dir = os.path.join(processed_dir, "cleaner")
feature_dir = os.path.join(processed_dir, "features")
filtered_dir = os.path.join(processed_dir, "filtered")

# Model directories
arima_model_dir = os.path.join(models_dir, "arima")
lstm_model_dir = os.path.join(models_dir, "lstm")
xgboost_model_dir = os.path.join(models_dir, "xgboost")

# Figure directories
arima_figure_dir = os.path.join(figures_dir, "arima")
lstm_figure_dir = os.path.join(figures_dir, "lstm")
xgboost_figure_dir = os.path.join(figures_dir, "xgboost")

# Results directories
arima_results_dir = os.path.join(results_dir, "arima")
lstm_results_dir = os.path.join(results_dir, "lstm")
xgboost_results_dir = os.path.join(results_dir, "xgboost")

# Additional root
new_root_dir = os.path.join(project_root, "new_root")

# Mapping of names to paths
_path_dict = {
    "root": project_root,
    "notebooks": notebook_dir,
    "data": data_dir,
    "models": models_dir,
    "reports": reports_dir,
    "figures": figures_dir,
    "results": results_dir,
    "raw": raw_dir,
    "processed": processed_dir,
    "cleaner": cleaner_dir,
    "features": feature_dir,
    "filtered": filtered_dir,

    "arima_model": arima_model_dir,
    "lstm_model": lstm_model_dir,
    "xgboost_model": xgboost_model_dir,

    "arima_figures": arima_figure_dir,
    "lstm_figures": lstm_figure_dir,
    "xgboost_figures": xgboost_figure_dir,

    "arima_results": arima_results_dir,
    "lstm_results": lstm_results_dir,
    "xgboost_results": xgboost_results_dir,

    "new_root": new_root_dir,
}

def get_path(path_name: str, mkdir: bool = True) -> str:
    """
    Retrieve a project path by name. Optionally create the directory.
    Returns a string path.
    """
    if path_name not in _path_dict:
        valid = ", ".join(_path_dict.keys())
        raise ValueError(f"Invalid path name '{path_name}'. Valid options: {valid}")

    path = _path_dict[path_name]

    if mkdir:
        os.makedirs(path, exist_ok=True)

    return path
