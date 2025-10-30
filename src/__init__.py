# src/__init__.py
"""
Capstone Project Source Package

Contains utility modules for State of Health (SoH) analysis,
data preprocessing, and model pipelines.
"""

from .cs_soh_utils import compute_capacity, estimate_resistance, detect_cycles, plot_soh_trend
from .cs_data_preprocessing import preprocess_logs, clean_signals
from .cs_model_pipeline import train_model, evaluate_model

__all__ = [
    "calculate_soh1",
    "detect_cycles_from_voltage",
    "count_cycles_voltage",
    "merge_short_gaps", 
    "parse_log_date",
    "compute_log_soh", 
    "ensure_dt_index",
    "attach_soh1_series",
    "ensure_time_index_1s", 
    "ensure_time_utc"
]
