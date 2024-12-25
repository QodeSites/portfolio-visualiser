# app/utils/helpers.py
import pandas as pd
import numpy as np
from datetime import datetime

def convert_nan_to_none(data):
    """Recursively convert NaN and NaT to None in dictionaries or lists."""
    if isinstance(data, dict):
        return {k: convert_nan_to_none(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_nan_to_none(item) for item in data]
    elif isinstance(data, pd.Series):
        return data.apply(convert_nan_to_none).tolist()
    elif isinstance(data, pd.Timestamp):
        return data.strftime('%d-%m-%Y') if pd.notnull(data) else None
    elif isinstance(data, float) and np.isnan(data):
        return None
    else:
        return data
