# app/services/data_processing.py
import pandas as pd
import os
import logging
from datetime import datetime

def load_data(file_path):
    """Load and preprocess strategy data from a CSV file."""
    if not os.path.exists(file_path):
        logging.error(f"File {file_path} does not exist.")
        raise FileNotFoundError(f"File {file_path} not found.")
    
    df = pd.read_csv(file_path, parse_dates=['Date'], dayfirst=True)
    
    # Handle missing dates by forward filling
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
    df = df.dropna(subset=['Date'])
    df.set_index('Date', inplace=True)
    df = df.sort_index()
    df = df.ffill()
    
    logging.info(f"Data loaded successfully from {file_path}")
    return df

def required_df(df, start_date, end_date, systems):
    """Filter DataFrame based on date range and selected systems."""
    df_filtered = df[systems]
    if start_date and end_date:
        df_filtered = df_filtered.loc[start_date:end_date]
    logging.info(f"Data filtered from {start_date} to {end_date} for systems: {systems}")
    return df_filtered
