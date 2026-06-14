# src/return_calculator.py

import pandas as pd
import numpy as np
import os

def extract_close_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts long-format ETF price data to wide format using Close prices.
    
    Input: DataFrame with ['Date', 'Ticker', 'Close'] columns
    Output: Wide-format DataFrame with Date index and Ticker columns
    """
    df = df[['Date', 'Ticker', 'Close']].copy()
    df['Date'] = pd.to_datetime(df['Date'])
    wide_df = df.pivot(index='Date', columns='Ticker', values='Close').sort_index()
    return wide_df

def compute_log_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns for each ETF in wide-format price DataFrame.
    Output columns will be named '<TICKER>_ret'.
    """
    log_returns = np.log(price_df / price_df.shift(1)).dropna()
    log_returns.columns = [f"{col}_ret" for col in log_returns.columns]
    return log_returns

def generate_forward_returns(return_df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """
    Generate forward-shifted returns for prediction targets.
    Input: return_df with columns '<TICKER>_ret'
    Output: DataFrame with '<TICKER>_target' columns
    """
    forward_returns = return_df.shift(-horizon)
    forward_returns.columns = [col.replace("_ret", "") + "_target" for col in forward_returns.columns]
    return forward_returns.dropna()

def save_returns(return_df: pd.DataFrame, path: str):
    """
    Save a return DataFrame (either raw or shifted) to a CSV file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return_df.to_csv(path, index=True)
