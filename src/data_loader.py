# src/data_loader.py

import pandas as pd
import numpy as np
import yfinance as yf

def load_etf_prices(path: str) -> pd.DataFrame:
    """
    Load daily close prices for all ETFs.
    """
    df = pd.read_csv(path, parse_dates=['Date'])
    df = df[['Date', 'Ticker', 'Close']]
    return df.pivot(index='Date', columns='Ticker', values='Close').sort_index()

def load_fama_french_factors(path: str) -> pd.DataFrame:
    """
    Load Fama-French and QMJ factors.
    """
    df = pd.read_csv(path, parse_dates=['Date'])
    df = df.dropna()
    df = df.set_index('Date').sort_index()
    storedf = []
    for c in df.columns:
        temp = df[c].dropna().diff().fillna(0)
        storedf.append(temp)
    df = pd.concat(storedf,axis = 1)
    return df

def load_volatility_indices(path: str) -> pd.DataFrame:
    """
    Load VIX and VVIX.
    """
    df = pd.read_csv(path, parse_dates=['Date'])
    df = df.set_index('Date').sort_index()
    df = df.diff().fillna(0)
    return df

def load_macro_indicators(path: str) -> pd.DataFrame:
    """
    Load wide-format macro indicators (leading indicators).
    Forward-fill missing values.
    """
    df = pd.read_csv(path, parse_dates=['Date'])
    df = df.set_index('Date').sort_index()
    macro_df = []
    for c in df.columns:
        temp = df[c].dropna().diff().fillna(0)
        macro_df.append(temp)
    df = pd.concat(macro_df,axis = 1)

    return df.ffill()

def merge_all_factors(etf_prices: pd.DataFrame,
                      fama_french: pd.DataFrame,
                      vix: pd.DataFrame,
                      macro: pd.DataFrame) -> pd.DataFrame:
    """
    Merge Fama-French factors, VIX data, and macroeconomic indicators.
    Keep only the common date range across all dataframes.
    Forward fill missing data, then drop rows still containing NAs.
    Print out diagnostic information along the way.
    """

    # --- Step 1: Outer join all dataframes on the date index ---
    merged = macro.join(fama_french, how='outer')
    merged = merged.join(vix, how='outer')

    # --- Step 2: Get latest date available in each dataframe and take the earliest ---
    last_dates = [
        etf_prices.index.max(),
        fama_french.index.max(),
        vix.index.max(),
        macro.index.max()
    ]
    max_end_date = min(last_dates)
    print(f"✅ Latest common end date (earliest of last dates): {max_end_date.date()}")

    # Filter merged to not go beyond this date
    merged = merged[merged.index <= max_end_date]

    # --- Step 3: Get earliest date in each dataframe and take the latest ---
    first_dates = [
        etf_prices.index.min(),
        fama_french.index.min(),
        vix.index.min(),
        macro.index.min()
    ]
    min_start_date = max(first_dates)
    print(f"✅ Earliest common start date (latest of first dates): {min_start_date.date()}")

    # Filter merged to not go before this date
    merged = merged[merged.index >= min_start_date]

    # --- Step 3.5: Fill then drop rows with NAs, tracking dropped dates ---
    merged = merged.sort_index()  # Ensure sorted before ffill
    merged = merged.ffill()

    # Identify rows that still contain NAs
    rows_with_na = merged[merged.isna().any(axis=1)]
    if not rows_with_na.empty:
        print(f"⚠️ Dropping {len(rows_with_na)} row(s) with remaining NAs after forward fill:")
        print(rows_with_na.index.date.tolist())

    # Drop rows with any remaining NAs
    merged = merged.dropna(how='any')

    # --- Step 4: Final diagnostics ---
    print(f"\n📆 Final merged date range: {merged.index.min().date()} to {merged.index.max().date()}")
    print(f"📐 Shape: {merged.shape}")
    print(f"🧩 Columns: {list(merged.columns)}")
    print(f"❓ Any NAs: {merged.isna().any().any()}")

    return merged

def get_etf_data(tickers, start_date, end_date):
    """
    Fetch price and dividend data for a list of ETFs
    """
    # Download price data
    price_data = yf.download(tickers, start=start_date, end=end_date)['Close']
    
    # Download dividend data
    dividend_data = {}
    for ticker in tickers:
        etf = yf.Ticker(ticker)
        div = etf.history(start=start_date, end=end_date)['Dividends']
        dividend_data[ticker] = div
    
    dividends = pd.DataFrame(dividend_data)
    dividends.index = price_data.index
    
    return price_data, dividends