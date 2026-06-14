# fama_macbeth.py

import pandas as pd
import numpy as np
from typing import List, Tuple
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

def prepare_fama_macbeth_data(
    equity_X: pd.DataFrame,
    targets: pd.DataFrame,
    equity_etfs: List[str],
    drop_non_relevant_momentum: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Prepares data for Fama-MacBeth regression by aligning feature and target dates,
    filtering ETF targets, and constructing a panel dataset.

    Parameters:
    - equity_X: pd.DataFrame
        Feature matrix with datetime index.
    - targets: pd.DataFrame
        DataFrame with target returns per ETF (columns = {ETF}_target).
    - equity_etfs: List[str]
        ETFs considered for equity modeling.
    - drop_non_relevant_momentum: bool
        If True, drops momentum features from irrelevant ETFs.

    Returns:
    - X_panel: pd.DataFrame
        Repeated feature matrix (panel format).
    - y_panel: pd.Series
        Concatenated target returns (same index as X_panel).
    - etf_labels: pd.Series
        ETF identity per row (matches X_panel and y_panel index).
    """

    """
    Prepares data for Fama-MacBeth regression by aligning feature and target dates,
    filtering ETF targets, and constructing a panel dataset.
    """

    print("=== START: prepare_fama_macbeth_data ===")
    print(f"Initial equity_X shape: {equity_X.shape}")
    print(f"Initial targets shape: {targets.shape}")
    print(f"Equity ETFs used: {equity_etfs}\n")

    # --- Step 1: Drop irrelevant ETF momentum features ---
    if drop_non_relevant_momentum:
        keep_features = []
        for col in equity_X.columns:
            if "_ret_lag" in col:
                etf_prefix = col.split("_ret_lag")[0]
                if etf_prefix in equity_etfs:
                    keep_features.append(col)
            else:
                keep_features.append(col)
        equity_X = equity_X[keep_features]
        print(f"Step 1 ✅ Filtered equity_X shape: {equity_X.shape}")
        assert equity_X.shape[1] > 0, "No features retained after momentum filtering.\n"

    # --- Step 2: Align dates ---
    common_dates = equity_X.index.intersection(targets.index)
    print(f"Step 2 ✅ Common dates: {len(common_dates)}")
    assert len(common_dates) > 0, "No common dates between features and targets.\n"

    equity_X = equity_X.loc[common_dates]
    targets = targets.loc[common_dates]
    print(f"Aligned equity_X shape: {equity_X.shape}")
    print(f"Aligned targets shape: {targets.shape}\n")

    # --- Step 3: Filter and reshape targets ---
    relevant_cols = [f"{etf}_target" for etf in equity_etfs if f"{etf}_target" in targets.columns]
    print(f"Step 3 ✅ Relevant target columns: {relevant_cols}")
    assert len(relevant_cols) > 0, "No matching ETF target columns found.\n"

    targets_filtered = targets[relevant_cols].copy()
    targets_filtered.columns = [col.replace("_target", "") for col in targets_filtered.columns]

    # Melt into long format and convert to MultiIndex
    targets_long = (
        targets_filtered
        .melt(ignore_index=False, var_name="ETF", value_name="target")
        .dropna()
        .set_index('ETF', append=True)
        .reorder_levels(['Date', 'ETF'])
    )
    print(f"targets_long shape: {targets_long.shape}")
    print(f"targets_long index type: {type(targets_long.index)}\n")
    assert not targets_long.empty, "targets_long is empty after melt/dropna.\n"

    # --- Step 4: Repeat feature rows for each ETF ---
    X_repeated = pd.DataFrame(
        np.tile(equity_X.values, (len(equity_etfs), 1)),
        index=pd.MultiIndex.from_product([equity_X.index, equity_etfs], names=["Date", "ETF"]),
        columns=equity_X.columns
    )
    print(f"Step 4 ✅ X_repeated shape: {X_repeated.shape}")
    print(f"X_repeated index type: {type(X_repeated.index)}\n")
    assert not X_repeated.empty, "X_repeated is empty.\n"

    # --- Step 5: Align panels ---
    common_index = X_repeated.index.intersection(targets_long.index)
    print(f"Step 5 ✅ Common index length: {len(common_index)}")
    assert len(common_index) > 0, "No matching MultiIndex between features and targets.\n"

    X_panel = X_repeated.loc[common_index]
    y_panel = targets_long.loc[common_index, "target"]
    etf_labels = pd.Series([ix[1] for ix in common_index], index=common_index)

    print("✅ Final shapes:")
    print(f"X_panel: {X_panel.shape}")
    print(f"y_panel: {y_panel.shape}")
    print(f"etf_labels: {etf_labels.shape}")
    print("=== END: prepare_fama_macbeth_data ===\n")

    return X_panel, y_panel, etf_labels

def run_fama_macbeth(X, y, dates, etfs):
    """
    Run Fama-MacBeth two-pass regression.
    
    Parameters:
        X (DataFrame): Feature matrix (MultiIndex: Date, ETF)
        y (Series): Target returns (MultiIndex: Date, ETF)
        dates (Index): Unique sorted dates
        etfs (List[str]): List of ETF tickers

    Returns:
        lambda_df (DataFrame): Time series of factor premia (lambda_t)
        avg_lambda (Series): Average factor premia over time
        t_stats (Series): t-stats of average premia
        y_hat (Series): Fitted values (predicted returns)
    """
    # --- First-pass: Cross-sectional regression each period ---
    lambda_list = []
    y_hat = pd.Series(index=y.index, dtype=float)

    for date in dates:
        try:
            X_t = X.loc[date]
            y_t = y.loc[date]

            X_t = sm.add_constant(X_t)  # Add intercept
            model = sm.OLS(y_t, X_t).fit()
            lambda_t = model.params
            lambda_list.append(lambda_t)

            # Save fitted values
            y_hat.loc[date] = model.predict(X_t)
        except Exception as e:
            print(f"Skipping date {date} due to error: {e}")
            continue

    lambda_df = pd.DataFrame(lambda_list, index=dates)

    # --- Second-pass: Time-series average of betas ---
    avg_lambda = lambda_df.mean()
    t_stats = lambda_df.mean() / (lambda_df.std() / (len(lambda_df)**0.5))

    return lambda_df, avg_lambda, t_stats, y_hat

def plot_predicted_vs_actual(y_true, y_pred, title="Predicted vs Actual Returns"):
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Actual Return")
    plt.ylabel("Predicted Return")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_cumulative_returns(predicted_returns_df, title="Cumulative Return of Strategy"):
    cum_returns = (1 + predicted_returns_df).cumprod()
    plt.figure(figsize=(10, 4))
    for col in cum_returns.columns:
        plt.plot(cum_returns[col], label=col)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_factor_premia_over_time(lambda_df):
    plt.figure(figsize=(12, 6))
    for col in lambda_df.columns:
        plt.plot(lambda_df.index, lambda_df[col], label=col)
    plt.title("Time-Varying Factor Premia (λₜ)")
    plt.xlabel("Date")
    plt.ylabel("Factor Premium")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()