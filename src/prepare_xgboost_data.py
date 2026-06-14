# prepare_xgboost_data.py

import pandas as pd
import numpy as np
import os

import pandas as pd
import numpy as np

def prepare_xgboost_data(
    equity_X_path: str,
    targets_path: str,
    equity_etfs: list
) -> pd.DataFrame:
    """
    Prepares merged panel data for XGBoost training.
    Adds detailed logging to verify data consistency.
    """
    print("=== Loading CSVs ===")
    equity_X = pd.read_csv(equity_X_path, parse_dates=["Date"], index_col="Date")
    targets = pd.read_csv(targets_path, parse_dates=["Date"], index_col="Date")

    print(f"✅ equity_X shape: {equity_X.shape}")
    print(f"✅ targets shape: {targets.shape}")
    print("✅ equity_X preview:")
    print(equity_X.head())

    # Step 1: Drop irrelevant momentum columns not related to selected ETFs
    print("\n=== Filtering momentum features for selected ETFs ===")
    keep_cols = []
    for col in equity_X.columns:
        if "_ret_lag" in col:
            prefix = col.split("_ret_lag")[0]
            if prefix in equity_etfs:
                keep_cols.append(col)
        else:
            keep_cols.append(col)
    equity_X = equity_X[keep_cols]
    print(f"✅ Filtered equity_X shape: {equity_X.shape}")
    print("✅ Columns retained:")
    print(keep_cols)
    print("✅ Sample rows of filtered equity_X:")
    print(equity_X.head(3))

    assert not equity_X.empty, "❌ No features retained after filtering"

    # Step 2: Filter target columns to only selected ETFs
    print("\n=== Filtering target columns for selected ETFs ===")
    target_cols = [f"{etf}_target" for etf in equity_etfs if f"{etf}_target" in targets.columns]
    targets = targets[target_cols]
    print(f"✅ Filtered targets shape: {targets.shape}")
    print("✅ Sample of targets after filtering:")
    print(targets.head(3))

    assert not targets.empty, "❌ No target columns found for selected ETFs"

    # Step 3: Melt targets into long format (panel)
    print("\n=== Reshaping targets into long format ===")
    targets_panel = (
        targets
        .copy()
        .rename(columns=lambda x: x.replace("_target", ""))
        .melt(ignore_index=False, var_name="ETF", value_name="target")
        .dropna()
    )
    print(f"✅ targets_panel shape: {targets_panel.shape}")
    print("✅ Sample rows of targets_panel:")
    print(targets_panel.head(3))

    assert not targets_panel.empty, "❌ targets_panel is empty"

    # === Correct Step 4: Repeat equity_X per ETF using concat and assign ETF label ===
    print("\n=== Expanding feature matrix into panel format ===")
    frames = []
    for etf in equity_etfs:
        df = equity_X.copy()
        df["ETF"] = etf
        frames.append(df)

    equity_X_repeated = pd.concat(frames)
    equity_X_repeated.reset_index(inplace=True)  # Date is index now, restore as column
    equity_X_repeated = equity_X_repeated[["Date", "ETF"] + [col for col in equity_X.columns]]
    print(f"✅ equity_X_repeated shape: {equity_X_repeated.shape}")
    print(f"✅ Sample from equity_X_repeated:\n{equity_X_repeated.head()}")
    
    print("\n🔍 Verifying macro consistency for the same date:")
    sample_date = equity_X_repeated["Date"].iloc[0]
    print(equity_X_repeated[equity_X_repeated["Date"] == sample_date][
        ["ETF", "Date", "MKT-Rf", "SMB", "HML", "UMD", "VVIX"]
    ])

    # Spot check: Same date, different ETF — are values identical?
    print("\n🔍 Checking consistency of macro factors on same date across ETFs (e.g., MKT-Rf, SMB, HML)")
    sample_date = equity_X.index[-1]  # pick a recent date
    print(f"🔍 Sample Date: {sample_date}")
    print(equity_X_repeated[equity_X_repeated["Date"] == sample_date][
    ["ETF", "Date", "MKT-Rf", "SMB", "HML", "UMD", "QMJ", "VIX", "VVIX"]
])

    assert not equity_X_repeated.empty, "❌ equity_X_repeated is empty"

    # Step 5: Join features with target panel
    print("\n=== Joining feature and target panels ===")
    merged_panel = pd.merge(
        equity_X_repeated.reset_index(),
        targets_panel.reset_index(),
        on=["Date", "ETF"],
        how="inner"
    )
    print(f"✅ Final merged_panel shape: {merged_panel.shape}")
    print("✅ Sample rows of merged panel:")
    print(merged_panel.head(6))

    # Check if same-date rows have consistent macro factors
    print("\n🔍 Checking final consistency across same dates post-merge:")
    latest = merged_panel[merged_panel["Date"] == sample_date]
    print(latest[["ETF", "MKT-Rf", "SMB", "HML", "UMD", "QMJ", "VIX", "VVIX"]])

    assert "target" in merged_panel.columns, "❌ 'target' column missing in final panel"
    assert not merged_panel.isnull().any().any(), "❌ Null values detected after merge"

    print("✅ Merge successful and data is clean.")
    return merged_panel

