import pandas as pd
from configs.config import ROLLING_STANDARDIZE_WINDOW, MOMENTUM_LAGS

def rolling_standardize(df: pd.DataFrame, window: int = ROLLING_STANDARDIZE_WINDOW) -> pd.DataFrame:
    """
    Apply rolling z-score standardization to each feature column.
    Uses past `window` values (with shift) to prevent lookahead bias.
    """
    rolling_mean = df.shift(1).rolling(window=window).mean()
    rolling_std = df.shift(1).rolling(window=window).std()
    return (df - rolling_mean) / rolling_std

def add_momentum_features(return_df: pd.DataFrame, lags=MOMENTUM_LAGS) -> pd.DataFrame:
    """
    Generate momentum features by lagging each return column.
    Produces columns like '<TICKER>_ret_lag1', etc.
    """
    features = []
    for lag in lags:
        lagged = return_df.shift(lag).copy()
        lagged.columns = [f"{col}_lag{lag}" for col in return_df.columns]
        features.append(lagged)
    
    momentum_df = pd.concat(features, axis=1)
    return momentum_df

def get_equity_features(merged_factors: pd.DataFrame, return_df: pd.DataFrame) -> pd.DataFrame:
    """
    Features for equity ETFs: factor exposures, sentiment, macro, and momentum.
    """
    macro = merged_factors[[
        "MKT-Rf", "SMB", "HML", "UMD", "QMJ",
        "VIX", "VVIX",
        "Consumer_Confidence_Index",
        "Unemployment_Initial_Claims"
    ]].copy()
    macro = rolling_standardize(macro)

    momentum = add_momentum_features(return_df)
    momentum = rolling_standardize(momentum)

    return pd.concat([macro, momentum], axis=1).dropna()

def get_fixed_income_features(merged_factors: pd.DataFrame) -> pd.DataFrame:
    """
    Features for fixed income ETFs: rates, spreads, macro.
    """
    selected = merged_factors[[
        "10Y_Yield_Constant_Maturity",
        "20Y_Yield_Constant_Maturity",
        "Building_Permits_Contribution",
        "Consumer_Confidence_Expectations",
        "Unemployment_Initial_Claims"
    ]].copy()
    selected["Term_Spread"] = (
        selected["20Y_Yield_Constant_Maturity"] - selected["10Y_Yield_Constant_Maturity"]
    )
    return rolling_standardize(selected).dropna()

def get_commodity_features(merged_factors: pd.DataFrame, return_df: pd.DataFrame) -> pd.DataFrame:
    """
    Features for commodity ETFs: trend, macro, sentiment, and momentum.
    """
    macro = merged_factors[[
        'VIX', 'VVIX',
        'PMI_Manufacturing_Global',
        'PMI_Services_Business_Activity_US'
    ]].copy()
    macro = rolling_standardize(macro)

    momentum = add_momentum_features(return_df[['GLD_ret']])
    momentum = rolling_standardize(momentum)

    return pd.concat([macro, momentum], axis=1).dropna()

def generate_all_features(merged, returns, targets):
    """
    Runs all feature engineering steps and performs diagnostics.
    Returns aligned equity, fixed income, and commodity features, and aligned targets.
    """

    # === Run individual feature generators ===
    equity_X = get_equity_features(merged, returns)
    fixed_income_X = get_fixed_income_features(merged)
    commodity_X = get_commodity_features(merged, returns)

    # === Diagnostics: Shape ===
    print("✅ Equity features shape:", equity_X.shape)
    print("✅ Fixed income features shape:", fixed_income_X.shape)
    print("✅ Commodity features shape:", commodity_X.shape)

    # === Diagnostics: NaNs ===
    print("🔍 NaN in equity features?", equity_X.isnull().values.any())
    print("🔍 NaN in fixed income features?", fixed_income_X.isnull().values.any())
    print("🔍 NaN in commodity features?", commodity_X.isnull().values.any())

    # === Diagnostics: Feature names ===
    print("\n🧠 Equity feature columns:\n", equity_X.columns.tolist())
    print("\n🧠 Fixed income feature columns:\n", fixed_income_X.columns.tolist())
    print("\n🧠 Commodity feature columns:\n", commodity_X.columns.tolist())

    # === Date alignment check ===
    equity_dates = equity_X.index
    fixed_income_dates = fixed_income_X.index
    commodity_dates = commodity_X.index
    target_dates = targets.index

    common_dates = equity_dates.intersection(fixed_income_dates).intersection(commodity_dates)

    print("\n📅 Equity dates:        ", equity_dates.min().date(), "to", equity_dates.max().date())
    print("📅 Fixed income dates: ", fixed_income_dates.min().date(), "to", fixed_income_dates.max().date())
    print("📅 Commodity dates:    ", commodity_dates.min().date(), "to", commodity_dates.max().date())
    print("📅 Common date range:  ", common_dates.min().date(), "to", common_dates.max().date())
    print("📅 Targets date range: ", target_dates.min().date(), "to", target_dates.max().date())
    print("📏 # of common dates:  ", len(common_dates))

    # === Align with targets (final intersection with equity_X only)
    final_common_dates = equity_X.index.intersection(targets.index)
    equity_X = equity_X.loc[final_common_dates]
    fixed_income_X = fixed_income_X.loc[final_common_dates]
    commodity_X = commodity_X.loc[final_common_dates]
    targets = targets.loc[final_common_dates]
    
    # === Date alignment check ===
    equity_dates = equity_X.index
    fixed_income_dates = fixed_income_X.index
    commodity_dates = commodity_X.index
    target_dates = targets.index

    print("\n✅ After fix:")
    print("🎯 Final equity_X shape:", equity_X.shape)
    print("🎯 Final targets shape:", targets.shape)
    print("\n📅 Equity dates:        ", equity_dates.min().date(), "to", equity_dates.max().date())
    print("📅 Fixed income dates: ", fixed_income_dates.min().date(), "to", fixed_income_dates.max().date())
    print("📅 Commodity dates:    ", commodity_dates.min().date(), "to", commodity_dates.max().date())
    print("📅 Common date range:  ", common_dates.min().date(), "to", common_dates.max().date())
    print("📅 Targets date range: ", target_dates.min().date(), "to", target_dates.max().date())
    print("📏 # of common dates:  ", len(common_dates))

    return equity_X, fixed_income_X, commodity_X, targets

