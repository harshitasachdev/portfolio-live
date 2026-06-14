# src/models/xgboost_trainer.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from configs.config import MODEL_PARAMS, PLOT_OUTPUT_DIR, DEFAULT_SPLIT_DATE, MODEL_OUTPUT_DIR, LOG_OUTPUT_DIR
import os

def train_xgboost_regressor(panel_df, split_date=DEFAULT_SPLIT_DATE):
    """
    Train and evaluate XGBoostRegressor on ETF panel data.

    Parameters:
        panel_df (pd.DataFrame): ETF panel DataFrame
        split_date (str): Date to split in-sample and out-of-sample sets

    Returns:
        dict: Mapping from ETF to trained model
    """
    print("=== Loading Data ===")
    df = panel_df.copy()
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    assert "Date" in df.columns and "ETF" in df.columns and "target" in df.columns

    print("✅ Input shape:", df.shape)

    # Ensure Date is datetime
    df["Date"] = pd.to_datetime(df["Date"])

    # Create directory for plots if it doesn't exist
    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

    # Store models
    models = {}

    # Loop over unique ETFs
    for etf in df["ETF"].unique():
        print(f"\n=== Training model for ETF: {etf} ===")

        etf_df = df[df["ETF"] == etf].copy()
        etf_df = etf_df.sort_values("Date")

        # Split into in-sample and out-of-sample sets
        train_df = etf_df[etf_df["Date"] < split_date]
        test_df = etf_df[etf_df["Date"] >= split_date]

        X_train = train_df.drop(columns=["Date", "ETF", "target"])
        y_train = train_df["target"]
        X_test = test_df.drop(columns=["Date", "ETF", "target"])
        y_test = test_df["target"]

        print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        # Train XGBoost model
        model = xgb.XGBRegressor(**MODEL_PARAMS)
        model.fit(X_train, y_train)
        
        # Save model to disk
        model_path = os.path.join(MODEL_OUTPUT_DIR, f"xgb_model_{etf}.json")
        model.save_model(model_path)

        # Predict and evaluate
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        print(f"Train R^2: {train_r2:.4f}, MSE: {train_mse:.6f}")
        print(f"Test  R^2: {test_r2:.4f}, MSE: {test_mse:.6f}")

        # Plot
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(y_train.values, label="Actual", alpha=0.7)
        plt.plot(y_train_pred, label="Predicted", alpha=0.7)
        plt.title(f"In-Sample: {etf}")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(y_test.values, label="Actual", alpha=0.7)
        plt.plot(y_test_pred, label="Predicted", alpha=0.7)
        plt.title(f"Out-of-Sample: {etf}")
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_OUTPUT_DIR, f"xgb_pred_vs_actual_{etf}.png"))
        plt.close()

        models[etf] = model

    return models

