# src/models/xgboost_trainer.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
import joblib
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from configs.config import MODEL_PARAMS, PLOT_OUTPUT_DIR, MODEL_OUTPUT_DIR, OPTUNA_LOG_PATH, DEFAULT_SPLIT_DATE

def objective(trial, X_train, y_train, X_valid, y_valid):
    """Optuna objective function for hyperparameter tuning."""
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 1.0),
        "random_state": 42,
        "n_jobs": -1
    }

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_squared_error(y_valid, preds)

def filter_features(columns, etf):
    """Return only macro/factor features and return lags for the given ETF."""
    selected = []
    for col in columns:
        if "_ret_lag" in col:
            if col.startswith(etf):
                selected.append(col)
        else:
            selected.append(col)
    return selected

def train_xgboost_regressor(panel_df, split_date=DEFAULT_SPLIT_DATE):
    """Train per-ETF XGBoostRegressor using Optuna tuning and export model, logs, and predictions."""
    print("=== Loading Data ===")
    df = panel_df.copy()
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    assert "Date" in df.columns and "ETF" in df.columns and "target" in df.columns
    df["Date"] = pd.to_datetime(df["Date"])
    print("✅ Input shape:", df.shape)

    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    all_preds = []
    hyperparam_log = []

    for etf in df["ETF"].unique():
        print(f"\n=== Training model for ETF: {etf} ===")
        etf_df = df[df["ETF"] == etf].sort_values("Date")
        train_df = etf_df[etf_df["Date"] < split_date]
        test_df = etf_df[etf_df["Date"] >= split_date]

        candidate_cols = train_df.columns.difference(["Date", "ETF", "target"])
        selected_features = filter_features(candidate_cols, etf)

        X_train = train_df[selected_features]
        y_train = train_df["target"]
        X_test = test_df[selected_features]
        y_test = test_df["target"]

        print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        print("📌 Feature columns:", selected_features)
        print("📅 Train date range:", train_df["Date"].min().date(), "to", train_df["Date"].max().date())
        print("📅 Test date range:", test_df["Date"].min().date(), "to", test_df["Date"].max().date())

        # === Step 1: Optuna tuning ===
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=30)

        best_params = study.best_params
        best_params["random_state"] = 42
        best_params["n_jobs"] = -1

        print("✅ Best hyperparameters:", best_params)

        # === Step 2: Train model with best params ===
        model = xgb.XGBRegressor(**best_params)
        model.fit(X_train, y_train)

        # === Step 3: Predict and evaluate ===
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)

        print(f"Train R²: {train_r2:.4f}, MSE: {train_mse:.6f}")
        print(f"Test  R²: {test_r2:.4f}, MSE: {test_mse:.6f}")

        # === Step 4: Plot ===
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

        # === Step 5: Save model ===
        model_path = os.path.join(MODEL_OUTPUT_DIR, f"xgb_model_{etf}.pkl")
        joblib.dump(model, model_path)

        # === Step 6: Log best hyperparameters ===
        best_params["ETF"] = etf
        best_params["train_r2"] = train_r2
        best_params["test_r2"] = test_r2
        best_params["train_mse"] = train_mse
        best_params["test_mse"] = test_mse
        hyperparam_log.append(best_params)

        # === Step 7: Store predictions ===
        preds_df = test_df[["Date", "ETF"]].copy()
        preds_df["y_true"] = y_test.values
        preds_df["y_pred"] = y_test_pred
        all_preds.append(preds_df)

    # === Final Output ===
    log_df = pd.DataFrame(hyperparam_log)
    log_df.to_csv(OPTUNA_LOG_PATH, index=False)
    print(f"\n✅ Saved Optuna hyperparameter log to {OPTUNA_LOG_PATH}")

    pred_df = pd.concat(all_preds).reset_index(drop=True)
    pred_output_path = os.path.join(MODEL_OUTPUT_DIR, "xgb_predictions.csv")
    pred_df.to_csv(pred_output_path, index=False)
    print(f"✅ Final prediction DataFrame shape: {pred_df.shape}")
    print(f"✅ Saved predictions to {pred_output_path}")

    return pred_df
