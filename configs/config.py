# configs/config.py

# === Rolling Standardization Window ===
ROLLING_STANDARDIZE_WINDOW = 252  # ~1 year of trading days (can also use 63 for 3-month window)

# === Momentum Feature Lags ===
MOMENTUM_LAGS = [1, 5, 20]  # 1-day, 1-week, 1-month lagged returns

# === Model Parameters for Each Model Type ===
# MODEL_PARAMS = {
#     "ridge": {
#         "alpha": 1.0
#     },
#     "lasso": {
#         "alpha": 0.1
#     },
#     "elasticnet": {
#         "alpha": 1.0,
#         "l1_ratio": 0.5
#     },
#     "theilsen": {
#         "fit_intercept": True
#     },
#     "randomforest": {
#         "n_estimators": 100,
#         "max_depth": 5,
#         "random_state": 42
#     },
#     "xgboost": {
#         "n_estimators": 100,
#         "max_depth": 3,
#         "learning_rate": 0.1,
#         "subsample": 0.8,
#         "colsample_bytree": 0.8,
#         "random_state": 42
#     } }
# MODEL_PARAMS = {
#     "objective": "reg:squarederror",
#     "n_estimators": 100,
#     "max_depth": 3,
#     "learning_rate": 0.05,
#     "subsample": 0.8,
#     "colsample_bytree": 0.8,
#     "random_state": 42
# }
MODEL_PARAMS = {
    "random_state": 42,
    "n_jobs": -1
}

# === Paths ===
PLOT_OUTPUT_DIR = "outputs/plots"
MODEL_OUTPUT_DIR = "models/xgboost"
LOG_OUTPUT_DIR = "logs"
OPTUNA_LOG_PATH = "reports/hyperparameter_log.csv"

# === Training Control ===
DEFAULT_SPLIT_DATE = "2024-01-01"


# === Feature Engineering Configuration ===
STANDARDIZE_FEATURES = True  # set to False to skip rolling_standardize()

# === Output Paths ===
ALPHA_OUTPUT_PATH = "outputs/alpha_factors.csv"
#PLOT_OUTPUT_DIR = "outputs/"
