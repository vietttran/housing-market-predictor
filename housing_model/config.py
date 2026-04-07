"""
Central configuration for the housing price direction model.

All constants live here: file paths, backtest parameters, model
hyperparameters, and feature lists. Changing a value here propagates
to every other module automatically.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR: Path = Path(__file__).parent.parent
DATA_DIR: Path = ROOT_DIR
OUTPUTS_DIR: Path = ROOT_DIR / "outputs"

# ---------------------------------------------------------------------------
# Data files
# ---------------------------------------------------------------------------

# FRED series: filename -> column name in the merged dataset
FRED_CONFIGS: dict[str, str] = {
    "MORTGAGE30US.csv": "interest",
    "RRVRUSQ156N.csv":  "vacancy",
    "CPIAUCSL.csv":     "cpi",
}

UNRATE_FILE          = "UNRATE.csv"
ZILLOW_PRICE_FILE    = "Metro_median_sale_price_uc_sfrcondo_sm_week.csv"
ZILLOW_ZHVI_FILE     = "Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_month.csv"
ZILLOW_ZORI_FILE     = "Metro_zori_uc_sfrcondomfr_sm_sa_month.csv"
ZILLOW_INVENTORY_FILE = "Metro_invt_fs_uc_sfrcondo_sm_month.csv"

# ---------------------------------------------------------------------------
# Backtest parameters
# ---------------------------------------------------------------------------
BACKTEST_START: int    = 260   # minimum training rows (~5 years of weekly data)
BACKTEST_STEP: int     = 52    # rows advanced per fold (1 year)
FORECAST_HORIZON: int  = 13    # weeks ahead to predict (one quarter)

# ---------------------------------------------------------------------------
# Model hyperparameters  (fixed params shared by all folds)
# ---------------------------------------------------------------------------
RF_PARAMS: dict = {
    "random_state": 1,
    "n_jobs":       -1,
}

XGB_PARAMS: dict = {
    "eval_metric":  "logloss",
    "random_state": 1,
}

# ---------------------------------------------------------------------------
# Hyperparameter search grids  (tuned once via TimeSeriesSplit before backtest)
# Optimisation metric: Matthews Correlation Coefficient — robust to class imbalance.
# ---------------------------------------------------------------------------
RF_PARAM_GRID: dict = {
    "n_estimators":      [200, 400],
    "min_samples_split": [5, 10, 20],
    "min_samples_leaf":  [1, 2],
    "max_features":      ["sqrt", "log2"],
}

XGB_PARAM_GRID: dict = {
    "n_estimators":   [200, 400],
    "max_depth":      [3, 4, 6],
    "learning_rate":  [0.01, 0.05, 0.1],
    "subsample":      [0.7, 0.8],
    "scale_pos_weight": [1.0, 1.55],   # 1.55 ≈ n_down / n_up in training data
}

# ---------------------------------------------------------------------------
# Feature lists
# ---------------------------------------------------------------------------

# Original features from the v1 model
BASE_PREDICTORS: list[str] = [
    "interest", "vacancy",
    "adj_price", "adj_value",
    "zori", "inventory",
    "zori_lag1", "zori_lag3",
    "inventory_lag1", "inventory_lag3",
    "zori_yoy", "inventory_yoy",
    "unrate", "unrate_lag1", "unrate_lag3", "unrate_yoy",
]

# New momentum, mean-reversion, rate velocity, and seasonality features
MOMENTUM_PREDICTORS: list[str] = [
    "price_4w_return",
    "price_26w_return",
    "price_52w_return",
    "price_vs_value_ratio",
    "interest_4w_change",
    "interest_52w_change",
    "month_of_year",
    "week_of_year",
]

PREDICTORS: list[str] = BASE_PREDICTORS + MOMENTUM_PREDICTORS
TARGET: str = "change"
