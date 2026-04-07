"""
Model training, walk-forward backtesting, and multi-model comparison.

Design decisions
----------------
1. Hyperparameters are tuned ONCE on the initial training window using
   TimeSeriesSplit before the backtest begins.  Optimisation metric is MCC.

2. Three models are compared: Random Forest, XGBoost, and a soft-voting
   ensemble that averages their predicted probabilities.  Ensembling
   consistently outperforms either individual model on tabular data.

3. NaN rows at the start of training are dropped via dropna() in predict().

4. SHAP analyses use the final fold's trained model over the full backtested
   period — never a full-data refit.

5. XGBoost is an optional dependency.  If not installed, only RF runs.
"""
from __future__ import annotations

from itertools import product
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.model_selection import TimeSeriesSplit

try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

from housing_model.config import (
    BACKTEST_START,
    BACKTEST_STEP,
    PREDICTORS,
    TARGET,
    RF_PARAMS,
    RF_PARAM_GRID,
    XGB_PARAMS,
    XGB_PARAM_GRID,
)


# ---------------------------------------------------------------------------
# Default model factory (used internally and in tests)
# ---------------------------------------------------------------------------

def _make_rf() -> RandomForestClassifier:
    """Return a RandomForestClassifier with the project's fixed base params."""
    return RandomForestClassifier(**RF_PARAMS)


# ---------------------------------------------------------------------------
# Soft-voting ensemble
# ---------------------------------------------------------------------------

class SoftVotingEnsemble:
    """
    Averages the predicted probabilities of a Random Forest and XGBoost
    classifier before thresholding at 0.5.  Soft voting outperforms hard
    voting because it preserves confidence information from both models.
    """

    def __init__(self, rf_params: dict, xgb_params: dict) -> None:
        self.rf_params  = rf_params
        self.xgb_params = xgb_params
        self.rf_:  RandomForestClassifier | None = None
        self.xgb_: XGBClassifier | None          = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SoftVotingEnsemble":
        self.rf_  = RandomForestClassifier(**self.rf_params)
        self.xgb_ = XGBClassifier(**self.xgb_params)
        self.rf_.fit(X, y)
        self.xgb_.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p_rf  = self.rf_.predict_proba(X)[:, 1]
        p_xgb = self.xgb_.predict_proba(X)[:, 1]
        avg   = (p_rf + p_xgb) / 2
        return np.column_stack([1 - avg, avg])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------

def tune_hyperparameters(
    data: pd.DataFrame,
    model_cls,
    fixed_params: dict,
    param_grid: dict,
    predictors: list[str] = PREDICTORS,
    target: str = TARGET,
    n_splits: int = 3,
) -> dict:
    """
    Grid search over param_grid using TimeSeriesSplit on the initial training
    window only.  Optimises MCC averaged across folds.

    Returns the best hyperparameter dict merged with fixed_params.
    """
    train_data = data.iloc[:BACKTEST_START].dropna(subset=predictors + [target])
    X = train_data[predictors].values
    y = train_data[target].astype(int).values

    tscv = TimeSeriesSplit(n_splits=n_splits)
    keys, values = list(param_grid.keys()), list(param_grid.values())

    best_score, best_params = -np.inf, {}

    for combo in product(*values):
        params = dict(zip(keys, combo))
        scores = []
        for tr_idx, val_idx in tscv.split(X):
            model = model_cls(**{**fixed_params, **params})
            model.fit(X[tr_idx], y[tr_idx])
            scores.append(matthews_corrcoef(y[val_idx], model.predict(X[val_idx])))
        score = float(np.mean(scores))
        if score > best_score:
            best_score, best_params = score, params

    return {**fixed_params, **best_params}


# ---------------------------------------------------------------------------
# Core prediction step
# ---------------------------------------------------------------------------

def predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    model_factory: Callable,
    predictors: list[str] = PREDICTORS,
    target: str = TARGET,
) -> tuple[np.ndarray, object]:
    """
    Train a model on train, predict on test.
    Drops NaN training rows; fills residual NaN in test with training median.

    Returns (predictions, fitted_model).
    """
    train_clean = train.dropna(subset=predictors + [target])
    model = model_factory()
    model.fit(train_clean[predictors], train_clean[target].astype(int))

    X_test = test[predictors].copy()
    if X_test.isna().any().any():
        X_test = X_test.fillna(train_clean[predictors].median())

    return model.predict(X_test), model


# ---------------------------------------------------------------------------
# Walk-forward backtest
# ---------------------------------------------------------------------------

def backtest(
    data: pd.DataFrame,
    model_factory: Callable,
    predictors: list[str] = PREDICTORS,
    target: str = TARGET,
    start: int = BACKTEST_START,
    step: int = BACKTEST_STEP,
) -> tuple[np.ndarray, list[float], object, pd.DataFrame]:
    """
    Walk-forward expanding-window backtest.
    No future data ever enters a training window.

    Returns (preds, window_accuracies, final_model, final_test).
    """
    all_preds: list[np.ndarray] = []
    window_accuracies: list[float] = []
    final_model = None
    final_test: pd.DataFrame = pd.DataFrame()

    for i in range(start, data.shape[0], step):
        train = data.iloc[:i]
        test  = data.iloc[i : i + step]
        if test.empty:
            break

        preds, model = predict(train, test, model_factory, predictors, target)
        y_window = test[target].dropna().astype(int).values
        n = min(len(preds), len(y_window))
        window_accuracies.append(accuracy_score(y_window[:n], preds[:n]))

        all_preds.append(preds)
        final_model = model
        final_test  = test.copy()

    return np.concatenate(all_preds), window_accuracies, final_model, final_test


# ---------------------------------------------------------------------------
# Multi-model comparison
# ---------------------------------------------------------------------------

def run_comparison(
    data: pd.DataFrame,
    predictors: list[str] = PREDICTORS,
    target: str = TARGET,
) -> dict[str, dict]:
    """
    Tune RF and XGBoost, then run the walk-forward backtest for:
      - Random Forest (tuned)
      - XGBoost (tuned)
      - Soft-Voting Ensemble (tuned RF + tuned XGB probabilities averaged)

    Returns a dict keyed by model name with preds, window_accuracies,
    final_model, final_test, and best_params.
    """
    results: dict[str, dict] = {}

    # ---- Tune RF --------------------------------------------------------
    print("  Tuning Random Forest (TimeSeriesSplit, n_splits=3)...")
    rf_best = tune_hyperparameters(
        data, RandomForestClassifier, RF_PARAMS, RF_PARAM_GRID, predictors, target,
    )
    print(f"    Best RF params : {rf_best}")

    print("  Running Random Forest backtest...")
    rf_preds, rf_accs, rf_model, rf_test = backtest(
        data, lambda: RandomForestClassifier(**rf_best), predictors, target,
    )
    results["Random Forest"] = dict(
        preds=rf_preds, window_accuracies=rf_accs,
        final_model=rf_model, final_test=rf_test, best_params=rf_best,
    )

    # ---- Tune XGBoost ---------------------------------------------------
    if not _XGBOOST_AVAILABLE:
        print("  XGBoost not installed; skipping. Run: pip install xgboost")
        return results

    print("  Tuning XGBoost (TimeSeriesSplit, n_splits=3)...")
    xgb_best = tune_hyperparameters(
        data, XGBClassifier, XGB_PARAMS, XGB_PARAM_GRID, predictors, target,
    )
    print(f"    Best XGB params: {xgb_best}")

    print("  Running XGBoost backtest...")
    xgb_preds, xgb_accs, xgb_model, xgb_test = backtest(
        data, lambda: XGBClassifier(**xgb_best), predictors, target,
    )
    results["XGBoost"] = dict(
        preds=xgb_preds, window_accuracies=xgb_accs,
        final_model=xgb_model, final_test=xgb_test, best_params=xgb_best,
    )

    # ---- Soft-Voting Ensemble -------------------------------------------
    print("  Running Soft-Voting Ensemble backtest (RF + XGB probabilities)...")
    ens_preds, ens_accs, ens_model, ens_test = backtest(
        data,
        lambda: SoftVotingEnsemble(rf_best, xgb_best),
        predictors, target,
    )
    results["Ensemble (RF + XGB)"] = dict(
        preds=ens_preds, window_accuracies=ens_accs,
        final_model=ens_model, final_test=ens_test,
        best_params={"rf": rf_best, "xgb": xgb_best},
    )

    return results
