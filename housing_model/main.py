"""
Entry point for the housing price direction model.

Usage
-----
    # From the project root:
    python -m housing_model
    python housing_model/main.py

Pipeline
--------
1. Load raw data from CSV files (data_loader)
2. Engineer all features and build the binary target (features)
3. Run walk-forward backtest for Random Forest and XGBoost (model)
4. Print full evaluation metrics: accuracy, F1, MCC, confusion matrix (evaluate)
5. Generate and save all visualizations to outputs/ (visualize)
"""
from __future__ import annotations

import pandas as pd

from housing_model.config import PREDICTORS, TARGET, BACKTEST_START, OUTPUTS_DIR
from housing_model.data_loader import load_raw_dataset
from housing_model.features import build_features
from housing_model.model import run_comparison, predict
from housing_model.evaluate import print_comparison_table
from housing_model import visualize


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Load data
    # ------------------------------------------------------------------
    print("=" * 55)
    print("  Housing Price Direction Model")
    print("=" * 55)
    print("\n[1/5] Loading data...")

    raw = load_raw_dataset()
    print(f"  Rows loaded  : {raw.shape[0]:,}")
    print(f"  Date range   : {raw.index.min().date()}  ->  {raw.index.max().date()}")
    print(f"  Columns      : {list(raw.columns)}")

    # ------------------------------------------------------------------
    # Step 2: Feature engineering
    # ------------------------------------------------------------------
    print("\n[2/5] Engineering features...")

    data = build_features(raw)

    # Trim to the period when Zillow price data exists (2008 onward).
    # Pre-2008 rows have NaN for price and all price-derived features,
    # making them useless for training.
    data = data.dropna(subset=["price", "adj_price"])

    # Drop the last FORECAST_HORIZON rows where the target is unknown
    data = data.dropna(subset=[TARGET])

    n_train_rows = data.shape[0] - BACKTEST_START
    print(f"  Total rows after feature engineering : {data.shape[0]:,}")
    print(f"  Predictors used                      : {len(PREDICTORS)}")
    print(f"  Rows available for backtesting       : {n_train_rows:,}")

    # ------------------------------------------------------------------
    # Step 3: Walk-forward backtest
    # ------------------------------------------------------------------
    print("\n[3/5] Tuning hyperparameters + walk-forward backtest...")

    results = run_comparison(data, PREDICTORS, TARGET)

    # ------------------------------------------------------------------
    # Step 4: Evaluation
    # ------------------------------------------------------------------
    print("\n[4/5] Evaluating models...")

    y_true = data[TARGET].iloc[BACKTEST_START:].astype(int).values
    print_comparison_table(results, y_true)

    # ------------------------------------------------------------------
    # Step 5: Visualizations
    # ------------------------------------------------------------------
    print("[5/5] Generating plots...")

    print("  Prediction scatter...")
    visualize.plot_predictions(data, results)

    print("  Rolling accuracy chart...")
    visualize.plot_rolling_accuracy(results, data)

    print("  Cumulative return chart...")
    visualize.plot_cumulative_return(data, results)

    print("  Correlation heatmap...")
    visualize.plot_correlation_heatmap(data, PREDICTORS)

    # SHAP: run the final fold's model over every row in the backtested period.
    # The final model was never trained on any post-BACKTEST_START row within
    # its own fold, so these explanations are leak-free.  Using all ~648 rows
    # gives far more statistical power than the last fold's 24-row test window.
    print("  SHAP summary plots...")
    X_shap = data.iloc[BACKTEST_START:][PREDICTORS].copy()
    X_shap = X_shap.ffill().bfill().fillna(X_shap.median()).dropna()

    for name, res in results.items():
        # Skip ensemble: it wraps RF + XGB internally; the RF and XGBoost
        # SHAP plots already explain the two underlying models individually.
        if "Ensemble" in name:
            continue
        final_model = res["final_model"]
        if final_model is None or X_shap.empty:
            continue
        visualize.plot_shap_summary(final_model, X_shap, model_name=name)

    print(f"\nAll outputs saved to: {OUTPUTS_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
