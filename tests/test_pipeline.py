"""
Smoke tests for the model pipeline.

Uses synthetic data only — no CSV files required.
Tests that the backtest loop runs without crashing and that outputs
have the correct shapes, types, and value ranges.

Run with:
    pytest tests/test_pipeline.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from housing_model.config import (
    BACKTEST_START,
    BACKTEST_STEP,
    PREDICTORS,
    TARGET,
)
from housing_model.model import backtest, predict, _make_rf


# ---------------------------------------------------------------------------
# Synthetic data factory
# ---------------------------------------------------------------------------

def _synthetic_dataset(n: int = 380) -> pd.DataFrame:
    """
    Build a minimal synthetic DataFrame with all required PREDICTOR columns
    and the TARGET column, indexed by weekly dates.

    Values are random but shaped so the backtest loop can run end-to-end
    without any real CSV data being present.
    """
    rng = np.random.default_rng(42)
    idx = pd.date_range("2010-01-04", periods=n, freq="W")

    data: dict = {}
    for col in PREDICTORS:
        data[col] = rng.uniform(0.0, 100.0, n).astype(float)
    data[TARGET] = rng.integers(0, 2, n).astype(int)

    return pd.DataFrame(data, index=idx)


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------

class TestPredict:
    def test_output_length_matches_test(self):
        data = _synthetic_dataset(80)
        train, test = data.iloc[:50], data.iloc[50:]
        preds, _ = predict(train, test, _make_rf)
        assert len(preds) == len(test)

    def test_predictions_are_binary(self):
        data = _synthetic_dataset(80)
        train, test = data.iloc[:50], data.iloc[50:]
        preds, _ = predict(train, test, _make_rf)
        assert set(preds).issubset({0, 1})

    def test_returns_fitted_model(self):
        data = _synthetic_dataset(80)
        train, test = data.iloc[:50], data.iloc[50:]
        _, model = predict(train, test, _make_rf)
        # A fitted RandomForest has feature_importances_
        assert hasattr(model, "feature_importances_")

    def test_model_can_predict_after_return(self):
        data = _synthetic_dataset(80)
        train, test = data.iloc[:50], data.iloc[50:]
        _, model = predict(train, test, _make_rf)
        # Should be able to predict again on new data
        new_preds = model.predict(test[PREDICTORS])
        assert len(new_preds) == len(test)

    def test_handles_nan_in_test(self):
        """Residual NaN in test features should be filled, not cause a crash."""
        data = _synthetic_dataset(80)
        train = data.iloc[:50].copy()
        test  = data.iloc[50:].copy()
        # Introduce NaN in test features
        test.iloc[0, 0] = np.nan
        preds, _ = predict(train, test, _make_rf)
        assert len(preds) == len(test)
        assert set(preds).issubset({0, 1})


# ---------------------------------------------------------------------------
# backtest()
# ---------------------------------------------------------------------------

class TestBacktest:
    def test_predictions_are_binary(self):
        data = _synthetic_dataset()
        preds, _, _, _ = backtest(data, _make_rf)
        assert set(preds).issubset({0, 1})

    def test_predictions_do_not_exceed_available_rows(self):
        data = _synthetic_dataset()
        preds, _, _, _ = backtest(data, _make_rf)
        max_possible = len(data) - BACKTEST_START
        assert len(preds) <= max_possible
        assert len(preds) > 0

    def test_window_accuracies_are_bounded(self):
        data = _synthetic_dataset()
        _, accs, _, _ = backtest(data, _make_rf)
        for acc in accs:
            assert 0.0 <= acc <= 1.0, f"Accuracy out of range: {acc}"

    def test_number_of_folds(self):
        data = _synthetic_dataset()
        _, accs, _, _ = backtest(data, _make_rf)
        expected = len(range(BACKTEST_START, len(data), BACKTEST_STEP))
        assert len(accs) == expected

    def test_final_model_is_fitted(self):
        data = _synthetic_dataset()
        _, _, final_model, _ = backtest(data, _make_rf)
        assert final_model is not None
        assert hasattr(final_model, "feature_importances_")

    def test_final_test_is_nonempty_dataframe(self):
        data = _synthetic_dataset()
        _, _, _, final_test = backtest(data, _make_rf)
        assert isinstance(final_test, pd.DataFrame)
        assert len(final_test) > 0

    def test_final_test_contains_predictor_columns(self):
        data = _synthetic_dataset()
        _, _, _, final_test = backtest(data, _make_rf)
        for col in PREDICTORS:
            assert col in final_test.columns

    def test_custom_start_and_step(self):
        """backtest() respects explicit start and step parameters."""
        data = _synthetic_dataset(200)
        _, accs, _, _ = backtest(data, _make_rf, start=100, step=20)
        expected = len(range(100, len(data), 20))
        assert len(accs) == expected

    def test_small_dataset_does_not_crash(self):
        """When there is just barely enough data for one fold."""
        n = BACKTEST_START + BACKTEST_STEP + 5
        data = _synthetic_dataset(n)
        preds, accs, _, _ = backtest(data, _make_rf)
        assert len(accs) >= 1
        assert len(preds) > 0
