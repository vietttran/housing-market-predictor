"""
Unit tests for housing_model.features.

All tests use small, synthetic pandas objects constructed inline.
No CSV files or internet access required.

Run with:
    pytest tests/test_features.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from housing_model.features import (
    add_momentum_features,
    inflation_adjust,
    make_lags,
    make_yoy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _series(values: list, name: str = "x", freq: str = "W") -> pd.Series:
    """Build a weekly-indexed Series from a list of values."""
    idx = pd.date_range("2020-01-06", periods=len(values), freq=freq)
    return pd.Series(values, index=idx, dtype=float, name=name)


def _df_for_momentum(n: int = 60) -> pd.DataFrame:
    """Build a minimal DataFrame with the columns add_momentum_features() needs."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-06", periods=n, freq="W")
    return pd.DataFrame(
        {
            "adj_price": rng.uniform(200_000, 400_000, n),
            "adj_value": rng.uniform(200_000, 400_000, n),
            "interest":  rng.uniform(3.0, 7.5, n),
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# inflation_adjust
# ---------------------------------------------------------------------------

class TestInflationAdjust:
    def test_basic_division(self):
        price = _series([100.0, 200.0, 300.0], "price")
        cpi   = _series([100.0, 100.0, 150.0], "cpi")
        result = inflation_adjust(price, cpi)
        expected = pd.Series([100.0, 200.0, 200.0], index=price.index)
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_constant_cpi_scales_linearly(self):
        price = _series([50.0, 100.0, 150.0], "price")
        cpi   = _series([50.0, 50.0, 50.0],   "cpi")
        result = inflation_adjust(price, cpi)
        expected = pd.Series([100.0, 200.0, 300.0], index=price.index)
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_cpi_equals_100_is_identity(self):
        price = _series([123.45, 678.90], "price")
        cpi   = _series([100.0,  100.0],  "cpi")
        result = inflation_adjust(price, cpi)
        pd.testing.assert_series_equal(result, price, check_names=False)

    def test_returns_series(self):
        price = _series([1.0], "price")
        cpi   = _series([1.0], "cpi")
        assert isinstance(inflation_adjust(price, cpi), pd.Series)


# ---------------------------------------------------------------------------
# make_lags
# ---------------------------------------------------------------------------

class TestMakeLags:
    def test_lag1_first_value_is_nan(self):
        s = _series([10, 20, 30, 40, 50], "x")
        result = make_lags(s, [1])
        assert pd.isna(result["x_lag1"].iloc[0])

    def test_lag1_correct_shift(self):
        s = _series([10, 20, 30, 40, 50], "x")
        result = make_lags(s, [1])
        assert result["x_lag1"].iloc[1] == 10.0
        assert result["x_lag1"].iloc[4] == 40.0

    def test_lag3_has_exactly_three_leading_nans(self):
        s = _series([1, 2, 3, 4, 5, 6], "x")
        result = make_lags(s, [3])
        assert result["x_lag3"].isna().sum() == 3

    def test_lag3_first_nonnan_is_correct(self):
        s = _series([10, 20, 30, 40, 50, 60], "x")
        result = make_lags(s, [3])
        # Position 3 should equal position 0 of original
        assert result["x_lag3"].iloc[3] == 10.0

    def test_multiple_lags_produce_multiple_columns(self):
        s = _series(list(range(10)), "x")
        result = make_lags(s, [1, 3])
        assert "x_lag1" in result.columns
        assert "x_lag3" in result.columns
        assert result.shape[1] == 2

    def test_returns_dataframe(self):
        s = _series([1, 2, 3], "x")
        assert isinstance(make_lags(s, [1]), pd.DataFrame)

    def test_empty_lags_list_returns_empty_dataframe(self):
        s = _series([1, 2, 3], "x")
        result = make_lags(s, [])
        assert result.empty

    def test_column_naming(self):
        s = _series([1, 2, 3], "my_col")
        result = make_lags(s, [1, 2])
        assert "my_col_lag1" in result.columns
        assert "my_col_lag2" in result.columns


# ---------------------------------------------------------------------------
# make_yoy
# ---------------------------------------------------------------------------

class TestMakeYoy:
    def test_doubling_over_12_periods_gives_1(self):
        # First 12 values = 100, next 12 = 200; at position 12: 200/100 - 1 = 1.0
        values = [100.0] * 12 + [200.0] * 12
        s = _series(values, "x")
        yoy = make_yoy(s)
        assert abs(yoy.iloc[12] - 1.0) < 1e-9

    def test_flat_series_gives_zero_yoy(self):
        s = _series([50.0] * 24, "x")
        yoy = make_yoy(s)
        non_nan = yoy.dropna()
        assert (non_nan.abs() < 1e-9).all()

    def test_first_12_values_are_nan(self):
        s = _series(list(range(1, 25)), "x")
        yoy = make_yoy(s)
        assert yoy.iloc[:12].isna().all()

    def test_value_after_12_is_not_nan(self):
        s = _series(list(range(1, 25)), "x")
        yoy = make_yoy(s)
        assert not pd.isna(yoy.iloc[12])

    def test_output_name_is_suffixed(self):
        s = _series([1.0, 2.0], "my_col")
        assert make_yoy(s).name == "my_col_yoy"

    def test_halving_gives_minus_half(self):
        values = [100.0] * 12 + [50.0] * 12
        s = _series(values, "x")
        yoy = make_yoy(s)
        assert abs(yoy.iloc[12] - (-0.5)) < 1e-9


# ---------------------------------------------------------------------------
# add_momentum_features
# ---------------------------------------------------------------------------

class TestAddMomentumFeatures:
    EXPECTED_COLS = [
        "price_4w_return", "price_26w_return", "price_52w_return",
        "price_vs_value_ratio",
        "interest_4w_change", "interest_52w_change",
        "month_of_year", "week_of_year",
    ]

    def test_all_expected_columns_added(self):
        df = _df_for_momentum()
        result = add_momentum_features(df)
        for col in self.EXPECTED_COLS:
            assert col in result.columns, f"Missing column: {col}"

    def test_does_not_modify_input(self):
        df = _df_for_momentum()
        cols_before = set(df.columns)
        add_momentum_features(df)
        assert set(df.columns) == cols_before  # original unchanged

    def test_price_vs_value_ratio_correctness(self):
        idx = pd.date_range("2020-01-06", periods=3, freq="W")
        df = pd.DataFrame(
            {
                "adj_price": [200.0, 300.0, 400.0],
                "adj_value": [100.0, 100.0, 200.0],
                "interest":  [4.0,   4.0,   4.0],
            },
            index=idx,
        )
        result = add_momentum_features(df)
        expected = [2.0, 3.0, 2.0]
        for i, exp in enumerate(expected):
            assert abs(result["price_vs_value_ratio"].iloc[i] - exp) < 1e-9

    def test_month_of_year_in_valid_range(self):
        df = _df_for_momentum(52)
        result = add_momentum_features(df)
        assert result["month_of_year"].between(1, 12).all()

    def test_week_of_year_in_valid_range(self):
        df = _df_for_momentum(52)
        result = add_momentum_features(df)
        assert result["week_of_year"].between(1, 53).all()

    def test_price_4w_return_first_4_are_nan(self):
        df = _df_for_momentum(20)
        result = add_momentum_features(df)
        assert result["price_4w_return"].iloc[:4].isna().all()

    def test_interest_4w_change_first_4_are_nan(self):
        df = _df_for_momentum(20)
        result = add_momentum_features(df)
        assert result["interest_4w_change"].iloc[:4].isna().all()

    def test_interest_4w_change_correctness(self):
        idx = pd.date_range("2020-01-06", periods=6, freq="W")
        df = pd.DataFrame(
            {
                "adj_price": [1.0] * 6,
                "adj_value": [1.0] * 6,
                "interest":  [3.0, 3.5, 4.0, 4.5, 5.0, 5.5],
            },
            index=idx,
        )
        result = add_momentum_features(df)
        # interest_4w_change at index 4: 5.0 - 3.0 = 2.0
        assert abs(result["interest_4w_change"].iloc[4] - 2.0) < 1e-9
