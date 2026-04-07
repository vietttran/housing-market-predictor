"""
Feature engineering for the housing price direction model.

All functions are pure: given a DataFrame, return an augmented DataFrame.
No I/O, no global state, no side effects.

Design note on NaN handling
----------------------------
Sparse base columns (vacancy, cpi, zori, etc.) are forward-filled in
data_loader.load_raw_dataset() before reaching this module.  The lag and
year-over-year features derived here therefore inherit those filled values.
The only remaining NaN rows are in the first N rows of momentum features
(e.g. price_52w_return is NaN for the first 52 rows).  Those rows fall
well within the initial training window and are dropped inside
model.predict() via dropna().
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from housing_model.config import FORECAST_HORIZON, TARGET


# ---------------------------------------------------------------------------
# Primitive transformations (tested individually)
# ---------------------------------------------------------------------------

def inflation_adjust(series: pd.Series, cpi: pd.Series) -> pd.Series:
    """
    Return an inflation-adjusted Series in real dollars (CPI base = 100).

    Parameters
    ----------
    series : pd.Series
        Nominal price values.
    cpi : pd.Series
        Consumer Price Index aligned to the same index as series.

    Returns
    -------
    pd.Series
        Real values: series / cpi * 100.
    """
    return series / cpi * 100


def make_lags(series: pd.Series, lags: list[int]) -> pd.DataFrame:
    """
    Create lag features for a Series.

    Parameters
    ----------
    series : pd.Series
        The input time series (must have a name).
    lags : list[int]
        Lag periods (in rows) to generate.

    Returns
    -------
    pd.DataFrame
        One column per lag, named '<series.name>_lag<n>'.
    """
    return pd.DataFrame(
        {f"{series.name}_lag{n}": series.shift(n) for n in lags},
        index=series.index,
    )


def make_yoy(series: pd.Series) -> pd.Series:
    """
    Compute year-over-year fractional change: (value / value_12_periods_ago) - 1.

    Works on monthly or weekly data (shift=12 rows).  For weekly data this
    approximates a 12-week change; pass the monthly-resampled series for a
    true annual comparison.

    Returns
    -------
    pd.Series
        Named '<series.name>_yoy'.  First 12 values are NaN.
    """
    return (series / series.shift(12) - 1).rename(f"{series.name}_yoy")


def add_momentum_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add price momentum, mean-reversion, interest rate velocity,
    and calendar seasonality features.

    Requires columns: adj_price, adj_value, interest.

    New columns added
    -----------------
    price_4w_return      : 4-week percent change in real price
    price_26w_return     : 26-week (half-year) percent change in real price
    price_52w_return     : 52-week percent change in real price
    price_vs_value_ratio : adj_price / adj_value (elevated = mean-reversion signal)
    interest_4w_change   : 4-week first difference in mortgage rate
    interest_52w_change  : 52-week first difference in mortgage rate
    month_of_year        : calendar month (1-12, captures spring seasonality)
    week_of_year         : ISO week number (1-53)
    """
    data = data.copy()

    # Price momentum
    data["price_4w_return"]  = data["adj_price"].pct_change(4,  fill_method=None)
    data["price_26w_return"] = data["adj_price"].pct_change(26, fill_method=None)
    data["price_52w_return"] = data["adj_price"].pct_change(52, fill_method=None)

    # Mean-reversion signal: how far sale prices are from estimated value
    data["price_vs_value_ratio"] = data["adj_price"] / data["adj_value"]

    # Interest rate velocity: level alone misses acceleration
    data["interest_4w_change"]  = data["interest"].diff(4)
    data["interest_52w_change"] = data["interest"].diff(52)

    # Seasonality: housing demand peaks in spring and summer
    data["month_of_year"] = data.index.month
    data["week_of_year"]  = data.index.isocalendar().week.astype(int)

    return data


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering to the raw dataset and build the
    binary prediction target.

    Parameters
    ----------
    raw : pd.DataFrame
        Output of data_loader.load_raw_dataset().

    Returns
    -------
    pd.DataFrame
        Original columns plus all engineered features and the target column.
        The last FORECAST_HORIZON rows will have NaN for the target because
        the future price is unknown; drop them before training.
    """
    data = raw.copy()

    # Inflation-adjusted price and home value
    data["adj_price"] = inflation_adjust(data["price"], data["cpi"])
    data["adj_value"] = inflation_adjust(data["value"], data["cpi"])

    # Binary target: will inflation-adjusted price be higher N weeks from now?
    data["next_quarter"] = data["adj_price"].shift(-FORECAST_HORIZON)
    data[TARGET] = (data["next_quarter"] > data["adj_price"]).astype("Int64")

    # Lag and year-over-year features for rental, inventory, unemployment
    for col in ["zori", "inventory", "unrate"]:
        lags_df = make_lags(data[col], lags=[1, 3])
        yoy     = make_yoy(data[col])
        data    = pd.concat([data, lags_df, yoy], axis=1)

    # Momentum, mean-reversion, rate velocity, seasonality
    data = add_momentum_features(data)

    return data
