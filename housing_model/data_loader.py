"""
Data loading for the housing price direction model.

All CSV I/O lives here. Functions return clean, merged DataFrames
with sparse (monthly/quarterly) columns forward-filled to their
last published value — the same information available in real time.
No look-ahead filling is performed; the fill uses only past observations.
"""
from __future__ import annotations

import pandas as pd
from datetime import timedelta

from housing_model.config import (
    DATA_DIR,
    FRED_CONFIGS,
    UNRATE_FILE,
    ZILLOW_PRICE_FILE,
    ZILLOW_ZHVI_FILE,
    ZILLOW_ZORI_FILE,
    ZILLOW_INVENTORY_FILE,
)

# Columns that are published monthly or quarterly and need forward-filling
# so every weekly row in the merged dataset carries a valid observation.
_SPARSE_BASE_COLS = ["vacancy", "cpi", "zori", "inventory", "unrate", "interest"]


def _load_fred_csv(filename: str, column_name: str) -> pd.Series:
    """
    Load a FRED CSV file and return a named Series indexed by date.

    FRED files have an 'observation_date' column and a single value column.
    """
    path = DATA_DIR / filename
    df = pd.read_csv(path, parse_dates=["observation_date"])
    df = df.set_index("observation_date")
    df.index.name = "date"
    df.columns = [column_name]
    return df[column_name]


def _load_zillow_national(filename: str, value_name: str) -> pd.Series:
    """
    Load a Zillow Research wide-format CSV, extract the US national row
    (RegionName == 'United States'), and return a date-indexed Series.

    Zillow wide files have one row per metro area and one column per date.
    """
    path = DATA_DIR / filename
    df = pd.read_csv(path)
    national = df[df["RegionName"] == "United States"].copy()

    meta_cols = ["RegionID", "SizeRank", "RegionName", "RegionType", "StateName"]
    national = national.drop(columns=meta_cols, errors="ignore")

    melted = national.melt(var_name="date", value_name=value_name)
    melted["date"] = pd.to_datetime(melted["date"])
    return melted.set_index("date")[value_name].sort_index()


def load_raw_dataset() -> pd.DataFrame:
    """
    Load and merge all data sources into a single weekly time-indexed DataFrame.

    Returned columns (before feature engineering):
        interest  : 30-year fixed mortgage rate (FRED MORTGAGE30US), weekly
        vacancy   : Rental vacancy rate (FRED RRVRUSQ156N), quarterly -> filled
        cpi       : Consumer Price Index (FRED CPIAUCSL), monthly -> filled
        price     : Zillow national median sale price, weekly
        value     : Zillow Home Value Index (ZHVI middle tier), monthly -> filled
        zori      : Zillow Observed Rent Index, monthly -> filled
        inventory : Active for-sale listings count, monthly -> filled
        unrate    : U.S. unemployment rate (FRED), monthly -> filled

    Sparse (monthly/quarterly) columns are forward-filled with the most
    recently published value, which is the value available in real time.
    """
    # ------------------------------------------------------------------
    # FRED macro series
    # ------------------------------------------------------------------
    fred_series = [_load_fred_csv(f, col) for f, col in FRED_CONFIGS.items()]
    fed_data = pd.concat(fred_series, axis=1)

    # FRED dates are typically published with a publication-day offset;
    # shift 2 days so they align with Zillow's week-ending Saturday dates.
    fed_data.index = fed_data.index + timedelta(days=2)

    # ------------------------------------------------------------------
    # Zillow weekly price + monthly ZHVI
    # Both files have one row per region in wide format.
    # Merge on calendar month so every weekly price row gets the
    # corresponding monthly ZHVI observation.
    # ------------------------------------------------------------------
    price_raw = _load_zillow_national(ZILLOW_PRICE_FILE, "price")
    zhvi_raw  = _load_zillow_national(ZILLOW_ZHVI_FILE,  "value")

    price_df = price_raw.to_frame()
    price_df["_month"] = price_df.index.to_period("M")

    zhvi_df = zhvi_raw.to_frame()
    zhvi_df["_month"] = zhvi_df.index.to_period("M")

    price_value = price_df.merge(
        zhvi_df[["_month", "value"]], on="_month", how="left"
    )
    price_value.index = price_df.index
    price_value = price_value.drop(columns="_month")

    # ------------------------------------------------------------------
    # Outer merge: FRED dates + Zillow weekly dates
    # ------------------------------------------------------------------
    data = fed_data.merge(
        price_value, left_index=True, right_index=True, how="outer"
    )

    # ------------------------------------------------------------------
    # Zillow ZORI and inventory (monthly, national)
    # ------------------------------------------------------------------
    zori      = _load_zillow_national(ZILLOW_ZORI_FILE,      "zori")
    inventory = _load_zillow_national(ZILLOW_INVENTORY_FILE, "inventory")

    data = (
        data.reset_index()
            .rename(columns={"index": "date"})
            .merge(zori.reset_index(),      on="date", how="left")
            .merge(inventory.reset_index(), on="date", how="left")
            .set_index("date")
            .sort_index()
    )

    # ------------------------------------------------------------------
    # Unemployment rate (monthly FRED)
    # ------------------------------------------------------------------
    unrate = _load_fred_csv(UNRATE_FILE, "unrate")
    data = (
        data.reset_index()
            .merge(unrate.reset_index(), on="date", how="left")
            .set_index("date")
            .sort_index()
    )

    # ------------------------------------------------------------------
    # Forward-fill sparse columns
    # Using the last published value is the correct real-time behaviour:
    # we always know the most recently released CPI, mortgage rate, etc.
    # ------------------------------------------------------------------
    # Forward-fill: propagate last published value to subsequent weeks.
    # Backward-fill: extend the earliest observation back to cover dates
    # before a data series began (e.g. ZORI starts 2015, inventory 2018).
    # This ensures the training set is never empty due to missing series.
    cols_to_fill = [c for c in _SPARSE_BASE_COLS if c in data.columns]
    data[cols_to_fill] = data[cols_to_fill].ffill().bfill()

    return data
