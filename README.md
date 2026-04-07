# Housing Price Direction Model

A machine learning pipeline that predicts whether U.S. home prices will be higher or lower 13 weeks from now, using macroeconomic and housing market data from FRED and Zillow.

---

## Output Plots

After running the model, six charts are saved to `outputs/`:

| Chart | What it shows |
|-------|--------------|
| `predictions_scatter.png` | Inflation-adjusted prices over time, colored green (correct) or red (wrong) |
| `rolling_accuracy.png` | Per-fold accuracy across the backtest timeline for both models |
| `cumulative_return.png` | Simulated return of following each model's signal vs baselines |
| `shap_random_forest.png` | SHAP beeswarm for Random Forest: direction and magnitude of each feature's impact |
| `shap_xgboost.png` | SHAP beeswarm for XGBoost: direction and magnitude of each feature's impact |
| `correlation_heatmap.png` | Predictor correlation matrix |

*Run the model locally to generate these plots (see How to Run It below).*

![Prediction Scatter](outputs/predictions_scatter.png)
![Rolling Accuracy](outputs/rolling_accuracy.png)
![Cumulative Return](outputs/cumulative_return.png)
![SHAP Summary — Random Forest](outputs/shap_random_forest.png)
![SHAP Summary — XGBoost](outputs/shap_xgboost.png)
![Correlation Heatmap](outputs/correlation_heatmap.png)

---

## What It Does

The model takes weekly national housing data going back to 2008 and asks one question: will inflation-adjusted home prices be higher 13 weeks from now than they are today? The answer is binary (up or down), and the model is a Random Forest and XGBoost classifier trained on 24 economic features.

The data comes from two places. FRED (the Federal Reserve's data portal) provides the 30-year mortgage rate, rental vacancy rate, consumer price index, and unemployment rate. Zillow Research provides the national median sale price, home value index (ZHVI), observed rent index (ZORI), and active for-sale inventory count.

Features include the raw economic indicators, 1-month and 3-month lags of rent, inventory, and unemployment, year-over-year changes for those same series, price momentum over 4, 26, and 52 weeks, the ratio of sale price to home value, mortgage rate velocity (how fast rates are rising or falling), and calendar seasonality (month and week of year to capture the spring buying peak).

The model is evaluated with a walk-forward backtest: for each fold, training uses only data from before the test window, so no future information ever reaches the model during evaluation. Results are reported as accuracy, F1, Matthews Correlation Coefficient, precision, recall, and a confusion matrix — not just accuracy, which is insufficient on its own.

SHAP values are computed by running the final fold's trained model over the entire backtested period (648 rows). The final model was never trained on any of those rows within its own fold, so the explanations are leak-free and statistically meaningful.

---

## Features

- Walk-forward (expanding-window) backtest with no data leakage
- Hyperparameter tuning via TimeSeriesSplit grid search optimising MCC
- Random Forest, XGBoost, and a soft-voting ensemble trained and compared side-by-side
- Full evaluation suite: accuracy, F1, MCC, precision, recall, confusion matrix
- 24 engineered features: macro indicators, lags, year-over-year changes, momentum, mean-reversion, rate velocity, seasonality
- SHAP beeswarm plots for both models showing feature direction and magnitude
- Rolling per-fold accuracy chart to detect regime-specific performance
- Simulated cumulative return chart vs always-long and random baselines
- Predictor correlation heatmap
- Modular codebase: each concern lives in its own file
- 40 pytest unit tests: 26 covering feature engineering functions, 14 pipeline smoke tests

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Core language |
| pandas | Data loading, merging, and all time-series manipulation |
| NumPy | Array operations |
| scikit-learn | Random Forest, accuracy/F1/MCC metrics, classification report |
| XGBoost | Gradient-boosted tree model for comparison |
| SHAP | Model explainability (beeswarm summary plot) |
| seaborn + matplotlib | All visualizations |
| pytest | Unit tests for feature engineering and pipeline |

---

## How to Run It Locally

**1. Clone the repo**

```bash
git clone <your-repo-url>
cd "Housing Model"
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Place the data files**

Download the CSV files listed in the Data Files section below and place them in the project root (same directory as `requirements.txt`).

**4. Run the model**

```bash
python -m housing_model
```

The script prints a summary table of metrics for each model, then saves all six plots to `outputs/`.

**5. Run the tests**

```bash
pytest tests/ -v
```

Tests use synthetic data only — no CSV files needed.

---

## Environment Variables / API Keys

None required. All data is loaded from local CSV files.

---

## Data Files

Download these and place them in the project root. They are not committed to the repo due to file size.

| File | Source | Description |
|------|--------|-------------|
| `MORTGAGE30US.csv` | FRED | 30-year fixed mortgage rate, weekly |
| `RRVRUSQ156N.csv` | FRED | Rental vacancy rate, quarterly |
| `CPIAUCSL.csv` | FRED | Consumer Price Index, monthly |
| `UNRATE.csv` | FRED | U.S. unemployment rate, monthly |
| `Metro_median_sale_price_uc_sfrcondo_sm_week.csv` | Zillow Research | National median sale price, weekly |
| `Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_month.csv` | Zillow Research | Home Value Index (middle tier), monthly |
| `Metro_zori_uc_sfrcondomfr_sm_sa_month.csv` | Zillow Research | Observed Rent Index, monthly |
| `Metro_invt_fs_uc_sfrcondo_sm_month.csv` | Zillow Research | Active for-sale listings, monthly |

FRED data: [fred.stlouisfed.org](https://fred.stlouisfed.org)
Zillow data: [zillow.com/research/data](https://www.zillow.com/research/data/)

---

## Project Structure

```
Housing Model/
├── housing_model/
│   ├── __init__.py        Package entry point
│   ├── config.py          All constants: paths, hyperparameters, feature lists
│   ├── data_loader.py     CSV loading and merging for all 8 data sources
│   ├── features.py        Feature engineering: lags, YoY, momentum, seasonality
│   ├── model.py           Walk-forward backtest, RF + XGBoost comparison
│   ├── evaluate.py        Accuracy, F1, MCC, confusion matrix, comparison table
│   ├── visualize.py       All six plot functions, saved to outputs/
│   └── main.py            Orchestrates the full pipeline end-to-end
│
├── tests/
│   ├── test_features.py   Unit tests for inflation_adjust, make_lags, make_yoy,
│   │                      add_momentum_features (26 tests, no CSV required)
│   └── test_pipeline.py   Smoke tests for predict() and backtest() using
│                          synthetic data (14 tests, no CSV required)
│
├── outputs/               Auto-created; all plot PNGs saved here
├── requirements.txt       Pinned dependencies
├── housing_model_v1.py    Original single-file version (preserved for reference)
└── README.md
```

---

## Model Card

**Task:** Binary classification — predict whether the U.S. national inflation-adjusted median home sale price will be higher 13 weeks (one quarter) from the current week.

**Training data:** FRED and Zillow national aggregates, weekly granularity, approximately 2008 to 2024 depending on data source coverage.

**Evaluation method:** Walk-forward (expanding-window) backtest. The first training window is 260 rows (~5 years). Each subsequent fold advances by 52 rows (1 year). No future data is ever present in any training window. Hyperparameters are tuned once on the initial training window using TimeSeriesSplit (3 splits) optimising Matthews Correlation Coefficient, then held fixed across all backtest folds.

**Metrics (backtest, national aggregate):**

| Model | Accuracy | F1 (Up) | MCC | Precision | Recall |
|-------|----------|---------|-----|-----------|--------|
| Random Forest (tuned) | 77.62% | 0.713 | 0.530 | 0.717 | 0.709 |
| XGBoost (tuned) | 73.92% | 0.644 | 0.443 | 0.692 | 0.602 |
| Ensemble (RF + XGB) | 74.38% | 0.658 | 0.455 | 0.690 | 0.630 |

**Known limitations:**
- All data is national aggregate. The model says nothing about individual metros, neighborhoods, or property types.
- The training period spans a prolonged bull market (2012-2021) followed by a rate shock (2022-2023). Performance during future regimes not represented in training is unknown.
- The binary target loses information about the magnitude of price changes. A prediction of "up" could mean +0.1% or +5%.
- The model does not account for transaction costs, taxes, or the practical illiquidity of housing.

---

## Built by

Viet
