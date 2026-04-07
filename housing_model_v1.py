# housing_model.py
import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance

# -----------------------------
# STEP 1: Load Federal Reserve Data
# -----------------------------
fed_files = ["MORTGAGE30US.csv", "RRVRUSQ156N.csv", "CPIAUCSL.csv"]
dfs = [pd.read_csv(f, parse_dates=True, index_col=0) for f in fed_files]
fed_data = pd.concat(dfs, axis=1).ffill().dropna()

# -----------------------------
# STEP 2: Load Zillow Sale Price & Home Value Index
# -----------------------------
zillow_files = [
    "Metro_median_sale_price_uc_sfrcondo_sm_week.csv",
    "Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_month.csv"
]
dfs = [pd.read_csv(f) for f in zillow_files]
dfs = [pd.DataFrame(df.iloc[0, 5:]) for df in dfs]
for df in dfs:
    df.index = pd.to_datetime(df.index)
    df["month"] = df.index.to_period("M")
price_data = dfs[0].merge(dfs[1], on="month")
price_data.index = dfs[0].index
del price_data["month"]
price_data.columns = ["price", "value"]

# Align dates
fed_data.index = fed_data.index + timedelta(days=2)
price_data = fed_data.merge(price_data, left_index=True, right_index=True, how="outer")
price_data.columns = ["interest", "vacancy", "cpi", "price", "value"]

# Inflation-adjusted values
price_data["adj_price"] = price_data["price"] / price_data["cpi"] * 100
price_data["adj_value"] = price_data["value"] / price_data["cpi"] * 100

# Target: price change next quarter
price_data["next_quarter"] = price_data["adj_price"].shift(-13)
price_data["change"] = (price_data["next_quarter"] > price_data["adj_price"]).astype(int)

# -----------------------------
# STEP 3: Load Zillow ZORI & Inventory
# -----------------------------
def reshape_zillow(df, value_name):
    df = df.drop(columns=['RegionID','SizeRank','RegionName','RegionType','StateName'], errors='ignore')
    df = df.melt(var_name='date', value_name=value_name)
    df['date'] = pd.to_datetime(df['date'])
    return df

# ZORI
zori = pd.read_csv("Metro_zori_uc_sfrcondomfr_sm_sa_month.csv")
zori = zori[zori["RegionName"]=="United States"]
zori = reshape_zillow(zori, "zori")

# Inventory
inventory = pd.read_csv("Metro_invt_fs_uc_sfrcondo_sm_month.csv")
inventory = inventory[inventory["RegionName"]=="United States"]
inventory = reshape_zillow(inventory, "inventory")

# Merge ZORI & Inventory into price_data
price_data = price_data.reset_index().rename(columns={"index":"date"})
price_data = price_data.merge(zori, on="date", how="left")
price_data = price_data.merge(inventory, on="date", how="left")
price_data = price_data.set_index("date").sort_index()

# -----------------------------
# STEP 4: Load Unemployment Rate (UNRATE)
# -----------------------------
# Handles FRED CSV with columns: observation_date,UNRATE
unrate = pd.read_csv("UNRATE.csv", parse_dates=["observation_date"])
unrate = unrate.rename(columns={"observation_date":"date", "UNRATE":"unrate"})
unrate = unrate.set_index("date")
price_data = price_data.reset_index().merge(unrate, on="date", how="left")
price_data = price_data.set_index("date").sort_index()

# -----------------------------
# STEP 5: Feature Engineering (Lags & YoY)
# -----------------------------
for col in ["zori","inventory","unrate"]:
    price_data[f"{col}_lag1"] = price_data[col].shift(1)
    price_data[f"{col}_lag3"] = price_data[col].shift(3)
    price_data[f"{col}_yoy"] = price_data[col] / price_data[col].shift(12) - 1

# -----------------------------
# STEP 6: Define Predictors and Target
# -----------------------------
predictors = ["interest","vacancy","adj_price","adj_value",
              "zori","inventory","zori_lag1","zori_lag3",
              "inventory_lag1","inventory_lag3","zori_yoy","inventory_yoy",
              "unrate","unrate_lag1","unrate_lag3","unrate_yoy"]

target = "change"

# -----------------------------
# STEP 7: Fill NaNs and convert predictors to numeric
# -----------------------------
price_data[predictors] = price_data[predictors].apply(pd.to_numeric, errors='coerce')
price_data[predictors] = price_data[predictors].ffill()

# -----------------------------
# STEP 8: Backtesting Functions
# -----------------------------
START = 260
STEP = 52

def predict(train, test, predictors, target):
    rf = RandomForestClassifier(min_samples_split=10, random_state=1)
    rf.fit(train[predictors], train[target])
    return rf.predict(test[predictors])

def backtest(data, predictors, target):
    all_preds = []
    for i in range(START, data.shape[0], STEP):
        train = data.iloc[:i]
        test = data.iloc[i:i+STEP]
        preds = predict(train, test, predictors, target)
        all_preds.append(preds)
    preds = np.concatenate(all_preds)
    return preds, accuracy_score(data.iloc[START:][target], preds)

# -----------------------------
# STEP 9: Run Backtest
# -----------------------------
preds, accuracy = backtest(price_data, predictors, target)
print(f"Backtest Accuracy: {accuracy:.2%}")

# -----------------------------
# STEP 10: Plot Predictions (Full Timeline)
# -----------------------------
pred_match = (preds == price_data[target].iloc[START:])
pred_colors = np.where(pred_match,"green","red")

plot_data = price_data.iloc[START:].copy()
plot_data.reset_index().plot.scatter(x="date", y="adj_price", color=pred_colors)
plt.title("Predicted Up vs. Down Price Movements (Full Timeline)")
plt.show()

# -----------------------------
# STEP 11: Feature Importance
# -----------------------------
rf = RandomForestClassifier(min_samples_split=10, random_state=1)
rf.fit(price_data[predictors], price_data[target])
result = permutation_importance(rf, price_data[predictors], price_data[target], n_repeats=10, random_state=1)
importance = pd.Series(result["importances_mean"], index=predictors).sort_values(ascending=False)
print("\nFeature Importance:")
print(importance)
