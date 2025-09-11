# ml_train.py

from datetime import date
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, mean_squared_error

from feature_engineering import get_features_with_partner
from data_splits import split_model_holdout
from model_selection import ModelSelector
from portfoliosim import simulate_long_short_portfolio

import config
from config import BASE_TICKER

# 1️⃣ Feature engineering
merged_df, partner = get_features_with_partner(
    BASE_TICKER,
    config.START_DATE,
    config.END_DATE,
    auto_adjust=config.AUTO_ADJUST
)

print(f"Most correlated ticker to {BASE_TICKER}: {partner}")
print(merged_df.head())

# 2️⃣ Train/holdout split
X_model, Y_model, X_holdout, Y_holdout = split_model_holdout(merged_df, BASE_TICKER, partner)
print("Model set shape:", X_model.shape, Y_model.shape)
print("Holdout set shape:", X_holdout.shape, Y_holdout.shape)

# 3️⃣ Model selection & tuning
selector = ModelSelector(X_model, Y_model, base_ticker=BASE_TICKER, random_iter=20)
selector.tune_models()
results, ranked, top_two = selector.compare_models()

print("\nTuned Model Performance:", results)
print("\nRanked Models:", ranked)
print(f"\nTop 2 Models: {top_two[0][0]} and {top_two[1][0]}")

# 4️⃣ Retrain top 2 and evaluate on holdout
for name in [m[0] for m in top_two]:
    model = selector.tuned_models[name]
    model.fit(X_model, Y_model)
    holdout_preds = model.predict(X_holdout)

    acc = accuracy_score(Y_holdout, holdout_preds)
    prec = precision_score(Y_holdout, holdout_preds, zero_division=0)
    mse = mean_squared_error(Y_holdout, holdout_preds)

    print(f"{name} on Holdout (2023) → "
          f"Accuracy: {acc:.4f}, "
          f"Precision: {prec:.4f}, "
          f"MSE: {mse:.4f}")

# 5️⃣ Portfolio simulations (long–short + buy & hold)
holdout_dates = X_holdout.index
dt_preds = selector.tuned_models['DecisionTree'].predict(X_holdout)
gb_preds = selector.tuned_models['GradientBoosting'].predict(X_holdout)

dt_ls = simulate_long_short_portfolio(
    merged_df, holdout_dates, dt_preds, base_ticker=BASE_TICKER,
    starting_capital=10000, trans_cost_bp=10, annual_borrow_rate=0.03
)
gb_ls = simulate_long_short_portfolio(
    merged_df, holdout_dates, gb_preds, base_ticker=BASE_TICKER,
    starting_capital=10000, trans_cost_bp=10, annual_borrow_rate=0.03
)

# 6️⃣ Plot equity curves
plt.figure(figsize=(11, 6))
plt.plot(dt_ls["long_short"], label="DecisionTree (Long-Short)")
plt.plot(gb_ls["long_short"], label="GradientBoosting (Long-Short)")
plt.plot(dt_ls["buy_hold"], label="Buy & Hold", linestyle="--", color="gray")
plt.axhline(10000, color="black", linestyle=":", linewidth=1)
plt.title(f"Long–Short vs Buy & Hold — 2023 Holdout ({BASE_TICKER})")
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.legend()
plt.grid(True)
plt.show()

# 7️⃣ Summary stats helper
def summarize(name, series):
    total_ret = series.iloc[-1] / series.iloc[0] - 1.0
    daily = series.pct_change().dropna()
    ann_ret = (1.0 + daily.mean())**252 - 1.0
    ann_vol = daily.std() * np.sqrt(252)
    dd = (series / series.cummax() - 1.0).min()
    print(f"{name}: Total={total_ret:.2%}, AnnRet={ann_ret:.2%}, AnnVol={ann_vol:.2%}, MaxDD={dd:.2%}")

summarize("DT Long-Short", dt_ls["long_short"])
summarize("GB Long-Short", gb_ls["long_short"])
summarize("Buy & Hold", dt_ls["buy_hold"])
