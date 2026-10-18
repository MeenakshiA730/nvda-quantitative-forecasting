"""Regenerate all report charts for the NVDA sentiment/price project.

Usage:
    python make_charts.py                      # uses CSVs in the current folder
    python make_charts.py --tweets clean_tweets.csv   # also draws the sentiment-vs-price chart

Inputs
    nvda_walk_forward_results.csv  (from the notebook: date, close, actual_next_close,
                                    predicted_next_close, actual_return, strategy_return,
                                    equity, drawdown, position)
    nvda_forecast.csv              (Horizon, Forecast Date, Predicted Close)
    clean_tweets.csv   (optional)  (createdAt, sentiment)  one row per post
Outputs: PNG files in ./charts/
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--walk", default="nvda_walk_forward_results.csv")
ap.add_argument("--forecast", default="nvda_forecast.csv")
ap.add_argument("--tweets", default=None)
ap.add_argument("--out", default="charts")
ap.add_argument("--folds", type=int, default=5, help="TimeSeriesSplit n_splits used in the notebook")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GREEN, RED, GREY = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#7f7f7f"

w = pd.read_csv(args.walk, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
f = pd.read_csv(args.forecast, parse_dates=["Forecast Date"])

# Price series: closes plus the final realised next close
last_date = f.loc[f["Horizon"] == "n+0", "Forecast Date"].iloc[0]
price = pd.concat([w[["date", "close"]],
                   pd.DataFrame({"date": [last_date], "close": [w["actual_next_close"].iloc[-1]]})],
                  ignore_index=True)

# Benchmarks
w["bh_equity"] = (1 + w["actual_return"]).cumprod()
w["bh_drawdown"] = w["bh_equity"] / w["bh_equity"].cummax() - 1


def sharpe(x):
    return x.mean() / x.std(ddof=1) * np.sqrt(252)


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, name), bbox_inches="tight")
    plt.close()
    print("saved", name)


# 1. Price and forecast
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.plot(price["date"], price["close"], color=BLUE, lw=2, label="NVDA close")
ax.plot(f["Forecast Date"], f["Predicted Close"], "o--", color=ORANGE, label="Forecast n+0 to n+5")
ax.axvline(last_date, color=GREY, ls=":")
for _, r in f.iterrows():
    ax.annotate(r["Horizon"], (r["Forecast Date"], r["Predicted Close"]),
                textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7)
ax.set_title("NVDA close and 5-day forecast")
ax.set_ylabel("USD")
ax.legend()
save("01_price_and_forecast.png")

# 2. Equity curve vs buy and hold
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.plot(w["date"], w["equity"], color=GREEN, lw=2,
        label=f"Strategy (Sharpe {sharpe(w['strategy_return']):.2f})")
ax.plot(w["date"], w["bh_equity"], color=GREY, lw=2,
        label=f"Buy & hold (Sharpe {sharpe(w['actual_return']):.2f})")
ax.axhline(1, color="black", lw=0.8)
ax.set_title("Walk-forward equity curve (out-of-sample, no costs)")
ax.set_ylabel("Growth of 1.0")
ax.legend()
save("02_equity_vs_buy_and_hold.png")

# 3. Drawdown
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.fill_between(w["date"], w["drawdown"] * 100, 0, color=GREEN, alpha=0.4,
                label=f"Strategy (max {w['drawdown'].min()*100:.2f}%)")
ax.plot(w["date"], w["bh_drawdown"] * 100, color=GREY, lw=1.5,
        label=f"Buy & hold (max {w['bh_drawdown'].min()*100:.2f}%)")
ax.set_title("Drawdown")
ax.set_ylabel("%")
ax.legend(loc="lower left")
save("03_drawdown.png")

# 4. Price with long signals shaded
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.plot(w["date"], w["close"], color=BLUE, lw=1.8, label="NVDA close")
pos = w["position"].to_numpy().astype(bool)
ax.scatter(w.loc[pos, "date"], w.loc[pos, "close"], color=GREEN, s=22, zorder=3,
           label=f"Model says long next day ({pos.sum()} of {len(w)} days)")
ax.set_title("Where the model went long")
ax.set_ylabel("USD")
ax.legend()
save("04_long_signals_on_price.png")

# 5. Predicted vs actual next close
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].plot(w["date"], w["actual_next_close"], color=BLUE, lw=1.8, label="Actual next close")
axs[0].plot(w["date"], w["predicted_next_close"], color=ORANGE, lw=1.4, label="Model prediction")
axs[0].set_title("n+1 prediction vs actual")
axs[0].set_ylabel("USD")
axs[0].legend()
axs[0].tick_params(axis="x", rotation=30)
mae_m = (w["actual_next_close"] - w["predicted_next_close"]).abs().mean()
mae_b = (w["actual_next_close"] - w["close"]).abs().mean()
axs[1].bar(["Persistence", "Random Forest"], [mae_b, mae_m], color=[GREY, ORANGE])
for i, v in enumerate([mae_b, mae_m]):
    axs[1].text(i, v + 0.05, f"{v:.3f}", ha="center")
axs[1].set_title("n+1 MAE (lower is better)")
axs[1].set_ylabel("USD")
save("05_prediction_vs_actual_and_mae.png")

# 6. Return by fold
n = len(w)
size = n // args.folds
rows = []
for k in range(args.folds):
    g = w.iloc[k * size:(k + 1) * size if k < args.folds - 1 else n]
    rows.append({"fold": f"Fold {k+1}\n{g['date'].iloc[0]:%b %d}",
                 "strategy": ((1 + g["strategy_return"]).prod() - 1) * 100,
                 "bh": ((1 + g["actual_return"]).prod() - 1) * 100})
fd = pd.DataFrame(rows)
x = np.arange(len(fd))
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - 0.2, fd["strategy"], 0.4, color=GREEN, label="Strategy")
ax.bar(x + 0.2, fd["bh"], 0.4, color=GREY, label="Buy & hold")
ax.set_xticks(x)
ax.set_xticklabels(fd["fold"])
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("Return, %")
ax.set_title("Out-of-sample return by walk-forward fold")
ax.legend()
save("06_return_by_fold.png")

# 7. Daily return distribution when long vs flat
fig, ax = plt.subplots(figsize=(9, 5))
bins = np.linspace(w["actual_return"].min() * 100, w["actual_return"].max() * 100, 22)
ax.hist(w.loc[pos, "actual_return"] * 100, bins=bins, alpha=0.7, color=GREEN, label="Model long")
ax.hist(w.loc[~pos, "actual_return"] * 100, bins=bins, alpha=0.6, color=RED, label="Model flat")
ax.set_xlabel("Next-day NVDA return, %")
ax.set_ylabel("Days")
ax.set_title("Next-day returns by model signal")
ax.legend()
save("07_returns_by_signal.png")

# 8. (optional) Sentiment vs price, needs the tweet file
if args.tweets and os.path.exists(args.tweets):
    t = pd.read_csv(args.tweets)
    t["date"] = pd.to_datetime(t["createdAt"], utc=True).dt.tz_localize(None).dt.normalize()
    d = t.groupby("date").agg(sent=("sentiment", "mean"), n=("sentiment", "size")).reset_index()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1.3]})
    a1.plot(price["date"], price["close"], color=BLUE, lw=2, label="NVDA close")
    a1.set_ylabel("USD")
    b = a1.twinx()
    b.plot(d["date"], d["sent"].rolling(3, min_periods=1).mean(), color=ORANGE, lw=1.8,
           marker="o", ms=3, label="Mean X sentiment (3-day avg)")
    b.set_ylabel("VADER compound")
    b.axhline(0, color=GREY, lw=0.6)
    h1, l1 = a1.get_legend_handles_labels()
    h2, l2 = b.get_legend_handles_labels()
    a1.legend(h1 + h2, l1 + l2, loc="upper left")
    a1.set_title("X sentiment vs NVDA price")
    a2.bar(d["date"], d["n"], color=BLUE, alpha=0.6, width=0.8)
    a2.set_ylabel("Posts / day")
    save("08_sentiment_vs_price.png")
else:
    print("No --tweets file given: skipped 08_sentiment_vs_price.png")
