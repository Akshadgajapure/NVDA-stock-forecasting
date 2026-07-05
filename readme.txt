# NVDA Stock Price Forecasting

A time-series forecasting pipeline that predicts NVIDIA's next-day closing price using
15+ years of daily price data alongside macroeconomic indicators (S&P 500, Gold, Oil,
10-Year Treasury yield, VIX). Four modeling approaches are implemented and compared:
ARIMA, SARIMAX (with exogenous macro features), XGBoost, and LSTM.

## Pipeline

```
data/raw/stock_market_dataset.csv
        │
        ▼
clean.ipynb                  → handles multi-index columns, missing values, dedup
        │
        ▼
feature_engineering.ipynb    → SMA/EMA, MACD, rolling volatility, lag features, target
        │
        ▼
eda.ipynb                    → trend/seasonality inspection, correlation analysis
        │
        ▼
arima_model.ipynb            → walk-forward ARIMA(5,1,0)
sarima_model.ipynb           → SARIMAX with macro exogenous variables, walk-forward
xgboost_model.ipynb          → gradient-boosted trees, return-based target
LSTM_model.ipynb             → 2-layer LSTM, return-based target, 60-day sequences
```

## Repository structure

```
.
├── data/
│   ├── raw/                        # stock_market_dataset.csv (via yfinance)
│   └── processed/                  # cleaned / feature-engineered CSVs
├── main.ipynb                      # data collection (yfinance)
├── clean.ipynb
├── feature_engineering.ipynb
├── eda.ipynb
├── arima_model.ipynb
├── sarima_model.ipynb
├── xgboost_model.ipynb
├── LSTM_model.ipynb
├── requirements.txt
└── README.md
```

## Results

| Model | MAE | RMSE | R² | MAPE |
|---|---|---|---|---|
| Naive baseline (tomorrow = today) | — | — | — | — |
| Walk-Forward ARIMA | 2.09 | 3.18 | 0.9967 | 2.28 |
| Walk-Forward SARIMAX | 1.81 | 2.77 | 0.9975 | 1.91 |
| XGBoost (return-based) | 2.36 | 3.55 | 0.9959 | 2.56 |
| LSTM (return-based) | 2.74 | 3.93 | 0.9944 | 2.55 |

*Naive baseline row: fill in once computed (see Methodology notes below) — for a
near-random-walk series like a stock's close price, this baseline is deceptively hard
to beat, which is why it's included as the true reference point rather than "0" or "no model."*

SARIMAX with macro exogenous variables performs best across all metrics among the
modeled approaches.

## Methodology notes

- **Naive baseline**: every model above is compared against simply predicting
  "tomorrow's close = today's close." Raw stock prices are highly autocorrelated
  day-to-day, so this baseline is intentionally hard to beat — it's what separates
  genuine forecasting skill from an artifact of the series' own smoothness.
- **Walk-forward validation** (ARIMA, SARIMAX) refits the model at each step using all
  history up to that point, which is the honest way to validate a time-series model —
  it never sees future information at fit time.
- **Return-based framing** (XGBoost, LSTM): rather than predicting the next raw price,
  these models predict the next-day percentage return, which is then applied to the
  current price. This is a harder, more honest target than raw price, since it can't
  lean on the series' near-random-walk behavior as much.
- **Single train/test split** (XGBoost, LSTM): unlike ARIMA/SARIMAX, these use one
  80/20 split rather than walk-forward validation. This is a known limitation —
  see "Next Steps" below.

## Known limitations / next steps

- [ ] Add walk-forward (rolling-origin) cross-validation for XGBoost and LSTM to match
      the rigor of the ARIMA/SARIMAX evaluation.
- [ ] Hyperparameters (ARIMA order, XGBoost depth/learning rate, LSTM units) were set
      manually rather than tuned via grid/random search or Optuna.
- [ ] No statistical significance testing between model results — differences in MAE
      of a few cents may not be distinguishable from noise given the test set size.
- [ ] No transaction-cost or trading-strategy simulation — this is a price/return
      forecasting exercise, not a validated trading strategy.
- [ ] Add directional accuracy (up/down correctness) alongside MAE/RMSE for the
      return-based models, since that's often the more decision-relevant metric.

## Setup

```bash
pip install -r requirements.txt
```

`requirements.txt`:
```
pandas
numpy
matplotlib
seaborn
scikit-learn
statsmodels
xgboost
tensorflow
yfinance
```

## How to run

1. `main.ipynb` — pulls NVDA + macro data via `yfinance` into `data/raw/`
2. `clean.ipynb` → `feature_engineering.ipynb` → `eda.ipynb`
3. Any of the four model notebooks: `arima_model.ipynb`, `sarima_model.ipynb`,
   `xgboost_model.ipynb`, `LSTM_model.ipynb`

## Disclaimer

This is a forecasting research exercise, not investment advice. No trading strategy,
backtest with transaction costs, or live deployment is implemented or validated here.
