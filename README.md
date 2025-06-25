
# 📈 Algo-Trading System with ML & Automation

A Python-based automated trading system that uses rule-based logic (RSI + Moving Average Crossover) and basic machine learning (Logistic Regression) to generate and analyze trade signals. The system is fully automated to ingest data, log results to Google Sheets, perform backtesting, and trigger alerts.

---

## 🚀 Features

- ✅ Fetches stock data using `yfinance`
- 📊 Implements RSI < 30 + 20-DMA & 50-DMA crossover strategy
- 🧠 ML model (Logistic Regression) predicts next-day stock movement
- 📉 Backtests strategy using `backtesting` module
- 📁 Logs all outputs to **Google Sheets** (Ingested Data, Summary, Buy Signals)
- 📡 (Optional) Sends alerts to Telegram bot
- 📅 Designed for daily automation & portfolio monitoring

---

## 🧱 Project Structure

```
Algo_Trading/
│
├── run_trading_bot.py             # Master script: runs full pipeline
├── backtest.py                    # Runs backtest for all tickers
├── requirements.txt               # Dependencies
├── README.md                      # Project overview
│
├── src/
│   ├── ingestion.py               # Fetches and merges stock data
│   ├── simple_strategy.py         # Applies technical strategy logic
│   ├── sheets_logger.py           # Logs outputs to Google Sheets
│   ├── ml_model.py                # Trains Logistic Regression model
│   └── algo_sheets_api.json       # Google Sheets API credentials
```

---

## ⚙️ Setup Instructions

### 🔧 Install Requirements

```bash
pip install -r requirements.txt
```

> Ensure you have Python 3.9+ and enable Google Sheets API (Service Account with `algo_sheets_api.json`).

---

### 📥 Data Ingestion

In `run_trading_bot.py`, you can configure:
```python
tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
start_date = "2022-01-01"
end_date = "2025-06-24"
```

---

## 📐 Strategy Logic

- Buy signal is generated if:
  - RSI < 30
  - AND 20-DMA crosses above 50-DMA

- Signals logged in Google Sheet under **Buy_Signals** tab

---

## 🧠 ML Component

- Input: RSI, MACD, Volume, etc.
- Output: Predicts next-day price direction (UP/DOWN)
- Model: Logistic Regression (`sklearn`)
- Accuracy is printed in the console

---

## 📉 Backtesting

Run this file for backtest:

```bash
python backtest.py
```

- Strategy: Same as above
- Output: Trade logs saved to `/trade_logs/{ticker}_trade_log.csv`

---

## 📤 Google Sheets Automation

- Requires `algo_sheets_api.json` for credentials
- Auto-uploads:
  - `Ingested_Data` – Raw daily price data
  - `Summary` – Trades, profit, win/loss ratio
  - `Buy_Signals` – All valid trade signals


## 🖥️ Console Output Sample

```
📊 RELIANCE.NS - Lowest RSI: 24.80
✅ Logged data to Google Sheet: Ingested_Data
✅ Buy signals found and logged
🔍 Training ML model for INFY.NS... Accuracy: 0.79
📉 Backtest summary saved: RELIANCE.NS_trade_log.csv
```


## 📚 References

- [yfinance](https://github.com/ranaroussi/yfinance)
- [Google Sheets API](https://developers.google.com/sheets/api)

---

## 📬 Contact

> Built by **Dhruv Saini** for H TO H  assignment and real-world strategy prototyping.  
Radhe Radhe 🙏
