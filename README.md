
```markdown
# 📈 Algo-Trading System with ML & Automation

A Python-based modular algorithmic trading system that:
- Uses rule-based RSI + Moving Average crossover strategy
- Logs data and analytics to Google Sheets
- Trains machine learning models for stock movement prediction
- Supports Streamlit dashboard visualization
- Sends Telegram alerts for buy signals

---

## ✅ Features

### 📊 Trading Strategy
- Buy signal:
  - RSI < 30
  - 20-DMA crosses above 50-DMA
- Sell signal:
  - 20-DMA crosses below 50-DMA
- Backtesting for 3 NIFTY stocks (`RELIANCE.NS`, `TCS.NS`, `INFY.NS`) using `backtesting.py`

### 🧠 ML Module
- Features: RSI, MACD, ATR, Bollinger Bands, Williams %R
- Model: Decision Tree with GridSearchCV
- Predicts whether stock will go up next day
- Trained model saved under `final_model/`

### 📈 Streamlit Dashboard
- Run backtest for selected stock
- Train and evaluate ML model
- View buy signals and strategy performance
- Live equity curve, trade logs, stats

### 📤 Google Sheets Integration
- Trade logs
- Summary sheet (profit, win ratio, etc.)
- Buy signal sheet

### 🔔 Telegram Alerts
- Instant alerts when a buy signal is triggered
- Configured using BotFather and your chat ID

---

## 📁 Project Structure

```.
├── app.py                        # Streamlit UI
├── run_trading_bot.py           # Main script (scheduler, automation)
├── requirements.txt             # Dependencies
├── README.md                    # Project documentation
├── trade_logs/                  # CSV logs of backtests
├── final_model/                 # Trained ML models and scalers
├── src/
│   ├── backtest.py              # RSI + SMA backtesting logic
│   ├── ingestion.py             # Fetch stock data
│   ├── simple_strategy.py       # Rule-based buy signal generator
│   ├── ml_model.py              # ML training, prediction
│   ├── sheets_logger.py         # Google Sheets logging
│   ├── telegram_alert.py        # Buy signal Telegram bot
│   ├── indicators.py            # Technical indicators
│   ├── config.py                # Configuration constants
│   └── algo_sheets_api.json     # Google Sheets API credentials

````

---

## 🚀 How to Run

### 1. 🔧 Install Dependencies

```bash
pip install -r requirements.txt
````

### 2. 🧪 Run Main Bot Logic

```bash
python run_trading_bot.py
```

### 3. 📊 Launch Streamlit UI

```bash
streamlit run app.py
```



## 📌 Highlights

| Feature                | Done ✅ |
| ---------------------- | ------ |
| RSI + SMA Strategy     | ✅      |
| ML Model Integration   | ✅      |
| Backtesting & Logging  | ✅      |
| Google Sheets Logging  | ✅      |
| Streamlit Dashboard    | ✅      |
| Modular Code Structure | ✅      |


## ✍️ Author

**Dhruv Saini**
B.Tech Ai&ML | Machine Learning Enthusiast
Radhe Radhe 

```

