
# 📈 Algo Trading System with ML & Google Sheets Logging

This project is a **Python-based mini-algorithmic trading system** that:

* Fetches stock data (RELIANCE, TCS, INFY)
* Computes RSI and SMA indicators to generate buy signals
* Logs trades, profits, summaries, and signals to **Google Sheets**
* Trains an ML model (Decision Tree) on technical indicators to predict future price direction

---

## 🔧 Project Structure

```
Algo_Trading_System/
├── run_trading_bot.py          # Master script to run everything
├── src/
│   ├── ingestion.py            # Stock data fetcher
│   ├── simple_strategy.py      # RSI + Moving Average buy strategy
│   ├── ml_model.py             # ML model training (Decision Tree)
│   ├── sheets_logger.py        # Google Sheets logging logic
│   └── algo_sheets_api.json    # Google Sheets API service account key
├── requirements.txt            # List of dependencies
└── README.md                   # This file
```

---

## 🚀 How to Run

### 1. ✅ Clone the Repo

```bash
git clone <your_repo_url>
cd Algo_Trading_System
```

### 2. ✅ Create and Activate Virtual Environment (optional but recommended)

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac
```

### 3. ✅ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. ✅ Setup Google Sheets API

* Go to [Google Cloud Console](https://console.cloud.google.com/)
* Enable **Google Sheets API** and **Google Drive API**
* Create a **Service Account Key (JSON)**
* Share your Google Sheet with the **service account email** (e.g., `abc@project.iam.gserviceaccount.com`) with **Editor** permissions
* Rename the file to `algo_sheets_api.json` and place it in `src/` folder

### 5. ✅ Add Your Google Sheet Info

Edit the following in `run_trading_bot.py`:

```python
sheet_name = "Algo_Trading_Sheets"      # Your Google Sheet name
data_sheet = "Ingested_Data"
summary_sheet = "Summary"
signal_sheet = "Buy_Signals"
```

### 6. ✅ Run the Bot

```bash
python run_trading_bot.py
```

---

## 📊 Output to Google Sheets

Your Google Sheet will have:

* **Ingested\_Data** – Raw stock data with profits
* **Summary** – Win/Loss/Profit stats per ticker
* **Buy\_Signals** – Detected trade signals from RSI + SMA strategy

---

## 💡 Strategy Logic

The strategy buys when:

* RSI < 30 (oversold)
* SMA20 crosses above SMA50 (bullish signal)

---

## 🤖 ML Model

A Decision Tree model is trained on:

* RSI, MACD, ATR, Bollinger Bands, Williams %R
* Predicts whether next day's close will be up or down

---

## 📦 requirements.txt

```txt
pandas
numpy
matplotlib
pandas_ta
scikit-learn
gspread
oauth2client
yfinance
```

Install using:

```bash
pip install -r requirements.txt
```

---

## 📌 Notes

* Ensure no `NaN`, `inf`, or invalid values are passed to Google Sheets — already handled in `sanitize_dataframe()` function.
* If you hit `InvalidJSONError`, ensure all numeric values are finite and all timestamps are formatted.

