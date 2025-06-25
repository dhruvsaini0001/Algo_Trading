import os
import sys
import pandas as pd
import numpy as np
import math
import json
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.test import SMA
from backtesting.lib import crossover


# Add 'src' folder to sys.path



from src.ingestion import fetch_data
from src.simple_strategy import get_signals_for_tickers
from src.sheets_logger import log_to_named_sheet
from src.ml_model import fetch_and_prepare, train_model ,predict_next_signal

# Create logs folder for backtest results
BACKTEST_LOG_DIR = "trade_logs"
os.makedirs(BACKTEST_LOG_DIR, exist_ok=True)


def sanitize_dataframe(df, debug=False):
    def clean_value(val):
        if isinstance(val, float):
            if np.isnan(val) or np.isinf(val) or abs(val) > 1e308:
                return None
            return round(val, 4)
        elif isinstance(val, (np.integer, int)):
            return int(val)
        elif isinstance(val, (np.floating,)):
            return round(float(val), 4)
        elif isinstance(val, pd.Timestamp):
            return val.strftime("%Y-%m-%d %H:%M:%S")
        return val

    cleaned_df = df.astype(object).applymap(clean_value)

    if debug:
        valid_rows = []
        for i, row in cleaned_df.iterrows():
            try:
                json.dumps(row.to_dict(), allow_nan=False)
                valid_rows.append(row)
            except ValueError:
                print(f"\n❌ Invalid JSON in row {i}:", row.to_dict())
        cleaned_df = pd.DataFrame(valid_rows)

    return cleaned_df.reset_index(drop=True)


# --- Custom Strategy for Backtesting ---
class MyStrategy(Strategy):
    def init(self):
        close = pd.Series(self.data.Close, index=self.data.index)
        self.rsi = self.I(ta.rsi, close, length=14)
        self.sma20 = self.I(SMA, close, 20)
        self.sma50 = self.I(SMA, close, 50)

    def next(self):
        if pd.isna(self.rsi[-1]) or pd.isna(self.sma20[-1]) or pd.isna(self.sma50[-1]):
            return
        if self.rsi[-1] < 30 and crossover(self.sma20, self.sma50):
            self.buy()
        elif self.position.is_long and crossover(self.sma50, self.sma20):
            self.position.close()


def run_backtest(ticker):
    print(f"\n📉 Backtesting {ticker}")
    df = fetch_data(ticker, "2020-01-01", "2025-06-24")

    df = df.rename(columns={"Date": "datetime"})
    df = df[["datetime", "Open", "High", "Low", "Close", "Volume"]]
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)

    bt = Backtest(df, MyStrategy, cash=10000, commission=0.002, exclusive_orders=True)
    stats = bt.run()
    print(f"\n📊 {ticker} Backtest Summary:\n{stats}")

    trades = stats._trades
    trade_log_path = os.path.join(BACKTEST_LOG_DIR, f"{ticker}_trade_log.csv")
    trades.to_csv(trade_log_path, index=False)
    print(f"✅ Trade log saved to: {trade_log_path}")


def main():
    tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
    start_date = "2022-01-01"
    end_date = "2025-06-24"
    json_path = "src/algo_sheets_api.json"
    sheet_name = "Algo_Trading_Sheets"
    data_sheet = "Ingested_Data"
    summary_sheet = "Summary"
    signal_sheet = "Buy_Signals"

    # --- 1. Fetch Data ---
    all_data = fetch_data(tickers, start=start_date, end=end_date)

    # --- 2. Add Profit Columns ---
    all_data["Buy_Price"] = all_data["Open"]
    all_data["Sell_Price"] = all_data["Close"]
    all_data["Profit"] = all_data["Sell_Price"] - all_data["Buy_Price"]

    # --- 3. Log Trade Data ---
    log_to_named_sheet(sheet_name, data_sheet, sanitize_dataframe(all_data), json_path)

    # --- 4. Generate Summary ---
    summary_list = []
    for ticker, df in all_data.groupby("Ticker"):
        total_trades = len(df)
        win_trades = (df["Profit"] > 0).sum()
        loss_trades = (df["Profit"] <= 0).sum()
        total_profit = df["Profit"].sum()
        avg_profit = df["Profit"].mean()
        win_ratio = round((win_trades / total_trades) * 100, 2) if total_trades > 0 else 0.0

        summary_list.append({
            "Ticker": ticker,
            "Total Trades": total_trades,
            "Winning Trades": win_trades,
            "Losing Trades": loss_trades,
            "Total Profit": round(total_profit, 2),
            "Average Profit": round(avg_profit, 2),
            "Win Ratio (%)": win_ratio
        })

    summary_df = pd.DataFrame(summary_list)
    log_to_named_sheet(sheet_name, summary_sheet, sanitize_dataframe(summary_df), json_path)

    # --- 5. Generate Buy Signals ---
    signal_df = get_signals_for_tickers(tickers, start_date, end_date, rsi_threshold=30)
    signal_df["Date"] = pd.to_datetime(signal_df["Date"]).dt.strftime("%Y-%m-%d")
    buy_signals = signal_df[signal_df["signal"] == 1][["Ticker", "Date", "Close", "RSI", "SMA20", "SMA50", "signal"]]
    buy_signals = buy_signals.dropna(subset=["SMA20", "SMA50"])

    if not buy_signals.empty:
        log_to_named_sheet(sheet_name, signal_sheet, sanitize_dataframe(buy_signals, debug=True), json_path)
    else:
        print("⚠️ No Buy Signals found to log.")
    # --- 6. Train ML Models ---
    for ticker in tickers:
        print(f"\n🔍 Training ML model for {ticker}...")
        df = fetch_and_prepare(ticker)
        train_model(df)

    # --- 6.1 Predict Next Signal ---
    for ticker in tickers:
        print(f"\n🧠 Predicting next signal for {ticker} using ML model...")
        try:
            pred, prob = predict_next_signal(ticker)
            signal_type = "📈 BUY" if pred == 1 else "📉 SELL/HOLD"
            print(f"🔮 Prediction for {ticker}: {signal_type} (Confidence: {round(prob, 2)}%)")
        except Exception as e:
            print(f"❌ Prediction failed for {ticker}: {e}")


    # --- 7. Run Backtests ---
    for ticker in tickers:
        run_backtest(ticker)


if __name__ == "__main__":
    main()
