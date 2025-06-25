# run_trading_bot.py

import os
import sys
import pandas as pd
import numpy as np

# Ensure the 'src' folder is accessible
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from ingestion import fetch_data
from simple_strategy import get_signals_for_tickers
from sheets_logger import log_to_named_sheet
from ml_model import fetch_and_prepare, train_model

import math
import json

def sanitize_dataframe(df, debug=False):
    """
    Ensures all values are JSON-serializable, safe for Google Sheets.
    """
    def clean_value(val):
        if isinstance(val, float):
            if math.isnan(val) or math.isinf(val) or abs(val) > 1e308:
                return None
            return round(val, 4)
        elif isinstance(val, (np.integer, int)):
            return int(val)
        elif isinstance(val, (np.floating,)):
            return round(float(val), 4)
        elif isinstance(val, pd.Timestamp):
            return val.strftime("%Y-%m-%d %H:%M:%S")
        return val

    cleaned_df = df.applymap(clean_value)

    if debug:
        for i, row in cleaned_df.iterrows():
            try:
                json.dumps(row.to_dict(), allow_nan=False)
            except ValueError as e:
                print(f"\n Invalid JSON in row {i}:")
                print(row.to_dict())
                raise

    return cleaned_df

def main():
    # --- Configuration ---
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
    if not buy_signals.empty:
        log_to_named_sheet(sheet_name, signal_sheet, sanitize_dataframe(buy_signals), json_path)
    else:
        print("⚠️ No Buy Signals found to log.")

    # --- 6. Train ML Model for each ticker ---
    for ticker in tickers:
        print(f"\n🔍 Training ML model for {ticker}...")
        df = fetch_and_prepare(ticker)
        train_model(df)

if __name__ == "__main__":
    main()
