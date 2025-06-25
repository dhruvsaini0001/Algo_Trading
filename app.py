import os
import sys
import streamlit as st
import pandas as pd
import joblib

# Set up paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, 'src')
sys.path.append(SRC_DIR)

from backtest import run_and_log
from ingestion import fetch_data
from simple_strategy import get_signals_for_tickers
from ml_model import fetch_and_prepare, train_model, predict_next_signal

st.set_page_config(page_title="Algo Trading Dashboard", layout="wide")
st.title("📈 Algo-Trading System with ML & Automation")

tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
selected_ticker = st.selectbox("Select a stock to analyze", tickers)

# --- Backtesting ---
if st.button("📉 Run Backtest"):
    with st.spinner(f"Running backtest for {selected_ticker}..."):
        stats, trades = run_and_log(selected_ticker)
        if stats is not None and trades is not None:
            st.subheader("📊 Backtest Statistics")
            st.dataframe(stats.to_frame().T)

            st.subheader("📋 Trade Log")
            st.dataframe(trades)

            st.subheader("📈 Equity Curve")
            st.line_chart(stats['_equity_curve']['Equity'])
        else:
            st.warning("⚠️ No trades executed. Try a different stock or strategy.")

# --- ML Model Training ---
if st.button("🧠 Train ML Model"):
    df = fetch_and_prepare(selected_ticker)
    with st.spinner(f"Training ML model for {selected_ticker}..."):
        train_model(df)
        st.success("✅ ML model trained and saved successfully!")

# --- Buy Signal Detection ---
if st.button("🔍 Show Buy Signals"):
    with st.spinner("Fetching buy signals..."):
        signal_df = get_signals_for_tickers([selected_ticker], "2022-01-01", "2025-06-24")
        buy_signals = signal_df[signal_df["signal"] == 1]
        if not buy_signals.empty:
            st.dataframe(buy_signals[["Date", "Close", "RSI", "SMA20", "SMA50"]])
        else:
            st.warning("⚠️ No buy signals found.")

# --- Predict Next Signal ---
if st.button("📡 Predict Next Signal using ML"):
    try:
        signal, prob = predict_next_signal(selected_ticker)
        if signal == 1:
            st.success(f"📈 Prediction: BUY Signal ({prob*100:.2f}% confidence)")
        else:
            st.warning(f"📉 Prediction: SELL / HOLD Signal ({prob*100:.2f}% confidence)")
    except FileNotFoundError:
        st.error("❌ Model or Scaler not found. Please train the model first.")
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
