# simple_strategy.py

import pandas as pd
import pandas_ta as ta
from ingestion import fetch_data

def add_indicators(df):
    """
    Adds RSI, SMA20, SMA50 indicators to a single-ticker DataFrame.
    """
    df['RSI'] = df.ta.rsi(length=14)
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    return df

def generate_signals(df, rsi_threshold=30):
    """
    Adds 'signal' column with value 1 when RSI < threshold.
    """
    df = add_indicators(df)
    df['signal'] = 0

    # Only check RSI for now (less strict)
    df.loc[df['RSI'] < rsi_threshold, 'signal'] = 1

    return df

def get_signals_for_tickers(tickers, start_date, end_date, rsi_threshold=30):
    """
    Applies the strategy to each ticker and combines results.
    Works with flat DataFrame that includes 'Ticker'.
    """
    all_data = fetch_data(tickers, start=start_date, end=end_date)
    all_signals = []

    for ticker in tickers:
        df = all_data[all_data["Ticker"] == ticker].copy()

        if df.empty or "Close" not in df.columns:
            print(f"⚠️ No data for {ticker}")
            continue

        df = generate_signals(df, rsi_threshold=rsi_threshold)
        df['Ticker'] = ticker
        df['Date'] = df.index
        all_signals.append(df)

        # ✅ Show debug info
        print(f"\n📊 {ticker} - Lowest RSI: {df['RSI'].min():.2f}")
        print(df[df["signal"] == 1][['Date', 'Close', 'RSI', 'signal']].tail())

    if not all_signals:
        print("\n❌ No RSI < threshold signals generated.")
        return pd.DataFrame()

    return pd.concat(all_signals).reset_index(drop=True)

# --- Optional CLI Test ---
if __name__ == "__main__":
    tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
    df = get_signals_for_tickers(tickers, "2022-01-01", "2025-06-24", rsi_threshold=30)

    print("\n✅ Final Buy Signals:")
    signal_df = df[df["signal"] == 1].copy()
    signal_df["Date"] = signal_df["Date"].dt.strftime("%d-%b-%Y")
    print(signal_df[['Date', 'Ticker', 'Close', 'RSI', 'signal']].tail())
