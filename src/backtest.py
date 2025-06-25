import os
import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.test import SMA
from backtesting.lib import crossover
from src.ingestion import fetch_data

# Output folder for logs
OUTPUT_FOLDER = "trade_logs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


class MyStrategy(Strategy):
    def init(self):
        close = pd.Series(self.data.Close, index=self.data.index)
        self.rsi = self.I(ta.rsi, close, length=14)
        self.sma20 = self.I(SMA, close, 20)
        self.sma50 = self.I(SMA, close, 50)
        self.oversold_flag = False

    def next(self):
        if pd.isna(self.rsi[-1]) or pd.isna(self.sma20[-1]) or pd.isna(self.sma50[-1]):
            return

        # Step 1: Monitor RSI condition
        if self.rsi[-1] < 30:
            self.oversold_flag = True
            print(f"⚠️ RSI below 30 at {self.data.index[-1]}")

        # Step 2: Buy when crossover happens after RSI < 30
        if self.oversold_flag and crossover(self.sma20, self.sma50):
            print(f"✅ BUY at {self.data.index[-1]} | RSI: {self.rsi[-1]:.2f}")
            self.buy()
            self.oversold_flag = False  # Reset after buying

        # Step 3: Sell condition
        elif self.position.is_long and crossover(self.sma50, self.sma20):
            print(f"🔻 SELL at {self.data.index[-1]}")
            self.position.close()


def run_and_log(ticker):
    print(f"\n🚀 Running backtest for {ticker}")
    df = fetch_data(ticker, "2024-01-01", "2025-06-20")

    # Format the DataFrame correctly
    df = df.rename(columns={"Date": "datetime"})
    df = df[["datetime", "Open", "High", "Low", "Close", "Volume"]]
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)

    bt = Backtest(df, MyStrategy, cash=10000, commission=0.002, exclusive_orders=True)
    stats = bt.run()
    print(f"\n📊 {ticker} Performance Summary:\n{stats}")

    # Save trade log
    trades = stats._trades
    log_path = os.path.join(OUTPUT_FOLDER, f"{ticker}_trade_log.csv")
    trades.to_csv(log_path, index=False)
    print(f"✅ Saved trade log: {log_path}")
    return stats, trades


def main():
    tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
    for ticker in tickers:
        run_and_log(ticker)


if __name__ == "__main__":
    main()
