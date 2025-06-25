import os
import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.test import SMA
from backtesting.lib import crossover
from ingestion import fetch_data

OUTPUT_FOLDER = "trade_logs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

class MyStrategy(Strategy):

    def init(self):
        # indicator setup
        close = pd.Series(self.data.Close, index=self.data.index)
        self.rsi = self.I(ta.rsi, close, length=14)
        self.sma1 = self.I(SMA, close, 20)
        self.sma2 = self.I(SMA, close, 50)

    def next(self):
        # trade logic
        if pd.isna(self.rsi[-1]):
            return

        if crossover(self.sma1, self.sma2):
            print("BUY ...")
            self.buy()
        elif self.position.is_long and crossover(self.sma2, self.sma1):
            print("SELL ...")
            self.position.close()


def run_and_optimize(ticker):
    df = fetch_data(ticker, "2020-01-01", "2025-06-20")
    bt = Backtest(df, MyStrategy, cash=10000, commission=0.002, exclusive_orders=True)
    stats = bt.run()
    print(f"\n*** {ticker} Base Stats ***\n{stats}")

    trades = stats._trades.copy()
    filename = f"{OUTPUT_FOLDER}/{ticker}_trade_log.csv"
    trades.to_csv(filename, index=False)
    print(f"✅ Trade log saved: {filename}")

    # Optional optimize()
    # ...

def main():
    for t in ["RELIANCE.NS", "TCS.NS", "INFY.NS"]:
        run_and_optimize(t)

if __name__ == "__main__":
    main()
