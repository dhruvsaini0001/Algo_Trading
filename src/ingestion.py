import pandas as pd
import yfinance as yf

def fetch_data(tickers, start, end, interval="1d"):
    if isinstance(tickers, str):
        tickers = [tickers]

    df = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by='ticker'
    )

    df.index = df.index.tz_localize(None)
    combined = []

    for ticker in tickers:
        try:
            # Try multi-ticker format
            data = df[ticker][['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        except (KeyError, TypeError):
            try:
                # Fallback: single-ticker format
                data = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            except:
                print(f"⚠️ No valid data for {ticker}, filling empty frame.")
                data = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])

        data['Ticker'] = ticker
        data['Date'] = data.index
        combined.append(data)

    final_df = pd.concat(combined).reset_index(drop=True)
    return final_df[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]
