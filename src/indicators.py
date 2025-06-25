import pandas_ta as ta
def add_indicators(df):
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['DMA20'] = df['Close'].rolling(20).mean()
    df['DMA50'] = df['Close'].rolling(50).mean()
    macd = ta.macd(df['Close'])
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_Sig'] = macd['MACDs_12_26_9']
    return df
