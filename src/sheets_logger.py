import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
from ingestion import fetch_data
from simple_strategy import get_signals_for_tickers  # ✅ Import your strategy results

# --- 1. Google Auth ---
def authenticate_google(json_key_path):
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name(json_key_path, scope)
    client = gspread.authorize(creds)
    return client

# --- 2. Upload DataFrame to Specific Sheet ---
def log_to_named_sheet(sheet_name, worksheet_name, df, json_key_path):
    client = authenticate_google(json_key_path)

    try:
        sheet = client.open(sheet_name)
    except gspread.SpreadsheetNotFound:
        sheet = client.create(sheet_name)

    try:
        worksheet = sheet.worksheet(worksheet_name)
        worksheet.clear()
    except gspread.WorksheetNotFound:
        worksheet = sheet.add_worksheet(title=worksheet_name, rows=str(len(df)+10), cols=str(len(df.columns)+5))

    # Handle Timestamp columns for Google Sheets
    df = df.applymap(lambda x: x.strftime("%Y-%m-%d %H:%M:%S") if isinstance(x, pd.Timestamp) else x)

    worksheet.insert_row(df.columns.tolist(), index=1)
    worksheet.insert_rows(df.values.tolist(), row=2)

    print(f"✅ Logged data to {worksheet_name} in Google Sheet: {sheet_name}")

# --- 3. Main Execution ---
if __name__ == "__main__":
    json_path = "algo_sheets_api.json"
    sheet_name = "Algo_Trading_Sheets"

    # Config
    tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
    start_date = "2024-01-01"
    end_date = "2025-06-24"

    # 🔁 Raw Ingested Price Data
    all_data = pd.concat([
        fetch_data(ticker, start=start_date, end=end_date).assign(Ticker=ticker)
        for ticker in tickers
    ])
    all_data.reset_index(inplace=True)
    all_data = all_data[['Ticker'] + [col for col in all_data.columns if col != 'Ticker']]
    all_data["Buy_Price"] = all_data["Open"]
    all_data["Sell_Price"] = all_data["Close"]
    all_data["Profit"] = all_data["Sell_Price"] - all_data["Buy_Price"]

    log_to_named_sheet(sheet_name, "Ingested_Data", all_data, json_path)

    # 📊 Summary Per Ticker
    summary_list = []
    for ticker, df in all_data.groupby("Ticker"):
        total_trades = len(df)
        win_trades = (df["Profit"] > 0).sum()
        loss_trades = (df["Profit"] <= 0).sum()
        total_profit = df["Profit"].sum()
        avg_profit = df["Profit"].mean()
        win_ratio = round((win_trades / total_trades) * 100, 2)

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
    log_to_named_sheet(sheet_name, "Summary", summary_df, json_path)

    # 📈 Strategy Signals (RSI + MA Crossover)
    signal_df = get_signals_for_tickers(tickers, start_date, end_date)
    signal_df = signal_df[signal_df["signal"] == 1]  # Filter only buy signals

    log_to_named_sheet(sheet_name, "Buy_Signals", signal_df, json_path)
