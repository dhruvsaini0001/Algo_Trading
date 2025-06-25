import os
import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

from src.ingestion import fetch_data

# Constants
MODEL_DIR = "final_model"
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_COLS = ['RSI', 'MACD', 'ATR', 'BBU', 'BBL', 'WILLR', 'RSI_prev', 'MACD_prev']


def fetch_and_prepare(ticker):
    df = fetch_data(ticker, start="2024-01-01", end="2025-06-20")
    if df is None or df.empty:
        raise ValueError(f"Failed to fetch data for {ticker}")

    df.dropna(inplace=True)

    # Technical indicators
    df['RSI'] = ta.rsi(df['Close'], length=14)
    macd = ta.macd(df['Close'])
    if macd is None or 'MACD_12_26_9' not in macd:
        raise ValueError("MACD calculation failed.")

    df['MACD'] = macd['MACD_12_26_9']
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'])
    bb = ta.bbands(df['Close'], length=20)
    if bb is not None and bb.shape[1] >= 3:
        df['BBU'] = bb.iloc[:, 2]
        df['BBL'] = bb.iloc[:, 0]
    else:
        raise ValueError("Bollinger Bands calculation failed.")

    df['WILLR'] = ta.willr(df['High'], df['Low'], df['Close'])

    # Lag features
    df['RSI_prev'] = df['RSI'].shift(1)
    df['MACD_prev'] = df['MACD'].shift(1)

    # Target variable
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    df.dropna(inplace=True)
    return df


def train_model(df, ticker):
    X = df[FEATURE_COLS]
    y = df['Target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    params = {
        'max_depth': [4, 6, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 3, 5]
    }

    clf = GridSearchCV(DecisionTreeClassifier(random_state=42), params, cv=5)
    clf.fit(X_train, y_train)

    best_model = clf.best_estimator_
    y_pred = best_model.predict(X_test)

    print(f"\n📊 {ticker} - Best Parameters: {clf.best_params_}")
    print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("✅ Report:\n", classification_report(y_test, y_pred))

    # Save model and scaler
    joblib.dump(best_model, os.path.join(MODEL_DIR, f"{ticker}_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, f"{ticker}_scaler.pkl"))

    # Plot feature importances
    importances = pd.Series(best_model.feature_importances_, index=FEATURE_COLS)
    importances.nlargest(10).plot(kind='barh')
    plt.title(f"Top Features - {ticker}")
    plt.tight_layout()
    plt.show()

    return best_model, scaler


def predict_next_day(df_row, ticker):
    model_path = os.path.join(MODEL_DIR, f"{ticker}_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, f"{ticker}_scaler.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Model or scaler not found for {ticker}. Please train the model first.")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    X = scaler.transform([df_row[FEATURE_COLS].values])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]  # Probability of class 1 (buy)

    return pred, prob


def predict_next_signal(ticker):
    df = fetch_and_prepare(ticker)
    latest_row = df.iloc[-1]
    return predict_next_day(latest_row, ticker)
