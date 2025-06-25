# ml_model.py

import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

from ingestion import fetch_data


def fetch_and_prepare(ticker):
    df = fetch_data(ticker, start="2024-01-01", end="2025-06-20")
    df.dropna(inplace=True)

    df['RSI'] = ta.rsi(df['Close'], length=14)
    macd = ta.macd(df['Close'])
    df['MACD'] = macd['MACD_12_26_9']
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'])
    bb = ta.bbands(df['Close'], length=20)
    df['BBU'] = bb.iloc[:, 2]
    df['BBL'] = bb.iloc[:, 0]
    df['WILLR'] = ta.willr(df['High'], df['Low'], df['Close'])

    df['RSI_prev'] = df['RSI'].shift(1)
    df['MACD_prev'] = df['MACD'].shift(1)
    df.dropna(inplace=True)

    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)

    return df


def train_model(df, model_path='src/ml_model.pkl'):
    feature_cols = ['RSI', 'MACD', 'ATR', 'BBU', 'BBL', 'WILLR', 'RSI_prev', 'MACD_prev']
    X = df[feature_cols]
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

    print("✅ Best Parameters:", clf.best_params_)
    print("✅ Accuracy:", accuracy_score(y_test, y_pred))
    print("✅ Report:\n", classification_report(y_test, y_pred))

    # Save model and scaler
    joblib.dump(best_model, model_path)
    joblib.dump(scaler, 'src/scaler.pkl')

    # Plot feature importance
    importances = pd.Series(best_model.feature_importances_, index=feature_cols)
    importances.nlargest(10).plot(kind='barh')
    plt.title("Top Features")
    plt.tight_layout()
    plt.show()

    return best_model, scaler


def predict_next_day(df_row, model_path='src/ml_model.pkl'):
    model = joblib.load(model_path)
    scaler = joblib.load('src/scaler.pkl')
    features = ['RSI', 'MACD', 'ATR', 'BBU', 'BBL', 'WILLR', 'RSI_prev', 'MACD_prev']
    X = scaler.transform([df_row[features].values])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]  # Probability of class 1 (buy)
    return pred, prob

def predict_next_signal(ticker, model_path='src/ml_model.pkl'):
    df = fetch_and_prepare(ticker)
    latest_row = df.iloc[-1]
    return predict_next_day(latest_row, model_path)

