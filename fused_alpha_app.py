# Fused Alpha Engine - Streamlit Dashboard

# === Imports ===
import yfinance as yf
from nsetools import Nse
import pandas as pd
import numpy as np
import ta
from ta.momentum import RSIIndicator
from ta.trend import MACD, ADXIndicator
from ta.volatility import BollingerBands
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import streamlit as st
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# === Step 1: Fetch Price + Technicals ===
def get_price_data(symbol='RELIANCE.NS', period='30d'):
    df = yf.download(symbol, period=period, interval='1d')
    df.dropna(inplace=True)
    df['RSI'] = RSIIndicator(df['Close']).rsi()
    df['MACD'] = MACD(df['Close']).macd_diff()
    df['ADX'] = ADXIndicator(df['High'], df['Low'], df['Close']).adx()
    bb = BollingerBands(df['Close'])
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_lower'] = bb.bollinger_lband()
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
    return df

# === Step 2: Fundamentals (Mocked/Static) ===
def get_fundamentals(symbol='RELIANCE'):
    try:
        nse = Nse()
        info = nse.get_quote(symbol)
        return {
            'ROCE': info.get('returnOnCapitalEmployed', 12.5),
            'PEG': info.get('priceToEarnings', 18) / info.get('eps', 45),
            'DebtEquity': info.get('debt', 0.6),
            'PromoterHolding': info.get('promotorHolding', 50)
        }
    except:
        return {
            'ROCE': 12.5,
            'PEG': 0.4,
            'DebtEquity': 0.6,
            'PromoterHolding': 50
        }

# === Step 3: Sentiment ===
def get_sentiment_score(texts):
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(text)['compound'] for text in texts]
    return np.mean(scores)

# === Step 4: Macro Triggers ===
def get_macro_triggers():
    return {'repo_rate': 6.5, 'cpi': 5.2, 'credit_growth': 14.3}

# === Step 5: Label Generation + Model Training ===
def train_models(df):
    X = df[['RSI', 'MACD', 'ADX', 'BB_lower', 'BB_upper']].dropna()
    y = np.where(df['Close'].shift(-5).fillna(method='ffill') > df['Close'] * 1.03, 1, 0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    svm = SVC()
    svm.fit(X_scaled[:-5], y[:-5])
    df['Future_Close'] = df['Close'].shift(-5)
    df['Target'] = df['Future_Close'] - df['Close']
    reg = LinearRegression()
    reg.fit(X_scaled[:-5], df['Target'].dropna()[:-5])
    return svm, reg, scaler

# === Step 6: Rule-Based Engine ===
def rule_based_signal(row):
    if row['Close'] <= row['BB_lower'] and row['RSI'] < 35:
        return 'BUY'
    elif row['RSI'] > 70:
        return 'SELL'
    else:
        return 'HOLD'

# === Streamlit UI ===
st.set_page_config(page_title="Fused Alpha Engine", layout="wide")
st.title("📊 Fused Alpha Engine - Swing Trade Dashboard")

symbol = st.text_input("Enter NSE Symbol (e.g., RELIANCE.NS)", value="RELIANCE.NS")

if st.button("Run Analysis"):
    df = get_price_data(symbol)
    fundamentals = get_fundamentals(symbol.split('.')[0])
    sentiment = get_sentiment_score([
        "Company expects higher margins",
        "Strong growth in telecom sector",
        "Management optimistic on expansion"
    ])
    macros = get_macro_triggers()
    svm, reg, scaler = train_models(df)
    df['Signal'] = df.apply(rule_based_signal, axis=1)
    df['StopLoss'] = df['Close'] - 1.5 * df['ATR']
    df['PredTarget'] = reg.predict(scaler.transform(df[['RSI', 'MACD', 'ADX', 'BB_lower', 'BB_upper']].dropna()))
    latest = df.iloc[-1]
    swing_score = svm.predict(scaler.transform([latest[['RSI', 'MACD', 'ADX', 'BB_lower', 'BB_upper']]]))[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("📈 Signal", latest['Signal'])
    col2.metric("🎯 Predicted Target", f"₹{latest['PredTarget']:.2f}")
    col3.metric("🛑 Stop Loss", f"₹{latest['StopLoss']:.2f}")

    st.subheader("📊 Technical Chart")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df['Close'], label='Close Price')
    ax.plot(df['BB_upper'], linestyle='--', label='BB Upper')
    ax.plot(df['BB_lower'], linestyle='--', label='BB Lower')
    ax.legend()
    st.pyplot(fig)

    st.subheader("📉 RSI and MACD")
    st.line_chart(df[['RSI', 'MACD']])

    st.subheader("📋 Fundamentals")
    st.write(fundamentals)

    st.subheader("🧠 Sentiment Score")
    st.metric("Sentiment (VADER)", round(sentiment, 2))

    st.subheader("🌐 Macro Triggers")
    st.write(macros)
