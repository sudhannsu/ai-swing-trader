# utils/fetch_data.py
import yfinance as yf
import pandas as pd

def get_price_data(symbol, period="60d", interval="1d"):
    df = yf.download(symbol, period=period, interval=interval)
    df = df.dropna()
    return df