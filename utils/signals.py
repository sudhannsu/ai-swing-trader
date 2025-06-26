# utils/signals.py
import ta

def generate_trade_signals(df, sentiment, config):
    close = df['Close']
    rsi = ta.momentum.RSIIndicator(close).rsi().iloc[-1]
    bb = ta.volatility.BollingerBands(close)
    lower_band = bb.bollinger_lband().iloc[-1]
    upper_band = bb.bollinger_hband().iloc[-1]
    atr = ta.volatility.AverageTrueRange(df['High'], df['Low'], close).average_true_range().iloc[-1]

    last_price = close.iloc[-1]
    buy_price = last_price if rsi < 35 and last_price < lower_band else None
    target = round(last_price + 2 * atr, 2)
    stop_loss = round(last_price - 1.5 * atr, 2)

    features = {
        'RSI': rsi,
        'Sentiment': sentiment,
        'Price_vs_LowerBand': last_price - lower_band,
        'Price_vs_UpperBand': upper_band - last_price,
        'ATR': atr
    }

    return {
        'Buy Price': round(buy_price, 2) if buy_price else None,
        'Target Price': target,
        'Stop Loss': stop_loss,
        'RSI': round(rsi, 2),
        'features': features
    }