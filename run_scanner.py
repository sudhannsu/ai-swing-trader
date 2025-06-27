# run_scanner.py

import pandas as pd
import yaml
from utils.fetch_data import get_price_data
from utils.signals import generate_trade_signals
from utils.sentiment import get_sentiment_score
from model.model_loader import load_model
from alerts.telegram_alerts import send_telegram
import joblib

def scan_stocks_and_alert():
    try:
        # Load watchlist safely
        df_symbols = pd.read_csv("watchlist.csv")
        if 'Symbol' not in df_symbols.columns:
            raise ValueError("❌ 'Symbol' column not found in watchlist.csv.")
        symbols = df_symbols['Symbol'].dropna().unique().tolist()
    except Exception as e:
        print(f"🚫 Failed to load watchlist: {e}")
        return

    try:
        model, scaler = load_model()
        with open("config.yaml") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"🚫 Failed to load model or config: {e}")
        return

    print(f"🔍 Scanning {len(symbols)} symbols...\n")

    for symbol in symbols:
        try:
            df = get_price_data(symbol)
            sentiment = get_sentiment_score(symbol)
            signal = generate_trade_signals(df, sentiment, config)

            if not signal or signal.get('Buy Price') is None:
                print(f"⚪ No signal for {symbol}")
                continue

            X = pd.DataFrame([signal['features']])
            X_scaled = scaler.transform(X)
            prob = model.predict_proba(X_scaled)[0][1]

            if prob >= config.get('alert_threshold', 0.75):
                buy = round(signal['Buy Price'], 2)
                tgt = round(signal['Target Price'], 2)
                sl = round(signal['Stop Loss'], 2)
                score = round(prob, 3)

                message = (
                    f"📈 {symbol} → Buy ₹{buy} | 🎯 Target ₹{tgt} | "
                    f"⛔ SL ₹{sl} | 🧠 Score: {score}"
                )

                if config.get('enable_alerts', False):
                    send_telegram(message, config['telegram_token'], config['telegram_chat_id'])

                print(f"✅ Alert: {symbol} → {score}")

            else:
                print(f"⚠️ Low confidence for {symbol} → Score: {round(prob, 3)}")

        except Exception as e:
            print(f"❌ {symbol} failed: {e}")

if __name__ == "__main__":
    scan_stocks_and_alert()
