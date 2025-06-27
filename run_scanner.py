# run_scanner.py
import pandas as pd
from utils.fetch_data import get_price_data
from utils.signals import generate_trade_signals
from utils.sentiment import get_sentiment_score
from model.model_loader import load_model
from alerts.telegram_alerts import send_telegram
import joblib
import yaml

def scan_stocks_and_alert():
    # Load watchlist
    symbols = pd.read_csv("watchlist.csv")["Symbol"]

    # Load model and config
    model, scaler = load_model()
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    for symbol in symbols:
        try:
            df = get_price_data(symbol)
            sentiment = get_sentiment_score(symbol)
            signal = generate_trade_signals(df, sentiment, config)

            if signal['Buy Price'] is None:
                continue  # Skip if no valid setup

            # Prepare features and score
            X = pd.DataFrame([signal['features']])
            X_scaled = scaler.transform(X)
            prob = model.predict_proba(X_scaled)[0][1]

            if prob >= config['alert_threshold']:
                message = f"📈 {symbol} → Buy ₹{signal['Buy Price']} | 🎯 Target ₹{signal['Target Price']} | ⛔ SL ₹{signal['Stop Loss']} | 🧠 Score: {round(prob, 3)}"
                if config['enable_alerts']:
                    send_telegram(message, config['telegram_token'], config['telegram_chat_id'])
                print(f"✅ Alert: {symbol} → {round(prob,3)}")

        except Exception as e:
            print(f"❌ {symbol} failed: {e}")

if __name__ == "__main__":
    scan_stocks_and_alert()
