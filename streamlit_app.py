# streamlit_app.py
import streamlit as st
import pandas as pd
from utils.fetch_data import get_price_data
from utils.signals import generate_trade_signals
from utils.sentiment import get_sentiment_score
from model.model_loader import load_model
from alerts.telegram_alerts import send_telegram
import yaml
from utils.nifty_symbols import fetch_top_nifty_symbols


st.set_page_config(page_title="AI Swing Trader", layout="wide")
st.title("🧠 AI-Powered Swing Trade Scanner")

# uploaded_file = st.file_uploader("📤 Upload stock list CSV (must contain column 'Symbol')", type="csv")
auto_scan = st.checkbox("🔄 Auto-scan Nifty stocks (no CSV)", value=True)

symbols = []

if auto_scan:
    symbols = fetch_top_nifty_symbols(limit=50)  # adjust limit if needed
    st.success(f"📈 Scanning top {len(symbols)} NSE stocks")
else:
    uploaded_file = st.file_uploader("📤 Upload your stock list CSV", type="csv")
    if uploaded_file:
        symbols = pd.read_csv(uploaded_file)["Symbol"].tolist()
    # Load model and config
    model, scaler = load_model()
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    results = []

    for symbol in symbols:
        try:
            df = get_price_data(symbol)
            sentiment = get_sentiment_score(symbol)
            signal = generate_trade_signals(df, sentiment, config)
            print(df)
            print(sentiment)
            print(signal)

            X = pd.DataFrame([signal['features']])
            X_scaled = scaler.transform(X)
            prob = model.predict_proba(X_scaled)[0][1]

            signal.update({
                'Symbol': symbol,
                'Swing Probability': round(prob, 3),
                'Confidence': '🔵 High' if prob > 0.8 else '🟢 Medium' if prob > 0.6 else '🟡 Low'
            })

            results.append(signal)

            if config['enable_alerts'] and prob > config['alert_threshold']:
                message = f"📈 {symbol} → Buy @ ₹{signal['Buy Price']} | Target: ₹{signal['Target Price']} | SL: ₹{signal['Stop Loss']}"
                send_telegram(message, config['telegram_token'], config['telegram_chat_id'])

        except Exception as e:
            st.error(f"❌ {symbol} failed: {e}")

    df_out = pd.DataFrame(results)
    st.dataframe(df_out.sort_values('Swing Probability', ascending=False), use_container_width=True)
