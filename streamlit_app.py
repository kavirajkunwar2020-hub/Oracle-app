import streamlit as st
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import ccxt

# --- INITIAL SETUP ---
st.set_page_config(page_title="K-ALPHA COMMAND CENTER", layout="wide")

# --- STEP 2: THE ENGINE FUNCTIONS (Put these in the middle) ---

def fetch_live_data(symbol):
    try:
        exchange = ccxt.binance()
        # Fetches last 100 hours of price data
        bars = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=100)
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        st.error(f"Data Fetch Error: {e}")
        return pd.DataFrame()

def run_ai_logic(df):
    # Create Technical Indicators
    df['RSI'] = ta.rsi(df['close'], length=14)
    df['EMA_20'] = ta.ema(df['close'], length=20)
    # Target: 1 if price went up, 0 if down
    df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
    df = df.dropna()
    
    # Random Forest Setup
    X = df[['RSI', 'EMA_20']]
    y = df['target']
    
    model = RandomForestClassifier(n_estimators=100, max_depth=5)
    model.fit(X[:-1], y[:-1]) # Train on all data except the last row
    
    # Predict the future
    prediction = model.predict(X.tail(1))[0]
    prob = model.predict_proba(X.tail(1))[0]
    return "BULLISH" if prediction == 1 else "BEARISH", max(prob) * 100

# --- THE USER INTERFACE (The Front Door) ---

st.title("⚔️ K-ALPHA COMMAND CENTER")

# Sidebar for controls
symbol = st.sidebar.selectbox("SELECT ASSET", ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
run_btn = st.sidebar.button("EXECUTE ANALYSIS")

if run_btn:
    with st.spinner("Oracle is analyzing live markets..."):
        # Use the functions from Step 2
        df = fetch_live_data(symbol)
        
        if not df.empty:
            signal, confidence = run_ai_logic(df)
            
            # Show Results in Columns (Not Basic!)
            col1, col2, col3 = st.columns(3)
            current_price = df['close'].iloc[-1]
            
            col1.metric("LIVE PRICE", f"${current_price:,}")
            col2.metric("SENTIMENT", signal)
            col3.metric("CONFIDENCE", f"{confidence:.1f}%")
            
            # Show the Chart
            st.subheader(f"{symbol} 1-Hour Momentum")
            st.line_chart(df.set_index('timestamp')['close'])
            
            if confidence > 80:
                st.success(f"🔥 STRONG {signal} SIGNAL DETECTED")
        else:
            st.error("Could not fetch data. Check your internet or symbol.")

