import streamlit as st
import pandas as pd
import pandas_ta as ta  # Swapped from talib for easier cloud install
from tradingview_ta import TA_Handler, Interval

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="K_ALPHA ORACLE", layout="wide")

# Securely fetch secrets from Streamlit's dashboard settings
try:
    API_KEY = st.secrets["BYBIT_KEY"]
    API_SECRET = st.secrets["BYBIT_SECRET"]
except:
    st.warning("API Keys not found. Please add them to Streamlit Secrets.")

# --- 2. THE DATA ENGINE ---
def get_market_data(symbol):
    try:
        handler = TA_Handler(
            symbol=symbol,
            screener="crypto",
            exchange="BINANCE",
            interval=Interval.INTERVAL_1_HOUR
        )
        # Fetching analysis and OHLCV data
        analysis = handler.get_analysis().indicators
        # We simulate the dataframe for the indicators
        df = pd.DataFrame([analysis]) 
        
        # Adding Indicators using pandas_ta (Better for Mobile/Cloud)
        # Note: In a real scenario, you'd fetch a full OHLCV list
        return analysis
    except Exception as e:
        st.error(f"Error fetching {symbol}: {e}")
        return None

# --- 3. UI LAYOUT ---
st.title("⚔️ K_ALPHA_ENGINE: WAR ROOM")

col1, col2 = st.columns(2)

with col1:
    st.header("Financial Assets")
    symbol = st.selectbox("Select Asset", ["BTCUSDT", "ETHUSDT", "AAPL"])
    data = get_market_data(symbol)
    if data:
        st.json(data) # Visualizing raw signal data first

with col2:
    st.header("Oracle Controls")
    risk = st.slider("Risk Tolerance %", 0.1, 5.0, 2.0)
    if st.button("Run Full Scan"):
        st.success(f"Scanning {symbol} at {risk}% Risk...")
