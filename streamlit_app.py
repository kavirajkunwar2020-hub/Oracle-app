import streamlit as st
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import time

# 1. SETUP MEMORY
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()

st.set_page_config(page_title="K_ALPHA ORACLE", layout="wide")
st.title("⚔️ K_ALPHA_ENGINE: PREDICTIVE MODE")

# 2. THE AI BRAIN (Random Forest)
def run_ai_engine(df):
    # Adding Indicators (Features)
    df['RSI'] = ta.rsi(df['close'], length=14)
    df['EMA'] = ta.ema(df['close'], length=20)
    df = df.dropna()
    
    # Target: Predict if next price is higher (1) or lower (0)
    X = df[['RSI', 'EMA']].values
    y = np.where(df['close'].shift(-1) > df['close'], 1, 0)
    
    # The Forest Committee (100 Trees)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X[:-1], y[:-1])
    
    # Final Prediction
    last_row = X[-1].reshape(1, -1)
    prediction = model.predict(last_row)[0]
    confidence = model.predict_proba(last_row)[0][prediction]
    
    return "UP" if prediction == 1 else "DOWN", round(confidence * 100, 2)

# 3. SIDEBAR CONTROLS
symbol = st.sidebar.selectbox("Select Market", ["BTCUSDT", "ETHUSDT", "SOLUSDT"])

if st.sidebar.button("🚀 RUN ORACLE"):
    # THE SPINNER
    with st.spinner("Consulting the Forest... Please wait."):
        time.sleep(2) # Simulating complex calculation
        
        # Mock Data (Replace with your Bybit fetch function)
        data = {
            'close': np.random.randint(40000, 45000, size=100),
            'high': np.random.randint(45000, 46000, size=100),
            'low': np.random.randint(39000, 40000, size=100)
        }
        st.session_state.df = pd.DataFrame(data)
        
        # Run AI
        signal, conf = run_ai_engine(st.session_state.df)
        st.session_state.signal = signal
        st.session_state.conf = conf

# 4. DISPLAY RESULTS
if not st.session_state.df.empty:
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(label="AI PREDICTION", value=st.session_state.signal)
        st.write(f"Confidence Level: **{st.session_state.conf}%**")
        st.progress(st.session_state.conf / 100)
        
    with col2:
        st.line_chart(st.session_state.df['close'].tail(20))
        
    st.success("Analysis Complete!")
else:
    st.info("👈 Use the Sidebar to launch the Engine.")
import streamlit as st
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import ccxt # For LIVE data

# 1. SETUP THE LOOK
st.set_page_config(page_title="K_ALPHA ORACLE", layout="wide", initial_sidebar_state="expanded")

# Custom CSS to make it look like a Terminal
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; padding: 10px; border_radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# 2. LIVE DATA ENGINE
def fetch_live_data(symbol):
    exchange = ccxt.binance()
    bars = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=100)
    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    return df

# 3. THE "FOREST" LOGIC
def run_ai_logic(df):
    df['RSI'] = ta.rsi(df['close'], length=14)
    df['EMA_20'] = ta.ema(df['close'], length=20)
    df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
    df = df.dropna()
    
    X = df[['RSI', 'EMA_20']]
    y = df['target']
    
    model = RandomForestClassifier(n_estimators=100, max_depth=5)
    model.fit(X[:-1], y[:-1])
    
    prediction = model.predict(X.tail(1))[0]
    prob = model.predict_proba(X.tail(1))[0]
    return "BULLISH" if prediction == 1 else "BEARISH", max(prob) * 100

# --- UI LAYOUT ---
st.title("⚔️ K-ALPHA COMMAND CENTER")

with st.sidebar:
    st.header("CONTROL PANEL")
    symbol = st.selectbox("ASSET", ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
    run_btn = st.button("EXECUTE ANALYSIS", use_container_width=True)

if run_btn:
    with st.status("Initializing Neural Forest...", expanded=True) as status:
        st.write("📡 Fetching Live Binance Data...")
        df = fetch_live_data(symbol)
        
        st.write("🌳 Generating 100 Decision Trees...")
        signal, confidence = run_ai_logic(df)
        
        status.update(label="Analysis Complete!", state="complete", expanded=False)

    # BIG NUMBERS AT THE TOP
    current_price = df['close'].iloc[-1]
    price_change = current_price - df['close'].iloc[-2]
    
    m1, m2, m3 = st.columns(3)
    m1.metric("CURRENT PRICE", f"${current_price:,}", f"{price_change:.2f}")
    m2.metric("AI SENTIMENT", signal)
    m3.metric("CONFIDENCE", f"{confidence:.1f}%")

    # THE CHART
    st.subheader("Market Momentum")
    st.area_chart(df.set_index('timestamp')['close'])
    
    # THE ALERTS
    if confidence > 80:
        st.success(f"🔥 HIGH CONFIDENCE SIGNAL: {signal}")
    else:
        st.warning("⚠️ Low Confidence - Neutral Market")

