import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# --- INITIALIZATION ---
try:
    nltk.data.find('vaders_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

st.set_page_config(page_title="Aurora 2026 | Student Lab", layout="wide")

# Custom UI Styling
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div[data-testid="stMetric"] { background-color: #1c2128; border: 1px solid #30363d; padding: 15px; border-radius: 10px; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: #161b22; border-radius: 5px; color: white; }
    .stTabs [aria-selected="true"] { background-color: #58a6ff !important; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("🛡️ Finova x ISTE")
    ticker = st.text_input("Enter Ticker", value="AAPL").upper()
    time_period = st.selectbox("Historical Range", ["3mo", "6mo", "1y", "2y"])
    st.divider()
    rsi_window = st.slider("RSI Window", 5, 30, 14)

# --- PHASE 1: DATA ENGINEERING ---
@st.cache_data
def get_data(symbol, period):
    # TODO: Use yfinance to download data for the given symbol and period
    # Hint: data = yf.download(...)
    data = pd.DataFrame() # Placeholder
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

df = get_data(ticker, time_period)

# --- PHASE 2: THE QUANTITATIVE ENGINE ---
def compute_indicators(data):
    # TODO: Calculate RSI (Relative Strength Index)
    # 1. Calculate the price difference (delta)
    # 2. Separate gains and losses
    # 3. Calculate the average gain/loss and apply the RSI formula
    data['RSI'] = 50 # Placeholder
    
    # TODO: Calculate Bollinger Bands
    # Middle Band = 20-day Simple Moving Average
    # Upper/Lower Band = Middle Band +/- (2 * Standard Deviation)
    data['Upper_BB'] = data['Close'] * 1.1 # Placeholder
    data['Lower_BB'] = data['Close'] * 0.9 # Placeholder
    
    # TODO: Calculate MACD
    # MACD = 12-day EMA - 26-day EMA
    # Signal = 9-day EMA of MACD
    data['MACD'] = 0 # Placeholder
    data['Signal'] = 0 # Placeholder
    return data

df = compute_indicators(df)

# --- PHASE 3: THE AI BRAIN (SENTIMENT) ---
def get_sentiment_data(symbol):
    sid = SentimentIntensityAnalyzer()
    headlines = [
        f"{symbol} exceeds quarterly revenue expectations",
        f"Concerns grow over {symbol} supply chain issues",
        f"New AI integration announced for {symbol} products",
        f"Analysts maintain 'Hold' rating on {symbol} stock"
    ]
    
    results = []
    for h in headlines:
        # TODO: Use sid.polarity_scores(h) to get the sentiment dictionary
        # Extract the 'compound' score
        score = 0 # Placeholder
        
        results.append({
            "Headline": h, 
            "Score": score, 
            "Mood": "✅ Positive" if score > 0.05 else "❌ Negative" if score < -0.05 else "⚖️ Neutral"
        })
    return pd.DataFrame(results)

sentiment_df = get_sentiment_data(ticker)
avg_sentiment = sentiment_df['Score'].mean()

# --- PHASE 4: THE VERDICT ENGINE ---
def get_verdict(data, sentiment):
    # Get the most recent values
    last_rsi = float(data['RSI'].iloc[-1])
    last_close = float(data['Close'].iloc[-1])
    lower_bb = float(data['Lower_BB'].iloc[-1])
    upper_bb = float(data['Upper_BB'].iloc[-1])
    
    score = 0
    signals = []

    # TODO: Add logic for RSI scoring (+2 if oversold < 35, -2 if overbought > 65)
    
    # TODO: Add logic for Bollinger Bands (+1 if price < lower_bb)
    
    # TODO: Add logic for Sentiment (+1 if avg_sentiment > 0.05)
    
    # Determine Final Verdict
    v_text = "NEUTRAL"
    v_color = "#ffd700"
    if score >= 2:
        v_text, v_color = "STRONG BUY", "#00ff00"
    elif score <= -1:
        v_text, v_color = "SELL", "#ff4b4b"
        
    return v_text, v_color, signals

verdict, v_color, signals_list = get_verdict(df, avg_sentiment)

# --- UI TABS (Layout Pre-built for Students) ---
t1, t2, t3, t4 = st.tabs(["📊 Market", "📈 Indicators", "🤖 AI Brain", "🎯 Verdict"])

with t1:
    fig_p = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
    fig_p.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']), row=1, col=1)
    fig_p.add_trace(go.Bar(x=df.index, y=df['Volume'], opacity=0.3), row=2, col=1)
    fig_p.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_p, use_container_width=True)

with t2:
    st.subheader("Technical Deep-Dive")
    c1, c2 = st.columns(2)
    with c1:
        fig_bb = go.Figure()
        fig_bb.add_trace(go.Scatter(x=df.index, y=df['Upper_BB'], line=dict(color='rgba(173,216,230,0.4)'), name="Upper"))
        fig_bb.add_trace(go.Scatter(x=df.index, y=df['Lower_BB'], line=dict(color='rgba(173,216,230,0.4)'), fill='tonexty', name="Lower"))
        fig_bb.add_trace(go.Scatter(x=df.index, y=df['Close'], line=dict(color='#58a6ff'), name="Price"))
        st.plotly_chart(fig_bb, use_container_width=True)
    with c2:
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], name="MACD"))
        fig_macd.add_trace(go.Scatter(x=df.index, y=df['Signal'], name="Signal"))
        st.plotly_chart(fig_macd, use_container_width=True)

with t3:
    st.subheader("Headline Sentiment Breakdown")
    st.table(sentiment_df)

with t4:
    st.markdown(f"<h1 style='text-align: center; color: {v_color};'>{verdict}</h1>", unsafe_allow_html=True)
    for s in signals_list:
        st.write(f"✔️ {s}")
