import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup
import requests
import nltk

# --- INITIALIZATION ---
try:
    nltk.data.find('vaders_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

st.set_page_config(page_title="Aurora 2026 | Quant Lab", layout="wide")

# Custom UI Styling
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: #161b22; border-radius: 5px; color: white; }
    .stTabs [aria-selected="true"] { background-color: #58a6ff !important; }
    div[data-testid="stMetric"] { background-color: #1c2128; border: 1px solid #30363d; padding: 15px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("🛡️ Finova x ISTE")
    ticker = st.text_input("Enter Ticker (e.g., TSLA, BTC-USD)", value="AAPL").upper()
    time_period = st.selectbox("Historical Range", ["3mo", "6mo", "1y", "2y"])
    st.divider()
    st.write("**Engine Settings**")
    rsi_window = st.slider("RSI Window", 5, 30, 14)

# --- PHASE 1: DATA ENGINEERING ---
@st.cache_data
def get_data(symbol, period):
    # TODO: Download data using yfinance for the selected ticker and period
    data = pd.DataFrame() # Replace this line
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

df = get_data(ticker, time_period)

# --- PHASE 2: THE QUANTITATIVE ENGINE ---
def compute_indicators(data):
    # TODO: Calculate RSI (Relative Strength Index)
    # Hint: Use .diff() and .rolling()
    data['RSI'] = 0 # Placeholder
    
    # TODO: Calculate Bollinger Bands
    # Middle Band = 20-day MA | STD = 20-day Rolling Standard Deviation
    data['MA20'] = 0 # Placeholder
    data['Upper_BB'] = 0 # Placeholder
    data['Lower_BB'] = 0 # Placeholder
    
    # TODO: Calculate MACD and Signal Line
    # MACD = EMA12 - EMA26
    data['MACD'] = 0 # Placeholder
    data['Signal'] = 0 # Placeholder
    return data

df = compute_indicators(df)

# --- PHASE 3: THE AI BRAIN (LIVE NEWS SCRAPING) ---
def get_sentiment_analysis(symbol):
    sid = SentimentIntensityAnalyzer()
    
    # TODO: Implement Web Scraping using BeautifulSoup
    # 1. Define URL and Headers
    # 2. Use requests.get() to fetch the page
    # 3. Use BeautifulSoup to parse 'h3' tags
    headlines = [] 
    
    # Fallback Headlines (Keep these for stability)
    if not headlines:
        headlines = [f"{symbol} market outlook steady", f"Analysts watch {symbol}"]

    # TODO: Generate Sentiment Scores for each headline
    sentiment_data = []
    # Hint: Use sid.polarity_scores(h)
    
    sentiment_df = pd.DataFrame(sentiment_data)
    avg_compound = sentiment_df['Compound'].mean() if not sentiment_df.empty else 0
    return avg_compound, sentiment_df

avg_sentiment, sentiment_results_df = get_sentiment_analysis(ticker)

# --- PHASE 4: THE VERDICT ENGINE ---
def get_verdict(data, sentiment):
    last = data.iloc[-1]
    prev = data.iloc[-2]
    
    score = 0
    signals = []
    
    # TODO: Build the weighted logic
    # 1. If RSI < 30 -> +2 score
    # 2. If Price < Lower_BB -> +1 score
    # 3. If MACD crosses above Signal -> +2 score
    # 4. If Sentiment > 0.05 -> +1 score
    
    # TODO: Assign final verdict based on score
    # Example: score >= 3 -> STRONG BUY
    return "NEUTRAL", "#ffd700", signals 

verdict, v_color, signals_list = get_verdict(df, avg_sentiment)

# --- UI TABS (Layout Pre-Built) ---
t1, t2, t3, t4 = st.tabs(["📊 Market View", "📈 Indicators", "🤖 AI Sentiment", "🎯 Final Verdict"])

with t1:
    st.subheader(f"{ticker} Price & Volume Action")
    fig_market = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    fig_market.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
    fig_market.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Volume", marker_color='#1f77b4', opacity=0.5), row=2, col=1)
    fig_market.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False, showlegend=False)
    st.plotly_chart(fig_market, use_container_width=True)

with t2:
    st.subheader("Quantitative Technical Analysis")
    col_a, col_b = st.columns(2)
    with col_a:
        fig_bb = go.Figure()
        fig_bb.add_trace(go.Scatter(x=df.index, y=df['Upper_BB'], line=dict(color='rgba(173,216,230,0.2)'), name="Upper BB"))
        fig_bb.add_trace(go.Scatter(x=df.index, y=df['Lower_BB'], line=dict(color='rgba(173,216,230,0.2)'), fill='tonexty', fillcolor='rgba(173,216,230,0.1)', name="Lower BB"))
        fig_bb.add_trace(go.Scatter(x=df.index, y=df['Close'], line=dict(color='#58a6ff'), name="Price"))
        fig_bb.update_layout(title="Bollinger Bands", template="plotly_dark", height=400)
        st.plotly_chart(fig_bb, use_container_width=True)

    with col_b:
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], name="MACD", line=dict(color='cyan')))
        fig_macd.add_trace(go.Scatter(x=df.index, y=df['Signal'], name="Signal", line=dict(color='orange')))
        fig_macd.update_layout(title="MACD", template="plotly_dark", height=400)
        st.plotly_chart(fig_macd, use_container_width=True)
    
    st.divider()
    # RSI Gauge
    fig_rsi = go.Figure(go.Indicator(mode="gauge+number", value=df['RSI'].iloc[-1] if not df['RSI'].empty else 0, title={'text': "RSI Gauge"},
                        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#58a6ff"},
                               'steps': [{'range': [0, 30], 'color': "green"}, {'range': [70, 100], 'color': "red"}]}))
    fig_rsi.update_layout(template="plotly_dark", height=300)
    st.plotly_chart(fig_rsi, use_container_width=True)

with t3:
    st.subheader("AI Sentiment Breakdown")
    st.metric("Compound Mood Score", f"{avg_sentiment:.2f}")
    if not sentiment_results_df.empty:
        st.table(sentiment_results_df) 

with t4:
    st.markdown(f"<h1 style='text-align: center; color: {v_color};'>{verdict}</h1>", unsafe_allow_html=True)
    st.divider()
    st.write("### Engine Signal Breakdown")
    for s in signals_list:
        st.write(f"✔️ {s}")
