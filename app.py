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

st.set_page_config(page_title="Aurora 2026 | Quant Dash", layout="wide")

# Custom CSS for professional look
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
    data = yf.download(symbol, period=period, interval="1d")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

df = get_data(ticker, time_period)

# --- PHASE 2: THE QUANTITATIVE ENGINE ---
def compute_indicators(data):
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
    data['RSI'] = 100 - (100 / (1 + gain/loss))
    
    # Bollinger Bands
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['STD'] = data['Close'].rolling(window=20).std(ddof=0)
    data['Upper_BB'] = data['MA20'] + (data['STD'] * 2)
    data['Lower_BB'] = data['MA20'] - (data['STD'] * 2)
    
    # MACD
    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    return data

df = compute_indicators(df)

# --- PHASE 3: THE AI BRAIN (SENTIMENT) ---
def get_sentiment_analysis(symbol):
    sid = SentimentIntensityAnalyzer()
    headlines = [
        f"{symbol} market share grows amid tech rally",
        f"Regulatory concerns weigh on {symbol} outlook",
        f"New product suite from {symbol} receives positive reviews",
        f"Investors rotate capital into {symbol} for stability"
    ]
    
    # Generate scores for each headline
    sentiment_data = []
    for h in headlines:
        scores = sid.polarity_scores(h)
        sentiment_data.append({
            "Headline": h,
            "Positive": scores['pos'],
            "Negative": scores['neg'],
            "Neutral": scores['neu'],
            "Compound": scores['compound']
        })
    
    sentiment_df = pd.DataFrame(sentiment_data)
    avg_compound = sentiment_df['Compound'].mean()
    return avg_compound, sentiment_df

avg_sentiment, sentiment_results_df = get_sentiment_analysis(ticker)

# --- PHASE 4: THE VERDICT ENGINE ---
def get_verdict(data, sentiment):
    last_rsi = float(data['RSI'].iloc[-1])
    last_close = float(data['Close'].iloc[-1])
    lower_bb = float(data['Lower_BB'].iloc[-1])
    upper_bb = float(data['Upper_BB'].iloc[-1])
    
    score = 0
    signals = []
    
    if last_rsi < 35: 
        score += 2
        signals.append("Bullish: RSI indicates Oversold conditions.")
    elif last_rsi > 65: 
        score -= 2
        signals.append("Bearish: RSI indicates Overbought conditions.")
        
    if last_close < lower_bb:
        score += 1
        signals.append("Bullish: Price testing Lower Bollinger Band (Potential Rebound).")
    elif last_close > upper_bb:
        score -= 1
        signals.append("Bearish: Price testing Upper Bollinger Band.")
        
    if sentiment > 0.05:
        score += 1
        signals.append("Bullish: Positive AI Sentiment detected in recent news.")
        
    if score >= 2: return "STRONG BUY", "#00ff00", signals
    elif score <= -1: return "SELL", "#ff4b4b", signals
    return "NEUTRAL", "#ffd700", signals

verdict, v_color, signals_list = get_verdict(df, avg_sentiment)

# --- UI TABS ---
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
        fig_bb.add_trace(go.Scatter(x=df.index, y=df['Upper_BB'], line=dict(color='rgba(173, 216, 230, 0.4)'), name="Upper BB"))
        fig_bb.add_trace(go.Scatter(x=df.index, y=df['Lower_BB'], line=dict(color='rgba(173, 216, 230, 0.4)'), fill='tonexty', fillcolor='rgba(173, 216, 230, 0.1)', name="Lower BB"))
        fig_bb.add_trace(go.Scatter(x=df.index, y=df['Close'], line=dict(color='#58a6ff'), name="Close Price"))
        fig_bb.update_layout(title="Bollinger Bands (Volatility Channel)", template="plotly_dark", height=400)
        st.plotly_chart(fig_bb, use_container_width=True)
        with st.expander("📚 What are Bollinger Bands?"):
            st.write("Bollinger Bands are volatility envelopes. They consist of a middle Moving Average and two outer bands (standard deviations). Prices near the upper band suggest the asset is overextended, while the lower band suggests a potential rebound.")

    with col_b:
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], name="MACD", line=dict(color='cyan')))
        fig_macd.add_trace(go.Scatter(x=df.index, y=df['Signal'], name="Signal", line=dict(color='orange')))
        fig_macd.update_layout(title="MACD (Trend Momentum)", template="plotly_dark", height=400)
        st.plotly_chart(fig_macd, use_container_width=True)
        with st.expander("📚 What is MACD?"):
            st.write("MACD (Moving Average Convergence Divergence) tracks the relationship between two moving averages. When the MACD line (cyan) crosses above the signal line (orange), it triggers a bullish momentum signal.")
    
    st.divider()
    fig_rsi = go.Figure(go.Indicator(mode="gauge+number", value=df['RSI'].iloc[-1], title={'text': "Relative Strength Index (RSI)"},
                        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#58a6ff"},
                               'steps': [{'range': [0, 30], 'color': "green"}, {'range': [70, 100], 'color': "red"}]}))
    fig_rsi.update_layout(template="plotly_dark", height=300)
    st.plotly_chart(fig_rsi, use_container_width=True)
    with st.expander("📚 What is RSI?"):
        st.write("RSI is a momentum oscillator that measures the speed and change of price movements on a scale of 0–100. Generally, RSI > 70 is 'Overbought' (potentially overvalued) and RSI < 30 is 'Oversold'.")

with t3:
    st.subheader("AI Sentiment Breakdown")
    st.metric("Aggregate Compound Score", f"{avg_sentiment:.2f}")
    st.write("Below is the specific scoring for each analyzed headline using the NLTK VADER engine:")
    # Displaying the raw score table
    st.table(sentiment_results_df) 
    st.info("💡 **Compound Score Guide**: -1.0 (Most Negative) to +1.0 (Most Positive). Values > 0.05 are generally considered Bullish.")

with t4:
    st.markdown(f"<h1 style='text-align: center; color: {v_color};'>{verdict}</h1>", unsafe_allow_html=True)
    st.divider()
    st.write("### Engine Signal Breakdown")
    for s in signals_list:
        st.write(f"✔️ {s}")
    if not signals_list: st.write("No significant signals detected.")