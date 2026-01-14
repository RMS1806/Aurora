import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import google.generativeai as genai
import finnhub
import nltk

# ---------------- INITIALISATION ----------------
try:
    nltk.data.find("vader_lexicon")
except LookupError:
    nltk.download("vader_lexicon")

st.set_page_config(page_title="Aurora 2026 | Quant Lab", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main { background: radial-gradient(circle, #1a1a2e 0%, #0f0f1b 100%); color: #e0e0e0; }
[data-testid="stSidebar"] { background-color: rgba(255, 255, 255, 0.05) !important; backdrop-filter: blur(15px); border-right: 1px solid #00ffcc; }
div[data-testid="stMetric"] { background-color: #1c2128; border: 1px solid #30363d; padding: 15px; border-radius: 10px; }
.stTabs [aria-selected="true"] { background-color: #58a6ff !important; }
.stButton>button { width: 100%; border-radius: 12px; border: 2px solid #00ffcc; background: transparent; color: #00ffcc; font-weight: bold; }
.stButton>button:hover { background: #00ffcc !important; color: #1a1a2e !important; box-shadow: 0 0 30px #00ffcc; }
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("Finova x ISTE")
    ticker = st.text_input("Enter Ticker", value="NVDA").upper()
    time_period = st.selectbox("Historical Range", ["3mo", "6mo", "1y", "2y"])
    
    st.divider()
    st.write("**API Configurations**")
    gemini_api_key = st.text_input("Gemini API Key", type="password")
    finnhub_api_key = st.text_input("Finnhub API Key", type="password")
    
    st.divider()
    # Parameters for indicators
    rsi_window = st.slider("RSI Window", 5, 30, 14)
    # TODO: Students can add more parameter sliders here for their custom indicators

# ---------------- PHASE 1: DATA ENGINEERING ----------------
@st.cache_data
def get_data(symbol, period):
    # TODO: Use yfinance to download data for the ticker and period
    data = pd.DataFrame() # Replace this line
    
    # Handle the yfinance Multi-Index columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

df = get_data(ticker, time_period)

# ---------------- PHASE 2: THE QUANTITATIVE ENGINE ----------------
def compute_indicators(data, rsi_win):
    # TODO: Calculate RSI (Relative Strength Index)
    # Hint: Use price diffs, average gains, and average losses
    data["RSI"] = 0 
    
    # TODO: Calculate Bollinger Bands
    # Middle Band (20-day MA), Upper Band, and Lower Band
    data["MA20"] = 0
    data["Upper_BB"] = 0
    data["Lower_BB"] = 0

    # TODO: Calculate MACD and Signal Line
    # MACD = EMA12 - EMA26
    data["MACD"] = 0
    data["Signal"] = 0

    # ---------------- CUSTOM INDICATOR SPACE ----------------
    # TODO: Students! Add your own indicators here (e.g., ADX, ATR, Fibonacci)
    # Reference the provided notebook for math logic
    
    return data

df = compute_indicators(df, rsi_window)

# ---------------- PHASE 3: AI NEWS & SENTIMENT ----------------
@st.cache_data
def get_finnhub_news(symbol, api_key, limit=10):
    # TODO: Connect to Finnhub and fetch news headlines
    # Return a list of headline strings
    return []

def get_sentiment_analysis(symbol, api_key):
    sid = SentimentIntensityAnalyzer()
    headlines = get_finnhub_news(symbol, api_key)
    
    # Handle empty news
    if not headlines:
        headlines = [f"{symbol} outlook stable", "Market monitoring volatility"]
    
    # TODO: Iterate through headlines and calculate VADER scores
    rows = []
    # Hint: Use sid.polarity_scores() and extract 'compound'
    
    sent_df = pd.DataFrame(rows)
    # Return average compound score and the dataframe
    return 0, sent_df

avg_sentiment, sentiment_df = get_sentiment_analysis(ticker, finnhub_api_key)

# ---------------- PHASE 4: VERDICT & GEN-AI REPORT ----------------
def get_verdict_score(data, sentiment):
    # TODO: Create your own weighted scoring logic
    # Combine RSI, BB, MACD, and Sentiment to create a final recommendation
    
    # Placeholder return
    return "NEUTRAL", "#ffd700"

verdict_str, verdict_color = get_verdict_score(df, avg_sentiment)

def generate_gemini_report(api_key, ticker, price, rsi, sentiment, verdict):
    # TODO: Configure Gemini and create a prompt for a Quant Analyst report
    # Pass market data as context to the model
    return "AI Report will appear here after implementation."

# ---------------- UI TABS ----------------
t1, t2, t3, t4 = st.tabs(["Market View", "Indicators", "AI Sentiment", "AI Analyst Report"])

with t1:
    # Price Chart Implementation
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.8, 0.2], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume"), row=2, col=1)
    fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with t2:
    # Unified Indicators Chart
    ind_fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.4, 0.3, 0.3])
    
    # TODO: Plot Bollinger Bands on Row 1
    # TODO: Plot RSI Oscillator on Row 2
    # TODO: Plot MACD and Signal Line on Row 3
    
    # TODO: (Extra) If you added custom indicators, add a Row 4 here!
    
    ind_fig.update_layout(template="plotly_dark", height=800, showlegend=False)
    st.plotly_chart(ind_fig, use_container_width=True)

with t3:
    st.metric("Aggregate Sentiment Score", f"{avg_sentiment:.2f}")
    st.dataframe(sentiment_df, use_container_width=True)

with t4:
    st.markdown(f"<h1 style='text-align:center;color:{verdict_color};'>{verdict_str}</h1>", unsafe_allow_html=True)
    st.divider()
    if st.button("Generate AI Analyst Report"):
        with st.spinner("Analyzing market microstructure..."):
            # TODO: Call the gemini report function
            report = "Function not implemented yet"
            st.markdown(report)
