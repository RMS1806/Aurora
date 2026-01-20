import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import finnhub
import nltk
from datetime import datetime, timedelta

# ---------------- INITIALISATION ----------------
try:
    nltk.data.find("vader_lexicon")
except LookupError:
    nltk.download("vader_lexicon")

st.set_page_config(page_title="Aurora 2026 | SentimentX Elite", layout="wide")

# ---------------- PROFESSIONAL TERMINAL UI ----------------
st.markdown("""
<style>
    .main { background-color: #0b0e14; color: #c9d1d9; font-family: 'Courier New', Courier, monospace; }
    [data-testid="stSidebar"] { background-color: #161b22 !important; border-right: 1px solid #30363d; }
    div[data-testid="stMetric"] { background-color: #0d1117; border: 1px solid #30363d; padding: 10px; border-radius: 4px; }
    .stTabs [data-baseweb="tab-list"] { background-color: #0b0e14; border-bottom: 1px solid #30363d; gap: 10px; }
    .stTabs [aria-selected="true"] { background-color: #1f6feb !important; color: white !important; border-radius: 4px 4px 0 0; }
    .stButton>button { width: 100%; border-radius: 4px; border: 1px solid #30363d; background: #21262d; color: #c9d1d9; transition: 0.3s; }
    .stButton>button:hover { border-color: #58a6ff; color: #58a6ff; background: #30363d; }
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR: CONFIGURATION ----------------
with st.sidebar:
    st.title("FINOVA X ISTE")
    st.subheader("TERMINAL CONTROLS")
    ticker = st.text_input("SYMBOL TICKER", value="NVDA").upper()
    time_period = st.selectbox("TIME HORIZON", ["3mo", "6mo", "1y", "2y"])
    
    st.divider()
    st.write("**API DATA FEED**")
    finnhub_key = st.text_input("FINNHUB API KEY", type="password")
    
    st.divider()
    rsi_period = st.slider("RSI LOOKBACK PERIOD", 5, 30, 14)

# ---------------- DATA ENGINE ----------------
@st.cache_data
def get_market_data(symbol, period):
    # Fetching OHLCV Data
    data = yf.download(symbol, period=period, interval="1d")
    # Flattening multi-index columns for consistent access
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

def add_technical_indicators(data, rsi_win):
    # RSI Calculation
    delta = data["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(rsi_win).mean()
    loss = -delta.where(delta < 0, 0).rolling(rsi_win).mean()
    data["RSI"] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    
    # Bollinger Bands
    data["MA20"] = data["Close"].rolling(20).mean()
    data["STD20"] = data["Close"].rolling(20).std(ddof=0)
    data["Upper_BB"] = data["MA20"] + 2 * data["STD20"]
    data["Lower_BB"] = data["MA20"] - 2 * data["STD20"]

    # MACD
    data["EMA12"] = data["Close"].ewm(span=12, adjust=False).mean()
    data["EMA26"] = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = data["EMA12"] - data["EMA26"]
    data["Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
    return data

# Process Pipeline
raw_df = get_market_data(ticker, time_period)
df = add_technical_indicators(raw_df, rsi_period)

# ---------------- NEWS & SENTIMENT ----------------
def get_sentiment(symbol, api_key):
    sid = SentimentIntensityAnalyzer()
    headlines = []
    
    if api_key:
        try:
            client = finnhub.Client(api_key=api_key)
            news = client.company_news(symbol, _from=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'), 
                                       to=datetime.now().strftime('%Y-%m-%d'))
            headlines = [item["headline"] for item in news if item.get("headline")][:10]
        except Exception: pass

    if not headlines:
        headlines = [f"{symbol} outlook stable", "Sector monitoring for volatility", "Institutional flow remains consistent"]
    
    results = []
    for h in headlines:
        score = sid.polarity_scores(h)["compound"] # Score between -1 and 1
        results.append({"Headline": h, "Score": score})
    
    sent_df = pd.DataFrame(results)
    return sent_df["Score"].mean(), sent_df

avg_sent, news_df = get_sentiment(ticker, finnhub_key)

# ---------------- VERDICT ENGINE (FULL TRANSPARENCY) ----------------
def get_detailed_verdict(data, sentiment):
    last = data.iloc[-1]
    prev = data.iloc[-2]
    score = 0
    breakdown = []
    
    # 1. RSI ANALYSIS
    rsi_val = last["RSI"]
    if rsi_val < 35:
        score += 2
        breakdown.append(f"RSI STATUS: OVERSOLD ({rsi_val:.2f}) | IMPACT: +2 PTS")
    elif rsi_val > 65:
        score -= 2
        breakdown.append(f"RSI STATUS: OVERBOUGHT ({rsi_val:.2f}) | IMPACT: -2 PTS")
    else:
        breakdown.append(f"RSI STATUS: NEUTRAL ({rsi_val:.2f}) | IMPACT: 0 PTS")

    # 2. BOLLINGER ANALYSIS
    price = last["Close"]
    if price <= last["Lower_BB"]:
        score += 1
        breakdown.append(f"PRICE POSITION: TESTING LOWER BAND (${price:.2f}) | IMPACT: +1 PT")
    elif price >= last["Upper_BB"]:
        score -= 1
        breakdown.append(f"PRICE POSITION: TESTING UPPER BAND (${price:.2f}) | IMPACT: -1 PT")
    else:
        breakdown.append(f"PRICE POSITION: STABLE WITHIN CHANNELS | IMPACT: 0 PTS")

    # 3. MACD CROSSOVER
    if prev["MACD"] < prev["Signal"] and last["MACD"] > last["Signal"]:
        score += 2
        breakdown.append("MACD SIGNAL: BULLISH CROSSOVER DETECTED | IMPACT: +2 PTS")
    elif prev["MACD"] > prev["Signal"] and last["MACD"] < last["Signal"]:
        score -= 2
        breakdown.append("MACD SIGNAL: BEARISH CROSSOVER DETECTED | IMPACT: -2 PTS")
    else:
        state = "BULLISH" if last["MACD"] > last["Signal"] else "BEARISH"
        breakdown.append(f"MACD SIGNAL: NO CROSSOVER ({state} BIAS) | IMPACT: 0 PTS")

    # 4. SENTIMENT ANALYSIS
    if sentiment > 0.05:
        score += 1
        breakdown.append(f"NEWS SENTIMENT: POSITIVE ({sentiment:.2f}) | IMPACT: +1 PT")
    elif sentiment < -0.05:
        score -= 1
        breakdown.append(f"NEWS SENTIMENT: NEGATIVE ({sentiment:.2f}) | IMPACT: -1 PT")
    else:
        breakdown.append(f"NEWS SENTIMENT: NEUTRAL ({sentiment:.2f}) | IMPACT: 0 PTS")
        
    color = "#00ffcc" if score >= 2 else "#ff4b4b" if score <= -2 else "#ffd700"
    verdict = "STRONG BUY" if score >= 2 else "STRONG SELL" if score <= -2 else "NEUTRAL/HOLD"
    
    return verdict, color, breakdown, score

# ---------------- BACKTESTING ENGINE ----------------
def run_strategy_backtest(data, capital=10000):
    results = data.copy()
    results['Cash'] = float(capital)
    results['Position'] = 0.0 
    results['Total_Value'] = float(capital)
    
    for i in range(1, len(results)):
        curr = results.iloc[i]
        buy_signal = curr['RSI'] < 35 or curr['Close'] < curr['Lower_BB']
        sell_signal = curr['RSI'] > 65 or curr['Close'] > curr['Upper_BB']
        
        if buy_signal and results.at[results.index[i-1], 'Cash'] > 0:
            results.at[results.index[i], 'Position'] = results.at[results.index[i-1], 'Cash'] / curr['Close']
            results.at[results.index[i], 'Cash'] = 0
        elif sell_signal and results.at[results.index[i-1], 'Position'] > 0:
            results.at[results.index[i], 'Cash'] = results.at[results.index[i-1], 'Position'] * curr['Close']
            results.at[results.index[i], 'Position'] = 0
        else:
            results.at[results.index[i], 'Cash'] = results.at[results.index[i-1], 'Cash']
            results.at[results.index[i], 'Position'] = results.at[results.index[i-1], 'Position']
            
        results.at[results.index[i], 'Total_Value'] = results.at[results.index[i], 'Cash'] + (results.at[results.index[i], 'Position'] * curr['Close'])
    
    return results

# Calculation logic
final_verdict, v_color, signals, q_score = get_detailed_verdict(df, avg_sent)

# ---------------- UI TABS ----------------
tabs = st.tabs(["MARKET VIEW", "QUANT ENGINE", "SENTIMENT", "VERDICT", "BACKTEST LAB"])

with tabs[0]:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.8, 0.2], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color="#1f6feb"), row=2, col=1)
    fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False, paper_bgcolor="#0b0e14", plot_bgcolor="#0b0e14")
    st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    ind_fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.4, 0.3, 0.3],
                            subplot_titles=("VOLATILITY (BOLLINGER)", "MOMENTUM (RSI)", "TREND (MACD)"))
    ind_fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price", line=dict(color="white")), row=1, col=1)
    ind_fig.add_trace(go.Scatter(x=df.index, y=df["Upper_BB"], line=dict(color='rgba(0, 255, 204, 0.2)'), name="Upper BB"), row=1, col=1)
    ind_fig.add_trace(go.Scatter(x=df.index, y=df["Lower_BB"], line=dict(color='rgba(0, 255, 204, 0.2)'), fill='tonexty', name="Lower BB"), row=1, col=1)
    ind_fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], line=dict(color="#bfff00")), row=2, col=1)
    ind_fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="#58a6ff")), row=3, col=1)
    ind_fig.add_trace(go.Scatter(x=df.index, y=df["Signal"], name="Signal", line=dict(color="#ff7b72")), row=3, col=1)
    ind_fig.update_layout(template="plotly_dark", height=800, paper_bgcolor="#0b0e14", plot_bgcolor="#0b0e14", showlegend=False)
    st.plotly_chart(ind_fig, use_container_width=True)

with tabs[2]:
    st.metric("AGGREGATE SENTIMENT", f"{avg_sent:.2f}")
    st.dataframe(news_df, use_container_width=True)

with tabs[3]:
    st.markdown(f"<h1 style='text-align:center;color:{v_color};'>{final_verdict}</h1>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align:center;'>QUANTITATIVE SCORE: {q_score}</h3>", unsafe_allow_html=True)
    st.divider()
    st.write("### ANALYST LOGIC BREAKDOWN")
    for s in signals:
        st.code(s, language="bash")

with tabs[4]:
    st.write("### PERFORMANCE SIMULATION")
    if st.button("EXECUTE BACKTEST"):
        bt_results = run_strategy_backtest(df)
        final_val = bt_results['Total_Value'].iloc[-1]
        
        c1, c2 = st.columns(2)
        c1.metric("FINAL EQUITY", f"${final_val:,.2f}")
        c2.metric("NET RETURN", f"{((final_val - 10000)/10000)*100:.2f}%")
        
        perf_fig = go.Figure()
        perf_fig.add_trace(go.Scatter(x=bt_results.index, y=bt_results['Total_Value'], name="Strategy Equity", line=dict(color="#00ffcc")))
        benchmark = (bt_results['Close'] / bt_results['Close'].iloc[0]) * 10000
        perf_fig.add_trace(go.Scatter(x=bt_results.index, y=benchmark, name="Benchmark (Buy & Hold)", line=dict(color="gray", dash="dash")))
        perf_fig.update_layout(template="plotly_dark", title="Equity Curve", paper_bgcolor="#0b0e14", plot_bgcolor="#0b0e14")
        st.plotly_chart(perf_fig, use_container_width=True)
