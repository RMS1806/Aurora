# 📈 SentimentX Elite: AI-Powered Quant Dashboard
### Finova x ISTE | Aurora 2026 Interactive Workshop

Welcome to the **SentimentX Elite** workshop! Today, you aren't just building a chart; you are engineering a **Wall Street-grade Quant Terminal**. We are combining raw market physics with automated sentiment analysis to eliminate emotional bias from trading.

---

## 🚀 The Mission
We are building a **Verdict Engine** that synthesizes multiple data streams into a single, logic-backed decision. By the end of this session, your terminal will:

1.  **Analyze Physics**: Process Volatility, Momentum, and Trend through math.
2.  **Analyze Mood**: Use NLP to "read" institutional news and determine market sentiment.
3.  **Simulate History**: Run a **Backtest Engine** to see if your strategy would have actually made money in the past.

---

## 🛠 The Tech Stack
* **UI Framework**: [Streamlit](https://streamlit.io/) (Finance Terminal Custom CSS Theme).
* **Market Data**: `yfinance` (Live Equity/Crypto Pricing).
* **Alternative Data**: `Finnhub API` (Institutional News Feed).
* **NLP Engine**: `NLTK / VADER` (Lexicon-based Sentiment Scoring).
* **Visuals**: `Plotly` (High-Fidelity Multi-track Subplots).

---

## 🗺 Project Roadmap

### Phase 1: Market Data Pipeline
* Connect to `yfinance` and handle **Multi-Index** dataframes.
* Clean and flatten time-series data for mathematical processing.

### Phase 2: The Quantitative Engine
Hand-code a multi-track indicator workspace:
* **Bollinger Bands**: Measuring volatility using Standard Deviation.
* **RSI Oscillator**: Identifying overbought (>70) and oversold (<30) momentum.
* **MACD Trend**: Tracking signal line crossovers to find trend reversals.



### Phase 3: AI Sentiment Analysis
* Implement live news fetching via the **Finnhub API**.
* Process headlines through the **VADER Sentiment Engine**.
* Map human narrative to a numerical score from -1.0 (Panic) to +1.0 (Euphoria).

### Phase 4: The Verdict Engine
* Build a **Weighted Decision Matrix**.
* Synthesize RSI, Bollinger, MACD, and Sentiment into a final terminal output.
* Provide a transparent **Logic Breakdown** for every decision.

### Phase 5: Backtest Lab
* Simulate your strategy on historical data.
* Compare your **Equity Curve** against a **Buy & Hold** benchmark.



---

## 🔧 Installation & Setup

Follow these steps to get the terminal running on your local machine:

### 1. Install Python & Environment
* Ensure you have **Python 3.10+** installed.
* **Important**: During installation, check the box **"Add Python to PATH"**.

### 2. Install Required Libraries
Open your terminal (CMD, PowerShell, or Bash) and run:
```bash
pip install streamlit yfinance pandas numpy nltk plotly finnhub-python
