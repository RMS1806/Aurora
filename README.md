# 📈 SentimentX Elite: AI-Powered Quant Dashboard
### Finova x ISTE | Aurora 2026 Interactive Workshop

Welcome to the **SentimentX Elite** workshop! Today, you aren't just building a chart; you are engineering a **Wall Street-grade Quant Terminal**. We are combining raw market physics (Indicators) with the power of Large Language Models (Gemini) to automate financial analysis.

---

##  The Mission
We are building a **Verdict Engine** that eliminates human bias. By the end of this session, your app will:
1.  **Analyze Physics:** Process Volatility, Momentum, and Trend through math.
2.  **Analyze Mood:** Use NLP to "read" institutional news from Finnhub.
3.  **Generate Thesis:** Use Google Gemini to write a professional Analyst Report automatically.

---

## The Tech Stack
* **Core:** Python 3.10+ & [Streamlit](https://streamlit.io/) (High-performance UI)
* **Market Data:** `yfinance` (Live Pricing) & `Finnhub API` (Institutional News)
* **The "Brain":** `Google Gemini 2.5 Flash` (Generative AI Analyst)
* **Financial Math:** `Pandas` (Vectorized Operations)
* **Visuals:** `Plotly` (High-Fidelity Multi-track Subplots)

---

##  Project Roadmap

### Phase 1: Market Data Pipeline
* Connect to `yfinance` and handle **Multi-Index** dataframes.
* Implement **API Authentication** for Finnhub and Google Gemini.
* Build a high-performance reactive sidebar.

### Phase 2: The Unified Quant Engine
Hand-code a multi-track indicator workspace:
* **Bollinger Bands:** Volatility envelopes for mean-reversion.
* **RSI Oscillator:** Identifying overbought/oversold momentum.
* **MACD Trend:** Tracking signal line crossovers for entry/exit timing.

### Phase 3: AI Sentiment Analysis
* Implement live news fetching via **Finnhub**.
* Process headlines through the **VADER Sentiment Engine**.
* Calculate an **Aggregate Mood Score** from -1.0 (Panic) to +1.0 (Euphoria).

### Phase 4: Agentic Report Generation
* Build the **Weighted Verdict Logic** (Combining RSI, BB, MACD, and Sentiment).
* Engineer a **Prompt Template** for Google Gemini.
* **Outcome:** A button that generates a professional Wall Street Quant Report on demand.

---

## Installation & Setup

1. **Clone & Install:**
```bash
git clone [https://github.com/](https://github.com/)[RMS1806]/Aurora.git
cd Aurora
pip install streamlit yfinance pandas numpy nltk plotly finnhub-python google-generativeai
