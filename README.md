# 📈 AI-Powered Quant Trading Dashboard
### Finova x ISTE | Aurora 2026 Interactive Workshop

Welcome to the **AI-Powered Quant Trading Dashboard** workshop! In this session, we are moving beyond simple price charts. You will build a production-grade financial tool that combines **Quantitative Math** with **Artificial Intelligence** to generate real-time trading signals.

---

## 🚀 The Mission
Most traders fail because they trade on emotion. Our goal today is to build a **"Verdict Engine"** that uses logic and data to decide whether to **Buy, Sell, or Hold**.

**We will bridge two worlds:**
1.  **The Math:** Using Technical Indicators (RSI, Bollinger Bands, MACD).
2.  **The Narrative:** Using AI (NLP) to "read" the news and gauge market sentiment.

---

## 🛠 The Tech Stack
* **Language:** Python 3.10+
* **UI/Web Framework:** [Streamlit](https://streamlit.io/)
* **Data Sourcing:** `yfinance` & `BeautifulSoup4`
* **Financial Math:** `Pandas` & `NumPy`
* **AI/NLP:** `NLTK / VADER` (Sentiment Analysis)
* **Visuals:** `Plotly` (Interactive High-Fidelity Charts)

---

## 🗺 Project Roadmap

### Phase 1: Financial Data Engineering
* Connecting to the Yahoo Finance API.
* Handling Time-Series dataframes and cleaning null values.
* **Outcome:** A reactive sidebar that fetches live data for any ticker (AAPL, BTC-USD, etc.).

### Phase 2: The Quantitative Engine
We will hand-code the following professional indicators:
* **RSI (Relative Strength Index):** Identifying momentum exhaustion.
* **Bollinger Bands:** Measuring volatility and price channels.
* **MACD (Moving Average Convergence Divergence):** Finding trend reversals.
* **Volume Profile:** Confirming the strength of price moves.

### Phase 3: The AI Brain - NLP Sentiment
* Scraping real-time market news headlines.
* Processing text through the **VADER Sentiment Engine**.
* **Outcome:** A "Mood Score" from -1 (Panic) to +1 (Euphoria).

### Phase 4: The Synthesis (Verdict Engine)
* Building a **Weighted Decision Matrix**.
* **Example Logic:** `(RSI Score * 0.3) + (Sentiment Score * 0.4) + (Trend Score * 0.3)`
* **Outcome:** A beautiful UI Gauge showing: **STRONG BUY / NEUTRAL / SELL**.

---

## 🔧 Installation & Setup

Ensure you have Python installed, then run:

```bash
# Clone the repository
git clone [https://github.com/](https://github.com/)[your-username]/quant-trading-dashboard.git
cd quant-trading-dashboard

# Install required libraries
pip install streamlit yfinance pandas numpy nltk plotly beautifulsoup4
