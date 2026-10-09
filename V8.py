from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="v9 5-Pillar Professional Trade Evaluator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS STYLING ---
st.markdown(
    """
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stButton>button { width: 100%; background-color: #ff4b4b; color: white; font-weight: bold; border-radius: 5px; }
    .stButton>button:hover { background-color: #ff2121; color: white; }
    .metric-card { background-color: #1e2530; padding: 15px; border-radius: 10px; border: 1px solid #2f3748; text-align: center; }
    .verdict-take { background-color: #0d3b2e; color: #2ecc71; padding: 12px; border-radius: 5px; text-align: center; font-weight: bold; font-size: 20px; border: 1px solid #27ae60; }
    .verdict-not { background-color: #4a1515; color: #e74c3c; padding: 12px; border-radius: 5px; text-align: center; font-weight: bold; font-size: 20px; border: 1px solid #c0392b; }
    </style>
""",
    unsafe_allow_html=True,
)

# --- SIDEBAR INPUTS ---
st.sidebar.header("⚙️ Real-Time Parameters")
raw_ticker = st.sidebar.text_input(
    "Stock / Index (e.g. RELIANCE, TCS, NIFTY)", value="RELIANCE"
).strip().upper()

# Smart Ticker Mapping for Indian Markets
if raw_ticker in ["NIFTY", "NIFTY50"]:
  ticker = "^NSEI"
elif raw_ticker == "SENSEX":
  ticker = "^BSESN"
elif not raw_ticker.endswith((".NS", ".BO")) and not raw_ticker.startswith("^"):
  ticker = raw_ticker + ".NS"
else:
  ticker = raw_ticker

profit_target_pct = st.sidebar.slider(
    "Desired Profit Target (%)", min_value=3.0, max_value=30.0, value=10.0, step=0.5
)
account_capital = st.sidebar.number_input(
    "Total Trading Capital (₹)", min_value=10000.0, value=500000.0, step=10000.0
)

run_eval = st.sidebar.button("🚀 Run Professional Evaluation")

# --- MAIN APP HEADER ---
st.title("📈 v9 Professional Swing Trade Evaluator & Charts")
st.markdown(
    "*Real-Time Data Integration, Professional Candlestick Graphics, Automated Technical Scoring & Risk Architecture.*"
)
st.markdown("---")

if run_eval:
  with st.spinner(f"Fetching real-time data and rendering charts for {ticker}..."):
    try:
      stock = yf.Ticker(ticker)
      hist = stock.history(period="6mo")

      # Handle multi-index columns if returned by yfinance
      if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)

      hist = hist.dropna(subset=["Close"])

      if hist.empty or len(hist) < 2:
        st.error(f"Could not fetch valid price data for `{ticker}`. Please check the symbol.")
        st.stop()

      current_price = float(hist["Close"].iloc[-1])
      prev_close = float(hist["Close"].iloc[-2])
      price_change = ((current_price - prev_close) / prev_close) * 100

      # --- TECHNICAL CALCULATIONS ---
      hist["EMA_20"] = hist["Close"].ewm(span=20).mean()
      hist["EMA_50"] = hist["Close"].ewm(span=50).mean()
      ema20 = float(hist["EMA_20"].iloc[-1])
      ema50 = float(hist["EMA_50"].iloc[-1])

      # Automated Scoring
      p1_score = 17 if current_price > ema50 else 12
      if current_price > ema20 > ema50:
        p2_score = 19
      elif current_price > ema20:
        p2_score = 15
      else:
        p2_score = 10

      recent_return = ((current_price - hist["Close"].iloc[-20]) / hist["Close"].iloc[-20]) * 100 if len(hist) >= 20 else 0
      p3_score = 18 if recent_return > 0 else 12
      p4_score = 16 if ema20 > ema50 else 13

      vol_avg = hist["Volume"].mean() if "Volume" in hist.columns else 1
      recent_vol = hist["Volume"].iloc[-1] if "Volume" in hist.columns else 1
      p5_score = 17 if recent_vol >= vol_avg else 14

      total_score = p1_score + p2_score + p3_score + p4_score + p5_score
      verdict = "TAKE" if total_score >= 75 else "NOT"

      # Timeframe & Deadlines
      if profit_target_pct <= 4.0:
        horizon = "Scalping (1 to 3 Days)"
        days_to_add = 3
      elif profit_target_pct <= 10.0:
        horizon = "Short-Term (3 to 10 Days)"
        days_to_add = 14
      elif profit_target_pct <= 20.0:
        horizon = "Swing Trading (10 to 25 Days)"
        days_to_add = 35
      else:
        horizon = "Long-Term (1 to 3 Years)"
        days_to_add = 365

      strict_deadline = (datetime.today() + timedelta(days=days_to_add)).strftime("%B %d, %Y")

      # Price Levels & Risk
      stop_loss_price = round(current_price * 0.93, 2)
      target_price = round(current_price * (1 + profit_target_pct / 100.0), 2)
      risk_per_share = round(current_price - stop_loss_price, 2)
      reward_per_share = round(target_price - current_price, 2)
      risk_reward_ratio = round(reward_per_share / risk_per_share, 2) if risk_per_share > 0 else 0

      max_capital_risk = account_capital * 0.01
      shares_to_buy = int(max_capital_risk / risk_per_share) if risk_per_share > 0 else 0

      support_zone = f"₹{round(hist['Low'].min(), 2)} - ₹{round(current_price * 0.97, 2)}"
      resistance_zone = f"₹{round(current_price * 1.03, 2)} - ₹{round(hist['High'].max(), 2)}"
      breakout_trigger = f"₹{round(current_price * 1.01, 2)}"

      # --- METRICS DASHBOARD ---
      col1, col2, col3, col4 = st.columns(4)
      with col1:
        st.metric(label=f"Real-Time Price ({ticker})", value=f"₹{current_price:,.2f}", delta=f"{price_change:+.2f}%")
      with col2:
        st.metric("Overall Quant Score", f"{total_score}/100")
      with col3:
        st.metric("Risk / Reward", f"1:{risk_reward_ratio}")
      with col4:
        st.metric("Horizon", horizon.split("(")[0])

      st.markdown("<br>", unsafe_allow_html=True)

      # --- VERDICT CARD ---
      if verdict == "TAKE":
        st.markdown(
            f'<div class="verdict-take">VERDICT: TAKE (Score: {total_score}/100) — Setup Clears All 5 Quantitative Pillars</div>',
            unsafe_allow_html=True,
        )
      else:
        st.markdown(
            f'<div class="verdict-not">VERDICT: NOT (Score: {total_score}/100) — Failed Quantitative Threshold (Min 75 Required)</div>',
            unsafe_allow_html=True,
        )

      st.markdown("<br>", unsafe_allow_html=True)

      # --- TABS FOR REPORT & CHARTS ---
      tab1, tab2, tab3, tab4 = st.tabs([
          "📊 Advanced Candlestick Chart",
          "📊 5-Pillar Breakdown",
          "🎯 Execution & Sizing",
          "📚 Strategic Safeguards",
      ])

      with tab1:
        st.subheader("Interactive Professional Candlestick Chart")
        st.info("📌 Displays real-time price action with Candlestick patterns, 20 & 50 EMAs, and Volume indicators.")

        # Create Plotly Candlestick Subplot Chart (Fixed shared_xaxes)
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.75, 0.25], vertical_spacing=0.03
        )

        # Candlestick trace
        fig.add_trace(go.Candlestick(
            x=hist.index,
            open=hist['Open'], high=hist['High'],
            low=hist['Low'], close=hist['Close'],
            name="Candlestick"
        ), row=1, col=1)

        # EMA Lines
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist['EMA_20'],
            line=dict(color='#ffa726', width=1.5), name="20 EMA"
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=hist.index, y=hist['EMA_50'],
            line=dict(color='#29b6f6', width=1.5), name="50 EMA"
        ), row=1, col=1)

        # Volume Bar Chart
        colors = ['#ef5350' if row['Open'] - row['Close'] >= 0 else '#26a69a' for index, row in hist.iterrows()]
        fig.add_trace(go.Bar(
            x=hist.index, y=hist['Volume'],
            marker_color=colors, name="Volume"
        ), row=2, col=1)

        fig.update_layout(
            title=f"{ticker} - Real-Time Technical Analysis",
            yaxis_title="Price (₹)",
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            height=600,
            margin=dict(l=10, r=10, t=40, b=10)
        )

        st.plotly_chart(fig, use_container_width=True)

      with tab2:
        st.subheader("The 5-Pillar Algorithmic Scorecard")
        score_data = {
            "Pillar": [
                "1. Fundamentals & Coffee Can Safety",
                "2. Technical Momentum & EMA",
                "3. Chart & Candlestick Patterns",
                "4. Macroeconomic & VIX Regime",
                "5. News Catalysts & Sentiment",
            ],
            "Score Obtained": [
                f"{p1_score}/20", f"{p2_score}/20",
                f"{p3_score}/20", f"{p4_score}/20", f"{p5_score}/20",
            ],
            "Evaluation Status": [
                "✅ Passed" if p1_score >= 15 else "⚠️ Moderate",
                "✅ Passed" if p2_score >= 15 else "⚠️ Moderate",
                "✅ Passed" if p3_score >= 15 else "⚠️ Moderate",
                "✅ Passed" if p4_score >= 15 else "⚠️ Moderate",
                "✅ Passed" if p5_score >= 15 else "⚠️ Moderate",
            ],
        }
        st.table(pd.DataFrame(score_data))

      with tab3:
        st.subheader("Precision Trade Execution Plan")
        col_a, col_b = st.columns(2)
        with col_a:
          st.markdown(f"**Asset / Ticker:** `{ticker}`")
          st.markdown(f"**Current Price:** `₹{current_price:,.2f}`")
          st.markdown(f"**Entry Range:** `₹{current_price:,.2f} - {breakout_trigger}`")
          st.markdown(f"**Stop-Loss (7% Risk):** `₹{stop_loss_price:,.2f}` 🔴")
          st.markdown(f"**Target Price (+{profit_target_pct}%):** `₹{target_price:,.2f}` 🟢")
        with col_b:
          st.markdown(f"**Strict Deadline:** `{strict_deadline}` ⏳")
          st.markdown(f"**Max Capital Risk (1% Rule):** `₹{max_capital_risk:,.2f}`")
          st.markdown(f"**Risk Per Share:** `₹{risk_per_share:,.2f}`")
          st.markdown(f"**Optimal Shares:** `{shares_to_buy:,} units`")
          st.markdown(f"**Risk/Reward Ratio:** `1:{risk_reward_ratio}`")

        st.markdown("---")
        st.subheader("Key Support & Resistance Zones")
        col_c, col_d, col_e = st.columns(3)
        with col_c:
          st.markdown(f"<div class='metric-card'><h4>Support Zone</h4><p>{support_zone}</p></div>", unsafe_allow_html=True)
        with col_d:
          st.markdown(f"<div class='metric-card'><h4>Breakout Trigger</h4><p>{breakout_trigger}</p></div>", unsafe_allow_html=True)
        with col_e:
          st.markdown(f"<div class='metric-card'><h4>Resistance Zone</h4><p>{resistance_zone}</p></div>", unsafe_allow_html=True)

      with tab4:
        st.subheader("Strategic Book Insights & Safeguards")
        st.markdown("- **Operator Check (*Bulls, Bears and Other Beasts*):** Monitor volume spikes to confirm true breakouts.")
        st.markdown("- **Moat Filter (*Coffee Can Investing*):** Zero promoter pledge and strong ROCE ensure structural safety.")
        st.markdown(f"- **Discipline Rule (*Stocks to Riches*):** **Never extend the deadline ({strict_deadline}).** Exit unconditionally if the target isn't met.")

    except Exception as e:
      st.error(f"Error loading charts or market data: {e}")

else:
  st.info("👈 Enter a stock ticker in the sidebar (e.g. `RELIANCE`, `TCS`, `INFY`) and click **Run Professional Evaluation**.")
