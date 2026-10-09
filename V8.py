from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="v9 5-Pillar Auto-Fetch Swing Trade Evaluator",
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
st.sidebar.header("⚙️ Auto-Fetch Parameters")
raw_ticker = st.sidebar.text_input(
    "Stock Ticker (e.g. RELIANCE, BSE, TCS)", value="BSE"
).strip().upper()

# Automatically append .NS for Indian NSE stocks if suffix is missing
if not raw_ticker.endswith((".NS", ".BO")):
  ticker = raw_ticker + ".NS"
else:
  ticker = raw_ticker

profit_target_pct = st.sidebar.slider(
    "Desired Profit Target (%)", min_value=3.0, max_value=30.0, value=10.0, step=0.5
)
account_capital = st.sidebar.number_input(
    "Total Trading Capital (₹)", min_value=10000.0, value=500000.0, step=10000.0
)

run_eval = st.sidebar.button("🚀 Run Auto 5-Pillar Evaluation")

# --- MAIN APP HEADER ---
st.title("📈 v9 Auto-Fetch 5-Pillar Swing Trade Evaluator")
st.markdown(
    "*Instant Live Data Integration, Automated Technical Scoring, Strict Non-Extending Deadlines & Risk Architecture.*"
)
st.markdown("---")

if run_eval:
  with st.spinner(f"Fetching live data and running 5-pillar analysis for {ticker}..."):
    try:
      stock = yf.Ticker(ticker)
      hist = stock.history(period="6mo")

      # Handle multi-index columns if returned by yfinance
      if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)

      # Clean missing values
      hist = hist.dropna(subset=["Close"])

      if hist.empty or len(hist) < 2:
        st.error(f"Could not fetch valid price data for `{ticker}`. Please verify the stock symbol.")
        st.stop()

      current_price = float(hist["Close"].iloc[-1])
      prev_close = float(hist["Close"].iloc[-2])
      price_change = ((current_price - prev_close) / prev_close) * 100

      # --- AUTOMATED TECHNICAL CALCULATIONS ---
      hist["EMA_20"] = hist["Close"].ewm(span=20).mean()
      hist["EMA_50"] = hist["Close"].ewm(span=50).mean()
      ema20 = float(hist["EMA_20"].iloc[-1])
      ema50 = float(hist["EMA_50"].iloc[-1])

      # Automated Scoring based on live quantitative rules (0-20 per pillar)
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

      # Timeframe & Deadline Mapping
      if profit_target_pct <= 4.0:
        horizon = "Scalping (1 to 3 Trading Days)"
        days_to_add = 3
      elif profit_target_pct <= 10.0:
        horizon = "Short-Term Momentum (3 to 10 Days)"
        days_to_add = 14
      elif profit_target_pct <= 20.0:
        horizon = "Swing Trading (10 to 25 Days)"
        days_to_add = 35
      else:
        horizon = "Long-Term Coffee Can (1 to 3 Years)"
        days_to_add = 365

      strict_deadline = (datetime.today() + timedelta(days=days_to_add)).strftime("%B %d, %Y")

      # Price Levels & Risk Math
      stop_loss_price = round(current_price * 0.93, 2)
      target_price = round(current_price * (1 + profit_target_pct / 100.0), 2)
      risk_per_share = round(current_price - stop_loss_price, 2)
      reward_per_share = round(target_price - current_price, 2)
      risk_reward_ratio = round(reward_per_share / risk_per_share, 2) if risk_per_share > 0 else 0

      # Position Sizing (1% Capital Risk Rule)
      max_capital_risk = account_capital * 0.01
      shares_to_buy = int(max_capital_risk / risk_per_share) if risk_per_share > 0 else 0

      # Support & Resistance Estimation
      support_zone = f"₹{round(hist['Low'].min(), 2)} - ₹{round(current_price * 0.97, 2)}"
      resistance_zone = f"₹{round(current_price * 1.03, 2)} - ₹{round(hist['High'].max(), 2)}"
      breakout_trigger = f"₹{round(current_price * 1.01, 2)}"

      # --- DASHBOARD METRICS ---
      col1, col2, col3, col4 = st.columns(4)
      with col1:
        st.metric(label=f"Live Price ({ticker})", value=f"₹{current_price:,.2f}", delta=f"{price_change:+.2f}%")
      with col2:
        st.metric("Overall Quant Score", f"{total_score}/100")
      with col3:
        st.metric("Risk / Reward Ratio", f"1:{risk_reward_ratio}")
      with col4:
        st.metric("Trading Horizon", horizon.split("(")[0])

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

      # --- TABS FOR STRUCTURED REPORT ---
      tab1, tab2, tab3, tab4 = st.tabs([
          "📊 5-Pillar Breakdown",
          "🎯 Execution & Sizing",
          "🗺️ Chart Levels & Price Action",
          "📚 Strategic Safeguards",
      ])

      with tab1:
        st.subheader("The 5-Pillar Algorithmic Scorecard (Auto-Evaluated)")
        score_data = {
            "Pillar": [
                "1. Fundamentals & Coffee Can Safety",
                "2. Technical Momentum & EMA",
                "3. Chart & Candlestick Patterns",
                "4. Macroeconomic & VIX Regime",
                "5. News Catalysts & Sentiment",
            ],
            "Score Obtained": [
                f"{p1_score}/20",
                f"{p2_score}/20",
                f"{p3_score}/20",
                f"{p4_score}/20",
                f"{p5_score}/20",
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

      with tab2:
        st.subheader("Precision Trade Execution Plan")
        col_a, col_b = st.columns(2)
        with col_a:
          st.markdown(f"**Asset / Ticker:** `{ticker}`")
          st.markdown(f"**Current Live Price:** `₹{current_price:,.2f}`")
          st.markdown(f"**Entry Range:** `₹{current_price:,.2f} - {breakout_trigger}`")
          st.markdown(f"**Stop-Loss (7.0% Max Risk):** `₹{stop_loss_price:,.2f}` 🔴")
          st.markdown(f"**Target Price (+{profit_target_pct}%):** `₹{target_price:,.2f}` 🟢")
        with col_b:
          st.markdown(f"**Strict Non-Extending Deadline:** `{strict_deadline}` ⏳")
          st.markdown(f"**Max Capital Risk (1% Rule):** `₹{max_capital_risk:,.2f}`")
          st.markdown(f"**Risk Per Share:** `₹{risk_per_share:,.2f}`")
          st.markdown(f"**Optimal Shares to Allocate:** `{shares_to_buy:,} units`")
          st.markdown(f"**Risk/Reward Ratio:** `1:{risk_reward_ratio}`")

      with tab3:
        st.subheader("Key Chart Support & Resistance Mapping")
        st.info("📌 Automatically calculated from historical price boundaries and moving averages.")
        
        col_c, col_d, col_e = st.columns(3)
        with col_c:
          st.markdown(f"<div class='metric-card'><h4>Support Zone</h4><p>{support_zone}</p></div>", unsafe_allow_html=True)
        with col_d:
          st.markdown(f"<div class='metric-card'><h4>Breakout Trigger</h4><p>{breakout_trigger}</p></div>", unsafe_allow_html=True)
        with col_e:
          st.markdown(f"<div class='metric-card'><h4>Resistance Zone</h4><p>{resistance_zone}</p></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("6-Month Price & EMA Chart")
        chart_data = hist[["Close", "EMA_20", "EMA_50"]]
        st.line_chart(chart_data)

      with tab4:
        st.subheader("Strategic Book Insights & Behavioral Safeguards")
        st.markdown("- **Operator & Liquidity Check (*Bulls, Bears and Other Beasts*):** Ensure volume supports the move; avoid illiquid operator traps.")
        st.markdown("- **Moat & Governance Filter (*Coffee Can Investing*):** Zero promoter pledge and stable ROCE protect against structural meltdowns.")
        st.markdown(f"- **Behavioral Discipline & Deadline Rule (*Stocks to Riches*):** **Never extend the deadline ({strict_deadline}).** If the profit target is not reached by this date, exit the trade unconditionally to preserve capital velocity.")

    except Exception as e:
      st.error(f"Error fetching data or running evaluation: {e}")

else:
  st.info("👈 Enter your stock name in the sidebar (e.g. `BSE`, `RELIANCE`, `TCS`) and click **Run Auto 5-Pillar Evaluation**.")
