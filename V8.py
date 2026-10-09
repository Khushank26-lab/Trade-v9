from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="v9.3 Clean Swing Trade Evaluator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- ULTRA-SMOOTH CUSTOM CSS ---
st.markdown(
    """
    <style>
    .main { background-color: #0b0f19; color: #f8fafc; font-family: -apple-system, BlinkMacSystemFont, sans-serif; }
    .stButton>button { width: 100%; background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); color: white; font-weight: 600; border-radius: 8px; border: none; padding: 10px; transition: all 0.3s ease; }
    .stButton>button:hover { background: linear-gradient(135deg, #f87171 0%, #ef4444 100%); box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3); }
    .metric-card { background: #111827; padding: 16px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.08); text-align: center; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
    .verdict-take { background: rgba(16, 185, 129, 0.1); color: #34d399; padding: 14px; border-radius: 10px; text-align: center; font-weight: 700; font-size: 20px; border: 1px solid rgba(52, 211, 153, 0.3); }
    .verdict-not { background: rgba(239, 68, 68, 0.1); color: #f87171; padding: 14px; border-radius: 10px; text-align: center; font-weight: 700; font-size: 20px; border: 1px solid rgba(248, 113, 113, 0.3); }
    </style>
""",
    unsafe_allow_html=True,
)

# --- CACHED DATA FETCHER FOR SMOOTH PERFORMANCE ---
@st.cache_data(ttl=1800, show_spinner=False)
def get_stock_history(ticker_symbol):
  stock = yf.Ticker(ticker_symbol)
  df = stock.history(period="max")
  if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
  return df.dropna(subset=["Close"])


# --- SIDEBAR INPUTS ---
st.sidebar.header("⚙️ Execution Parameters")
raw_ticker = st.sidebar.text_input(
    "Stock / Index (e.g. RELIANCE, TCS, NIFTY)", value="RELIANCE"
).strip().upper()

# Smart Ticker Mapping
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

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Pillar Score Tweaks")
enable_custom = st.sidebar.checkbox("Manually Override Scores", value=False)

if enable_custom:
  cust_p1 = st.sidebar.slider("Pillar 1: Fundamentals (/20)", 0, 20, 16)
  cust_p2 = st.sidebar.slider("Pillar 2: Technical Momentum (/20)", 0, 20, 15)
  cust_p3 = st.sidebar.slider("Pillar 3: Chart Patterns (/20)", 0, 20, 15)
  cust_p4 = st.sidebar.slider("Pillar 4: Macro & VIX (/20)", 0, 20, 16)
  cust_p5 = st.sidebar.slider("Pillar 5: News Sentiment (/20)", 0, 20, 14)

run_eval = st.sidebar.button("🚀 Run Evaluation")

# --- MAIN APP HEADER ---
st.title("📈 v9.3 Clean Swing Trade Evaluator")
st.markdown(
    "*Lightning-fast real-time data caching, multi-year trend analysis & sleek risk management.*"
)
st.markdown("---")

if run_eval:
  with st.spinner(f"Analyzing {ticker} with high-speed caching..."):
    try:
      hist = get_stock_history(ticker)

      if hist.empty or len(hist) < 50:
        st.error(f"⚠️ Could not fetch valid data for `{ticker}`. Please verify the symbol.")
        st.stop()

      current_price = float(hist["Close"].iloc[-1])
      prev_close = float(hist["Close"].iloc[-2])
      price_change = ((current_price - prev_close) / prev_close) * 100

      # --- TECHNICAL & MULTI-YEAR INDICATORS ---
      hist["EMA_20"] = hist["Close"].ewm(span=20).mean()
      hist["EMA_50"] = hist["Close"].ewm(span=50).mean()
      hist["EMA_200"] = hist["Close"].ewm(span=200).mean()

      ema20 = float(hist["EMA_20"].iloc[-1])
      ema50 = float(hist["EMA_50"].iloc[-1])
      ema200 = float(hist["EMA_200"].iloc[-1]) if not np.isnan(hist["EMA_200"].iloc[-1]) else ema50

      # RSI Calculation (14-period)
      delta = hist["Close"].diff()
      gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
      loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
      rs = gain / loss
      hist["RSI"] = 100 - (100 / (1 + rs))
      current_rsi = float(hist["RSI"].iloc[-1]) if not np.isnan(hist["RSI"].iloc[-1]) else 50.0

      # --- AUTOMATED 5-PILLAR ALGORITHM ---
      if not enable_custom:
        p1_score = 18 if current_price > ema200 else 12
        if current_price > ema20 > ema50 > ema200:
          p2_score = 19
        elif current_price > ema20:
          p2_score = 15
        else:
          p2_score = 10

        recent_return = ((current_price - hist["Close"].iloc[-20]) / hist["Close"].iloc[-20]) * 100
        if 50 <= current_rsi <= 70 and recent_return > 0:
          p3_score = 18
        elif recent_return > 0:
          p3_score = 15
        else:
          p3_score = 10

        p4_score = 17 if ema50 > ema200 else 13
        vol_avg = hist["Volume"].mean() if "Volume" in hist.columns else 1
        recent_vol = hist["Volume"].iloc[-1] if "Volume" in hist.columns else 1
        p5_score = 18 if recent_vol >= vol_avg * 1.1 else 14
      else:
        p1_score, p2_score, p3_score, p4_score, p5_score = cust_p1, cust_p2, cust_p3, cust_p4, cust_p5

      total_score = p1_score + p2_score + p3_score + p4_score + p5_score
      verdict = "TAKE" if total_score >= 75 else "NOT"

      # Timeframe & Deadlines
      if profit_target_pct <= 4.0:
        horizon, days_to_add = "Scalping (1-3 Days)", 3
      elif profit_target_pct <= 10.0:
        horizon, days_to_add = "Short-Term (3-10 Days)", 14
      elif profit_target_pct <= 20.0:
        horizon, days_to_add = "Swing Trading (10-25 Days)", 35
      else:
        horizon, days_to_add = "Long-Term (1-3 Years)", 365

      strict_deadline = (datetime.today() + timedelta(days=days_to_add)).strftime("%B %d, %Y")

      # Risk & Price Math
      stop_loss_price = round(current_price * 0.93, 2)
      target_price = round(current_price * (1 + profit_target_pct / 100.0), 2)
      risk_per_share = round(current_price - stop_loss_price, 2)
      reward_per_share = round(target_price - current_price, 2)
      risk_reward_ratio = round(reward_per_share / risk_per_share, 2) if risk_per_share > 0 else 0

      max_capital_risk = account_capital * 0.01
      shares_to_buy = int(max_capital_risk / risk_per_share) if risk_per_share > 0 else 0

      hist_1y = hist.tail(252)
      support_zone = f"₹{round(hist_1y['Low'].min(), 2)} - ₹{round(current_price * 0.97, 2)}"
      resistance_zone = f"₹{round(current_price * 1.03, 2)} - ₹{round(hist_1y['High'].max(), 2)}"
      breakout_trigger = f"₹{round(current_price * 1.015, 2)}"

      # --- CLEAN METRICS DASHBOARD ---
      col1, col2, col3, col4 = st.columns(4)
      with col1:
        st.metric("📊 Live Price", f"₹{current_price:,.2f}", f"{price_change:+.2f}%")
      with col2:
        st.metric("🎯 Quant Score", f"{total_score}/100")
      with col3:
        st.metric("⚖️ Risk / Reward", f"1:{risk_reward_ratio}")
      with col4:
        st.metric("⏳ Horizon", horizon)

      st.markdown("<br>", unsafe_allow_html=True)

      # --- VERDICT BANNER ---
      if verdict == "TAKE":
        st.markdown(f'<div class="verdict-take">✅ VERDICT: TAKE (Score: {total_score}/100) — High Probability Setup</div>', unsafe_allow_html=True)
      else:
        st.markdown(f'<div class="verdict-not">❌ VERDICT: NOT (Score: {total_score}/100) — Fails 75-Point Threshold</div>', unsafe_allow_html=True)

      st.markdown("<br>", unsafe_allow_html=True)

      # --- SMOOTH TABS ---
      tab1, tab2, tab3, tab4 = st.tabs([
          "📈 Chart Graphics",
          "🛡️ 5-Pillar Scorecard",
          "🎯 Execution Plan",
          "📚 Strategic Rules",
      ])

      with tab1:
        st.subheader("Interactive Price & Volume Action")
        chart_df = hist.tail(252)

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.75, 0.25], vertical_spacing=0.03
        )

        fig.add_trace(go.Candlestick(
            x=chart_df.index, open=chart_df['Open'], high=chart_df['High'],
            low=chart_df['Low'], close=chart_df['Close'], name="Price"
        ), row=1, col=1)

        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['EMA_20'], line=dict(color='#fbbf24', width=1.5), name="20 EMA"), row=1, col=1)
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['EMA_50'], line=dict(color='#38bdf8', width=1.5), name="50 EMA"), row=1, col=1)
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['EMA_200'], line=dict(color='#c084fc', width=2), name="200 EMA"), row=1, col=1)

        colors = ['#f87171' if r['Open'] - r['Close'] >= 0 else '#34d399' for _, r in chart_df.iterrows()]
        fig.add_trace(go.Bar(x=chart_df.index, y=chart_df['Volume'], marker_color=colors, name="Volume"), row=2, col=1)

        fig.update_layout(
            yaxis_title="Price (₹)",
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            height=550,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="#111827",
            plot_bgcolor="#111827"
        )
        st.plotly_chart(fig, use_container_width=True)

      with tab2:
        st.subheader("Quantitative Scorecard")
        score_data = {
            "Pillar Category": [
                "1. Fundamental & 200-EMA Baseline",
                "2. Technical Momentum & EMA Stack",
                "3. Chart Patterns & RSI Health",
                "4. Macro & Trend Regime",
                "5. Volume & Catalyst Sentiment",
            ],
            "Score Obtained": [f"{p1_score}/20", f"{p2_score}/20", f"{p3_score}/20", f"{p4_score}/20", f"{p5_score}/20"],
            "Status": [
                "✅ Passed" if p1_score >= 15 else "⚠️ Moderate",
                "✅ Passed" if p2_score >= 15 else "⚠️ Moderate",
                "✅ Passed" if p3_score >= 15 else "⚠️ Moderate",
                "✅ Passed" if p4_score >= 15 else "⚠️ Moderate",
                "✅ Passed" if p5_score >= 15 else "⚠️ Moderate",
            ],
        }
        st.table(pd.DataFrame(score_data))

      with tab3:
        st.subheader("Precision Execution & Sizing")
        col_a, col_b = st.columns(2)
        with col_a:
          st.markdown(f"**Asset / Ticker:** `{ticker}`")
          st.markdown(f"**Current Price:** `₹{current_price:,.2f}`")
          st.markdown(f"**Entry Trigger:** `{breakout_trigger}`")
          st.markdown(f"**Stop-Loss (7% Risk):** `₹{stop_loss_price:,.2f}` 🔴")
          st.markdown(f"**Target Price (+{profit_target_pct}%):** `₹{target_price:,.2f}` 🟢")
        with col_b:
          st.markdown(f"**Strict Deadline:** `{strict_deadline}` ⏳")
          st.markdown(f"**Max Capital Risk (1% Rule):** `₹{max_capital_risk:,.2f}`")
          st.markdown(f"**Risk Per Share:** `₹{risk_per_share:,.2f}`")
          st.markdown(f"**Allocation Shares:** `{shares_to_buy:,} units`")
          st.markdown(f"**Risk/Reward Ratio:** `1:{risk_reward_ratio}`")

        st.markdown("---")
        col_c, col_d, col_e = st.columns(3)
        with col_c:
          st.markdown(f"<div class='metric-card'><h4>Support</h4><p>{support_zone}</p></div>", unsafe_allow_html=True)
        with col_d:
          st.markdown(f"<div class='metric-card'><h4>Breakout</h4><p>{breakout_trigger}</p></div>", unsafe_allow_html=True)
        with col_e:
          st.markdown(f"<div class='metric-card'><h4>Resistance</h4><p>{resistance_zone}</p></div>", unsafe_allow_html=True)

      with tab4:
        st.subheader("Discipline & Safeguards")
        st.markdown("- **Operator Check (*Bulls, Bears & Beasts*):** Use volume spikes to filter false breakouts.")
        st.markdown("- **Structural Moat (*Coffee Can Investing*):** Maintain positions above the long-term baseline.")
        st.markdown(f"- **Strict Deadline Rule (*Stocks to Riches*):** **Never extend the deadline ({strict_deadline}).** Exit unconditionally if the target isn't met.")

    except Exception as e:
      st.error(f"⚠️ Error executing analysis engine: {e}")

else:
  st.info("👈 Enter a stock ticker in the sidebar and click **Run Evaluation**.")
