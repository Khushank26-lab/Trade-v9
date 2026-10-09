import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="v9 Live AI Screener", page_icon="⚡", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #0b0f19; color: #f8fafc; font-family: -apple-system, BlinkMacSystemFont, sans-serif;}
    .stButton>button { background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; border-radius: 8px; font-weight: bold; padding: 12px; width: 100%;}
    .metric-card { background: #111827; padding: 15px; border-radius: 10px; border: 1px solid #1f2937; text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ v10 AI Live Market Screener")
st.markdown("*Filters the background AI scan against your strict timeline and target using live intraday ATR velocity.*")

# 1. Read the pre-scanned Top Stocks from the background robot
@st.cache_data(ttl=60)
def load_top_stocks():
    if os.path.exists("top_setups.csv"):
        return pd.read_csv("top_setups.csv")
    else:
        return pd.DataFrame([{"Ticker": "RELIANCE.NS", "AI_Score": 0.85}, {"Ticker": "BSE.NS", "AI_Score": 0.82}])

top_stocks = load_top_stocks()

# --- SIDEBAR: BRINGING BACK TIMELINE & TARGETS ---
st.sidebar.header("🎯 Live Execution Goals")

timeline_options = {
    "1 to 2 Days (Scalp / BTST)": 2,
    "1 Week (Momentum Swing)": 7,
    "2 to 3 Weeks (Core Swing)": 21,
    "1 to 3 Months (Position)": 90
}
selected_timeline = st.sidebar.selectbox("Maximum Time to Hold?", list(timeline_options.keys()))
days_to_hold = timeline_options[selected_timeline]

profit_target_pct = st.sidebar.slider("Desired Profit Target (%)", 1.0, 30.0, 5.0, 0.5)
max_results = st.sidebar.slider("Max Stocks to Show", 1, 10, 5, 1)
account_capital = st.sidebar.number_input("Capital (₹)", min_value=10000.0, value=500000.0, step=10000.0)
run_live = st.sidebar.button("Fetch Live Charts & Filter")

if run_live:
    st.success(f"Scanning the AI-approved list to find stocks that can hit **{profit_target_pct}%** in **{selected_timeline}**...")
    
    displayed_count = 0
    
    for index, row in top_stocks.iterrows():
        if displayed_count >= max_results:
            break
            
        ticker = row["Ticker"]
        ai_score = row["AI_Score"]
        
        try:
            # FETCH LIVE INTRADAY DATA
            df = yf.Ticker(ticker).history(period="1y")
            
            # Clean MultiIndex & NaNs
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna(subset=["Close"])
            
            if df.empty or len(df) < 200:
                continue
                
            curr_price = float(df["Close"].iloc[-1])
            
            # --- CALCULATE ATR & VELOCITY ---
            hl = df["High"] - df["Low"]
            hc = np.abs(df["High"] - df["Close"].shift())
            lc = np.abs(df["Low"] - df["Close"].shift())
            df["ATR"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
            atr = float(df["ATR"].iloc[-1])
            
            # The Magic Filter: Can it actually hit the target in this timeline?
            daily_pct_move = (atr / curr_price) * 100
            max_expected_move = daily_pct_move * (days_to_hold * 0.75) 
            
            if max_expected_move < profit_target_pct:
                continue # Skip! This stock is too slow for the user's timeline.
            
            # Live Indicators
            df["EMA_20"] = df["Close"].ewm(span=20).mean()
            df["EMA_50"] = df["Close"].ewm(span=50).mean()
            df["EMA_200"] = df["Close"].ewm(span=200).mean()
            
            delta = df["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = float(100 - (100 / (1 + rs)).iloc[-1])
            
            ema20, ema50, ema200 = float(df["EMA_20"].iloc[-1]), float(df["EMA_50"].iloc[-1]), float(df["EMA_200"].iloc[-1])
            
            p1 = 20 if curr_price > ema200 else 8
            if curr_price > ema20 > ema50 > ema200: p2 = 20
            elif curr_price > ema20 > ema50: p2 = 15
            else: p2 = 8
            
            p3 = 20 if 55 <= rsi <= 75 else 12
            p4 = 20 if ema50 > ema200 else 12
            p5 = int(ai_score * 20)
            total_score = p1 + p2 + p3 + p4 + p5
            
            # Trade Math
            target_price = round(curr_price * (1 + profit_target_pct / 100.0), 2)
            stop_loss_price = round(curr_price * 0.95, 2) if days_to_hold <= 7 else round(curr_price * 0.92, 2)
            risk_per_share = round(curr_price - stop_loss_price, 2)
            reward_per_share = round(target_price - curr_price, 2)
            rr_ratio = round(reward_per_share / risk_per_share, 2) if risk_per_share > 0 else 0
            shares_to_buy = int((account_capital * 0.015) / risk_per_share) if risk_per_share > 0 else 0
            strict_deadline = (datetime.today() + timedelta(days=days_to_hold)).strftime("%B %d, %Y")
            
            # UI Rendering
            with st.expander(f"🏆 Rank #{displayed_count+1}: {ticker} | Live Price: ₹{curr_price:,.2f} | Score: {total_score}/100", expanded=(displayed_count==0)):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Live Price", f"₹{curr_price:,.2f}")
                col2.metric("AI Prediction", f"{ai_score*100:.1f}%")
                col3.metric("Live RSI", f"{rsi:.1f}")
                col4.metric("Risk/Reward", f"1:{rr_ratio}")
                
                tabA, tabB, tabC = st.tabs(["📈 Live Chart", "🎯 Execution Plan", "🛡️ Scorecard"])
                
                with tabA:
                    chart_df = df.tail(90)
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
                    
                    fig.add_trace(go.Candlestick(x=chart_df.index, open=chart_df['Open'], high=chart_df['High'], low=chart_df['Low'], close=chart_df['Close'], name="Live Price"), row=1, col=1)
                    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['EMA_20'], line=dict(color='#fbbf24', width=1.5), name="20 EMA"), row=1, col=1)
                    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['EMA_50'], line=dict(color='#38bdf8', width=1.5), name="50 EMA"), row=1, col=1)
                    
                    fig.add_hline(y=target_price, line_dash="dash", line_color="#34d399", annotation_text="TARGET", row=1, col=1)
                    fig.add_hline(y=stop_loss_price, line_dash="dash", line_color="#f87171", annotation_text="STOP-LOSS", row=1, col=1)
                    
                    colors = ['#f87171' if r['Open'] - r['Close'] >= 0 else '#34d399' for _, r in chart_df.iterrows()]
                    fig.add_trace(go.Bar(x=chart_df.index, y=chart_df['Volume'], marker_color=colors, name="Volume"), row=2, col=1)
                    
                    fig.update_layout(template="plotly_dark", height=450, margin=dict(l=5, r=5, t=25, b=5), paper_bgcolor="#111827", plot_bgcolor="#111827", xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)
                
                with tabB:
                    c_a, c_b = st.columns(2)
                    with c_a:
                        st.markdown(f"**Target (+{profit_target_pct}%):** `₹{target_price:,.2f}` 🟢")
                        st.markdown(f"**Stop-Loss:** `₹{stop_loss_price:,.2f}` 🔴")
                    with c_b:
                        st.markdown(f"**Shares to Buy:** `{shares_to_buy:,}`")
                        st.markdown(f"**Deadline:** `{strict_deadline}` ⏳")
                        
                with tabC:
                    st.markdown(f"**1. Fundamentals:** `{p1}/20` | **2. Momentum:** `{p2}/20` | **3. RSI:** `{p3}/20` | **4. Macro:** `{p4}/20` | **5. AI:** `{p5}/20`")

            displayed_count += 1

        except Exception as e:
            continue
            
    if displayed_count == 0:
        st.error(f"⚠️ None of the AI's top picks have enough volatility (ATR) right now to hit **{profit_target_pct}%** in just **{selected_timeline}**. Please lower your profit target or increase your timeline.")
else:
    st.info("Select your timeline and target on the left, then click **Fetch Live Charts & Filter**.")
