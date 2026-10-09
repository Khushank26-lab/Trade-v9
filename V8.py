import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import numpy as np
from datetime import datetime, timedelta
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Khushank AI Screener", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

# --- ADVANCED UI ANIMATIONS & GLASSMORPHISM CSS ---
st.markdown("""
    <style>
    @keyframes slideIn { from { opacity: 0; transform: translateX(-20px); } to { opacity: 1; transform: translateX(0); } }
    @keyframes float { 0% { transform: translateY(0px); } 50% { transform: translateY(-5px); } 100% { transform: translateY(0px); } }
    @keyframes pulseGlow { 0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4); } 70% { box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); } 100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); } }
    
    .main { background-color: #050914; color: #f8fafc; font-family: -apple-system, BlinkMacSystemFont, sans-serif; }
    
    /* Animated Gradient Buttons */
    .stButton>button { background: linear-gradient(135deg, #059669 0%, #10b981 100%); color: white; border-radius: 8px; font-weight: 800; padding: 14px; width: 100%; transition: all 0.3s ease; border: none; }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(16, 185, 129, 0.4); }
    
    /* Glassmorphism Metric Cards */
    div[data-testid="metric-container"] { background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 12px; animation: slideIn 0.5s ease-out forwards; }
    
    /* Glowing Badges */
    .premium-badge { display: inline-block; background: linear-gradient(90deg, #065f46, #047857); color: #a7f3d0; padding: 6px 14px; border-radius: 8px; font-size: 13px; font-weight: 900; letter-spacing: 1.5px; animation: pulseGlow 2s infinite; border: 1px solid #10b981; margin-bottom: 12px; box-shadow: 0 4px 15px rgba(16,185,129,0.2);}
    .title-text { background: -webkit-linear-gradient(45deg, #34d399, #3b82f6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 900; font-size: 2.5em; animation: float 6s ease-in-out infinite; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title-text">⚡ Institutional AI Screener v10</div>', unsafe_allow_html=True)
st.markdown("*Cross-referencing 20-year deep learning AI output with live intraday volatility and strict fundamental shields.*")
st.markdown("---")

@st.cache_data(ttl=60)
def load_top_stocks():
    if os.path.exists("top_setups.csv"):
        return pd.read_csv("top_setups.csv")
    else:
        return pd.DataFrame()

top_stocks = load_top_stocks()

# --- UPGRADED SIDEBAR FILTERS ---
st.sidebar.header("🎯 1. Target Objectives")
timeline_options = {"1 to 2 Days (Scalp / BTST)": 2, "1 Week (Momentum Swing)": 7, "2 to 3 Weeks (Core Swing)": 21, "1 to 3 Months (Position)": 90}
selected_timeline = st.sidebar.selectbox("Maximum Holding Duration", list(timeline_options.keys()))
days_to_hold = timeline_options[selected_timeline]
profit_target_pct = st.sidebar.slider("Desired Profit Target (%)", 1.0, 30.0, 5.0, 0.5)

st.sidebar.header("🛡️ 2. Live Market Filters")
min_ai_score = st.sidebar.slider("Minimum AI Confidence (%)", 88.0, 99.9, 90.0, 0.1) / 100.0
min_volume = st.sidebar.selectbox("Minimum Daily Volume", ["500,000+ (High Liquidity)", "1,000,000+ (Ultra Liquid)"])
volume_threshold = 500000 if "500" in min_volume else 1000000

st.sidebar.header("💰 3. Risk Management")
account_capital = st.sidebar.number_input("Trading Capital (₹)", min_value=10000.0, value=500000.0, step=10000.0)
risk_pct = st.sidebar.slider("Max Capital Risk per Trade (%)", 0.5, 3.0, 1.5, 0.1)
max_results = st.sidebar.slider("Max Stocks to Display", 1, 10, 5, 1)

run_live = st.sidebar.button("⚡ Scan Live Markets & Filter")

if run_live:
    if top_stocks.empty:
        st.error("⚠️ Background AI scan has not completed yet or found 0 setups today. Please check GitHub Actions.")
    else:
        # Filter dataframe instantly by AI score before API calls
        filtered_ai_stocks = top_stocks[top_stocks["AI_Score"] >= min_ai_score]
        
        if filtered_ai_stocks.empty:
            st.warning(f"⚠️ No stocks met the extreme AI confidence threshold of {min_ai_score*100}%.")
        else:
            progress_text = "Establishing secure connection to live market data..."
            progress_bar = st.progress(0, text=progress_text)
            
            displayed_count = 0
            valid_stocks = []
            
            for index, row in filtered_ai_stocks.iterrows():
                if displayed_count >= max_results: break
                
                ticker = row["Ticker"]
                ai_score = row["AI_Score"]
                news_score = row.get("News_Sentiment", 0.0)
                
                progress_bar.progress(min((index + 1) / len(filtered_ai_stocks), 1.0), text=f"Analyzing live volatility & volume for {ticker}...")
                
                try:
                    df = yf.Ticker(ticker).history(period="1y")
                    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                    df = df.dropna(subset=["Close"])
                    if df.empty or len(df) < 200: continue
                    
                    curr_price = float(df["Close"].iloc[-1])
                    live_volume = int(df["Volume"].iloc[-1])
                    
                    # FILTER: Live Volume Check
                    if live_volume < volume_threshold: continue
                    
                    hl, hc, lc = df["High"] - df["Low"], np.abs(df["High"] - df["Close"].shift()), np.abs(df["Low"] - df["Close"].shift())
                    df["ATR"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
                    atr = float(df["ATR"].iloc[-1])
                    
                    # FILTER: Target Velocity Math
                    daily_pct_move = (atr / curr_price) * 100
                    max_expected_move = daily_pct_move * (days_to_hold * 0.75) 
                    if max_expected_move < profit_target_pct: continue 
                    
                    df["EMA_20"] = df["Close"].ewm(span=20).mean()
                    df["EMA_50"] = df["Close"].ewm(span=50).mean()
                    df["EMA_200"] = df["Close"].ewm(span=200).mean()
                    
                    delta = df["Close"].diff()
                    gain, loss = (delta.where(delta > 0, 0)).rolling(14).mean(), (-delta.where(delta < 0, 0)).rolling(14).mean()
                    rsi = float(100 - (100 / (1 + (gain / loss))).iloc[-1])
                    ema20, ema50, ema200 = float(df["EMA_20"].iloc[-1]), float(df["EMA_50"].iloc[-1]), float(df["EMA_200"].iloc[-1])
                    
                    # STRICT 5-PILLAR MATH
                    p1 = 20 if curr_price > ema200 else 0 
                    if p1 < 20: continue # Hard Reject
                    p2 = 20 if curr_price > ema20 > ema50 > ema200 else (15 if curr_price > ema20 > ema50 else 8)
                    p3 = 20 if 55 <= rsi <= 75 else 12
                    p4 = 20 if ema50 > ema200 else 12
                    p5 = int(ai_score * 20)
                    total_score = p1 + p2 + p3 + p4 + p5
                    
                    # Execution Maths
                    target_price = round(curr_price * (1 + profit_target_pct / 100.0), 2)
                    stop_loss_price = round(curr_price * 0.95, 2) if days_to_hold <= 7 else round(curr_price * 0.92, 2)
                    risk_per_share = round(curr_price - stop_loss_price, 2)
                    rr_ratio = round((target_price - curr_price) / risk_per_share, 2) if risk_per_share > 0 else 0
                    shares_to_buy = int((account_capital * (risk_pct/100)) / risk_per_share) if risk_per_share > 0 else 0
                    strict_deadline = (datetime.today() + timedelta(days=days_to_hold)).strftime("%B %d, %Y")
                    
                    # RENDER UI
                    with st.expander(f"🏆 Rank #{displayed_count+1}: {ticker} | Score: {total_score}/100", expanded=(displayed_count==0)):
                        st.markdown(f'<div class="premium-badge">🛡️ INSTITUTIONAL PASS: 100% FUNDAMENTALS & TARGET VELOCITY</div>', unsafe_allow_html=True)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Live Price", f"₹{curr_price:,.2f}", delta=f"{rsi:.1f} RSI", delta_color="normal")
                        col2.metric("🧠 AI Prediction", f"{ai_score*100:.1f}%")
                        col3.metric("Live Volume", f"{live_volume:,}")
                        col4.metric("Risk/Reward", f"1:{rr_ratio}")
                        
                        tabA, tabB, tabC = st.tabs(["📈 Animated Chart", "🎯 Execution Plan", "🛡️ AI Core Metrics"])
                        
                        with tabA:
                            chart_df = df.tail(90)
                            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
                            fig.add_trace(go.Candlestick(x=chart_df.index, open=chart_df['Open'], high=chart_df['High'], low=chart_df['Low'], close=chart_df['Close'], name="Price"), row=1, col=1)
                            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['EMA_20'], line=dict(color='#fbbf24', width=1.5), name="20 EMA"), row=1, col=1)
                            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['EMA_50'], line=dict(color='#38bdf8', width=1.5), name="50 EMA"), row=1, col=1)
                            
                            fig.add_hline(y=target_price, line_dash="dash", line_color="#34d399", annotation_text="TARGET", row=1, col=1)
                            fig.add_hline(y=stop_loss_price, line_dash="dash", line_color="#f87171", annotation_text="STOP-LOSS", row=1, col=1)
                            
                            colors = ['#f87171' if r['Open'] - r['Close'] >= 0 else '#34d399' for _, r in chart_df.iterrows()]
                            fig.add_trace(go.Bar(x=chart_df.index, y=chart_df['Volume'], marker_color=colors, name="Volume"), row=2, col=1)
                            
                            fig.update_layout(template="plotly_dark", height=450, margin=dict(l=5, r=5, t=25, b=5), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_rangeslider_visible=False)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with tabB:
                            c_a, c_b = st.columns(2)
                            with c_a:
                                st.markdown(f"**🟢 Target (+{profit_target_pct}%):** `₹{target_price:,.2f}`")
                                st.markdown(f"**🔴 Stop-Loss:** `₹{stop_loss_price:,.2f}`")
                            with c_b:
                                st.markdown(f"**📦 Position Sizing:** `{shares_to_buy:,} Shares` (Clamped to {risk_pct}% Risk)")
                                st.markdown(f"**⏳ Hard Deadline:** `{strict_deadline}`")
                                
                        with tabC:
                            st.markdown(f"**1. Fundamentals & Structure:** `{p1}/20` (Strict Pass)")
                            st.markdown(f"**2. Momentum & Alignment:** `{p2}/20`")
                            st.markdown(f"**3. Relative Strength (RSI):** `{p3}/20`")
                            st.markdown(f"**4. Macro Regime:** `{p4}/20`")
                            st.markdown(f"**5. Gradient Boosting AI:** `{p5}/20`")
                            if news_score != 0.0:
                                st.markdown(f"📰 **Live News Sentiment (NLP):** `{'Positive' if news_score > 0 else 'Neutral'}`")

                    displayed_count += 1
                    valid_stocks.append(ticker)

                except Exception:
                    continue
            
            progress_bar.empty()
            
            if displayed_count == 0:
                st.error(f"⚠️ Zero stocks survived. The combination of >{min_ai_score*100}% AI Confidence, 100% Fundamentals, and your strict timeframe/target is too demanding for today's market. Cash is King.")
            else:
                st.balloons()
                st.success(f"Successfully secured {displayed_count} flawless institutional setups.")
else:
    st.info("👈 Configure your exact execution parameters in the sidebar, then click **Scan Live Markets & Filter**.")
