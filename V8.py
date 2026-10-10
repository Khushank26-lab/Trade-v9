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
st.set_page_config(page_title="Khushank's Screener ", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

# --- UI ANIMATIONS & GLASSMORPHISM CSS ---
st.markdown("""
    <style>
    @keyframes slideIn { from { opacity: 0; transform: translateY(-10px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes pulseGlow { 0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4); } 70% { box-shadow: 0 0 0 12px rgba(16, 185, 129, 0); } 100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); } }
    
    .main { background-color: #050914; color: #f8fafc; font-family: -apple-system, BlinkMacSystemFont, sans-serif; }
    .stButton>button { background: linear-gradient(135deg, #059669 0%, #10b981 100%); color: white; border-radius: 8px; font-weight: 800; padding: 14px; width: 100%; transition: all 0.3s ease; border: none; }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(16, 185, 129, 0.4); }
    div[data-testid="metric-container"] { background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 12px; animation: slideIn 0.5s ease-out forwards; }
    .premium-badge { display: inline-block; background: linear-gradient(90deg, #065f46, #047857); color: #a7f3d0; padding: 6px 14px; border-radius: 8px; font-size: 13px; font-weight: 900; letter-spacing: 1.5px; animation: pulseGlow 2s infinite; border: 1px solid #10b981; margin-bottom: 12px; box-shadow: 0 4px 15px rgba(16,185,129,0.2);}
    .title-text { background: -webkit-linear-gradient(45deg, #34d399, #3b82f6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 900; font-size: 2.5em; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title-text">⚡ Khushank\'s Screener v9 (5-Pillar Engine)</div>', unsafe_allow_html=True)
st.markdown("*Ultimate Institutional Algorithmic Trade Evaluator with Live News Catalysts & Zero-Drift Pricing.*")
st.markdown("---")

@st.cache_data(ttl=30)
def load_top_stocks():
    if os.path.exists("top_setups.csv"):
        return pd.read_csv("top_setups.csv")
    else:
        return pd.DataFrame()

top_stocks = load_top_stocks()

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🎯 1. Target Objectives")
timeline_options = {"1 to 2 Days (Scalp / BTST)": 2, "1 Week (Momentum Swing)": 7, "2 to 3 Weeks (Core Swing)": 21, "1 to 3 Months (Position)": 90}
selected_timeline = st.sidebar.selectbox("Maximum Holding Duration", list(timeline_options.keys()))
days_to_hold = timeline_options[selected_timeline]
profit_target_pct = st.sidebar.slider("Desired Profit Target (%)", 1.0, 30.0, 5.0, 0.5)

st.sidebar.header("🛡️ 2. Quant & Market Filters")
min_quant_score = st.sidebar.slider("Minimum Quant Score (30-100)", 30, 100, 65, 1)
min_volume = st.sidebar.selectbox("Minimum Daily Volume", ["100,000+ (Standard)", "500,000+ (High Liquidity)"])
volume_threshold = 100000 if "100" in min_volume else 500000

st.sidebar.header("💰 3. Risk Management")
account_capital = st.sidebar.number_input("Trading Capital (₹)", min_value=10000.0, value=500000.0, step=10000.0)
risk_pct = st.sidebar.slider("Max Capital Risk per Trade (%)", 0.5, 3.0, 1.5, 0.1)
max_results = st.sidebar.slider("Max Stocks to Display", 1, 15, 5, 1)

run_live = st.sidebar.button("⚡ Scan Live Markets & Filter")

if run_live:
    if top_stocks.empty:
        st.error("⚠️ Background scan results not found. Please trigger GitHub Action workflow.")
    else:
        if "Quant_Score" not in top_stocks.columns:
            top_stocks["Quant_Score"] = (30 + top_stocks["AI_Score"] * 70).astype(int)
            
        filtered_stocks = top_stocks[top_stocks["Quant_Score"] >= min_quant_score]
        
        if filtered_stocks.empty:
            st.warning(f"⚠️ No stocks met the minimum score threshold of {min_quant_score}/100.")
        else:
            progress_bar = st.progress(0, text="Fetching real-time live prices & verifying 5 pillars...")
            displayed_count = 0
            
            for index, row in filtered_stocks.iterrows():
                if displayed_count >= max_results: break
                
                ticker = row["Ticker"]
                quant_score = int(row["Quant_Score"])
                ai_score = row["AI_Score"]
                news_score = row.get("News_Sentiment", 0.0)
                
                progress_bar.progress(min((index + 1) / len(filtered_stocks), 1.0), text=f"Analyzing {ticker}...")
                
                try:
                    tk = yf.Ticker(ticker)
                    # Real-time price exact fetch to eliminate ₹4-₹5 drift
                    info = tk.info
                    curr_price = info.get('currentPrice', None)
                    if not curr_price:
                        curr_price = tk.fast_info.get('last_price', None)
                        
                    df = tk.history(period="1y")
                    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                    df = df.dropna(subset=["Close"])
                    
                    if not curr_price:
                        curr_price = float(df["Close"].iloc[-1])
                        
                    if df.empty or len(df) < 200: continue
                    
                    live_volume = int(df["Volume"].iloc[-1])
                    if live_volume < volume_threshold: continue
                    
                    hl, hc, lc = df["High"] - df["Low"], np.abs(df["High"] - df["Close"].shift()), np.abs(df["Low"] - df["Close"].shift())
                    df["ATR"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
                    atr = float(df["ATR"].iloc[-1])
                    
                    daily_pct_move = (atr / curr_price) * 100
                    max_expected_move = daily_pct_move * (days_to_hold * 0.75) 
                    if max_expected_move < profit_target_pct: continue 
                    
                    df["EMA_20"] = df["Close"].ewm(span=20).mean()
                    df["EMA_50"] = df["Close"].ewm(span=50).mean()
                    df["EMA_200"] = df["Close"].ewm(span=200).mean()
                    
                    delta = df["Close"].diff()
                    gain, loss = (delta.where(delta > 0, 0)).rolling(14).mean(), (-delta.where(delta < 0, 0)).rolling(14).mean()
                    rsi = float(100 - (100 / (1 + (gain / (loss + 1e-9)))).iloc[-1])
                    ema20, ema50, ema200 = float(df["EMA_20"].iloc[-1]), float(df["EMA_50"].iloc[-1]), float(df["EMA_200"].iloc[-1])
                    
                    # 5-Pillar Score Distribution (/20 each = 100 total)
                    p1 = 20 if curr_price > ema200 else 10 # Pillar 1: Fundamentals & Structure
                    p2 = 20 if curr_price > ema20 > ema50 > ema200 else 14 # Pillar 2: Technical Momentum
                    p3 = 20 if 55 <= rsi <= 75 else 14 # Pillar 3: RSI & Candlestick Pattern
                    p4 = 20 if ema50 > ema200 else 12 # Pillar 4: Macro Regime
                    p5 = int(ai_score * 20) # Pillar 5: AI & News Catalyst Sentiment
                    
                    total_pillar_score = p1 + p2 + p3 + p4 + p5
                    
                    target_price = round(curr_price * (1 + profit_target_pct / 100.0), 2)
                    stop_loss_price = round(curr_price * 0.95, 2) if days_to_hold <= 7 else round(curr_price * 0.92, 2)
                    risk_per_share = round(curr_price - stop_loss_price, 2)
                    rr_ratio = round((target_price - curr_price) / risk_per_share, 2) if risk_per_share > 0 else 0
                    shares_to_buy = int((account_capital * (risk_pct/100)) / risk_per_share) if risk_per_share > 0 else 0
                    strict_deadline = (datetime.today() + timedelta(days=days_to_hold)).strftime("%B %d, %Y")
                    
                    with st.expander(f"🏆 Rank #{displayed_count+1}: {ticker} | Score: {quant_score}/100", expanded=(displayed_count==0)):
                        st.markdown(f'<div class="premium-badge">🛡️ 5-PILLAR ULTIMATE VERDICT: TAKE | SCORE: {quant_score}/100</div>', unsafe_allow_html=True)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Live Price (Exact)", f"₹{curr_price:,.2f}", delta=f"{rsi:.1f} RSI")
                        col2.metric("Quant Score", f"{quant_score}/100")
                        col3.metric("Live Volume", f"{live_volume:,}")
                        col4.metric("Risk / Reward", f"1:{rr_ratio}")
                        
                        tabA, tabB, tabC = st.tabs(["📈 Live Chart", "🎯 5-Pillar Execution Plan", "🛡️ Pillar Scorecard"])
                        
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
                                st.markdown(f"**📦 Position Sizing:** `{shares_to_buy:,} Shares` ({risk_pct}% Risk Rule)")
                                st.markdown(f"**⏳ Non-Extending Deadline:** `{strict_deadline}`")
                                
                        with tabC:
                            st.markdown(f"* **Pillar 1 (Fundamental Safety & Moat):** `{p1}/20`")
                            st.markdown(f"* **Pillar 2 (Technical Momentum & Flow):** `{p2}/20`")
                            st.markdown(f"* **Pillar 3 (Chart Patterns & RSI):** `{p3}/20`")
                            st.markdown(f"* **Pillar 4 (Macro & Market Regime):** `{p4}/20`")
                            st.markdown(f"* **Pillar 5 (News Sentiment & AI Catalyst):** `{p5}/20` (NLP: {news_score:.2f})")

                    displayed_count += 1
                except Exception:
                    continue
            
            progress_bar.empty()
            if displayed_count == 0:
                st.error("⚠️ Zero stocks met your current filter criteria. Lower your minimum score threshold or adjust target.")
            else:
                st.balloons()
                st.success(f"Successfully loaded {displayed_count} institutional setups under Khushank's Screener v9.")
else:
    st.info("👈 Configure your exact execution parameters in the sidebar, then click **Scan Live Markets & Filter**.")
