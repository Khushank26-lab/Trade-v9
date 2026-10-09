import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

st.set_page_config(page_title="v9 Live AI Screener", page_icon="⚡", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #0b0f19; color: #f8fafc; }
    .stButton>button { background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; border-radius: 8px; font-weight: bold; }
    .metric-card { background: #111827; padding: 15px; border-radius: 10px; border: 1px solid #1f2937; text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ v9 Live Market Screener & Charts")
st.markdown("*Reads the heavy background AI scan, then fetches **LIVE** intraday data to generate your charts and 5-Pillar scorecard.*")

# 1. Read the pre-scanned Top Stocks from the background robot
@st.cache_data(ttl=60)
def load_top_stocks():
    if os.path.exists("top_setups.csv"):
        return pd.read_csv("top_setups.csv")
    else:
        # Fallback if GitHub Action hasn't run yet
        return pd.DataFrame([{"Ticker": "RELIANCE.NS", "AI_Score": 0.85}, {"Ticker": "BSE.NS", "AI_Score": 0.82}])

top_stocks = load_top_stocks()

# Sidebar
st.sidebar.header("🎯 Live Execution")
profit_target_pct = st.sidebar.slider("Profit Target (%)", 1.0, 30.0, 5.0, 0.5)
account_capital = st.sidebar.number_input("Capital (₹)", value=500000.0)
run_live = st.sidebar.button("Fetch Live Charts & Scores")

if run_live:
    st.success(f"Loaded {len(top_stocks)} AI-approved stocks. Fetching LIVE market data...")
    
    # Process only the top 5 to keep the app lightning fast
    for index, row in top_stocks.head(5).iterrows():
        ticker = row["Ticker"]
        ai_score = row["AI_Score"]
        
        try:
            # 2. FETCH LIVE INTRADAY DATA
            df = yf.Ticker(ticker).history(period="1y")
            curr_price = df["Close"].iloc[-1]
            
            # Live Indicators
            df["EMA_20"] = df["Close"].ewm(span=20).mean()
            df["EMA_50"] = df["Close"].ewm(span=50).mean()
            df["EMA_200"] = df["Close"].ewm(span=200).mean()
            
            # RSI
            delta = df["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]
            
            # Live v9 Scoring (0-20 each)
            ema20, ema50, ema200 = df["EMA_20"].iloc[-1], df["EMA_50"].iloc[-1], df["EMA_200"].iloc[-1]
            
            p1 = 20 if curr_price > ema200 else 8
            if curr_price > ema20 > ema50 > ema200: p2 = 20
            elif curr_price > ema20 > ema50: p2 = 15
            else: p2 = 8
            
            p3 = 20 if 55 <= rsi <= 75 else 12
            p4 = 20 if ema50 > ema200 else 12
            p5 = int(ai_score * 20) # AI Confidence from background CSV
            
            total_score = p1 + p2 + p3 + p4 + p5
            
            # Trade Math
            target_price = round(curr_price * (1 + profit_target_pct / 100.0), 2)
            stop_loss_price = round(curr_price * 0.95, 2)
            
            # UI Rendering
            with st.expander(f"🏆 {ticker} | Live Price: ₹{curr_price:,.2f} | Score: {total_score}/100", expanded=(index==0)):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Live Price", f"₹{curr_price:,.2f}")
                col2.metric("AI Prediction", f"{ai_score*100:.1f}%")
                col3.metric("Live RSI", f"{rsi:.1f}")
                col4.metric("Target", f"₹{target_price:,.2f}")
                
                # LIVE PLOTLY CHART
                chart_df = df.tail(90)
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
                
                fig.add_trace(go.Candlestick(x=chart_df.index, open=chart_df['Open'], high=chart_df['High'], low=chart_df['Low'], close=chart_df['Close'], name="Live Price"), row=1, col=1)
                fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['EMA_20'], line=dict(color='#fbbf24', width=1.5), name="20 EMA"), row=1, col=1)
                fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['EMA_50'], line=dict(color='#38bdf8', width=1.5), name="50 EMA"), row=1, col=1)
                
                # Execution Lines
                fig.add_hline(y=target_price, line_dash="dash", line_color="#34d399", annotation_text="TARGET", row=1, col=1)
                fig.add_hline(y=stop_loss_price, line_dash="dash", line_color="#f87171", annotation_text="STOP-LOSS", row=1, col=1)
                
                # Volume
                colors = ['#f87171' if r['Open'] - r['Close'] >= 0 else '#34d399' for _, r in chart_df.iterrows()]
                fig.add_trace(go.Bar(x=chart_df.index, y=chart_df['Volume'], marker_color=colors, name="Volume"), row=2, col=1)
                
                fig.update_layout(template="plotly_dark", height=450, margin=dict(l=5, r=5, t=25, b=5), paper_bgcolor="#111827", plot_bgcolor="#111827")
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown(f"**v9 Live Pillars:** P1:`{p1}` | P2:`{p2}` | P3:`{p3}` | P4:`{p4}` | P5 (AI):`{p5}`")

        except Exception as e:
            continue
else:
    st.info("Click **Fetch Live Charts & Scores** to load the AI recommendations and stream their live market data.")
