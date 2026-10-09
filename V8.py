from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import yfinance as yf

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="v11 AI/ML Stock Screener",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- ULTRA-SMOOTH CUSTOM CSS ---
st.markdown(
    """
    <style>
    .main { background-color: #0b0f19; color: #f8fafc; font-family: -apple-system, BlinkMacSystemFont, sans-serif; }
    .stButton>button { width: 100%; background: linear-gradient(135deg, #8b5cf6 0%, #6d28d9 100%); color: white; font-weight: 600; border-radius: 8px; border: none; padding: 12px; transition: all 0.3s ease; }
    .stButton>button:hover { background: linear-gradient(135deg, #a78bfa 0%, #8b5cf6 100%); box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4); }
    .metric-card { background: #111827; padding: 16px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.08); text-align: center; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
    .verdict-take { background: rgba(139, 92, 246, 0.15); color: #c4b5fd; padding: 14px; border-radius: 10px; text-align: center; font-weight: 700; font-size: 20px; border: 1px solid rgba(139, 92, 246, 0.4); }
    .verdict-not { background: rgba(239, 68, 68, 0.1); color: #f87171; padding: 14px; border-radius: 10px; text-align: center; font-weight: 700; font-size: 20px; border: 1px solid rgba(248, 113, 113, 0.3); }
    </style>
""",
    unsafe_allow_html=True,
)

# --- WATCHLIST ---
WATCHLIST = [
    "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "TCS.NS",
    "ITC.NS", "LT.NS", "SBIN.NS", "BHARTIARTL.NS", "BAJFINANCE.NS",
    "M&M.NS", "MARUTI.NS", "SUNPHARMA.NS", "TATASTEEL.NS", "TATAMOTORS.NS",
    "NTPC.NS", "POWERGRID.NS", "ZOMATO.NS", "HAL.NS", "TRENT.NS"
]

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period="2y") # 2 years for better ML training
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(subset=["Close"])

def calculate_technicals_and_ml_features(df, days_to_hold, target_pct):
    # Standard EMAs & RSI
    df["EMA_20"] = df["Close"].ewm(span=20).mean()
    df["EMA_50"] = df["Close"].ewm(span=50).mean()
    df["EMA_200"] = df["Close"].ewm(span=200).mean()
    
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    
    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    df['ATR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()
    
    # --- MACHINE LEARNING FEATURE ENGINEERING ---
    df['F_EMA20_Dist'] = (df['Close'] - df['EMA_20']) / df['EMA_20']
    df['F_EMA50_Dist'] = (df['Close'] - df['EMA_50']) / df['EMA_50']
    df['F_RSI_Norm'] = df['RSI'] / 100.0
    df['F_ATR_Pct'] = df['ATR'] / df['Close']
    
    # Target Variable: Did it hit the target profit within 'days_to_hold' in the future?
    future_highs = df['High'].rolling(window=days_to_hold).max().shift(-days_to_hold)
    df['Target_Hit'] = np.where((future_highs - df['Close']) / df['Close'] >= (target_pct / 100.0), 1, 0)
    
    return df

def run_ml_prediction(df):
    ml_df = df.dropna().copy()
    features = ['F_EMA20_Dist', 'F_EMA50_Dist', 'F_RSI_Norm', 'F_ATR_Pct']
    
    # Need historical rows (excluding today which has no future data yet)
    X_train = ml_df[features].iloc[:-1]
    y_train = ml_df['Target_Hit'].iloc[:-1]
    X_today = ml_df[features].iloc[[-1]]
    
    # If target was never hit historically, or always hit, model can't train properly
    if len(np.unique(y_train)) < 2:
        return 10.0, 0.50 # Neutral default
        
    # Train Random Forest AI
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict Probability of Success for Today
    prob_success = model.predict_proba(X_today)[0][1]
    
    # Convert probability (0-1) to Pillar Score (0-20)
    ai_score = int(prob_success * 20)
    return ai_score, prob_success

# --- SIDEBAR INPUTS ---
st.sidebar.header("🎯 Investment Goal")

timeline_options = {
    "1 to 2 Days (Scalp / BTST)": 2,
    "1 Week (Short Swing)": 7,
    "2 to 3 Weeks (Swing)": 21,
    "1 to 3 Months (Position)": 90
}
selected_timeline = st.sidebar.selectbox("Maximum Time to Hold?", list(timeline_options.keys()))
days_to_hold = timeline_options[selected_timeline]

profit_target_pct = st.sidebar.slider(
    "Desired Profit Target (%)", min_value=1.0, max_value=30.0, value=5.0, step=0.5
)
account_capital = st.sidebar.number_input(
    "Total Trading Capital (₹)", min_value=10000.0, value=500000.0, step=10000.0
)

run_scan = st.sidebar.button("🧠 Run AI & ML Screener")

# --- MAIN APP HEADER ---
st.title("🧠 v11 Machine Learning Screener & Trade Engine")
st.markdown("*Trains a Random Forest AI model on-the-fly for every stock to predict the exact probability of hitting your target.*")
st.markdown("---")

if run_scan:
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    best_stock = None
    best_score = -1
    best_df = None
    best_metrics = {}

    # --- AI SCANNER ENGINE ---
    for i, ticker in enumerate(WATCHLIST):
        status_text.text(f"Training AI model for {ticker} ({i+1}/{len(WATCHLIST)})...")
        progress_bar.progress((i + 1) / len(WATCHLIST))
        
        try:
            df = fetch_stock_data(ticker)
            if len(df) < 200: continue
            
            df = calculate_technicals_and_ml_features(df, days_to_hold, profit_target_pct)
            
            curr_price = df["Close"].iloc[-1]
            ema20, ema50, ema200 = df["EMA_20"].iloc[-1], df["EMA_50"].iloc[-1], df["EMA_200"].iloc[-1]
            rsi, atr = df["RSI"].iloc[-1], df["ATR"].iloc[-1]
            
            # 1. ATR Velocity Check
            daily_pct_move = (atr / curr_price) * 100
            max_expected_move = daily_pct_move * (days_to_hold * 0.7)
            if max_expected_move < profit_target_pct:
                continue 
                
            # 2. Train ML & Get Prediction
            ai_pillar_score, ai_probability = run_ml_prediction(df)
            
            # 3. Traditional Chart Scoring
            p1 = 20 if curr_price > ema200 else 10
            if curr_price > ema20 > ema50 > ema200: p2 = 20
            elif curr_price > ema20 > ema50: p2 = 16
            else: p2 = 10
            
            if 55 <= rsi <= 75: p3 = 20 
            elif 40 <= rsi < 55: p3 = 14 
            else: p3 = 8
            
            p4 = 20 if ema50 > ema200 else 10 
            p5 = ai_pillar_score # AI PREDICTION REPLACES OLD VOLUME PILLAR
            
            total_score = p1 + p2 + p3 + p4 + p5
            
            if total_score > best_score:
                best_score = total_score
                best_stock = ticker
                best_df = df
                best_metrics = {
                    "p1": p1, "p2": p2, "p3": p3, "p4": p4, "p5": p5,
                    "curr_price": curr_price, "rsi": rsi, "atr": atr, "ai_prob": ai_probability
                }
                
        except Exception as e:
            continue
            
    progress_bar.empty()
    status_text.empty()
    
    # --- RENDER RESULTS ---
    if best_stock is None:
        st.error(f"⚠️ **No setup found.** The AI determined that achieving **{profit_target_pct}%** in **{selected_timeline}** is highly improbable right now based on historical volatility and current market structure.")
    else:
        st.success(f"🎉 Best AI Prediction Found: **{best_stock}**")
        
        curr_price = best_metrics["curr_price"]
        prev_price = best_df["Close"].iloc[-2]
        price_change = ((curr_price - prev_price) / prev_price) * 100
        
        stop_loss_price = round(curr_price * 0.95, 2) if days_to_hold <= 7 else round(curr_price * 0.92, 2)
        target_price = round(curr_price * (1 + profit_target_pct / 100.0), 2)
        risk_per_share = round(curr_price - stop_loss_price, 2)
        reward_per_share = round(target_price - curr_price, 2)
        risk_reward_ratio = round(reward_per_share / risk_per_share, 2) if risk_per_share > 0 else 0

        max_capital_risk = account_capital * 0.015
        shares_to_buy = int(max_capital_risk / risk_per_share) if risk_per_share > 0 else 0
        
        strict_deadline = (datetime.today() + timedelta(days=days_to_hold)).strftime("%B %d, %Y")
        
        support_zone = f"₹{round(best_df['Low'].tail(20).min(), 2)} - ₹{round(curr_price * 0.98, 2)}"
        breakout_trigger = f"₹{round(curr_price * 1.005, 2)}"

        # --- METRICS DASHBOARD ---
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Selected Stock", f"{best_stock}", f"{price_change:+.2f}%")
        with col2:
            st.metric("🤖 AI Confidence", f"{best_metrics['ai_prob']*100:.1f}%")
        with col3:
            st.metric("🎯 Total Score", f"{best_score}/100")
        with col4:
            st.metric("⚖️ Risk/Reward", f"1:{risk_reward_ratio}")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="verdict-take">✅ VERDICT: TAKE {best_stock} — AI identifies a {best_metrics["ai_prob"]*100:.1f}% probability of hitting {profit_target_pct}% profit in {selected_timeline}.</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # --- SMOOTH TABS ---
        tab1, tab2, tab3 = st.tabs([
            "📈 Advanced Candlestick Chart",
            "🛡️ 5-Pillar Scorecard",
            "🎯 Execution Plan"
        ])

        with tab1:
            st.subheader(f"Chart Analysis for {best_stock}")
            st.info(f"📌 The Random Forest ML model analyzed the historical patterns of this exact chart structure and output a {best_metrics['ai_prob']*100:.1f}% success rate.")
            
            chart_df = best_df.tail(120) 
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)

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
                yaxis_title="Price (₹)", xaxis_rangeslider_visible=False,
                template="plotly_dark", height=550, margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="#111827", plot_bgcolor="#111827"
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("Why was this stock selected?")
            score_data = {
                "Pillar Category (AI & Chart-Trained)": [
                    "1. Price > 200 EMA (Structural Trend)",
                    "2. EMA Stack Alignment (20>50>200)",
                    "3. RSI Breakout Momentum (55-75 Zone)",
                    "4. Macro Trend Validation",
                    "5. AI/ML Predictive Confidence (Random Forest)",
                ],
                "Score": [f"{best_metrics['p1']}/20", f"{best_metrics['p2']}/20", f"{best_metrics['p3']}/20", f"{best_metrics['p4']}/20", f"{best_metrics['p5']}/20"],
            }
            st.table(pd.DataFrame(score_data))

        with tab3:
            st.subheader("Automated Execution & Sizing")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**Best Matched Asset:** `{best_stock}`")
                st.markdown(f"**Current Price:** `₹{curr_price:,.2f}`")
                st.markdown(f"**Stop-Loss:** `₹{stop_loss_price:,.2f}` 🔴")
                st.markdown(f"**Target Price (+{profit_target_pct}%):** `₹{target_price:,.2f}` 🟢")
            with col_b:
                st.markdown(f"**Time Deadline:** `{strict_deadline}` ⏳")
                st.markdown(f"**Capital Risked:** `₹{max_capital_risk:,.2f}`")
                st.markdown(f"**Shares to Buy:** `{shares_to_buy:,} units`")
                st.markdown(f"**Risk/Reward Ratio:** `1:{risk_reward_ratio}`")
            
            st.markdown("---")
            st.markdown(f"**Chart Support Base:** `{support_zone}` | **Entry Trigger:** `{breakout_trigger}`")

else:
    st.info("👈 Select your desired timeline and profit target in the sidebar, then click **Run AI & ML Screener**.")
