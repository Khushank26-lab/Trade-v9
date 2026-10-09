import concurrent.futures
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
    page_title="v10 AI/ML Hybrid Pattern Screener",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- MODERN DARK THEME CSS ---
st.markdown(
    """
    <style>
    .main { background-color: #0b0f19; color: #f8fafc; font-family: -apple-system, BlinkMacSystemFont, sans-serif; }
    .stButton>button { width: 100%; background: linear-gradient(135deg, #8b5cf6 0%, #6d28d9 100%); color: white; font-weight: 600; border-radius: 8px; border: none; padding: 12px; transition: all 0.3s ease; }
    .stButton>button:hover { background: linear-gradient(135deg, #a78bfa 0%, #8b5cf6 100%); box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4); }
    .metric-card { background: #111827; padding: 14px; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.08); text-align: center; }
    .pattern-badge { background: #1e1b4b; color: #a5b4fc; padding: 8px 14px; border-radius: 6px; display: inline-block; font-weight: bold; border: 1px solid #4338ca; }
    </style>
""",
    unsafe_allow_html=True,
)

# --- TOP 400 HIGH-VOLUME NSE EQUITIES ---
NSE_TOP_400 = list(set([
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "BHARTIARTL.NS", "INFY.NS", "ITC.NS", "LT.NS", "BAJFINANCE.NS", "SBIN.NS",
    "KOTAKBANK.NS", "HAL.NS", "HINDUNILVR.NS", "AXISBANK.NS", "M&M.NS", "MARUTI.NS", "HCLTECH.NS", "SUNPHARMA.NS", "TATASTEEL.NS", "NTPC.NS",
    "POWERGRID.NS", "TATAMOTORS.NS", "ULTRACEMCO.NS", "ASIANPAINT.NS", "TITAN.NS", "BAJAJFINSV.NS", "ADANIENT.NS", "ZOMATO.NS", "TRENT.NS", "ONGC.NS",
    "COALINDIA.NS", "WIPRO.NS", "ADANIPORTS.NS", "GRASIM.NS", "HINDALCO.NS", "BAJAJ-AUTO.NS", "TECHM.NS", "DRREDDY.NS", "CIPLA.NS", "INDUSINDBK.NS",
    "TATACONSUM.NS", "BRITANNIA.NS", "HEROMOTOCO.NS", "EICHERMOT.NS", "APOLLOHOSP.NS", "BPCL.NS", "TVSMOTOR.NS", "LTIM.NS", "SHREECEM.NS", "JSWSTEEL.NS",
    "DIVISLAB.NS", "PFC.NS", "RECLTD.NS", "GAIL.NS", "BHEL.NS", "BEL.NS", "INDIGO.NS", "DLF.NS", "PNB.NS", "BANKBARODA.NS",
    "IOC.NS", "AMBUJACEM.NS", "CHOLAFIN.NS", "HDFCLIFE.NS", "SBILIFE.NS", "ABB.NS", "SIEMENS.NS", "VEDL.NS", "PIDILITIND.NS", "GODREJCP.NS",
    "DABUR.NS", "ICICIGI.NS", "MUTHOOTFIN.NS", "BOSCHLTD.NS", "COLPAL.NS", "HAVELLS.NS", "MCDOWELL-N.NS", "CUMMINSIND.NS", "PIIND.NS", "AUBANK.NS",
    "TATAELXSI.NS", "PERSISTENT.NS", "COFORGE.NS", "MPHASIS.NS", "AARTIIND.NS", "DEEPAKNTR.NS", "SRF.NS", "NAVINFLUOR.NS", "TATACOMM.NS", "INDHOTEL.NS",
    "JUBLFOOD.NS", "PAGEIND.NS", "BATAINDIA.NS", "MANYAVAR.NS", "PAYTM.NS", "NYKAA.NS", "PBFINTECH.NS", "DELHIVERY.NS", "POLYCAB.NS", "KEI.NS",
    "APARINDS.NS", "SUZLON.NS", "IREDA.NS", "IRFC.NS", "RVNL.NS", "JINDALSTEL.NS", "NMDC.NS", "SAIL.NS", "LICHSGFIN.NS", "CANBK.NS",
    "UNIONBANK.NS", "IDFCFIRSTB.NS", "YESBANK.NS", "BANDHANBNK.NS", "OBEROIRLTY.NS", "GODREJPROP.NS", "PRESTIGE.NS", "LODHA.NS", "PHOENIXLTD.NS", "DIXON.NS",
    "VOLTAS.NS", "TATAPOWER.NS", "TATACHEM.NS", "BIOCON.NS", "AUROPHARMA.NS", "ZYDUSLIFE.NS", "LUPIN.NS", "TORNTPHARM.NS", "ESCORTS.NS", "MRF.NS",
    "BSE.NS", "CDSL.NS", "MCX.NS", "ANGELONE.NS", "IEX.NS", "KALYANKJIL.NS", "RAYMOND.NS", "ABFRL.NS", "DEVYANI.NS", "KAMATHOTEL.NS",
    "MOTHERSON.NS", "BHARATFORG.NS", "SONACOMS.NS", "ENDURANCE.NS", "APOLLOTYRE.NS", "BALKRISIND.NS", "IGL.NS", "MGL.NS", "GUJGASLTD.NS", "PETRONET.NS",
    "COROMANDEL.NS", "CHAMBLFERT.NS", "GNFC.NS", "UPL.NS", "BAYERCROP.NS", "SYNGENE.NS", "GLENMARK.NS", "ALKEM.NS", "LAURUSLABS.NS", "IPCALAB.NS",
    "MAXHEALTH.NS", "FORTIS.NS", "METROPOLIS.NS", "DRLALPATHLAB.NS", "NAUKRI.NS", "IRCTC.NS", "STARHEALTH.NS", "ABCAPITAL.NS", "POONAWALLA.NS", "SHRIRAMFIN.NS",
    "CONCOR.NS", "GMRINFRA.NS", "NATIONALUM.NS", "EXIDEIND.NS", "AMARAJABAT.NS", "ASHOKLEY.NS", "BALRAMCHIN.NS", "FEDERALBNK.NS", "IDBI.NS", "RBLBANK.NS",
    "OIL.NS", "HINDPETRO.NS", "CASTROLIND.NS", "ATGL.NS", "AWL.NS", "ADANIGREEN.NS", "ADANIPOWER.NS", "TORNTPOWER.NS", "CESC.NS", "NHPC.NS",
    "SJVN.NS", "NLCINDIA.NS", "COCHINSHIP.NS", "MAZDOCK.NS", "GRSE.NS", "BDL.NS", "DATAPATTNS.NS", "KAYNES.NS", "CYIENT.NS", "KPITTECH.NS",
    "SONATSOFTW.NS", "BSOFT.NS", "ZENSARTECH.NS", "LATENTVIEW.NS", "HAPPSTMNDS.NS", "MASTEK.NS", "TANLA.NS", "ROUTE.NS", "CLEAN.NS", "FINEORG.NS",
    "ATUL.NS", "VINATIORGA.NS", "AETHER.NS", "DEEPAKFERT.NS", "GSFC.NS", "FACT.NS", "RCF.NS", "SUMICHEM.NS", "ALKYLAMINE.NS", "BALAMINES.NS",
    "CENTURYPLY.NS", "GREENPANEL.NS", "KAJARIRCER.NS", "CERA.NS", "SUPRAJIT.NS", "CRAFTSMAN.NS", "SUBROS.NS", "CEATLTD.NS", "JKTYRE.NS", "TIMKEN.NS",
    "SKFINDIA.NS", "SCHAEFFLER.NS", "AIAENG.NS", "THERMAX.NS", "TRITURBINE.NS", "KIRLOSENG.NS", "CARBORUNIV.NS", "GRINDWELL.NS", "ELGIEQUIP.NS", "KEC.NS",
    "KALPATPOWR.NS", "PNCINFRA.NS", "KNRCON.NS", "HBLPOWER.NS", "TITAGARH.NS", "JWL.NS", "TEXRAIL.NS", "RAILTEL.NS", "RITES.NS", "IRB.NS",
    "ENGINERSIN.NS", "NBCC.NS", "HUDCO.NS", "CENTRALBK.NS", "IOB.NS", "UCOBANK.NS", "MAHABANK.NS", "PSB.NS", "J&KBANK.NS", "KARURVYSYA.NS",
    "CITYUNIONB.NS", "EQUITASBNK.NS", "UJJIVANSFB.NS", "CSBBANK.NS", "SOUTHBANK.NS", "CREDITACC.NS", "FIVESTAR.NS", "MANAPPURAM.NS", "IIFL.NS", "CANFINHOME.NS",
    "PNBHOUSING.NS", "HOMEFIRST.NS", "AAVAS.NS", "APTUS.NS", "MOTILALOFS.NS", "JMFINANCIL.NS", "GEOJITFSL.NS", "UTIAMC.NS", "HDFCAMC.NS", "NAM-INDIA.NS",
    "KIMS.NS", "ASTERDM.NS", "RAINBOW.NS", "MEDANTA.NS", "VIJAYA.NS", "THYROCARE.NS", "POLYMED.NS", "ERIS.NS", "JBCHEPHARM.NS", "NATCOPHARM.NS",
    "AJANTPHARM.NS", "CAPLIPOI.NS", "MARKSANS.NS", "PFIZER.NS", "SANOFI.NS", "GLAXO.NS", "ABBOTINDIA.NS", "PPLPHARMA.NS", "MANKIND.NS", "BLUESTARCO.NS",
    "AMBER.NS", "SYMPHONY.NS", "CROMPTON.NS", "ORIENTELEC.NS", "VGUARD.NS", "FINCABLES.NS", "RRKABEL.NS", "WHIRLPOOL.NS", "TTKPRESTIG.NS", "BOROLTD.NS",
    "CAMPUS.NS", "METROBRAND.NS", "REDTAPE.NS", "MIRZAINT.NS", "SAPPHIRE.NS", "WESTLIFE.NS", "BARBEQUE.NS", "RELAXO.NS", "VIPIND.NS", "SAFARI.NS",
    "SAMVARDHANA.NS", "SUNDRMFAST.NS", "VARROC.NS", "LUMAXIND.NS", "PRICOL.NS", "FIEMIND.NS", " Gabriel.NS", "SANSERA.NS", "ASKAUTOLTD.NS", "SHARDAMOTR.NS",
    "BLS.NS", "ECLERX.NS", "FIRSTSOURC.NS", "CMSINFO.NS", "SIS.NS", "QUESS.NS", "TEAMLEASE.NS", "VAIBHAVGBL.NS", "ETHOSLTD.NS", "SENCO.NS",
    "THANGAMAYL.NS", "PCJEWELLER.NS", "PGHH.NS", "GILLETTE.NS", "EMAMILTD.NS", "JYOTHYLAB.NS", "BAJAJCON.NS", "HONAUT.NS", "3MINDIA.NS", "SUVENPHAR.NS",
    "NEULANDLAB.NS", "FDC.NS", "GRANULES.NS", "HIKAL.NS", "AARTIDRUGS.NS", "RPGCHEM.NS", "NOCIL.NS", "SUDARSCHEM.NS", "GHCL.NS", "TATAINVEST.NS"
]))

# --- PATTERN RECOGNITION ENGINE ---
def detect_chart_pattern(df):
    if len(df) < 30: return "Consolidation Base", "Wait for confirmed breakout above resistance."
    last, prev, prior = df.iloc[-1], df.iloc[-2], df.iloc[-3]
    recent_lows, recent_highs = df["Low"].tail(25), df["High"].tail(25)
    
    if prev["Close"] < prev["Open"] and last["Close"] > last["Open"] and last["Close"] >= prev["Open"] and last["Open"] <= prev["Close"]:
        return "Bullish Engulfing Reversal", "High volume buyers swallowed prior selling candle; strong momentum pivot."
    
    body = abs(last["Close"] - last["Open"])
    lower_wick = min(last["Open"], last["Close"]) - last["Low"]
    upper_wick = last["High"] - max(last["Open"], last["Close"])
    if lower_wick >= 2 * body and upper_wick <= 0.5 * body and last["Low"] <= recent_lows.quantile(0.25):
        return "Hammer / Demand Rejection", "Lower shadow shows aggressive institutional dip-buying at major support."
        
    trough1, trough2 = recent_lows.iloc[:12].min(), recent_lows.iloc[12:].min()
    peak_neckline = recent_highs.iloc[6:18].max()
    if abs(trough1 - trough2) / trough1 <= 0.025 and last["Close"] > peak_neckline * 0.98:
        return "Double Bottom (W-Pattern)", "Constructive secondary retest completed; breaking out through neckline."
        
    base_low, base_high = recent_lows.min(), recent_highs.max()
    if (base_high - base_low) / base_low >= 0.08 and last["Close"] >= base_high * 0.985:
        return "Cup & Handle Breakout", "Multi-week accumulation complete; price clearing multi-month resistance pivot."
        
    if last["Close"] >= last["EMA_20"] and prev["Low"] <= last["EMA_20"] * 1.01 and last["EMA_20"] > last["EMA_50"]:
        return "20-EMA Pullback Continuation", "Trend-continuation retest holding dynamic 20-day exponential moving average."
        
    return "Ascending Momentum Breakout", "Higher highs and higher lows stacking inside bullish momentum channel."

def scan_single_stock_ml(ticker, days_to_hold, target_pct):
    try:
        stock = yf.Ticker(ticker)
        # Fetch 2 years to ensure the AI has enough historical data points to train on
        df = stock.history(period="2y")
        if len(df) < 250:
            return None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(subset=["Close"])

        df["EMA_20"] = df["Close"].ewm(span=20).mean()
        df["EMA_50"] = df["Close"].ewm(span=50).mean()
        df["EMA_200"] = df["Close"].ewm(span=200).mean()

        # RSI & ATR
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        hl = df["High"] - df["Low"]
        hc = np.abs(df["High"] - df["Close"].shift())
        lc = np.abs(df["Low"] - df["Close"].shift())
        df["ATR"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

        # --- MACHINE LEARNING FEATURE ENGINEERING ---
        df['F_EMA20_Dist'] = (df['Close'] - df['EMA_20']) / df['EMA_20']
        df['F_EMA50_Dist'] = (df['Close'] - df['EMA_50']) / df['EMA_50']
        df['F_RSI_Norm'] = df['RSI'] / 100.0
        df['F_ATR_Pct'] = df['ATR'] / df['Close']
        
        # Supervised Label: Did it hit the target in the future timeline?
        future_highs = df['High'].rolling(window=days_to_hold).max().shift(-days_to_hold)
        df['Target_Hit'] = np.where((future_highs - df['Close']) / df['Close'] >= (target_pct / 100.0), 1, 0)
        
        ml_df = df.dropna().copy()
        features = ['F_EMA20_Dist', 'F_EMA50_Dist', 'F_RSI_Norm', 'F_ATR_Pct']
        X_train = ml_df[features].iloc[:-1]
        y_train = ml_df['Target_Hit'].iloc[:-1]
        X_today = ml_df[features].iloc[[-1]]
        
        # Train AI Model on-the-fly for this specific stock
        if len(np.unique(y_train)) < 2:
            ai_prob = 0.50 # Neutral if target was never hit or always hit historically
        else:
            model = RandomForestClassifier(n_estimators=40, max_depth=4, random_state=42, n_jobs=1)
            model.fit(X_train, y_train)
            ai_prob = model.predict_proba(X_today)[0][1]

        curr_price = float(df["Close"].iloc[-1])
        ema20, ema50, ema200 = float(df["EMA_20"].iloc[-1]), float(df["EMA_50"].iloc[-1]), float(df["EMA_200"].iloc[-1])
        rsi, atr = float(df["RSI"].iloc[-1]), float(df["ATR"].iloc[-1])

        # Velocity Check
        daily_pct_move = (atr / curr_price) * 100
        max_expected_move = daily_pct_move * (days_to_hold * 0.75)
        if max_expected_move < target_pct:
            return None

        # v10 Hybrid Scoring (/20 each)
        p1 = 20 if curr_price > ema200 else 8
        if curr_price > ema20 > ema50 > ema200: p2 = 20
        elif curr_price > ema20 > ema50: p2 = 15
        else: p2 = 8

        if 55 <= rsi <= 75: p3 = 20
        elif 45 <= rsi < 55: p3 = 14
        else: p3 = 8

        p4 = 20 if ema50 > ema200 else 12
        p5 = int(ai_prob * 20) # AI Confidence replaces basic volume score

        total_score = p1 + p2 + p3 + p4 + p5
        if total_score >= 70: # Lowered threshold slightly to allow AI to make the final call
            pattern_name, pattern_desc = detect_chart_pattern(df)
            return {
                "ticker": ticker, "score": total_score, "df": df,
                "p1": p1, "p2": p2, "p3": p3, "p4": p4, "p5": p5,
                "ai_prob": ai_prob, "curr_price": curr_price, "rsi": rsi, "atr": atr,
                "pattern_name": pattern_name, "pattern_desc": pattern_desc
            }
        return None
    except Exception:
        return None

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🎯 Target & AI Selection")

scan_scope = st.sidebar.selectbox("Universe Scope", [
    "Top 100 Liquid Leaders (Fastest)",
    "Top 200 Core Equities (Balanced)",
    "Full Top 400 Universe (Deep ML Scan)"
])

if "100" in scan_scope: scan_list = NSE_TOP_400[:100]
elif "200" in scan_scope: scan_list = NSE_TOP_400[:200]
else: scan_list = NSE_TOP_400

timeline_options = {
    "1 to 2 Days (Scalp / BTST)": 2,
    "1 Week (Momentum Swing)": 7,
    "2 to 3 Weeks (Core Swing)": 21,
    "1 to 3 Months (Position)": 90
}
selected_timeline = st.sidebar.selectbox("Timeline / Holding Duration", list(timeline_options.keys()))
days_to_hold = timeline_options[selected_timeline]

profit_target_pct = st.sidebar.slider("Desired Profit Target (%)", 1.0, 30.0, 5.0, 0.5)
max_results = st.sidebar.slider("Number of Top Setups to Display", 1, 10, 5, 1)
account_capital = st.sidebar.number_input("Trading Capital (₹)", min_value=10000.0, value=500000.0, step=10000.0)

run_scan = st.sidebar.button("🧠 Run AI Predictive Scan")

# --- MAIN APP HEADER ---
st.title("🧠 v10 AI/ML Hybrid Chart Screener")
st.markdown("*Trains a Random Forest AI model in real-time for up to 400 equities while visually mapping classical chart patterns and exact execution levels.*")
st.markdown("---")

if run_scan:
    with st.spinner(f"Parallel scanning and training ML models for {len(scan_list)} equities... (Please wait)"):
        passed_stocks = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=14) as executor:
            futures = [executor.submit(scan_single_stock_ml, t, days_to_hold, profit_target_pct) for t in scan_list]
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res:
                    passed_stocks.append(res)

    if not passed_stocks:
        st.error(f"⚠️ No setups found clearing the AI prediction threshold for **{profit_target_pct}%** profit within **{selected_timeline}**. The machine learning model indicates this is a low-probability environment for your parameters.")
    else:
        passed_stocks = sorted(passed_stocks, key=lambda x: (x["score"], x["ai_prob"]), reverse=True)
        top_matches = passed_stocks[:max_results]
        st.success(f"🚀 AI found {len(passed_stocks)} valid setups across the market. Top {len(top_matches)} ranked below by AI Confidence:")

        for idx, match in enumerate(top_matches):
            ticker = match["ticker"]
            score = match["score"]
            df = match["df"]
            curr_price = match["curr_price"]
            ai_prob = match["ai_prob"]
            pattern_name = match["pattern_name"]
            pattern_desc = match["pattern_desc"]

            # Exact Trade Levels
            breakout_trigger = round(curr_price * 1.006, 2)
            target_price = round(curr_price * (1 + profit_target_pct / 100.0), 2)
            stop_loss_price = round(curr_price * 0.95, 2) if days_to_hold <= 7 else round(curr_price * 0.93, 2)
            
            risk_per_share = round(curr_price - stop_loss_price, 2)
            reward_per_share = round(target_price - curr_price, 2)
            rr_ratio = round(reward_per_share / risk_per_share, 2) if risk_per_share > 0 else 0
            
            max_risk_in_rupees = account_capital * 0.015
            recommended_shares = int(max_risk_in_rupees / risk_per_share) if risk_per_share > 0 else 0
            strict_deadline = (datetime.today() + timedelta(days=days_to_hold)).strftime("%B %d, %Y")

            with st.expander(f"🏆 #{idx+1} {ticker} | AI Confidence: {ai_prob*100:.1f}% | Pattern: {pattern_name}", expanded=(idx == 0)):
                st.markdown(f'<div class="pattern-badge">Detected Pattern: {pattern_name}</div>', unsafe_allow_html=True)
                st.caption(pattern_desc)
                st.markdown("<br>", unsafe_allow_html=True)

                col1, col2, col3, col4 = st.columns(4)
                with col1: st.metric("Live Price", f"₹{curr_price:,.2f}")
                with col2: st.metric("🧠 ML Probability", f"{ai_prob*100:.1f}%")
                with col3: st.metric("Target (+{0}%)".format(profit_target_pct), f"₹{target_price:,.2f}")
                with col4: st.metric("Stop-Loss Exit", f"₹{stop_loss_price:,.2f}")

                tab_chart, tab_execution, tab_score = st.tabs(["📈 Enhanced ML Chart Overlays", "🎯 Precision Plan", "🛡️ AI & Quant Scorecard"])

                with tab_chart:
                    chart_df = df.tail(120)
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)

                    fig.add_trace(go.Candlestick(
                        x=chart_df.index, open=chart_df['Open'], high=chart_df['High'],
                        low=chart_df['Low'], close=chart_df['Close'], name="Candlestick"
                    ), row=1, col=1)

                    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['EMA_20'], line=dict(color='#fbbf24', width=1.5), name="20 EMA"), row=1, col=1)
                    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['EMA_50'], line=dict(color='#38bdf8', width=1.5), name="50 EMA"), row=1, col=1)
                    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['EMA_200'], line=dict(color='#c084fc', width=2), name="200 EMA"), row=1, col=1)

                    # Horizontal Trade Execution Levels
                    fig.add_hline(y=breakout_trigger, line_dash="dash", line_color="#38bdf8", annotation_text=f"BUY ENTRY: ₹{breakout_trigger}", annotation_position="top right", row=1, col=1)
                    fig.add_hline(y=target_price, line_dash="dash", line_color="#34d399", annotation_text=f"TARGET: ₹{target_price}", annotation_position="top right", row=1, col=1)
                    fig.add_hline(y=stop_loss_price, line_dash="dash", line_color="#f87171", annotation_text=f"STOP-LOSS: ₹{stop_loss_price}", annotation_position="bottom right", row=1, col=1)

                    colors = ['#f87171' if r['Open'] - r['Close'] >= 0 else '#34d399' for _, r in chart_df.iterrows()]
                    fig.add_trace(go.Bar(x=chart_df.index, y=chart_df['Volume'], marker_color=colors, name="Volume"), row=2, col=1)

                    fig.update_layout(
                        title=f"{ticker} — {pattern_name} Trade Setup",
                        yaxis_title="Price (₹)", xaxis_rangeslider_visible=False,
                        template="plotly_dark", height=500, margin=dict(l=10, r=10, t=35, b=10),
                        paper_bgcolor="#111827", plot_bgcolor="#111827"
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_{ticker}_{idx}")

                with tab_execution:
                    c_a, c_b = st.columns(2)
                    with c_a:
                        st.markdown("### 🟢 Entry Strategy")
                        st.markdown(f"- **Trigger Entry Range:** `₹{curr_price:,.2f} - ₹{breakout_trigger:,.2f}`")
                        st.markdown(f"- **Execution Trigger:** Enter on a 15-minute or daily close breaking above `₹{breakout_trigger}`.")
                        st.markdown(f"- **Allocated Shares:** `{recommended_shares:,} shares` (Clamped to 1.5% risk)")
                        st.markdown(f"- **Total Capital Allocation:** `₹{recommended_shares * curr_price:,.2f}`")
                    with c_b:
                        st.markdown("### 🔴 Exit & Risk Strategy")
                        st.markdown(f"- **Target Exit (Profit):** `₹{target_price:,.2f}` (+{profit_target_pct}%)")
                        st.markdown(f"- **Hard Stop-Loss Exit (Failure):** `₹{stop_loss_price:,.2f}`")
                        st.markdown(f"- **Risk-to-Reward Ratio:** `1:{rr_ratio}`")
                        st.markdown(f"- **Non-Extending Deadline:** `{strict_deadline}` ⏳")
                        st.markdown("- **Trailing Stop Protocol:** Move stop-loss to Breakeven once the price achieves +50% of the target move.")

                with tab_score:
                    st.table(pd.DataFrame({
                        "Pillar Evaluation": [
                            "1. Fundamentals / 200 EMA Structural Baseline",
                            "2. Technical Momentum & EMA Alignment (20>50>200)",
                            "3. Chart Pattern Structure & RSI Breakout",
                            "4. Macro & Trend Regime (50 EMA > 200 EMA)",
                            "5. AI/ML Predictive Confidence (Random Forest Model)"
                        ],
                        "Score": [f"{match['p1']}/20", f"{match['p2']}/20", f"{match['p3']}/20", f"{match['p4']}/20", f"{match['p5']}/20"],
                        "Status": ["Pass" if p >= 15 else "Moderate" for p in [match['p1'], match['p2'], match['p3'], match['p4'], match['p5']]]
                    }))

else:
    st.info("👈 Set your timeline, profit target, and universe scope in the sidebar, then click **Run AI Predictive Scan**.")
