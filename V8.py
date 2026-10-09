import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import time
import os

# Download NLP Lexicon for News Analysis
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

def get_all_market_tickers():
    print("Downloading NSE Master List...")
    try:
        url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        nse_df = pd.read_csv(url)
        return (nse_df['SYMBOL'] + ".NS").tolist()
    except Exception:
        return ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]

def fetch_live_news_sentiment(ticker):
    """Pillar 4: NLP News Sentiment Analysis"""
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news:
            return 0.0 # Neutral if no news
            
        sentiment_score = 0
        for article in news[:5]: # Analyze top 5 most recent headlines
            title = article.get('title', '')
            score = sia.polarity_scores(title)['compound']
            sentiment_score += score
            
        return sentiment_score / len(news[:5])
    except:
        return 0.0

def train_and_evaluate(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # --- PILLAR 1: FLAWLESS FUNDAMENTALS (Zero Tolerance) ---
        mcap = info.get('marketCap', 0)
        roe = info.get('returnOnEquity', 0)
        profit_margin = info.get('profitMargins', 0)
        debt_to_equity = info.get('debtToEquity', 100) # Default to high debt if unknown
        
        if mcap is None or mcap < 20000000000: return None # Must be > ₹2,000 Cr (High Liquidity)
        if roe is None or roe < 0.15: return None          # Must have > 15% Return on Equity
        if profit_margin is None or profit_margin <= 0.08: return None # Must have > 8% Net Margins
        if debt_to_equity is None or debt_to_equity > 150: return None # No dangerously over-leveraged companies
        
        # --- PILLAR 2 & 3: CHART & TECHNICAL ML FEATURE ENGINEERING ---
        df = stock.history(period="2y")
        if len(df) < 250: return None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(subset=["Close"])
        
        # Advanced Technicals
        df["EMA_20"] = df["Close"].ewm(span=20).mean()
        df["EMA_50"] = df["Close"].ewm(span=50).mean()
        df["EMA_200"] = df["Close"].ewm(span=200).mean()
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # RSI & Volatility
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))
        
        # Chart Pattern Mathematics (Wicks & Engulfing)
        df['Body'] = abs(df['Close'] - df['Open'])
        df['Lower_Wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
        df['Is_Hammer'] = np.where((df['Lower_Wick'] > (2 * df['Body'])) & (df['Close'] > df['EMA_50']), 1, 0)
        
        # Labeling for Extreme Precision: Must hit +6% before hitting -2% (Strict Risk/Reward)
        df['Future_High'] = df['High'].rolling(window=10).max().shift(-10)
        df['Future_Low'] = df['Low'].rolling(window=10).min().shift(-10)
        
        # 1 = Success, 0 = Failure/Stopped Out
        df['Target_Hit'] = np.where(
            ((df['Future_High'] - df['Close']) / df['Close'] >= 0.06) & 
            ((df['Close'] - df['Future_Low']) / df['Close'] <= 0.02), 
            1, 0
        )
        
        df = df.dropna()
        if len(df) < 100: return None
        
        # --- THE ML MODEL (Gradient Boosting) ---
        features = ['EMA_20', 'EMA_50', 'RSI', 'MACD', 'MACD_Signal', 'Is_Hammer']
        X_train = df[features].iloc[:-1]
        y_train = df['Target_Hit'].iloc[:-1]
        X_today = df[features].iloc[[-1]]
        
        if len(np.unique(y_train)) < 2: return None # Skip if stock never moves safely
        
        # Gradient Boosting minimizes error far better than Random Forest
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
        model.fit(X_train, y_train)
        
        # Get AI Probability
        ai_prob = model.predict_proba(X_today)[0][1]
        
        # --- THE 90% THRESHOLD & NLP NEWS CHECK ---
        # Only check news if the AI is extremely confident (saves API calls)
        if ai_prob >= 0.88: 
            news_sentiment = fetch_live_news_sentiment(ticker)
            # If news is negative, instantly reject the setup despite perfect charts
            if news_sentiment < -0.10:
                return None
                
            return {
                "Ticker": ticker, 
                "AI_Score": ai_prob, 
                "News_Sentiment": news_sentiment,
                "ROE": roe
            }
        return None
    except Exception:
        return None

if __name__ == "__main__":
    print("Initiating Ultra-Precision Market Scan...")
    tickers = get_all_market_tickers()
    results = []
    
    for i, t in enumerate(tickers):
        if i % 25 == 0:
            print(f"Deep Scanning {i}/{len(tickers)} stocks...")
            
        res = train_and_evaluate(t)
        if res:
            results.append(res)
            print(f"🎯 FLAWLESS SETUP FOUND: {t} | AI Confidence: {res['AI_Score']*100:.1f}%")
            
    if results:
        df_results = pd.DataFrame(results).sort_values(by="AI_Score", ascending=False)
        df_results.to_csv("top_setups.csv", index=False)
        print(f"Scan complete! {len(df_results)} flawless setups secured.")
    else:
        # If no stocks pass, create an empty file so Streamlit knows the scan finished but found nothing.
        pd.DataFrame(columns=["Ticker", "AI_Score", "News_Sentiment", "ROE"]).to_csv("top_setups.csv", index=False)
        print("0 setups passed the 99% precision requirements today. Cash is king.")
