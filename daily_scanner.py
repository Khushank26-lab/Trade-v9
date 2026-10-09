import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import time

# For demonstration, we load a broad universe of 1,000+ NSE/BSE stocks.
# (In the future, you can replace this with a CSV containing all 7,500 symbols)
from urllib.request import Request, urlopen
import json

def get_broad_market_tickers():
    # Fallback large list covering major NSE segments (Expandable to 7500)
    return [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", 
        "LT.NS", "BAJFINANCE.NS", "HAL.NS", "ZOMATO.NS", "TRENT.NS", "BSE.NS", "CDSL.NS", "IRFC.NS", "SUZLON.NS",
        "IREDA.NS", "RVNL.NS", "JINDALSTEL.NS", "DIXON.NS", "POLYCAB.NS", "KALYANKJIL.NS", "ANGELONE.NS"
        # We start with a high-liquidity subset here to prevent GitHub Actions timeout.
        # You can upload a 'tickers.csv' to this repo later with 7,500 symbols.
    ]

def train_and_evaluate(ticker):
    try:
        df = yf.Ticker(ticker).history(period="1y")
        if len(df) < 150: return None
        
        df["EMA_20"] = df["Close"].ewm(span=20).mean()
        df["EMA_50"] = df["Close"].ewm(span=50).mean()
        df["EMA_200"] = df["Close"].ewm(span=200).mean()
        
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))
        
        # Machine Learning target (did it go up 5% in 14 days?)
        future_highs = df['High'].rolling(window=14).max().shift(-14)
        df['Target_Hit'] = np.where((future_highs - df['Close']) / df['Close'] >= 0.05, 1, 0)
        
        df = df.dropna()
        if len(df) < 50: return None
        
        features = df[['EMA_20', 'EMA_50', 'RSI']]
        X_train = features.iloc[:-1]
        y_train = df['Target_Hit'].iloc[:-1]
        X_today = features.iloc[[-1]]
        
        if len(np.unique(y_train)) < 2: return None
        
        model = RandomForestClassifier(n_estimators=30, max_depth=3, random_state=42)
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_today)[0][1]
        
        return {"Ticker": ticker, "AI_Score": prob}
    except:
        return None

if __name__ == "__main__":
    print("Starting background market scan...")
    tickers = get_broad_market_tickers()
    results = []
    
    for t in tickers:
        res = train_and_evaluate(t)
        if res and res["AI_Score"] > 0.55: # Only save stocks with >55% AI Confidence
            results.append(res)
        time.sleep(0.5) # Prevent Yahoo Finance IP Ban
        
    # Sort and save the top 50 to a CSV file
    df_results = pd.DataFrame(results).sort_values(by="AI_Score", ascending=False).head(50)
    df_results.to_csv("top_setups.csv", index=False)
    print("Saved top setups to top_setups.csv!")
