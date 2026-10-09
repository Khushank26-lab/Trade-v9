import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import time

def get_all_market_tickers():
    print("Downloading official master list from NSE...")
    try:
        # Dynamically fetches the live master list of all active NSE companies
        url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        nse_df = pd.read_csv(url)
        # Add '.NS' to match Yahoo Finance formatting
        tickers = (nse_df['SYMBOL'] + ".NS").tolist()
        print(f"Successfully loaded {len(tickers)} NSE stocks.")
        return tickers
    except Exception as e:
        print(f"Failed to fetch NSE list: {e}")
        # Fallback list if NSE servers are down
        return ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]

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
    tickers = get_all_market_tickers()
    results = []
    
    # Scanning all thousands of stocks
    for i, t in enumerate(tickers):
        if i % 100 == 0:
            print(f"Scanned {i}/{len(tickers)} stocks...")
            
        res = train_and_evaluate(t)
        if res and res["AI_Score"] > 0.55: 
            results.append(res)
            
    if results:
        df_results = pd.DataFrame(results).sort_values(by="AI_Score", ascending=False).head(500)
        df_results.to_csv("top_setups.csv", index=False)
        print(f"Scan complete! Saved top {len(df_results)} setups to top_setups.csv.")
    else:
        print("No setups found today.")
