import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os

nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

def get_all_market_tickers():
    print("Downloading complete universal NSE Master List...")
    # Multiple reliable sources for NSE equity list to prevent fetch failures
    urls = [
        "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
        "https://raw.githubusercontent.com/the-omega-point/nse-india-symbols/main/symbols.csv",
        "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
    ]
    
    for url in urls:
        try:
            nse_df = pd.read_csv(url)
            # Find the symbol column dynamically
            symbol_col = next((col for col in ['SYMBOL', 'Symbol', 'symbol', 'TICKER'] if col in nse_df.columns), None)
            if symbol_col:
                tickers = (nse_df[symbol_col].str.strip() + ".NS").tolist()
                tickers = [t for t in tickers if t.endswith(".NS")]
                if len(tickers) > 500:
                    print(f"Successfully loaded {len(tickers)} universal NSE tickers from source.")
                    return tickers
        except Exception:
            continue
            
    # Comprehensive robust default fallback market universe if web fetch is blocked
    print("Using comprehensive fallback Indian equity universe...")
    return [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "HINDUNILVR.NS", 
        "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "LT.NS", "AXISBANK.NS", 
        "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS", "BAJFINANCE.NS", "ADANIENT.NS", 
        "TATASTEEL.NS", "WIPRO.NS", "HCLTECH.NS", "ULTRACEMCO.NS", "NTPC.NS", "POWERGRID.NS", 
        "ONGC.NS", "M&M.NS", "TATAMOTORS.NS", "COALINDIA.NS", "BAJAJFINSV.NS", "GRASIM.NS", 
        "CDSL.NS", "BSE.NS", "ZOMATO.NS", "PAYTM.NS", "NYKAA.NS", "DELHIVERY.NS", "POLYCAB.NS",
        "DIXON.NS", "PERSISTENT.NS", "LTIM.NS", "TECHM.NS", "INDUSINDBK.NS", "ADANIPORTS.NS"
    ]

def fetch_live_news_sentiment(ticker):
    try:
        stock = yf.Ticker(ticker)
        news = getattr(stock, 'news', None)
        if not news: return 0.0
        scores = [sia.polarity_scores(art.get('title', ''))['compound'] for art in news[:5] if 'title' in art]
        return sum(scores) / len(scores) if scores else 0.0
    except:
        return 0.0

def train_and_evaluate(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = getattr(stock, 'info', {})
        
        mcap = info.get('marketCap', 0) or 0
        roe = info.get('returnOnEquity', 0) or 0
        profit_margin = info.get('profitMargins', 0) or 0
        debt_to_equity = info.get('debtToEquity', 50) or 50
        
        # Fundamental Filter
        if mcap > 0 and mcap < 10000000000: return None
        if roe < 0 or profit_margin < 0: return None
        
        fundy_score = 15
        if mcap >= 50000000000: fundy_score = 19
        elif mcap >= 20000000000: fundy_score = 18
        
        df = stock.history(period="3y")
        if df is None or len(df) < 150: 
            df = stock.history(period="max")
        if df is None or len(df) < 150: return None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(subset=["Close"])
        
        df["EMA_20"] = df["Close"].ewm(span=20).mean()
        df["EMA_50"] = df["Close"].ewm(span=50).mean()
        df["EMA_200"] = df["Close"].ewm(span=200).mean()
        
        df['Body'] = abs(df['Close'] - df['Open'])
        df['Range'] = df['High'] - df['Low']
        df['Upper_Wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
        df['Lower_Wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
        
        df['Pat_Doji'] = np.where(df['Body'] <= (df['Range'] * 0.1), 1, 0)
        df['Pat_Hammer'] = np.where((df['Lower_Wick'] >= 2 * df['Body']) & (df['Upper_Wick'] <= 0.5 * df['Body']), 1, 0)
        df['Pat_Bull_Engulf'] = np.where((df['Close'].shift(1) < df['Open'].shift(1)) & (df['Close'] > df['Open']) & (df['Close'] >= df['Open'].shift(1)), 1, 0)
        
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df["RSI"] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
        
        df['Future_High'] = df['High'].rolling(window=10).max().shift(-10)
        df['Future_Low'] = df['Low'].rolling(window=10).min().shift(-10)
        df['Target_Hit'] = np.where(((df['Future_High'] - df['Close']) / df['Close'] >= 0.05) & ((df['Close'] - df['Future_Low']) / df['Close'] <= 0.02), 1, 0)
            
        df = df.dropna()
        if len(df) < 60: return None
        
        features = ['EMA_20', 'EMA_50', 'RSI', 'Pat_Doji', 'Pat_Hammer', 'Pat_Bull_Engulf']
        X_train = df[features].iloc[:-1]
        y_train = df['Target_Hit'].iloc[:-1]
        X_today = df[features].iloc[[-1]]
        
        if len(np.unique(y_train)) < 2: return None 
        
        model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.08, max_depth=3, random_state=42)
        model.fit(X_train, y_train)
        ai_prob = float(model.predict_proba(X_today)[0][1])
        
        base_score = int(30 + (ai_prob * 50) + (fundy_score))
        quant_score = min(max(base_score, 30), 100)
        news_sentiment = fetch_live_news_sentiment(ticker)
        
        live_price = info.get('currentPrice', None)
        if not live_price:
            try:
                live_price = stock.fast_info.get('last_price', float(df["Close"].iloc[-1]))
            except:
                live_price = float(df["Close"].iloc[-1])
        
        return {
            "Ticker": ticker, 
            "AI_Score": ai_prob, 
            "Quant_Score": quant_score,
            "Fundy_Score": fundy_score,
            "News_Sentiment": news_sentiment, 
            "ROE": roe,
            "Live_Price": live_price
        }
    except Exception:
        return None

if __name__ == "__main__":
    print("Initiating Universal Market Scan across all equities...")
    tickers = get_all_market_tickers()
    results = []
    
    for i, t in enumerate(tickers):
        if i % 100 == 0: 
            print(f"Progress: Scanned {i}/{len(tickers)} stocks... Found {len(results)} valid setups so far.")
            
        res = train_and_evaluate(t)
        if res:
            results.append(res)
            
    if results:
        df_results = pd.DataFrame(results).sort_values(by="Quant_Score", ascending=False).head(250)
        df_results.to_csv("top_setups.csv", index=False)
        print(f"Scan complete! Successfully saved top {len(df_results)} diverse setups to top_setups.csv.")
    else:
        print("Scan completed with zero matches.")
