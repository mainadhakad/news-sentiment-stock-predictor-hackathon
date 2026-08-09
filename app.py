# ============================================================
# FLASK APP — News Sentiment Stock Predictor
# Run: python app.py
# ============================================================

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
import requests
import re
import nltk
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords
from datetime import datetime, timedelta

import os
app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(__file__), 'templates'))

# ── Load all models ──────────────────────────────────────────
print("Loading models...")
model    = pickle.load(open("models/best_model.pkl",        "rb"))
tfidf    = pickle.load(open("models/tfidf.pkl",             "rb"))
scaler   = pickle.load(open("models/scaler.pkl",            "rb"))
pca      = pickle.load(open("models/pca.pkl",               "rb"))
pt       = pickle.load(open("models/power_transformer.pkl", "rb"))
le       = pickle.load(open("models/label_encoder.pkl",     "rb"))
num_cols = pickle.load(open("models/num_cols.pkl",          "rb"))
print("✅ All models loaded!")

# ── Constants ────────────────────────────────────────────────
stop_words = set(stopwords.words("english"))

COMPANY_TICKERS = {
    "Apple":     "AAPL",
    "Microsoft": "MSFT",
    "Tesla":     "TSLA",
    "Google":    "GOOGL",
    "Amazon":    "AMZN",
    "Infosys":   "INFY",
    "Wipro":     "WIT",
    "Reliance":  "RELIANCE.NS",
}

# ── Helper functions ─────────────────────────────────────────
def clean_text(text):
    if pd.isna(text): return ""
    text  = str(text).lower()
    text  = re.sub(r"[^a-z\s]", " ", text)
    words = [w for w in text.split()
             if w not in stop_words and len(w) > 2]
    return " ".join(words)
GNEWS_API_KEY = "acb22d2c28635e8c41fd67674434a0ba"  # apni key daalo

def fetch_news(company_name):
    """GNews API se live news fetch karo"""
    try:
        url = "https://gnews.io/api/v4/search"
        params = {
            "q":        f"{company_name} stock",
            "lang":     "en",
            "country":  "us",
            "max":      10,
            "apikey":   GNEWS_API_KEY,
        }
        r    = requests.get(url, params=params, timeout=15)
        data = r.json()

        articles = data.get("articles", [])
        if not articles:
            return [f"{company_name} market update — no news found"]

        headlines = []
        for a in articles:
            title = a.get("title", "")
            if title and title.isascii():
                headlines.append(title)

        print(f"✅ GNews: {len(headlines)} headlines for {company_name}")
        return headlines[:5] if headlines else [f"{company_name} market update"]

    except Exception as e:
        print(f"❌ GNews error: {e}")
        return [f"{company_name} market update"]
def get_stock_data(ticker):
    """yfinance se stock data fetch karo"""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        hist  = stock.history(period="10d")
        if hist.empty:
            return None
        hist = hist.reset_index()
        return {
            "close":          float(hist["Close"].iloc[-1]),
            "price_change":   float(hist["Close"].pct_change().iloc[-1]),
            "volume_change":  float(hist["Volume"].pct_change().iloc[-1]),
            "high_low_ratio": float(hist["High"].iloc[-1] /
                                   hist["Low"].iloc[-1]),
            "volume":         float(hist["Volume"].iloc[-1]),
        }
    except:
        return None

def preprocess_and_predict(company_name, ticker):
    """Full pipeline — news + stock → prediction"""

    # 1. News fetch
    headlines = fetch_news(company_name)
    combined  = " | ".join(headlines)
    cleaned   = clean_text(combined)

    # 2. Stock data
    stock_data = get_stock_data(ticker)
    if not stock_data:
        return None

    # 3. TF-IDF
    X_text = tfidf.transform([cleaned]).toarray()

    # 4. Sentiment proxy
    positive_words = ["gain","rise","up","profit","growth","surge",
                      "high","strong","beat","record","bull","positive"]
    negative_words = ["fall","drop","down","loss","decline","crash",
                      "low","weak","miss","bear","negative","risk"]
    words     = cleaned.split()
    pos       = sum(1 for w in words if w in positive_words)
    neg       = sum(1 for w in words if w in negative_words)
    sentiment = (pos - neg) / (len(words) + 1)

    # 5. Numerical features
    try:
        company_enc = le.transform([company_name])[0]
    except:
        company_enc = 0

    # PowerTransformer columns
    pt_input = np.array([[
        stock_data["high_low_ratio"],
        stock_data["volume_change"] if not np.isinf(
            stock_data["volume_change"]) else 0
    ]])
    pt_output = pt.transform(pt_input)

    # Volume log
    volume_log = np.log1p(stock_data["volume"])

    # Price change norm (simple)
    price_change_norm = stock_data["price_change"]

    # news count
    news_count_capped = min(np.log1p(np.log1p(len(headlines))), 0.527)

    # Close norm (simple)
    close_norm = 0.0

    num_features = np.array([[
        pt_output[0][1],      # volume_change (transformed)
        pt_output[0][0],      # high_low_ratio (transformed)
        np.log1p(volume_log), # volume_log
        datetime.now().weekday(),  # day_of_week
        datetime.now().month,      # month
        (datetime.now().month-1)//3 + 1,  # quarter
        1 if datetime.now().weekday() == 0 else 0,  # is_monday
        1 if datetime.now().weekday() == 4 else 0,  # is_friday
        company_enc,          # company_encoded
        close_norm,           # close_norm
        price_change_norm,    # price_change_norm
        news_count_capped,    # news_count_capped
        len(words),           # word_count
        sentiment,            # sentiment_proxy
        price_change_norm,    # lag1_price
        0.0,                  # lag2_price
        0.0,                  # lag3_price
        price_change_norm,    # rolling_mean3
        0.0,                  # rolling_std3
        price_change_norm,    # rolling_mean5
    ]])

    # 6. Scale
    X_num_scaled = scaler.transform(num_features)

    # 7. Combine
    X_combined = np.hstack([X_text, X_num_scaled])

    # 8. PCA
    X_pca = pca.transform(X_combined)

    # 9. Predict
    pred  = int(model.predict(X_pca)[0])
    proba = float(model.predict_proba(X_pca)[0][1])
    conf  = proba if pred == 1 else 1 - proba

    return {
        "prediction":   "UP" if pred == 1 else "DOWN",
        "confidence":   round(conf * 100, 1),
        "probability":  round(proba * 100, 1),
        "latest_price": round(stock_data["close"], 2),
        "price_change": round(stock_data["price_change"] * 100, 2),
        "sentiment_score": round(sentiment, 3),
        "positive_count": pos,
        "negative_count": neg,
        "news_count": len(headlines),
        "headlines":    headlines[:5],
        "company":      company_name,
        "ticker":       ticker,
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ── Routes ───────────────────────────────────────────────────
@app.route("/")
def home():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
              'templates', 'index.html'), 'r') as f:
        return f.read()

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data    = request.json
        company = data.get("company", "Apple")
        ticker  = COMPANY_TICKERS.get(company, "AAPL")

        result = preprocess_and_predict(company, ticker)

        if not result:
            return jsonify({"error": "Data fetch failed"}), 400

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/companies", methods=["GET"])
def companies():
    return jsonify({"companies": list(COMPANY_TICKERS.keys())})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":    "running",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models":    "loaded"
    })


# ── Run ──────────────────────────────────────────────────────
# NAYA
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
