# ----------------------------------------------------------
# 🔧 Bibliotheken
# ----------------------------------------------------------
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ta

# ----------------------------------------------------------
# 🧭 Parameter
# ----------------------------------------------------------
SYMBOL = "^GDAXI"
ALT_SYMBOL = "EXS1.DE"
ATR_PERIOD = 14
RSI_PERIOD = 7
SMA_SHORT = 10
SMA_LONG = 50
CHAIN_MAX = 30

END = datetime.now()
START = END - timedelta(days=3*365)

# ----------------------------------------------------------
# 📥 Daten laden
# ----------------------------------------------------------
def load_data(ticker):
    df = yf.download(ticker, start=START, end=END, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"Keine Daten für {ticker}")
    for col in ["Open","High","Low","Close"]:
        if col not in df.columns:
            df[col] = df["Close"]
    df = df.reset_index()
    df["Return"] = df["Close"].pct_change().fillna(0)
    return df

df = None
for ticker in [SYMBOL, ALT_SYMBOL]:
    try:
        df = load_data(ticker)
        print(f"✅ Daten geladen von: {ticker}")
        break
    except Exception as e:
        print(f"⚠️ Fehler beim Laden von {ticker}: {e}")
if df is None:
    raise SystemExit("❌ Keine Daten verfügbar.")

# ----------------------------------------------------------
# 📊 ATR berechnen
# ----------------------------------------------------------
def compute_atr(df, period=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().fillna(method="bfill")
    df["ATR"] = atr
    return df

df = compute_atr(df, ATR_PERIOD)

# ----------------------------------------------------------
# 🔮 Vorhersagefunktion
# ----------------------------------------------------------
def calculate_prediction(df_slice):
    # ⚡ WICHTIG: Close als Series
    close = df_slice["Close"]

    df_slice["rsi"] = ta.momentum.RSIIndicator(close, window=RSI_PERIOD).rsi()
    df_slice["sma_short"] = close.rolling(SMA_SHORT).mean()
    df_slice["sma_long"] = close.rolling(SMA_LONG).mean()

    last_close = close.iloc[-1]
    last_sma_short = df_slice["sma_short"].iloc[-1]
    last_sma_long = df_slice["sma_long"].iloc[-1]
    last_rsi = df_slice["rsi"].iloc[-1] if not np.isnan(df_slice["rsi"].iloc[-1]) else 50
    last_atr = df_slice["ATR"].iloc[-1]

    prob = 50

    # Trend
    if last_sma_short > last_sma_long:
        prob += 10
    else:
        prob -= 10

    # Momentum
    prob += (last_rsi - 50)/2

    # Tagesbewegung relativ zur ATR
    daily_move = df_slice["Return"].iloc[-1]
    prob += (daily_move / last_atr) * 10

    # Trendserie (Markov-Prinzip)
    recent_returns = df_slice["Return"].tail(CHAIN_MAX)
    streak_up = (recent_returns > 0).cumprod().sum()
    streak_down = (recent_returns < 0).cumprod().sum()
    if streak_up > streak_down:
        prob += min(streak_up,5)
    elif streak_down > streak_up:
        prob -= min(streak_down,5)

    prob = max(0, min(100, prob))
    return prob

# ----------------------------------------------------------
# 📊 Backtesting
# ----------------------------------------------------------
correct = 0
total = 0

# Start nach SMA_LONG, damit alle Indikatoren verfügbar sind
for i in range(SMA_LONG, len(df)-1):
    df_slice = df.iloc[:i+1]
    prob = calculate_prediction(df_slice)

    # Richtung des nächsten Tages
    next_return = df["Return"].iloc[i+1]
    predicted_up = prob >= 50
    actual_up = next_return > 0

    if predicted_up == actual_up:
        correct += 1
    total += 1

accuracy = correct / total * 100
print(f"📊 Trefferquote des Modells auf historischen Daten: {accuracy:.2f} % ({correct}/{total})")
