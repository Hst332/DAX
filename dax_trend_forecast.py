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
LAST_DAYS = 400  # ⬅️ jetzt 400 Tage

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
def compute_atr(df, period=ATR_PERIOD):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().bfill()
    df["ATR"] = atr
    return df

df = compute_atr(df, ATR_PERIOD)

# ----------------------------------------------------------
# 🔮 Vorhersagefunktion (mit optimierten Gewichtungen)
# ----------------------------------------------------------
def calculate_prediction(df_slice):
    close = df_slice["Close"].squeeze()

    df_slice["rsi"] = ta.momentum.RSIIndicator(close, window=RSI_PERIOD).rsi()
    df_slice["sma_short"] = close.rolling(SMA_SHORT).mean()
    df_slice["sma_long"] = close.rolling(SMA_LONG).mean()

    last_sma_short = df_slice["sma_short"].iloc[-1]
    last_sma_long = df_slice["sma_long"].iloc[-1]
    last_rsi = df_slice["rsi"].iloc[-1] if not np.isnan(df_slice["rsi"].iloc[-1]) else 50
    last_atr = df_slice["ATR"].iloc[-1]
    daily_move = df_slice["Return"].iloc[-1]

    # Baseline
    prob = 50

    # 🟩 SMA-Trend (mittelfristig): stärker gewichten
    if last_sma_short > last_sma_long:
        prob += 15
    else:
        prob -= 15

    # 🟦 RSI (kurzfristiges Momentum)
    prob += (last_rsi - 50) * 0.6  # vorher 0.5 → feinfühliger

    # 🟨 ATR-bezogene Bewegung (Volatilität)
    if last_atr > 0:
        prob += np.tanh((daily_move / last_atr) * 2) * 8  # begrenzt auf ±8 %

    # 🟥 Trendserie (Markov-artig)
    recent_returns = df_slice["Return"].tail(CHAIN_MAX)
    up_streak = 0
    down_streak = 0
    for r in reversed(recent_returns):
        if r > 0:
            if down_streak > 0: break
            up_streak += 1
        elif r < 0:
            if up_streak > 0: break
            down_streak += 1
        else:
            break
    prob += up_streak * 1.2
    prob -= down_streak * 1.2

    # Grenzen
    prob = max(0, min(100, prob))
    return prob

# ----------------------------------------------------------
# 📊 Trefferquote der letzten LAST_DAYS
# ----------------------------------------------------------
correct = 0
total = 0

for i in range(len(df)-LAST_DAYS, len(df)):
    df_slice = df.iloc[:i+1]
    prob = calculate_prediction(df_slice)

    current_return = df["Return"].iloc[i]
    predicted_up = prob >= 50
    actual_up = current_return > 0

    if predicted_up == actual_up:
        correct += 1
    total += 1

accuracy = correct / total * 100
print(f"📊 Trefferquote des Modells in den letzten {LAST_DAYS} Tagen: {accuracy:.2f} % ({correct}/{total})")
