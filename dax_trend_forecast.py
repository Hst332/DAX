# ----------------------------------------------------------
# 📦 Bibliotheken
# ----------------------------------------------------------
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ta

# ----------------------------------------------------------
# ⚙️ Parameter
# ----------------------------------------------------------
SYMBOL = "^GDAXI"
ALT_SYMBOL = "EXS1.DE"
ATR_PERIOD = 14
SMA_SHORT = 20
SMA_LONG = 50
RSI_PERIOD = 14
LAST_DAYS = 400  # Rückblick für Trefferquote

# Beste gefundene Gewichtungen
W_SMA = 10
W_RSI = 0.8
W_ATR = 6
W_STREAK = 1.5

END = datetime.now()
START = END - timedelta(days=3*365)

# ----------------------------------------------------------
# 📥 Daten laden
# ----------------------------------------------------------
def load_data(ticker):
    df = yf.download(ticker, start=START, end=END, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"Keine Daten für {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    for col in ["Open", "High", "Low", "Close"]:
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
    atr = tr.rolling(period).mean().bfill()
    df["ATR"] = atr
    return df

df = compute_atr(df, ATR_PERIOD)

# ----------------------------------------------------------
# 🔮 Prognoseberechnung
# ----------------------------------------------------------
def calculate_prediction(df):
    close = df["Close"].squeeze()
    df["rsi"] = ta.momentum.RSIIndicator(close, window=RSI_PERIOD).rsi()
    df["sma_short"] = close.rolling(SMA_SHORT).mean()
    df["sma_long"] = close.rolling(SMA_LONG).mean()

    last_sma_short = df["sma_short"].iloc[-1]
    last_sma_long = df["sma_long"].iloc[-1]
    last_rsi = df["rsi"].iloc[-1] if not np.isnan(df["rsi"].iloc[-1]) else 50
    last_atr = df["ATR"].iloc[-1]
    daily_move = df["Return"].iloc[-1]

    prob = 50

    # Trend
    prob += W_SMA if last_sma_short > last_sma_long else -W_SMA

    # RSI
    prob += (last_rsi - 50) * W_RSI

    # Volatilitätseinfluss
    if last_atr > 0:
        prob += np.tanh((daily_move / last_atr) * 2) * W_ATR

    # Trendserie
    recent_returns = list(df["Return"].tail(14))
    up_streak = down_streak = 0
    for r in reversed(recent_returns):
        if r > 0:
            if down_streak > 0:
                break
            up_streak += 1
        elif r < 0:
            if up_streak > 0:
                break
            down_streak += 1
        else:
            break
    prob += up_streak * W_STREAK
    prob -= down_streak * W_STREAK

    prob = max(0, min(100, prob))
    trend = "Steigend 📈" if prob >= 50 else "Fallend 📉"
    return trend, prob

# ----------------------------------------------------------
# 📈 Trefferquote berechnen (Backtest)
# ----------------------------------------------------------
correct = 0
total = 0

for i in range(len(df) - LAST_DAYS, len(df)):
    df_slice = df.iloc[:i+1].copy()
    trend, prob = calculate_prediction(df_slice)
    actual_up = df["Return"].iloc[i] > 0
    predicted_up = prob >= 50
    if actual_up == predicted_up:
        correct += 1
    total += 1

accuracy = correct / total * 100

# ----------------------------------------------------------
# 🧭 Aktuelle Prognose
# ----------------------------------------------------------
trend, prob = calculate_prediction(df)
last_close = df["Close"].iloc[-1]

msg = (
    f"📅 {datetime.now():%d.%m.%Y %H:%M}\n"
    f"📈 DAX: {round(last_close, 2)} €\n"
    f"🔮 Trend: {trend}\n"
    f"📊 Wahrscheinlichkeit: {round(prob, 2)} %\n"
    f"🎯 Trefferquote (letzte {LAST_DAYS} Tage): {round(accuracy, 2)} %\n"
    f"ℹ️ Modellgewichte → SMA={W_SMA}, RSI={W_RSI}, ATR={W_ATR}, Streak={W_STREAK}"
)

print(msg)

# Datei speichern
with open("result.txt", "w", encoding="utf-8") as f:
    f.write(msg)
print("📁 Ergebnis in result.txt gespeichert ✅")
