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
CHAIN_MAX = 30  # Anzahl Tage für Trendstreak

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
# 🔮 Vorhersage berechnen
# ----------------------------------------------------------
def calculate_prediction(df):
    close = df["Close"]
    df["rsi"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    df["sma50"] = close.rolling(50).mean()

    last_close = close.iloc[-1]
    last_sma = df["sma50"].iloc[-1]
    last_rsi = df["rsi"].iloc[-1] if not np.isnan(df["rsi"].iloc[-1]) else 50

    # Basiswahrscheinlichkeit
    prob = 50

    # SMA-Trend einfließen lassen
    if last_close > last_sma:
        trend = "Steigend"
        prob += 10  # leicht bullish
    else:
        trend = "Fallend"
        prob -= 10  # leicht bearish

    # RSI einfließen lassen
    prob += (last_rsi - 50)/2

    # Begrenzung auf 0-100%
    prob = max(0, min(100, prob))

    return trend, round(prob,2)

trend, prob = calculate_prediction(df)
last_close = df["Close"].iloc[-1]

# ----------------------------------------------------------
# 📅 Aktuelle Trendserie
# ----------------------------------------------------------
def get_streak(df, days=30):
    recent_returns = df["Return"].tail(days).values
    up = recent_returns[-1] > 0
    streak = 1
    for r in reversed(recent_returns[:-1]):
        if (r > 0 and up) or (r < 0 and not up):
            streak += 1
        else:
            break
    direction = "gestiegen 📈" if up else "gefallen 📉"
    return direction, streak

direction, streak = get_streak(df, CHAIN_MAX)

# ----------------------------------------------------------
# 📈 Ergebnis ausgeben & Datei speichern
# ----------------------------------------------------------
msg = (
    f"📅 {datetime.now():%d.%m.%Y %H:%M}\n"
    f"📈 DAX: {round(last_close,2)} €\n"
    f"🔮 Trend: {trend}\n"
    f"📊 Wahrscheinlichkeit: {prob} %\n"
    f"📏 Aktueller Trend: DAX ist {streak} Tage in Folge {direction}\n"
    f"ℹ️ Automatische tägliche Prognose"
)

print(msg)

# Datei speichern (überschreibt täglich)
with open("result.txt", "w", encoding="utf-8") as f:
    f.write(msg)
print("📁 Ergebnis in result.txt gespeichert")
