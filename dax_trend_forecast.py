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
RSI_PERIOD = 7        # kürzer für schnelleres Momentum
SMA_SHORT = 10
SMA_LONG = 50
CHAIN_MAX = 30        # Trendserie max Tage

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
# 🔮 Vorhersage berechnen
# ----------------------------------------------------------
def calculate_prediction(df):
    close = df["Close"]
    df["rsi"] = ta.momentum.RSIIndicator(close, window=RSI_PERIOD).rsi()
    df["sma_short"] = close.rolling(SMA_SHORT).mean()
    df["sma_long"] = close.rolling(SMA_LONG).mean()

    last_close = close.iloc[-1]
    last_sma_short = df["sma_short"].iloc[-1]
    last_sma_long = df["sma_long"].iloc[-1]
    last_rsi = df["rsi"].iloc[-1] if not np.isnan(df["rsi"].iloc[-1]) else 50
    last_atr = df["ATR"].iloc[-1]

    # Basiswahrscheinlichkeit
    prob = 50

    # SMA-Trend
    if last_sma_short > last_sma_long:
        trend = "Steigend"
        prob += 10
    else:
        trend = "Fallend"
        prob -= 10

    # Momentum (RSI)
    prob += (last_rsi - 50) / 2  # ±25%

    # Letzte Tagesbewegung relativ zur ATR
    daily_move = df["Return"].iloc[-1]
    prob += (daily_move / last_atr) * 10  # Anpassung basierend auf Volatilität

    # Trendserie (Markov-Ketten-Prinzip)
    recent_returns = df["Return"].tail(CHAIN_MAX)
    streak_up = (recent_returns > 0).cumprod().sum()
    streak_down = (recent_returns < 0).cumprod().sum()
    if streak_up > streak_down:
        prob += min(streak_up, 5)  # max +5%
    elif streak_down > streak_up:
        prob -= min(streak_down, 5)  # max -5%

    # Begrenzung
    prob = max(0, min(100, prob))

    return trend, round(prob, 2)

trend, prob = calculate_prediction(df)
last_close = df["Close"].iloc[-1]

# ----------------------------------------------------------
# 📅 Aktuelle Trendserie
# ----------------------------------------------------------
def get_streak(df, days=CHAIN_MAX):
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

with open("result.txt", "w", encoding="utf-8") as f:
    f.write(msg)
print("📁 Ergebnis in result.txt gespeichert")
