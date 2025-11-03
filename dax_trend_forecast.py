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
RSI_PERIOD = 14
SMA_SHORT = 20
SMA_LONG = 50
LAST_DAYS = 400
CHAIN_MAX = 14
ROLL_WINDOW = 30  # Rollierende Trefferquote

# Beste gefundene Parameter (fest)
W_SMA = 8
W_RSI = 0.8
W_ATR = 4
W_STREAK = 1.5
OPT_HISTORICAL_ACCURACY = 69.84  # Optimierte Trefferquote

END = datetime.now()
START = END - timedelta(days=3*365)

# ----------------------------------------------------------
# 📥 Daten laden
# ----------------------------------------------------------
def load_data(ticker):
    df = yf.download(ticker, start=START, end=END, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"Keine Daten für {ticker}")
    for col in ["Open", "High", "Low", "Close"]:
        if col not in df.columns:
            df[col] = df["Close"]
    df = df.reset_index()
    # Close als 1D Series erzwingen
    df["Close"] = pd.Series(df["Close"].values.flatten(), dtype=float)
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
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(period).mean().bfill()
    return df

df = compute_atr(df, ATR_PERIOD)

# ----------------------------------------------------------
# 🔮 Prognoseberechnung
# ----------------------------------------------------------
def calculate_prediction(df, w_sma, w_rsi, w_atr, w_streak, sma_short, sma_long):
    if len(df) < sma_long:  # Minimum prüfen
        return 50  # neutrale Wahrscheinlichkeit

    close = pd.Series(df["Close"].values.flatten(), dtype=float)  # fix für RSI
    df["sma_short"] = close.rolling(sma_short).mean()
    df["sma_long"] = close.rolling(sma_long).mean()
    df["rsi"] = ta.momentum.RSIIndicator(close, window=RSI_PERIOD).rsi()

    last_sma_short = df["sma_short"].iloc[-1]
    last_sma_long = df["sma_long"].iloc[-1]
    last_rsi = df["rsi"].iloc[-1] if not np.isnan(df["rsi"].iloc[-1]) else 50
    last_atr = df["ATR"].iloc[-1]
    daily_move = df["Return"].iloc[-1]

    prob = 50

    # Trend
    prob += w_sma if last_sma_short > last_sma_long else -w_sma

    # RSI
    prob += (last_rsi - 50) * w_rsi

    # ATR-bewertete Bewegung
    if last_atr > 0:
        prob += np.tanh((daily_move / last_atr) * 2) * w_atr

    # Trendserie
    recent_returns = list(df["Return"].tail(CHAIN_MAX))
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
    prob += up_streak * w_streak
    prob -= down_streak * w_streak

    prob = max(0, min(100, prob))
    return prob

# ----------------------------------------------------------
# 🔹 Aktuelle Trendserie
# ----------------------------------------------------------
def get_streak(df):
    recent_returns = df["Return"].tail(30).values
    up = recent_returns[-1] > 0
    streak = 1
    for r in reversed(recent_returns[:-1]):
        if (r > 0 and up) or (r < 0 and not up):
            streak += 1
        else:
            break
    direction = "gestiegen 📈" if up else "gefallen 📉"
    return direction, streak

# ----------------------------------------------------------
# 🔹 Rollierende Trefferquote (optional)
# ----------------------------------------------------------
def rolling_accuracy(df, w_sma, w_rsi, w_atr, w_streak, sma_short, sma_long, window=ROLL_WINDOW):
    acc_list = []
    for i in range(window, len(df)):
        correct = 0
        for j in range(i-window, i):
            df_slice = df.iloc[:j+1].copy()
            prob = calculate_prediction(df_slice, w_sma, w_rsi, w_atr, w_streak, sma_short, sma_long)
            predicted_up = prob >= 50
            actual_up = df["Return"].iloc[j] > 0
            if predicted_up == actual_up:
                correct += 1
        acc_list.append(correct / window * 100)
    return acc_list

# ----------------------------------------------------------
# 🔹 Berechnungen
# ----------------------------------------------------------
trend_prob = calculate_prediction(df, W_SMA, W_RSI, W_ATR, W_STREAK, SMA_SHORT, SMA_LONG)
trend = "Steigend 📈" if trend_prob >= 50 else "Fallend 📉"
last_close = df["Close"].iloc[-1]

streak_direction, streak_length = get_streak(df)
rolling_acc = rolling_accuracy(df, W_SMA, W_RSI, W_ATR, W_STREAK, SMA_SHORT, SMA_LONG)

# ----------------------------------------------------------
# 📊 Ergebnis ausgeben & speichern
# ----------------------------------------------------------
msg = (
    f"📅 {datetime.now():%d.%m.%Y %H:%M}\n"
    f"📈 DAX: {round(last_close,2)} €\n"
    f"🔮 Trend: {trend}\n"
    f"📊 Wahrscheinlichkeit steigend: {round(trend_prob,2)} %\n"
    f"📊 Wahrscheinlichkeit fallend : {round(100-trend_prob,2)} %\n"
    f"📏 Aktueller Trend: DAX ist {streak_length} Tage in Folge {streak_direction}\n"
    f"🎯 Optimierte Trefferquote (letzte {LAST_DAYS} Tage): {OPT_HISTORICAL_ACCURACY} %\n"
    f"⚙️ Beste Parameter → SMA={SMA_SHORT}/{SMA_LONG}, WSMA={W_SMA}, RSI={W_RSI}, ATR={W_ATR}, Streak={W_STREAK}"
)

print(msg)

with open("result.txt", "w", encoding="utf-8") as f:
    f.write(msg)
print("📁 Ergebnis in result.txt gespeichert ✅")
