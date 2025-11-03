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
RSI_PERIODS = [5, 14]          # Doppel-RSI: kurzfristig + mittelfristig
SMA_SHORT_LIST = [10, 15, 20]
SMA_LONG_LIST = [40, 50, 60]
LAST_DAYS = 400
CHAIN_MAX = 14
CONFIDENCE_THRESHOLD = 5        # ±5 % um 50 wird als „unsicher“ ausgefiltert

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
# 🔮 Prognose mit doppeltem RSI und Gewichtung
# ----------------------------------------------------------
def calculate_prediction(df, w_sma, w_rsi, w_atr, w_streak, sma_short, sma_long):
    close = df["Close"].squeeze()
    df["sma_short"] = close.rolling(sma_short).mean()
    df["sma_long"] = close.rolling(sma_long).mean()
    df["rsi_short"] = ta.momentum.RSIIndicator(close, window=RSI_PERIODS[0]).rsi()
    df["rsi_long"] = ta.momentum.RSIIndicator(close, window=RSI_PERIODS[1]).rsi()

    last_sma_short = df["sma_short"].iloc[-1]
    last_sma_long = df["sma_long"].iloc[-1]
    last_rsi_short = df["rsi_short"].iloc[-1] if not np.isnan(df["rsi_short"].iloc[-1]) else 50
    last_rsi_long = df["rsi_long"].iloc[-1] if not np.isnan(df["rsi_long"].iloc[-1]) else 50
    last_atr = df["ATR"].iloc[-1]
    daily_move = df["Return"].iloc[-1]

    prob = 50

    # SMA-Trend
    prob += w_sma if last_sma_short > last_sma_long else -w_sma

    # Doppel-RSI: Durchschnitt aus RSI5 & RSI14
    rsi_mix = (last_rsi_short + last_rsi_long) / 2
    prob += (rsi_mix - 50) * w_rsi

    # ATR-bewertete Bewegung
    if last_atr > 0:
        prob += np.tanh((daily_move / last_atr) * 2) * w_atr

    # Trendserie (Markov-ähnlich)
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
        else:
            break
    prob += up_streak * w_streak
    prob -= down_streak * w_streak

    prob = max(0, min(100, prob))
    return prob

# ----------------------------------------------------------
# 🧪 Testfunktion mit Signalfilter
# ----------------------------------------------------------
def test_weights(w_sma, w_rsi, w_atr, w_streak, sma_short, sma_long):
    correct = 0
    total = 0
    used_signals = 0

    for i in range(len(df) - LAST_DAYS, len(df)):
        df_slice = df.iloc[:i+1].copy()
        prob = calculate_prediction(df_slice, w_sma, w_rsi, w_atr, w_streak, sma_short, sma_long)
        current_return = df["Return"].iloc[i]
        predicted_up = prob >= 50
        actual_up = current_return > 0

        # Nur zählen, wenn Signal sicher genug
        if abs(prob - 50) >= CONFIDENCE_THRESHOLD:
            used_signals += 1
            if predicted_up == actual_up:
                correct += 1
            total += 1

    if total == 0:
        return 0, 0
    return correct / total * 100, used_signals / LAST_DAYS * 100

# ----------------------------------------------------------
# 🔍 Grid Search – beste Kombination finden
# ----------------------------------------------------------
best_acc = 0
best_params = None

for sma_short in SMA_SHORT_LIST:
    for sma_long in SMA_LONG_LIST:
        for w_sma in [8, 10, 12]:
            for w_rsi in [0.6, 0.8, 1.0]:
                for w_atr in [4, 6, 8]:
                    for w_streak in [1.0, 1.2, 1.5]:
                        acc, coverage = test_weights(w_sma, w_rsi, w_atr, w_streak, sma_short, sma_long)
                        if acc > best_acc and coverage > 20:  # mind. 20 % Signaltage
                            best_acc = acc
                            best_params = (w_sma, w_rsi, w_atr, w_streak, sma_short, sma_long)
                        print(f"🧩 SMA={sma_short}/{sma_long}, WSMA={w_sma}, RSI={w_rsi}, ATR={w_atr}, Streak={w_streak} → {acc:.2f}% (Signale: {coverage:.1f}%)")

print("\n✅ Beste Kombination gefunden:")
print(f"   SMA={best_params[4]}/{best_params[5]}, WSMA={best_params[0]}, RSI={best_params[1]}, ATR={best_params[2]}, Streak={best_params[3]}")
print(f"📊 Trefferquote: {best_acc:.2f} %")

# ----------------------------------------------------------
# 📈 Aktuelle Prognose mit bestem Set
# ----------------------------------------------------------
w_sma, w_rsi, w_atr, w_streak, sma_short, sma_long = best_params
trend_prob = calculate_prediction(df, w_sma, w_rsi, w_atr, w_streak, sma_short, sma_long)
trend = "Steigend 📈" if trend_prob >= 50 else "Fallend 📉"
last_close = df["Close"].iloc[-1]

msg = (
    f"📅 {datetime.now():%d.%m.%Y %H:%M}\n"
    f"📈 DAX: {round(last_close,2)} €\n"
    f"🔮 Trend: {trend}\n"
    f"📊 Wahrscheinlichkeit steigend: {round(trend_prob,2)} %\n"
    f"📊 Wahrscheinlichkeit fallend: {100-round(trend_prob,2)} %\n"
    f"🎯 Optimierte Trefferquote (letzte {LAST_DAYS} Tage): {round(best_acc,2)} %\n"
    f"⚙️ Beste Parameter → SMA={sma_short}/{sma_long}, WSMA={w_sma}, RSI={w_rsi}, ATR={w_atr}, Streak={w_streak}"
)

print(msg)

# Datei speichern
with open("result.txt", "w", encoding="utf-8") as f:
    f.write(msg + "\n")
print("📁 Ergebnis in result.txt gespeichert ✅")
