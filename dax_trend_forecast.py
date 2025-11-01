#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DAX Trend Forecast – Automatische tägliche Marktprognose
Includes TradingEconomics integration if TE_API_KEY environment variable is set.
"""

import os
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
import ta

# Try to import tradingeconomics; if not available or no key, use dummy econ signal
try:
    import tradingeconomics as te  # optional, only if TE_API_KEY provided
    TE_AVAILABLE = True
except Exception:
    TE_AVAILABLE = False

# ----------------------------------------------------------
# Parameters
# ----------------------------------------------------------
SYMBOL = "^GDAXI"
ALT_SYMBOL = "EXS1.DE"
ATR_PERIOD = 14
CHAIN_MAX = 14
TODAY = datetime.now()
END = TODAY
START = END - timedelta(days=3*365)

# ----------------------------------------------------------
# Helper: load data
# ----------------------------------------------------------
def load_data(ticker):
    df = yf.download(ticker, start=START, end=END, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    for col in ['Open','High','Low','Close']:
        if col not in df.columns:
            df[col] = df['Close']
    df = df.reset_index()
    df['Return'] = df['Close'].pct_change().fillna(0)
    return df

# Load DAX data (try primary then alt)
df = None
for ticker in [SYMBOL, ALT_SYMBOL]:
    try:
        df = load_data(ticker)
        print(f"✅ Loaded data from: {ticker}")
        break
    except Exception as e:
        print(f"⚠️ Failed to load {ticker}: {e}")
if df is None:
    raise SystemExit("❌ No DAX data available.")

# ATR
def compute_atr(df, period=14):
    high, low, close = df['High'], df['Low'], df['Close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(period).mean().bfill()
    return df

df = compute_atr(df, ATR_PERIOD)

# Kettenanalyse
df['Up'] = df['Return'] > 0
positive_array = df['Up'].values
results = []
for chain in range(1, CHAIN_MAX+1):
    pos_idx = [i for i in range(len(positive_array)-chain) if all(positive_array[i:i+chain])]
    neg_idx = [i for i in range(len(positive_array)-chain) if all(~positive_array[i:i+chain])]
    pos_next = [positive_array[i+chain] for i in pos_idx if i+chain < len(positive_array)]
    neg_next = [positive_array[i+chain] for i in neg_idx if i+chain < len(positive_array)]
    results.append({
        "Kettenlänge": chain,
        "Positiv-Kette ↑ (%)": round(np.mean(pos_next)*100,2) if pos_next else np.nan,
        "Negativ-Kette ↓ (%)": round((1-np.mean(neg_next))*100,2) if neg_next else np.nan,
        "#Pos-Fälle": len(pos_idx),
        "#Neg-Fälle": len(neg_idx)
    })
result_df = pd.DataFrame(results)

# Technische Analyse (RSI + SMA50)
def calculate_technical_signal(df):
    close = df['Close']
    df['rsi'] = ta.momentum.RSIIndicator(close, window=14).rsi()
    df['sma50'] = close.rolling(50).mean()
    last_close = close.iloc[-1]
    last_rsi = df['rsi'].iloc[-1] if not np.isnan(df['rsi'].iloc[-1]) else 50
    if last_close > df['sma50'].iloc[-1]:
        signal = 1
        prob = 55 + (last_rsi - 50)/2
    else:
        signal = -1
        prob = 55 + (50 - last_rsi)/2
    return signal, max(0, min(100, prob))

tech_signal, tech_prob = calculate_technical_signal(df)

# Economic calendar via TradingEconomics if key is present
TE_KEY = os.environ.get("TE_API_KEY")
econ_used = "Dummy"  # default
def get_economic_risk_signal(country_list=None, importance_threshold=2):
    global econ_used
    if country_list is None:
        country_list = ["Germany","Eurozone","United States"]
    # If TE key not set or tradingeconomics not available, return dummy -1
    if not TE_KEY or not TE_AVAILABLE:
        econ_used = "Dummy"
        return -1
    try:
        te.login(TE_KEY)
        econ_used = "API"
        df_cal = te.getCalendarData(output_type='df')
        df_cal['Date'] = pd.to_datetime(df_cal['Date']).dt.date
        today = datetime.now().date()
        df_today = df_cal[(df_cal['Date'] == today) & (df_cal['Country'].isin(country_list))]
        high_impact = df_today[df_today['Importance'] >= importance_threshold]
        if not high_impact.empty:
            return -1
        return 1
    except Exception as e:
        print(f"⚠️ TradingEconomics fetch failed: {e}")
        econ_used = "Dummy"
        return -1

# Global signals via yfinance
def get_global_signals():
    signals = {}
    # DAX-Future: use recent hourly movement of ^GDAXI as proxy
    try:
        dax_future = yf.download("^GDAXI", period="2d", interval="1h", progress=False)
        dax_future_change = dax_future['Close'].pct_change().iloc[-1]
        signals['DAX-Future'] = 1 if dax_future_change > 0 else -1
    except Exception:
        signals['DAX-Future'] = 1

    # US markets: Dow (^DJI) and Nasdaq (^IXIC) last daily change
    try:
        us_dow = yf.download("^DJI", period="5d", progress=False)
        us_nasdaq = yf.download("^IXIC", period="5d", progress=False)
        us_perf = (us_dow['Close'].pct_change().iloc[-1] + us_nasdaq['Close'].pct_change().iloc[-1]) / 2
        signals['US-Märkte'] = 1 if us_perf > 0 else -1
    except Exception:
        signals['US-Märkte'] = 1

    # Asia: Nikkei (^N225) and Hang Seng (^HSI)
    try:
        asia_nikkei = yf.download("^N225", period="5d", progress=False)
        asia_hsi = yf.download("^HSI", period="5d", progress=False)
        asia_perf = (asia_nikkei['Close'].pct_change().iloc[-1] + asia_hsi['Close'].pct_change().iloc[-1]) / 2
        signals['Asienmärkte'] = 1 if asia_perf > 0 else -1
    except Exception:
        signals['Asienmärkte'] = -1

    # Economic calendar
    signals['Wirtschaftskalender'] = get_economic_risk_signal()

    # Technical
    signals['Technische'] = tech_signal
    return signals

signals = get_global_signals()

# Combined forecast with given weights
def combined_forecast(signals):
    weights = {
        "DAX-Future": 0.40,
        "US-Märkte": 0.20,
        "Asienmärkte": 0.10,
        "Wirtschaftskalender": 0.15,
        "Technische": 0.15
    }
    score = sum(signals[k] * weights[k] for k in weights)
    trend = "Steigend 📈" if score > 0 else "Fallend 📉"
    prob = round(abs(score) * 100, 2)
    return trend, prob, score

final_trend, final_prob, score = combined_forecast(signals)

# Chain length calculation
def calculate_chain_length(df):
    last_sign = np.sign(df['Return'].iloc[-1])
    chain = 1
    for i in range(2, len(df)):
        if np.sign(df['Return'].iloc[-i]) == last_sign:
            chain += 1
        else:
            break
    direction = "Positiv 📈" if last_sign > 0 else "Negativ 📉"
    return chain, direction

chain_len, chain_dir = calculate_chain_length(df)

def get_chain_probability(chain_len, result_df):
    if chain_len <= result_df['Kettenlänge'].max():
        row = result_df[result_df['Kettenlänge'] == chain_len]
        if not row.empty:
            if "Positiv" in chain_dir:
                return row['Positiv-Kette ↑ (%)'].values[0]
            else:
                return row['Negativ-Kette ↓ (%)'].values[0]
    return np.nan

chain_prob = get_chain_probability(chain_len, result_df)

# Output formatting
timestamp = datetime.now().strftime("%d.%m.%Y %H:%M")
signal_texts = {1: "Ja ✅ / Steigend", -1: "Nein ❌ / Fallend"}
signals_readable = "\\n".join([f"{k}: {signal_texts[v]}  (aktuelle Daten {timestamp})" for k, v in signals.items()])

last_close = df['Close'].iloc[-1]
last_date = df['Date'].iloc[-1].strftime("%d.%m.%Y")

econ_status_line = "🟢 Wirtschaftskalender: Echt (API)" if econ_used == "API" else "⚪ Wirtschaftskalender: Dummy (kein API-Key oder Fehler)"

msg = (
    f"📅 Prognosedatum: {TODAY:%d.%m.%Y %H:%M}\\n"
    f"📆 Letzter Handelstag: {last_date}\\n"
    f"📈 DAX: {round(last_close, 2)} €\\n"
    f"🔗 Aktuelle Kettenlänge: {chain_len} Tage in Folge {chain_dir}\\n"
    f"📊 Historische Fortsetzungswahrscheinlichkeit: {chain_prob:.2f} %\\n"
    f"{econ_status_line}\\n"
    f"🔮 Kombinierter Trend: {final_trend}\\n"
    f"📈 Gesamtwahrscheinlichkeit: {final_prob:.2f} %\\n"
    f"⚙️ Technischer Anteil (RSI/SMA): {tech_prob:.1f} %\\n\\n"
    f"🌍 Einzel-Signale:\\n{signals_readable}\\n\\n"
    f"ℹ️ Automatische tägliche Prognose\\n"
)

print(msg)
with open('result.txt', 'w', encoding='utf-8') as f:
    f.write(msg)
print('📁 Ergebnis in result.txt gespeichert ✅')
