# DAX Trend Forecast

Automatische tägliche DAX‑Prognose (Python). Kombiniert DAX‑Future, US‑Schlusskurse, Asienmärkte, Wirtschaftskalender (TradingEconomics optional) und technische Indikatoren (RSI/SMA) mit benutzerdefinierten Gewichtungen.

## Projektstruktur
```
dax-trend-forecast/
├── dax_trend_forecast.py
├── README.md
├── requirements.txt
├── .gitignore
└── .github/
    └── workflows/
        └── schedule.yml
```

## Gewichtung (fest)
- DAX-Future: 40%
- US-Schlusskurse: 20%
- Asienmärkte: 10%
- Wirtschaftskalender: 15%
- Technische Indikatoren (RSI/SMA): 15%

## Setup (lokal)
1. Virtuelle Umgebung erstellen und aktivieren
```bash
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
.venv\\Scripts\\Activate.ps1  # Windows PowerShell
```

2. Abhängigkeiten installieren
```bash
pip install -r requirements.txt
```

3. Optional: TradingEconomics API-Key setzen (lokal)
```bash
export TE_API_KEY="user:secret"        # macOS / Linux
setx TE_API_KEY "user:secret"          # Windows (PowerShell)
```

4. Skript ausführen
```bash
python dax_trend_forecast.py
```

## GitHub Actions (täglicher Lauf)
Die Datei `.github/workflows/schedule.yml` führt das Skript täglich um **08:30 Wien‑Zeit (CET/CEST)** aus. 
Die Action verwendet das Repository‑Secret `TE_API_KEY`, wenn gesetzt.

## Hinweise
- Wenn kein gültiger `TE_API_KEY` gefunden wird, verwendet das Skript einen Dummy‑Wert für den Wirtschaftskalender und gibt dies im Ergebnis aus.
- Für Kettenlängen >10 sind die historischen Wahrscheinlichkeiten statistisch wenig aussagekräftig — das Skript gibt die historischen Werte trotzdem aus.

## Lizenz
Optional (z. B. MIT)
