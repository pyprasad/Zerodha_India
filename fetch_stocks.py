"""
fetch_stocks.py — Download 15 liquid NSE F&O stock futures histories from Yahoo Finance.
Run once: python fetch_stocks.py
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import pandas as pd
import yfinance as yf

Path("data/historical").mkdir(parents=True, exist_ok=True)

# 15 most liquid NSE F&O single-stock futures
# Format: internal_name -> yahoo_ticker
STOCKS = {
    "RELIANCE":   "RELIANCE.NS",
    "HDFCBANK":   "HDFCBANK.NS",
    "INFY":       "INFY.NS",
    "TCS":        "TCS.NS",
    "ICICIBANK":  "ICICIBANK.NS",
    "AXISBANK":   "AXISBANK.NS",
    "SBIN":       "SBIN.NS",
    "BAJFINANCE": "BAJFINANCE.NS",
    "TATAMOTORS": "TATAMOTORS.NS",
    "HCLTECH":    "HCLTECH.NS",
    "ITC":        "ITC.NS",
    "LT":         "LT.NS",
    "SUNPHARMA":  "SUNPHARMA.NS",
    "KOTAKBANK":  "KOTAKBANK.NS",
    "MARUTI":     "MARUTI.NS",
}

print(f"Downloading {len(STOCKS)} stocks from Yahoo Finance (max history)...\n")

for name, ticker in STOCKS.items():
    out = Path(f"data/historical/{name}_daily_extended.csv")
    if out.exists():
        df_existing = pd.read_csv(out, index_col="date", parse_dates=True)
        print(f"  {name:12s} already exists ({len(df_existing)} bars) — skipping")
        continue

    try:
        df = yf.download(ticker, period="max", interval="1d",
                         auto_adjust=True, progress=False)
        if df.empty:
            print(f"  {name:12s} [FAIL] — no data returned")
            continue

        df.columns = [c.lower() if isinstance(c, str) else c[0].lower()
                      for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]]
        df.index.name = "date"
        df = df.dropna(subset=["open", "high", "low", "close"])
        df = df[df.index.dayofweek < 5]
        # Drop phantom bars (zero-move rows)
        no_move = ((df["open"] == df["close"]) & (df["high"] == df["close"])
                   & (df["low"]  == df["close"]))
        df = df[~no_move]
        df.index = df.index.normalize()
        df.to_csv(out)
        print(f"  {name:12s} OK — {len(df)} bars  "
              f"{df.index[0].date()} → {df.index[-1].date()}")
    except Exception as e:
        print(f"  {name:12s} [ERROR] {e}")

print("\nDone.")
