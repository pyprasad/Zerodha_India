"""
fetch_all_instruments.py
========================
Downloads 7-year daily OHLCV history for all tradeable Indian index instruments.
Handles Yahoo Finance symbol fallbacks. Saves CSV + Parquet to data/historical/.

Instruments:
  NIFTY       — ^NSEI                   (NIFTY 50, lot 65)
  BANKNIFTY   — ^NSEBANK                (Bank Nifty, lot 30)
  MIDCPNIFTY  — ^NSMIDCP / ^CNXMIDCAP  (Midcap Select, lot 120)
  FINNIFTY    — ^CNXFIN / ^CNXFINANCE   (Fin Services, lot 60)
  NIFTYIT     — ^CNXIT                  (IT Sector, lot 30)
  INDIAVIX    — ^INDIAVIX               (Volatility index — signal only)

Usage:
  python fetch_all_instruments.py
"""

import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import yfinance as yf

from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
START_DATE = "2017-01-01"
END_DATE   = datetime.today().strftime("%Y-%m-%d")
DATA_DIR   = Path("data/historical")
MIN_ROWS   = 100   # Reject if fewer rows returned

INSTRUMENTS = [
    {
        "name":     "NIFTY",
        "symbols":  ["^NSEI"],
        "filename": "NIFTY_daily_extended",
        "lot_size": 65,
        "note":     "NIFTY 50 spot index",
    },
    {
        "name":     "BANKNIFTY",
        "symbols":  ["^NSEBANK"],
        "filename": "BANKNIFTY_daily_extended",
        "lot_size": 30,
        "note":     "Bank Nifty spot index",
    },
    {
        "name":     "MIDCPNIFTY",
        "symbols":  ["^NSMIDCP", "^CNXMIDCAP", "NIFTY_MIDCAP_100.NS"],
        "filename": "MIDCPNIFTY_daily_extended",
        "lot_size": 120,
        "note":     "Midcap Select / Midcap 100 proxy",
    },
    {
        "name":     "FINNIFTY",
        "symbols":  ["^CNXFIN", "^CNXFINANCE", "NIFTY_FIN_SERVICE.NS"],
        "filename": "FINNIFTY_daily_extended",
        "lot_size": 60,
        "note":     "Nifty Financial Services proxy",
    },
    {
        "name":     "NIFTYIT",
        "symbols":  ["^CNXIT"],
        "filename": "NIFTYIT_daily_extended",
        "lot_size": 30,
        "note":     "Nifty IT sector index",
    },
    {
        "name":     "INDIAVIX",
        "symbols":  ["^INDIAVIX"],
        "filename": "INDIAVIX_daily_extended",
        "lot_size": 0,
        "note":     "India VIX — signal instrument only",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def fetch_with_fallback(name: str, symbols: list, start: str, end: str) -> tuple:
    """
    Try each Yahoo Finance symbol until one returns >= MIN_ROWS rows.
    Returns (DataFrame, symbol_used) or raises RuntimeError.

    DataFrame columns: open, high, low, close, volume (lowercase)
    Index: date (DatetimeIndex)
    """
    for sym in symbols:
        try:
            raw = yf.download(sym, start=start, end=end,
                              auto_adjust=True, progress=False)
            if raw is None or raw.empty:
                console.print(f"    [dim]{sym}: empty response[/dim]")
                continue
            # Flatten MultiIndex columns (yfinance >= 0.2.x with group_by)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            # Keep only OHLCV
            available = [c for c in ["Open","High","Low","Close","Volume"]
                         if c in raw.columns]
            if len(available) < 4:
                console.print(f"    [dim]{sym}: missing columns {available}[/dim]")
                continue
            df = raw[available].copy()
            df.columns = [c.lower() for c in df.columns]
            if "volume" not in df.columns:
                df["volume"] = 0
            df.index = pd.to_datetime(df.index)
            df.index.name = "date"
            # Drop rows with NaN close or zero/negative close
            df = df.dropna(subset=["close"])
            df = df[df["close"] > 0]
            if len(df) < MIN_ROWS:
                console.print(f"    [dim]{sym}: only {len(df)} rows — skipping[/dim]")
                continue
            return df, sym
        except Exception as e:
            console.print(f"    [dim]{sym}: error — {e}[/dim]")
            continue
    raise RuntimeError(f"All symbols failed for {name}: {symbols}")


def check_existing(filename: str) -> tuple:
    """Returns (from_date, to_date, num_rows) if CSV exists, else (None, None, 0)."""
    path = DATA_DIR / f"{filename}.csv"
    if not path.exists():
        return None, None, 0
    try:
        df = pd.read_csv(path, index_col="date", parse_dates=True)
        return (
            str(df.index[0].date()),
            str(df.index[-1].date()),
            len(df)
        )
    except Exception:
        return None, None, 0


def save_instrument(df: pd.DataFrame, filename: str) -> None:
    """Save DataFrame to CSV and Parquet."""
    csv_path     = DATA_DIR / f"{filename}.csv"
    parquet_path = DATA_DIR / f"{filename}.parquet"
    df.to_csv(csv_path)
    try:
        df.to_parquet(parquet_path)
    except Exception as e:
        console.print(f"    [dim]Parquet save skipped: {e}[/dim]")


def print_summary(results: list) -> None:
    tbl = Table(
        title=f"Instrument Data Summary — {START_DATE} → {END_DATE}",
        box=box.DOUBLE_EDGE, show_header=True, header_style="bold cyan",
        min_width=110,
    )
    tbl.add_column("Instrument",  min_width=14)
    tbl.add_column("Lot Size",    justify="right", min_width=8)
    tbl.add_column("Symbol Used", min_width=26)
    tbl.add_column("Rows",        justify="right", min_width=7)
    tbl.add_column("From",        min_width=12)
    tbl.add_column("To",          min_width=12)
    tbl.add_column("Status",      min_width=10)
    tbl.add_column("Note",        min_width=30)

    for r in results:
        ok     = r["status"] == "OK"
        color  = "green" if ok else "red"
        status = f"[{color}]{r['status']}[/{color}]"
        tbl.add_row(
            r["name"],
            str(r["lot_size"]),
            r.get("symbol_used", "—"),
            str(r["rows"]) if ok else "—",
            r.get("from", "—"),
            r.get("to",   "—"),
            status,
            r.get("note", ""),
        )
    console.print(tbl)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    console.rule("[bold cyan]Fetching Indian Index Instruments — 7-Year History[/bold cyan]")
    console.print(f"[dim]Source: Yahoo Finance | Period: {START_DATE} → {END_DATE}[/dim]\n")

    results = []

    for inst in INSTRUMENTS:
        name     = inst["name"]
        symbols  = inst["symbols"]
        filename = inst["filename"]
        lot_size = inst["lot_size"]
        note     = inst["note"]

        existing_from, existing_to, existing_rows = check_existing(filename)
        if existing_rows > 0:
            console.print(f"[cyan]{name}[/cyan] (existing: {existing_rows} rows, "
                          f"{existing_from} → {existing_to})")
        else:
            console.print(f"[cyan]{name}[/cyan] (not cached yet)")

        try:
            df, sym_used = fetch_with_fallback(name, symbols, START_DATE, END_DATE)
            save_instrument(df, filename)
            console.print(f"  [green]✓ {name}: {len(df):,} rows saved "
                          f"({str(df.index[0].date())} → {str(df.index[-1].date())}) "
                          f"via {sym_used}[/green]")
            results.append({
                "name":       name,
                "lot_size":   lot_size,
                "symbol_used":sym_used,
                "rows":       len(df),
                "from":       str(df.index[0].date()),
                "to":         str(df.index[-1].date()),
                "status":     "OK",
                "note":       note,
            })
        except RuntimeError as e:
            console.print(f"  [red]✗ FAILED: {e}[/red]")
            # Keep existing data if available
            if existing_rows > 0:
                console.print(f"  [yellow]  Using existing cache ({existing_rows} rows)[/yellow]")
            results.append({
                "name":       name,
                "lot_size":   lot_size,
                "symbol_used":"FAILED",
                "rows":       existing_rows,
                "from":       existing_from or "N/A",
                "to":         existing_to   or "N/A",
                "status":     "FAILED" if existing_rows == 0 else "CACHED",
                "note":       note,
            })
        console.print()

    console.print()
    print_summary(results)

    ok_count   = sum(1 for r in results if r["status"] == "OK")
    fail_count = sum(1 for r in results if r["status"] == "FAILED")
    total      = len(results)

    console.print(f"\n[bold]Result:[/bold] {ok_count}/{total} instruments downloaded successfully.")
    if fail_count > 0:
        console.print(f"[yellow]{fail_count} instrument(s) failed — backtest will use available data.[/yellow]")

    console.print("\n[dim]Notes:[/dim]")
    console.print("[dim]  • INDIAVIX volume column will be 0 — expected from Yahoo Finance[/dim]")
    console.print("[dim]  • MIDCPNIFTY/FINNIFTY are spot index proxies (not futures prices)[/dim]")
    console.print("[dim]  • Futures lot sizes are applied at position-sizing time, not from data[/dim]")
    console.print("[dim]  • Run v6_backtest.py next to run all strategies[/dim]")


if __name__ == "__main__":
    main()
