"""
validate_data_sources.py
========================
Compares Zerodha Kite Connect daily OHLCV data against Yahoo Finance CSV data
to ensure data quality before running the Kite-based backtest.

Checks performed:
  - Row count comparison
  - Date coverage (common dates, gaps)
  - Close price correlation (must be >= 0.999)
  - Max absolute % divergence in close prices on common dates
  - Flagged dates where close diverges by > 0.5%

Only NIFTY and BANKNIFTY are compared (both sources have 7-year history).
FINNIFTY and MIDCPNIFTY are Kite-only — their date ranges are reported.

Usage:
  python validate_data_sources.py

Prerequisites:
  1. Run fetch_all_instruments.py first (Yahoo Finance CSVs)
  2. Run fetch_kite_daily.py first (Kite daily data)
"""

import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from rich.console import Console
from rich.table import Table
from rich import box

from data.store import OHLCVStore

console = Console()

DATA_DIR = Path("data/historical")
COMPARE_INSTRUMENTS = ["NIFTY", "BANKNIFTY"]
KITE_ONLY_INSTRUMENTS = ["FINNIFTY", "MIDCPNIFTY", "INDIAVIX"]
CORR_THRESHOLD = 0.999
DIVERGENCE_ALERT_PCT = 0.5   # Flag dates where close differs by more than this %


def load_yahoo_csv(instrument: str) -> pd.DataFrame:
    """Load Yahoo Finance CSV from data/historical/{INST}_daily_extended.csv."""
    path = DATA_DIR / f"{instrument}_daily_extended.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    df.columns = [c.lower() for c in df.columns]
    df = df[df.index.dayofweek < 5]
    df = df.dropna(subset=["close"])
    df.index = df.index.normalize()
    return df


def load_kite_store(store: OHLCVStore, instrument: str) -> pd.DataFrame:
    """Load Kite data from OHLCVStore, indexed by date."""
    df = store.load_parquet(instrument, "day")
    if df.empty:
        return pd.DataFrame()
    # Strip timezone if present (Kite returns tz-aware IST)
    if hasattr(df["date"].dtype, "tz") and df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df[df["date"].dt.dayofweek < 5]
    df = df.dropna(subset=["close"])
    df = df.set_index("date")
    return df


def compare_instrument(store: OHLCVStore, instrument: str) -> dict:
    """
    Run all comparison checks for a single instrument.
    Returns a result dict with all metrics.
    """
    yahoo = load_yahoo_csv(instrument)
    kite  = load_kite_store(store, instrument)

    result = {
        "instrument": instrument,
        "yahoo_rows": len(yahoo),
        "kite_rows":  len(kite),
        "yahoo_from": str(yahoo.index.min().date()) if not yahoo.empty else "N/A",
        "yahoo_to":   str(yahoo.index.max().date()) if not yahoo.empty else "N/A",
        "kite_from":  str(kite.index.min().date()) if not kite.empty else "N/A",
        "kite_to":    str(kite.index.max().date()) if not kite.empty else "N/A",
        "common_dates": 0,
        "yahoo_only": 0,
        "kite_only": 0,
        "correlation": None,
        "max_divergence_pct": None,
        "flagged_dates": [],
        "pass": False,
        "error": None,
    }

    if yahoo.empty:
        result["error"] = "Yahoo CSV not found — run fetch_all_instruments.py first"
        return result
    if kite.empty:
        result["error"] = "Kite data not found — run fetch_kite_daily.py first"
        return result

    # ── Date coverage ──────────────────────────────────────────────────────
    yahoo_dates = set(yahoo.index.normalize())
    kite_dates  = set(kite.index.normalize())
    common      = yahoo_dates & kite_dates

    result["common_dates"] = len(common)
    result["yahoo_only"]   = len(yahoo_dates - kite_dates)
    result["kite_only"]    = len(kite_dates - yahoo_dates)

    if not common:
        result["error"] = "No overlapping dates between Yahoo and Kite"
        return result

    # ── Align on common dates ──────────────────────────────────────────────
    common_sorted = sorted(common)
    y_close = yahoo.loc[yahoo.index.isin(common), "close"].sort_index()
    k_close = kite.loc[kite.index.isin(common), "close"].sort_index()

    # Reindex to exact same index to ensure alignment
    y_close = y_close.reindex(sorted(common))
    k_close = k_close.reindex(sorted(common))

    # Drop any remaining NaN (holidays etc.)
    mask = y_close.notna() & k_close.notna()
    y_close = y_close[mask]
    k_close = k_close[mask]

    if len(y_close) < 10:
        result["error"] = "Too few common non-null dates to compare"
        return result

    # ── Correlation ────────────────────────────────────────────────────────
    corr = float(y_close.corr(k_close))
    result["correlation"] = round(corr, 6)

    # ── Price divergence ───────────────────────────────────────────────────
    pct_diff = ((k_close - y_close).abs() / y_close.replace(0, np.nan) * 100)
    result["max_divergence_pct"] = round(float(pct_diff.max()), 4)

    # Flag dates with large divergence
    flagged = pct_diff[pct_diff > DIVERGENCE_ALERT_PCT]
    result["flagged_dates"] = [
        {"date": str(d.date()), "yahoo": round(float(y_close[d]), 2),
         "kite": round(float(k_close[d]), 2),
         "diff_pct": round(float(pct_diff[d]), 4)}
        for d in flagged.index
    ]

    result["pass"] = (corr >= CORR_THRESHOLD)
    return result


def report_kite_only(store: OHLCVStore, instrument: str):
    """Print date range for Kite-only instruments (no Yahoo comparison)."""
    df = store.load_parquet(instrument, "day")
    if df.empty:
        console.print(f"  [yellow]{instrument}: no Kite data — run fetch_kite_daily.py[/yellow]")
    else:
        if hasattr(df["date"].dtype, "tz") and df["date"].dt.tz is not None:
            df["date"] = df["date"].dt.tz_localize(None)
        console.print(
            f"  [green]{instrument}: {len(df):,} rows "
            f"({df['date'].min().date()} → {df['date'].max().date()})[/green]"
        )


def main():
    console.rule("[bold cyan]Data Source Validation: Kite vs Yahoo Finance[/bold cyan]")
    console.print(f"[dim]Comparing daily close prices | Correlation threshold: {CORR_THRESHOLD}[/dim]\n")

    store = OHLCVStore()

    # ── Compare NIFTY and BANKNIFTY ────────────────────────────────────────
    results = []
    for inst in COMPARE_INSTRUMENTS:
        console.print(f"[cyan]Comparing {inst}...[/cyan]")
        r = compare_instrument(store, inst)
        results.append(r)

        if r["error"]:
            console.print(f"  [red]ERROR: {r['error']}[/red]")
            continue

        color = "green" if r["pass"] else "red"
        console.print(
            f"  Correlation: [{color}]{r['correlation']:.6f}[/{color}]  "
            f"Max divergence: {r['max_divergence_pct']:.4f}%  "
            f"Common dates: {r['common_dates']:,}"
        )

        if r["flagged_dates"]:
            console.print(f"  [yellow]⚠ {len(r['flagged_dates'])} dates with >{DIVERGENCE_ALERT_PCT}% divergence:[/yellow]")
            for fd in r["flagged_dates"][:10]:  # Show first 10
                console.print(
                    f"    {fd['date']}  Yahoo={fd['yahoo']}  Kite={fd['kite']}  "
                    f"Diff={fd['diff_pct']:.3f}%"
                )
            if len(r["flagged_dates"]) > 10:
                console.print(f"    ... and {len(r['flagged_dates']) - 10} more")

    # ── Summary table ──────────────────────────────────────────────────────
    console.print()
    tbl = Table(
        title="Comparison Summary",
        box=box.DOUBLE_EDGE, header_style="bold cyan",
    )
    for col in ["Instrument", "Yahoo Rows", "Kite Rows", "Common", "Yahoo Only",
                "Kite Only", "Correlation", "Max Diff%", "Result"]:
        tbl.add_column(col, justify="right" if col not in ("Instrument", "Result") else "left",
                       min_width=9)

    for r in results:
        if r["error"]:
            tbl.add_row(r["instrument"], str(r["yahoo_rows"]), str(r["kite_rows"]),
                        "—", "—", "—", "—", "—", f"[red]ERROR[/red]")
            continue
        ok    = r["pass"]
        color = "green" if ok else "red"
        tbl.add_row(
            r["instrument"],
            f"{r['yahoo_rows']:,}",
            f"{r['kite_rows']:,}",
            f"{r['common_dates']:,}",
            str(r["yahoo_only"]),
            str(r["kite_only"]),
            f"{r['correlation']:.6f}",
            f"{r['max_divergence_pct']:.4f}%",
            f"[{color}]{'PASS' if ok else 'FAIL'}[/{color}]",
        )

    console.print(tbl)

    # ── Kite-only instruments ──────────────────────────────────────────────
    console.print("\n[bold]Kite-only instruments (no Yahoo comparison):[/bold]")
    for inst in KITE_ONLY_INSTRUMENTS:
        report_kite_only(store, inst)

    # ── Final verdict ──────────────────────────────────────────────────────
    passed  = [r for r in results if r.get("pass")]
    failed  = [r for r in results if r.get("error") or not r.get("pass")]
    console.print()

    if all(r.get("pass") for r in results):
        console.print("[bold green]✓ All data sources pass quality checks.[/bold green]")
        console.print("[dim]Next: python run_v6_kite_backtest.py[/dim]")
    else:
        console.print("[bold red]✗ Some data sources failed quality checks.[/bold red]")
        console.print("[yellow]Common causes:[/yellow]")
        console.print("[dim]  • Low correlation: instrument token mismatch in config/settings.py[/dim]")
        console.print("[dim]  • Missing Kite data: re-run fetch_kite_daily.py[/dim]")
        console.print("[dim]  • Missing Yahoo data: re-run fetch_all_instruments.py[/dim]")


if __name__ == "__main__":
    main()
