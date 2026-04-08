"""
run_v6_kite_backtest.py
=======================
Re-runs the v6 Williams%R backtest using Zerodha Kite Connect data
instead of Yahoo Finance data.

Steps:
  1. Loads Kite daily OHLCV from OHLCVStore (populated by fetch_kite_daily.py)
  2. Exports each instrument to data/historical/{INST}_daily_extended.csv
     in the exact format v6_backtest.load_data() expects
  3. Runs v6_backtest.main() which produces a full HTML report

The original Yahoo Finance CSVs are backed up to data/historical/yahoo_backup/
so they can be restored if needed.

Usage:
  python run_v6_kite_backtest.py [--no-backup]

  --no-backup   Skip CSV backup step (faster; use if already backed up)

Prerequisites:
  Run fetch_kite_daily.py first.
"""

import sys, os, warnings, argparse, shutil
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
import pandas as pd
from rich.console import Console
from rich import box
from rich.table import Table

from data.store import OHLCVStore

console = Console()

DATA_DIR    = Path("data/historical")
BACKUP_DIR  = DATA_DIR / "yahoo_backup"

# All instruments v6_backtest needs (must match load_data() filenames)
V6_INSTRUMENTS = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "INDIAVIX"]


def backup_yahoo_csvs():
    """Back up existing Yahoo Finance CSVs before overwriting with Kite data."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backed_up = []
    for inst in V6_INSTRUMENTS:
        src = DATA_DIR / f"{inst}_daily_extended.csv"
        if src.exists():
            dst = BACKUP_DIR / f"{inst}_daily_extended.csv"
            shutil.copy2(src, dst)
            backed_up.append(inst)
    if backed_up:
        console.print(f"[dim]Yahoo CSV backup → {BACKUP_DIR} ({', '.join(backed_up)})[/dim]")
    return backed_up


def export_kite_to_csv(store: OHLCVStore) -> list:
    """
    Load Kite data from OHLCVStore and write to the CSV format v6_backtest expects:
      - Index name: "date"
      - Columns: open, high, low, close, volume (lowercase)
      - Timezone-naive DatetimeIndex
    Returns list of successfully exported instruments.
    """
    exported = []
    failed   = []

    for inst in V6_INSTRUMENTS:
        df = store.load_parquet(inst, "day")
        if df.empty:
            console.print(f"  [red]✗ {inst}: no Kite data — run fetch_kite_daily.py first[/red]")
            failed.append(inst)
            continue

        # Normalise: strip timezone, set date as index
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            if hasattr(df["date"].dt, "tz") and df["date"].dt.tz is not None:
                df["date"] = df["date"].dt.tz_localize(None)
            df = df.set_index("date")
        else:
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

        df.index.name = "date"
        df.index = df.index.normalize()

        # Ensure lowercase columns
        df.columns = [c.lower() for c in df.columns]

        # Ensure all required columns exist
        for col in ["open", "high", "low", "close"]:
            if col not in df.columns:
                console.print(f"  [red]✗ {inst}: missing column '{col}'[/red]")
                failed.append(inst)
                break
        else:
            if "volume" not in df.columns:
                df["volume"] = 0.0
            df["volume"] = df["volume"].fillna(0.0)

            # Drop weekends (safety)
            df = df[df.index.dayofweek < 5]
            df = df.sort_index()

            out_path = DATA_DIR / f"{inst}_daily_extended.csv"
            df.to_csv(out_path)
            console.print(
                f"  [green]✓ {inst}: {len(df):,} rows → {out_path.name} "
                f"({df.index.min().date()} → {df.index.max().date()})[/green]"
            )
            exported.append(inst)

    return exported, failed


def main():
    parser = argparse.ArgumentParser(description="Run v6 backtest using Kite data")
    parser.add_argument("--no-backup", action="store_true",
                        help="Skip backing up Yahoo Finance CSVs")
    args = parser.parse_args()

    console.rule("[bold cyan]v6 Backtest — Zerodha Kite Data[/bold cyan]")

    store = OHLCVStore()

    # ── Step 1: Backup Yahoo CSVs ──────────────────────────────────────────
    if not args.no_backup:
        console.print("\n[cyan]Step 1: Backing up Yahoo Finance CSVs...[/cyan]")
        backed_up = backup_yahoo_csvs()
        if not backed_up:
            console.print("[yellow]  No Yahoo CSVs found to back up (may not have been fetched yet)[/yellow]")
    else:
        console.print("\n[dim]Skipping Yahoo CSV backup (--no-backup)[/dim]")

    # ── Step 2: Export Kite data to CSV format ─────────────────────────────
    console.print("\n[cyan]Step 2: Exporting Kite data to CSV format v6 expects...[/cyan]")
    exported, failed = export_kite_to_csv(store)

    if failed:
        console.print(f"\n[bold red]Cannot proceed — missing data for: {', '.join(failed)}[/bold red]")
        console.print("[yellow]Run fetch_kite_daily.py first, then re-run this script.[/yellow]")
        sys.exit(1)

    # ── Step 3: Run v6 backtest ────────────────────────────────────────────
    console.print("\n[cyan]Step 3: Running v6 backtest with Kite data...[/cyan]")
    console.print("[dim](This may take 30-60 seconds)[/dim]\n")

    try:
        import v6_backtest
        v6_backtest.main()
    except Exception as e:
        console.print(f"\n[bold red]Backtest failed: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ── Step 4: Restore Yahoo CSVs ─────────────────────────────────────────
    if not args.no_backup and BACKUP_DIR.exists():
        console.print("\n[cyan]Step 4: Restoring Yahoo Finance CSVs from backup...[/cyan]")
        for inst in V6_INSTRUMENTS:
            backup = BACKUP_DIR / f"{inst}_daily_extended.csv"
            if backup.exists():
                shutil.copy2(backup, DATA_DIR / f"{inst}_daily_extended.csv")
                console.print(f"  [dim]Restored {inst}_daily_extended.csv[/dim]")
        console.print("[green]Yahoo CSVs restored — original data preserved[/green]")
    else:
        console.print("\n[dim]Note: CSVs now contain Kite data (no backup to restore from)[/dim]")

    console.print("\n[bold green]✓ Kite-based backtest complete.[/bold green]")
    console.print("[dim]Check reports/ for the generated HTML report.[/dim]")
    console.print("\n[dim]Compare results:[/dim]")
    console.print("[dim]  • Yahoo-based: run python v6_backtest.py (uses original CSVs)[/dim]")
    console.print("[dim]  • Kite-based:  run python run_v6_kite_backtest.py[/dim]")
    console.print("[dim]  • Expect annualized returns within ±5% of each other[/dim]")


if __name__ == "__main__":
    main()
