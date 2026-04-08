"""
fetch_kite_daily.py
===================
Downloads 7-year daily OHLCV history for all v6 strategy instruments
from Zerodha Kite Connect API (requires paid ₹500/month subscription).

Instruments fetched:
  NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY, INDIAVIX

Data is saved to data/historical/ as SQLite (via OHLCVStore) + Parquet.

Usage:
  python fetch_kite_daily.py

Prerequisites:
  1. Copy .env.example → .env and fill in Zerodha credentials
  2. Ensure FINNIFTY and MIDCPNIFTY tokens in config/settings.py are correct
     (verify with: python -c "from data.fetcher import KiteAuth; k=KiteAuth().login();
      [print(i['instrument_token'], i['tradingsymbol']) for i in k.instruments('NSE')
       if 'FIN SERVICE' in i['name'] or 'MIDCAP SELECT' in i['name']]")
"""

import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich import box

Path("logs").mkdir(exist_ok=True)

from config import credentials as creds
from data.fetcher import KiteAuth, KiteDataFetcher
from data.store import OHLCVStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path("logs") / f"fetch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
    ],
)
# Keep noisy libraries quiet
logging.getLogger("kiteconnect").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
# Show fetcher debug logs to diagnose login flow
logging.getLogger("data.fetcher").setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)
console = Console()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
START_DATE  = datetime(2017, 1, 1)
END_DATE    = datetime.now()
INTERVAL    = "day"

# Instruments to fetch — keys must match config/settings.py INSTRUMENTS dict
INSTRUMENTS = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "INDIAVIX"]


def main():
    # ── Step 1: Validate credentials ──────────────────────────────────────
    logger.info("Validating credentials...")
    try:
        creds.validate()
        logger.info("Credentials OK")
    except EnvironmentError as e:
        console.print(f"[bold red]Credential error:[/bold red] {e}")
        sys.exit(1)

    # ── Step 2: Login to Zerodha ───────────────────────────────────────────
    console.rule("[bold cyan]Zerodha Kite Daily Data Fetch[/bold cyan]")
    console.print(f"[dim]Period: {START_DATE.date()} → {END_DATE.date()} | Interval: {INTERVAL}[/dim]\n")

    logger.info("Initiating Zerodha login (password + TOTP)...")
    console.print("[cyan]Logging in to Zerodha...[/cyan]")
    try:
        kite    = KiteAuth().login()
        fetcher = KiteDataFetcher(kite)
        logger.info("Login successful — access token acquired")
        console.print("[green]✓ Login successful[/green]\n")
    except Exception as e:
        logger.error(f"Login failed: {e}")
        console.print(f"[bold red]Login failed:[/bold red] {e}")
        console.print("[yellow]Tip: check ZERODHA_TOTP_SECRET in .env — it must be the base32 seed, not the 6-digit code[/yellow]")
        sys.exit(1)

    # ── Step 3: Initialise storage ─────────────────────────────────────────
    logger.info("Initialising local OHLCVStore (SQLite)...")
    store = OHLCVStore()
    logger.info("Storage ready")

    # ── Step 4: Fetch each instrument ─────────────────────────────────────
    total = len(INSTRUMENTS)
    results = []
    for idx, inst in enumerate(INSTRUMENTS, 1):
        logger.info(f"[{idx}/{total}] Starting fetch: {inst} | {START_DATE.date()} → {END_DATE.date()}")
        console.print(f"[cyan][{idx}/{total}] Downloading {inst} ({INTERVAL})...[/cyan]")
        try:
            df = fetcher.get_historical(
                instrument=inst,
                from_date=START_DATE,
                to_date=END_DATE,
                interval=INTERVAL,
            )

            if df.empty:
                logger.warning(f"{inst}: API returned empty DataFrame")
                console.print(f"  [red]✗ {inst}: no data returned[/red]")
                results.append({"name": inst, "rows": 0, "from": "—", "to": "—", "status": "EMPTY"})
                continue

            logger.info(f"{inst}: received {len(df):,} raw candles from API")

            # Strip timezone (Kite returns tz-aware IST) → tz-naive for v6 compatibility
            if hasattr(df["date"].dtype, "tz") and df["date"].dt.tz is not None:
                df["date"] = df["date"].dt.tz_localize(None)
                logger.info(f"{inst}: stripped IST timezone from timestamps")
            elif df["date"].dtype == "object":
                df["date"] = df["date"].dt.tz_localize(None) if hasattr(df["date"].iloc[0], "tzinfo") else df["date"]

            # Save to SQLite and export Parquet
            logger.info(f"{inst}: saving to SQLite...")
            new_rows = store.save(inst, INTERVAL, df)
            logger.info(f"{inst}: {new_rows} new rows inserted into SQLite")

            logger.info(f"{inst}: exporting to Parquet...")
            parquet_path = store.save_parquet(inst, INTERVAL)
            logger.info(f"{inst}: Parquet saved → {parquet_path}")

            min_dt, max_dt = store.get_date_range(inst, INTERVAL)
            console.print(
                f"  [green]✓ {inst}: {len(df):,} candles fetched, {new_rows} new rows saved "
                f"({min_dt.date() if min_dt else '?'} → {max_dt.date() if max_dt else '?'})[/green]"
            )
            logger.info(f"{inst}: DONE — stored range {min_dt.date() if min_dt else '?'} → {max_dt.date() if max_dt else '?'}")
            results.append({
                "name": inst,
                "rows": len(df),
                "new":  new_rows,
                "from": str(min_dt.date()) if min_dt else "?",
                "to":   str(max_dt.date()) if max_dt else "?",
                "status": "OK",
            })

        except KeyError as e:
            logger.error(f"{inst}: KeyError — instrument token missing in config: {e}")
            console.print(f"  [red]✗ {inst}: instrument not found in config — {e}[/red]")
            console.print(f"  [yellow]   Verify token in config/settings.py INSTRUMENTS['{inst}'][/yellow]")
            results.append({"name": inst, "rows": 0, "from": "—", "to": "—", "status": "MISSING_TOKEN"})
        except Exception as e:
            logger.error(f"{inst}: fetch failed — {e}", exc_info=True)
            console.print(f"  [red]✗ {inst}: fetch failed — {e}[/red]")
            results.append({"name": inst, "rows": 0, "from": "—", "to": "—", "status": "ERROR"})

    # ── Step 5: Print summary table ────────────────────────────────────────
    console.print()
    tbl = Table(
        title="Kite Daily Data Fetch Summary",
        box=box.DOUBLE_EDGE, header_style="bold cyan",
    )
    tbl.add_column("Instrument",  min_width=14)
    tbl.add_column("Rows",        justify="right", min_width=8)
    tbl.add_column("New Rows",    justify="right", min_width=10)
    tbl.add_column("From",        min_width=12)
    tbl.add_column("To",          min_width=12)
    tbl.add_column("Status",      min_width=8)

    for r in results:
        ok    = r["status"] == "OK"
        color = "green" if ok else "red"
        tbl.add_row(
            r["name"],
            str(r.get("rows", 0)),
            str(r.get("new", "—")),
            r.get("from", "—"),
            r.get("to",   "—"),
            f"[{color}]{r['status']}[/{color}]",
        )

    console.print(tbl)

    ok_count = sum(1 for r in results if r["status"] == "OK")
    logger.info(f"Fetch complete: {ok_count}/{len(results)} instruments successful")
    console.print(f"\n[bold]{ok_count}/{len(results)} instruments fetched successfully.[/bold]")

    if ok_count < len(INSTRUMENTS):
        console.print("\n[yellow]To fix MISSING_TOKEN errors:[/yellow]")
        console.print("[dim]Run this snippet after login to find correct tokens:[/dim]")
        console.print("""[dim]
  python3 -c "
  from data.fetcher import KiteAuth
  kite = KiteAuth().login()
  insts = kite.instruments('NSE')
  for i in insts:
      n = i.get('name','')
      if any(x in n for x in ['FIN SERVICE','MIDCAP SELECT','INDIA VIX']):
          print(i['instrument_token'], i['tradingsymbol'], n)
  "[/dim]""")

    console.print("\n[dim]Next steps:[/dim]")
    console.print("[dim]  1. python validate_data_sources.py  ← compare Kite vs Yahoo Finance data[/dim]")
    console.print("[dim]  2. python run_v6_kite_backtest.py   ← re-run v6 backtest with Kite data[/dim]")


if __name__ == "__main__":
    main()
