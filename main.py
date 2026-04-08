"""
NIFTY/BANKNIFTY Autonomous Trading System
==========================================
Unified CLI entry point.

Usage:
  python main.py --mode fetch-historical
  python main.py --mode backtest [--strategy ORB|EMA|SUPERTREND|MR|ALL] [--instrument NIFTY|BANKNIFTY|ALL]
  python main.py --mode optimize [--strategy ORB|EMA|SUPERTREND] [--instrument NIFTY|BANKNIFTY|ALL]
  python main.py --mode paper      (paper trading — no real orders)
  python main.py --mode live       (live trading — REAL ORDERS PLACED)
  python main.py --mode v6-paper   (Williams%R v6 paper trading daemon — no real orders)
  python main.py --mode test       (run unit tests)
"""
import argparse
import logging
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        RichHandler(rich_tracebacks=True, show_path=False),
        logging.FileHandler(
            Path(__file__).parent / "logs" / f"system_{datetime.now().strftime('%Y%m%d')}.log"
        ),
    ],
)
# Suppress verbose libraries
logging.getLogger("kiteconnect").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
console = Console()

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

# Ensure logs/reports dirs exist
Path("logs").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)


# ------------------------------------------------------------------
# Mode: fetch-historical
# ------------------------------------------------------------------
def run_fetch_historical(instruments: list, interval: str, days: int):
    from config import credentials as creds
    creds.validate()
    from data.fetcher import KiteAuth, KiteDataFetcher
    from data.store import OHLCVStore

    console.print(f"[cyan]Fetching {days} days of {interval} data for {instruments}[/cyan]")

    kite = KiteAuth().login()
    fetcher = KiteDataFetcher(kite)
    store = OHLCVStore()

    to_date = datetime.now()
    from_date = to_date - timedelta(days=days)

    for inst in instruments:
        console.print(f"[cyan]Downloading {inst} ({interval})...[/cyan]")
        df = fetcher.get_historical(inst, from_date, to_date, interval)
        if df.empty:
            console.print(f"[red]No data returned for {inst}[/red]")
            continue
        store.save(inst, interval, df)
        store.save_parquet(inst, interval)
        min_dt, max_dt = store.get_date_range(inst, interval)
        console.print(
            f"[green]{inst}: {len(df):,} candles stored "
            f"({min_dt.date() if min_dt else '?'} → {max_dt.date() if max_dt else '?'})[/green]"
        )


# ------------------------------------------------------------------
# Mode: backtest
# ------------------------------------------------------------------
STRATEGY_CLASSES = {
    "ORB": ("strategies.orb", "ORBStrategy"),
    "EMA": ("strategies.ema_crossover", "EMACrossoverStrategy"),
    "SUPERTREND": ("strategies.supertrend", "SuperTrendStrategy"),
    "MR": ("strategies.mean_reversion", "MeanReversionStrategy"),
}

STRATEGY_INTERVALS = {
    "ORB": "5minute",
    "EMA": "5minute",
    "SUPERTREND": "15minute",
    "MR": "15minute",
}


def load_strategy_cls(name: str):
    import importlib
    module_path, cls_name = STRATEGY_CLASSES[name]
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


def run_backtest(strategies: list, instruments: list, days_train: int, days_validation: int):
    from backtest.engine import run_backtest as bt_run
    from backtest.report import generate_report, compare_strategies
    from data.store import OHLCVStore

    store = OHLCVStore()
    all_results = []

    to_date = datetime.now() - timedelta(days=days_validation)
    from_date = to_date - timedelta(days=days_train)

    console.print(
        f"[cyan]Backtesting period: {from_date.date()} → {to_date.date()} "
        f"| OOS validation: last {days_validation} days[/cyan]"
    )

    for strategy_name in strategies:
        cls = load_strategy_cls(strategy_name)
        interval = STRATEGY_INTERVALS[strategy_name]
        for instrument in instruments:
            try:
                result = bt_run(
                    strategy_cls=cls,
                    instrument=instrument,
                    interval=interval,
                    from_date=from_date,
                    to_date=to_date,
                    store=store,
                )
                generate_report(result)
                all_results.append(result)
            except Exception as e:
                logger.error(f"Backtest failed for {strategy_name}/{instrument}: {e}")

    if len(all_results) > 1:
        compare_strategies(all_results)

    return all_results


# ------------------------------------------------------------------
# Mode: optimize
# ------------------------------------------------------------------
def run_optimize(strategies: list, instruments: list, days: int):
    from backtest.optimizer import run_all_optimizations
    import json

    to_date = datetime.now()
    from_date = to_date - timedelta(days=days)

    for instrument in instruments:
        console.print(f"[cyan]Optimizing {instrument}...[/cyan]")
        results = run_all_optimizations(instrument, "5minute", from_date, to_date)
        console.print(f"[green]Optimal params for {instrument}:[/green]")
        console.print(json.dumps(results, indent=2))


# ------------------------------------------------------------------
# Mode: test
# ------------------------------------------------------------------
def run_tests():
    import unittest
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir="tests", pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


# ------------------------------------------------------------------
# Mode: paper / live
# ------------------------------------------------------------------
def run_trading(mode: str):
    if mode == "live":
        console.print(
            "[bold red]WARNING: LIVE MODE — REAL ORDERS WILL BE PLACED ON YOUR ZERODHA ACCOUNT[/bold red]"
        )
        confirm = input("Type 'YES I UNDERSTAND' to continue: ")
        if confirm != "YES I UNDERSTAND":
            console.print("[yellow]Aborted.[/yellow]")
            return

    import os
    os.environ["TRADING_MODE"] = mode
    from scheduler.runner import run_live
    console.print(f"[cyan]Starting {mode.upper()} trading system...[/cyan]")
    run_live()


# ------------------------------------------------------------------
# Mode: v6-paper  (Williams%R v6 strategy, paper trading daemon)
# ------------------------------------------------------------------
def run_v6_paper():
    """
    Start the Williams%R v6 paper trading daemon.
    Runs the daily scheduler: login@08:55, entry@09:16,
    exit checks@12:00/14:00/15:25, signals@15:25, EOD@15:30.
    No real orders placed.
    """
    import os
    os.environ["TRADING_MODE"] = "paper"
    from scheduler.v6_paper_runner import run_v6_paper as _run
    _run()


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="NIFTY/BANKNIFTY Autonomous Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["fetch-historical", "backtest", "optimize", "paper", "live", "v6-paper", "test"],
        required=True,
        help="Operating mode",
    )
    parser.add_argument(
        "--strategy",
        choices=["ORB", "EMA", "SUPERTREND", "MR", "ALL"],
        default="ALL",
        help="Strategy to backtest/optimize (default: ALL)",
    )
    parser.add_argument(
        "--instrument",
        choices=["NIFTY", "BANKNIFTY", "ALL"],
        default="ALL",
        help="Instrument (default: ALL)",
    )
    parser.add_argument(
        "--interval",
        default="5minute",
        help="Candle interval for historical fetch (default: 5minute)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=1095,
        help="Days of historical data to fetch/backtest (default: 1095 = 3 years)",
    )
    parser.add_argument(
        "--validation-days",
        type=int,
        default=180,
        help="Days to hold out for out-of-sample validation (default: 180)",
    )

    args = parser.parse_args()

    strategies = list(STRATEGY_CLASSES.keys()) if args.strategy == "ALL" else [args.strategy]
    instruments = ["NIFTY", "BANKNIFTY"] if args.instrument == "ALL" else [args.instrument]

    if args.mode == "fetch-historical":
        run_fetch_historical(instruments, args.interval, args.days)

    elif args.mode == "backtest":
        run_backtest(strategies, instruments, args.days - args.validation_days, args.validation_days)

    elif args.mode == "optimize":
        run_optimize(strategies, instruments, args.days)

    elif args.mode == "test":
        sys.exit(run_tests())

    elif args.mode in ("paper", "live"):
        run_trading(args.mode)

    elif args.mode == "v6-paper":
        run_v6_paper()


if __name__ == "__main__":
    main()
