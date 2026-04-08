"""
scheduler/v6_paper_runner.py
=============================
Daily paper trading daemon for the Williams%R v6 strategy.

Daily schedule (IST):
  08:55 — Login to Zerodha, refresh access token
  09:16 — Execute pending signals from previous evening (morning entry)
  12:00 — Check exit conditions for open positions (stop / WR mid-exit)
  14:00 — Check exit conditions again
  15:25 — Final exit check + compute end-of-day signals for tomorrow
  15:30 — End-of-day summary: print P&L, append to equity log

Strategy logic (daily timeframe):
  Signals fire at close (15:25) → orders execute at next day open (09:16)
  This is identical to the v6 backtest mechanics (no lookahead bias).

State files (in logs/):
  v6_paper_pending.json   — signals queued for next morning entry
  v6_paper_positions.json — currently open paper positions (persists between runs)
  v6_paper_equity.csv     — daily running P&L log

No real orders are ever placed. PaperBroker simulates everything locally.
"""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

from config import credentials as creds
from config.settings import INSTRUMENTS as KITE_INSTRUMENTS
from data.fetcher import KiteAuth, KiteDataFetcher
from data.store import OHLCVStore
from execution.broker import PaperBroker
from execution.tracker import TradeTracker
from strategies.williams_r_v6 import WilliamsRSignalGenerator, V6_INSTRUMENTS, LOT_SIZES
from v6_backtest import RISK_PER_TRADE, CAPITAL as V6_CAPITAL

logger = logging.getLogger(__name__)
console = Console()

LOGS_DIR   = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

PENDING_FILE   = LOGS_DIR / "v6_paper_pending.json"
POSITIONS_FILE = LOGS_DIR / "v6_paper_positions.json"
EQUITY_FILE    = LOGS_DIR / "v6_paper_equity.csv"

# Days of daily history to load when computing signals (enough for EMA200 warm-up)
HISTORY_DAYS = 400


class V6PaperRunner:
    """
    Orchestrates the daily paper trading cycle for the Williams%R v6 strategy.

    All methods are designed to be called by the scheduler at fixed times.
    State (open positions, pending signals) persists across restarts via JSON files.
    """

    def __init__(self):
        self.kite_auth = KiteAuth()
        self.kite      = None
        self.fetcher   = None
        self.store     = OHLCVStore()
        self.broker    = PaperBroker()
        self.tracker   = TradeTracker()
        self.signal_gen = WilliamsRSignalGenerator()
        self._logged_in = False

    # ──────────────────────────────────────────────────────────────────────
    # LOGIN
    # ──────────────────────────────────────────────────────────────────────
    def login(self):
        """Login to Zerodha and refresh access token. Called at 08:55 IST."""
        logger.info("[v6-paper] Logging in to Zerodha...")
        try:
            self.kite     = self.kite_auth.login()
            self.fetcher  = KiteDataFetcher(self.kite)
            self._logged_in = True
            console.print("[bold green][v6-paper] Zerodha login successful[/bold green]")
        except Exception as e:
            logger.error(f"[v6-paper] Login failed: {e}")
            console.print(f"[bold red][v6-paper] Login failed: {e}[/bold red]")

    # ──────────────────────────────────────────────────────────────────────
    # DATA FETCH
    # ──────────────────────────────────────────────────────────────────────
    def _fetch_latest_daily(self) -> dict:
        """
        Fetch the latest N days of daily OHLCV from Kite for all 4 instruments + INDIAVIX.
        Falls back to OHLCVStore cache if Kite is unavailable.

        Returns:
            {instrument: pd.DataFrame} — OHLCV DataFrames with DatetimeIndex
        """
        data = {}
        to_date   = datetime.now()
        from_date = to_date - timedelta(days=HISTORY_DAYS)

        for inst in V6_INSTRUMENTS + ["INDIAVIX"]:
            try:
                if self._logged_in and self.fetcher:
                    df = self.fetcher.get_historical(
                        instrument=inst,
                        from_date=from_date,
                        to_date=to_date,
                        interval="day",
                    )
                    if not df.empty:
                        # Also update the local cache
                        self.store.save(inst, "day", df)
                        data[inst] = df
                        continue
            except Exception as e:
                logger.warning(f"[v6-paper] Kite fetch failed for {inst}: {e} — using cached data")

            # Fallback: load from OHLCVStore cache
            df = self.store.load(inst, "day", from_date=from_date)
            if not df.empty:
                data[inst] = df
            else:
                logger.warning(f"[v6-paper] No data available for {inst}")

        return data

    # ──────────────────────────────────────────────────────────────────────
    # END-OF-DAY SIGNAL CHECK  (called at 15:25 IST)
    # ──────────────────────────────────────────────────────────────────────
    def run_eod_signal_check(self):
        """
        Compute Williams%R signals on today's close.
        Signals are queued in logs/v6_paper_pending.json for tomorrow's entry.
        Called at 15:25 IST (after market close, before 15:30 cleanup).
        """
        console.print("\n[bold cyan][v6-paper] 15:25 IST — End-of-day signal check[/bold cyan]")

        data_dict = self._fetch_latest_daily()
        if not data_dict:
            console.print("[red][v6-paper] No data available — skipping signal check[/red]")
            return

        signals = self.signal_gen.get_signals(data_dict)

        # Save pending signals to JSON
        pending = {
            "date":    datetime.now().strftime("%Y-%m-%d"),
            "signals": signals,
        }
        with open(PENDING_FILE, "w") as f:
            json.dump(pending, f, indent=2)

        # Log to TradeTracker
        for sig in signals:
            self.tracker.record_signal(
                instrument=sig["instrument"],
                strategy="WR_v6",
                signal=sig["direction"],
            )

        # Console output
        if signals:
            console.print(f"[bold green][v6-paper] {len(signals)} signal(s) generated:[/bold green]")
            for sig in signals:
                direction = "LONG" if sig["direction"] == "L" else "SHORT"
                console.print(
                    f"  [cyan]{sig['instrument']}[/cyan] {direction} "
                    f"| close=₹{sig['entry_close']:,.0f} "
                    f"| stop_dist=₹{sig['stop_distance']:.0f} "
                    f"| tag={sig['tag']}"
                )
            console.print(f"[dim]Signals saved to {PENDING_FILE}[/dim]")
            console.print("[dim]Entry will execute at 09:16 IST tomorrow[/dim]")
        else:
            console.print("[dim][v6-paper] No signals today[/dim]")

    # ──────────────────────────────────────────────────────────────────────
    # MORNING ENTRY  (called at 09:16 IST)
    # ──────────────────────────────────────────────────────────────────────
    def run_morning_entry(self):
        """
        Execute pending signals from the previous evening.
        Uses live LTP at 09:16 as the entry price (open price proxy).
        Called at 09:16 IST — 1 minute after market open.
        """
        console.print("\n[bold cyan][v6-paper] 09:16 IST — Morning entry execution[/bold cyan]")

        if not PENDING_FILE.exists():
            console.print("[dim][v6-paper] No pending signals file found[/dim]")
            return

        with open(PENDING_FILE) as f:
            pending = json.load(f)

        signals      = pending.get("signals", [])
        signal_date  = pending.get("date", "")

        # Check if signals are from yesterday (not stale)
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        today     = datetime.now().strftime("%Y-%m-%d")
        if signal_date not in (yesterday, today):
            console.print(f"[yellow][v6-paper] Stale signals from {signal_date} — skipping[/yellow]")
            PENDING_FILE.unlink(missing_ok=True)
            return

        if not signals:
            console.print("[dim][v6-paper] No signals to execute[/dim]")
            PENDING_FILE.unlink(missing_ok=True)
            return

        positions = self._load_positions()

        for sig in signals:
            inst = sig["instrument"]

            # Skip if already have an open position in this instrument
            if inst in positions:
                console.print(f"[yellow][v6-paper] Already in {inst} — skipping new signal[/yellow]")
                continue

            # Get entry price: live LTP at 09:16 as open price proxy
            entry_price = self._get_entry_price(inst)
            if entry_price <= 0:
                console.print(f"[red][v6-paper] Could not get LTP for {inst} — skipping[/red]")
                continue

            direction     = sig["direction"]   # "L" or "S"
            stop_distance = sig["stop_distance"]

            if direction == "L":
                stop_price = entry_price - stop_distance
            else:
                stop_price = entry_price + stop_distance

            # Compute position size (lot-aligned, risk-based)
            qty = self._compute_quantity(inst, entry_price, stop_price, positions)
            if qty == 0:
                console.print(f"[yellow][v6-paper] Insufficient risk budget for {inst} — skipping[/yellow]")
                continue

            # Place paper order (simulation only)
            result = self.broker.place_order(
                instrument=inst,
                direction=direction,
                entry_price=entry_price,
                stop_price=stop_price,
                quantity=qty,
                tag=sig["tag"],
            )

            # Record entry
            self.tracker.record_entry(
                instrument=inst,
                strategy="WR_v6",
                direction="long" if direction == "L" else "short",
                entry_price=entry_price,
                quantity=qty,
                order_id=result.order_id,
            )

            # Persist position
            positions[inst] = {
                "direction":   direction,
                "entry_price": entry_price,
                "stop_price":  stop_price,
                "quantity":    qty,
                "tag":         sig["tag"],
                "entry_date":  datetime.now().strftime("%Y-%m-%d"),
                "order_id":    result.order_id,
            }
            console.print(
                f"[green][v6-paper] ENTERED {inst} {'LONG' if direction == 'L' else 'SHORT'} "
                f"@ ₹{entry_price:,.0f} | qty={qty} | stop=₹{stop_price:,.0f}[/green]"
            )

        self._save_positions(positions)
        # Clear pending signals after execution
        PENDING_FILE.unlink(missing_ok=True)

    # ──────────────────────────────────────────────────────────────────────
    # EXIT CHECKS  (called at 12:00, 14:00, 15:25 IST)
    # ──────────────────────────────────────────────────────────────────────
    def check_exits(self):
        """
        Check stop-loss hits and WR mid-exit signals for all open positions.
        Called multiple times during the day (12:00, 14:00, 15:25).
        """
        positions = self._load_positions()
        if not positions:
            return

        console.print(f"\n[cyan][v6-paper] Exit check — {len(positions)} open position(s)[/cyan]")

        # Get latest prices for all held instruments
        ltps = {}
        for inst in list(positions.keys()):
            ltps[inst] = self._get_entry_price(inst)   # reuses LTP fetch

        # Check stop-loss hits
        for inst, pos in list(positions.items()):
            ltp = ltps.get(inst, 0)
            if ltp <= 0:
                continue

            direction  = pos["direction"]
            stop_price = pos["stop_price"]
            hit_stop   = False

            if direction == "L" and ltp <= stop_price:
                hit_stop = True
            elif direction == "S" and ltp >= stop_price:
                hit_stop = True

            if hit_stop:
                self.tracker.record_exit(inst, ltp, reason="stop_hit")
                del positions[inst]
                console.print(
                    f"[red][v6-paper] STOP HIT {inst} @ ₹{ltp:,.0f} "
                    f"(stop was ₹{stop_price:,.0f})[/red]"
                )

        # Check WR mid-exit signals (requires latest daily data)
        if positions:
            data_dict = self._fetch_latest_daily()
            exit_signals = self.signal_gen.get_exit_signals(data_dict, positions)
            for ex in exit_signals:
                inst = ex["instrument"]
                if inst not in positions:
                    continue
                ltp = ltps.get(inst, 0)
                exit_price = ltp if ltp > 0 else positions[inst]["entry_price"]
                self.tracker.record_exit(inst, exit_price, reason=ex["reason"])
                del positions[inst]
                console.print(
                    f"[yellow][v6-paper] WR MID-EXIT {inst} @ ₹{exit_price:,.0f}[/yellow]"
                )

        self._save_positions(positions)

    # ──────────────────────────────────────────────────────────────────────
    # END OF DAY  (called at 15:30 IST)
    # ──────────────────────────────────────────────────────────────────────
    def end_of_day(self):
        """
        Print daily P&L summary and append to equity log.
        Called at 15:30 IST.
        """
        console.print("\n[bold cyan][v6-paper] 15:30 IST — End of day summary[/bold cyan]")
        self.tracker.print_summary()

        positions = self._load_positions()
        daily_pnl = self.tracker.total_pnl

        # Append to equity log
        row = {
            "date":            datetime.now().strftime("%Y-%m-%d"),
            "daily_pnl":       round(daily_pnl, 2),
            "open_positions":  len(positions),
            "total_trades":    len(self.tracker.trades),
        }
        df_row = pd.DataFrame([row])

        if EQUITY_FILE.exists():
            df_row.to_csv(EQUITY_FILE, mode="a", header=False, index=False)
        else:
            df_row.to_csv(EQUITY_FILE, mode="w", header=True, index=False)

        console.print(f"[dim]Daily P&L appended to {EQUITY_FILE}[/dim]")
        if positions:
            console.print(f"[dim]Open positions carried forward: {list(positions.keys())}[/dim]")

    # ──────────────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────────────
    def _get_entry_price(self, instrument: str) -> float:
        """Get current LTP from Kite. Falls back to last known close from cache."""
        if self._logged_in and self.fetcher:
            try:
                return self.fetcher.get_ltp(instrument)
            except Exception as e:
                logger.warning(f"[v6-paper] LTP fetch failed for {instrument}: {e}")

        # Fallback: use last close from OHLCVStore
        df = self.store.load(instrument, "day")
        if not df.empty:
            return float(df["close"].iloc[-1])
        return 0.0

    def _compute_quantity(self, instrument: str, entry: float, stop: float,
                          open_positions: dict) -> int:
        """
        Compute lot-aligned quantity using same risk logic as v6 backtest:
          qty = (capital × 4% risk) / (stop_distance × lot_size)
        Caps at MAX_LOTS_PER_INST from v6.
        Returns 0 if budget is insufficient.
        """
        from v6_backtest import MAX_LOTS_PER_INST, MAX_TOTAL_RISK

        # Estimate current capital (V6_CAPITAL is the baseline)
        # In paper trading, we track notional capital via equity log
        equity = self._get_current_equity()
        if equity <= 0:
            equity = V6_CAPITAL

        # Compute available risk budget
        current_risk = sum(
            abs(p["entry_price"] - p["stop_price"]) * p["quantity"] / equity
            for p in open_positions.values()
        )
        avail_risk = min(RISK_PER_TRADE, MAX_TOTAL_RISK - current_risk)
        if avail_risk <= 0:
            return 0

        lot        = LOT_SIZES.get(instrument, 65)
        min_stop   = entry * 0.004    # 0.4% minimum stop distance
        risk_unit  = max(abs(entry - stop), min_stop)
        risk_amt   = equity * avail_risk
        lots       = int(risk_amt / (risk_unit * lot))

        if lots <= 0:
            return 0

        max_lots = MAX_LOTS_PER_INST.get(instrument, 20)
        lots     = min(lots, max_lots)
        return lots * lot

    def _get_current_equity(self) -> float:
        """Read cumulative P&L from equity log to estimate current paper capital."""
        if not EQUITY_FILE.exists():
            return V6_CAPITAL
        try:
            df = pd.read_csv(EQUITY_FILE)
            if df.empty:
                return V6_CAPITAL
            total_pnl = df["daily_pnl"].sum()
            return V6_CAPITAL + total_pnl
        except Exception:
            return V6_CAPITAL

    def _load_positions(self) -> dict:
        """Load open positions from JSON state file."""
        if not POSITIONS_FILE.exists():
            return {}
        try:
            with open(POSITIONS_FILE) as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_positions(self, positions: dict):
        """Persist open positions to JSON state file."""
        with open(POSITIONS_FILE, "w") as f:
            json.dump(positions, f, indent=2)

    def print_status(self):
        """Print current open positions and equity status. Safe to call anytime."""
        positions = self._load_positions()
        equity    = self._get_current_equity()

        console.print(f"\n[bold]v6 Paper Trading Status — {datetime.now().strftime('%Y-%m-%d %H:%M')}[/bold]")
        console.print(f"Paper Capital: ₹{equity:,.0f} (started at ₹{V6_CAPITAL:,.0f})")
        console.print(f"Open Positions: {len(positions)}")

        if positions:
            tbl = Table(box=box.SIMPLE, header_style="bold cyan")
            for col in ["Instrument", "Dir", "Entry ₹", "Stop ₹", "Qty", "Entry Date", "Tag"]:
                tbl.add_column(col)
            for inst, p in positions.items():
                tbl.add_row(
                    inst,
                    "LONG" if p["direction"] == "L" else "SHORT",
                    f"₹{p['entry_price']:,.0f}",
                    f"₹{p['stop_price']:,.0f}",
                    str(p["quantity"]),
                    p.get("entry_date", "?"),
                    p.get("tag", "?"),
                )
            console.print(tbl)

        if EQUITY_FILE.exists():
            try:
                df = pd.read_csv(EQUITY_FILE)
                total_pnl = df["daily_pnl"].sum()
                color     = "green" if total_pnl >= 0 else "red"
                console.print(
                    f"Cumulative Paper P&L: [{color}]₹{total_pnl:+,.0f}[/{color}] "
                    f"over {len(df)} trading day(s)"
                )
            except Exception:
                pass


# ──────────────────────────────────────────────────────────────────────────────
# Scheduler setup
# ──────────────────────────────────────────────────────────────────────────────
def setup_v6_schedule(runner: V6PaperRunner):
    """Register all daily jobs for v6 paper trading."""
    import schedule

    schedule.every().day.at("08:55").do(runner.login)
    schedule.every().day.at("09:16").do(runner.run_morning_entry)
    schedule.every().day.at("12:00").do(runner.check_exits)
    schedule.every().day.at("14:00").do(runner.check_exits)
    schedule.every().day.at("15:25").do(runner.run_eod_signal_check)
    schedule.every().day.at("15:25").do(runner.check_exits)   # Final exit check same time as signal
    schedule.every().day.at("15:30").do(runner.end_of_day)

    logger.info("[v6-paper] Daily schedule registered")
    console.print("[dim][v6-paper] Schedule: login@08:55, entry@09:16, "
                  "exits@12:00/14:00/15:25, signals@15:25, EOD@15:30[/dim]")


def run_v6_paper():
    """Entry point for the v6 paper trading daemon."""
    creds.validate()

    console.rule("[bold cyan]v6 Williams%R Paper Trading Daemon[/bold cyan]")
    console.print("[yellow][PAPER MODE] No real orders will be placed[/yellow]\n")

    runner = V6PaperRunner()
    setup_v6_schedule(runner)

    console.print("[cyan]Scheduler running. Waiting for market hours (IST)...[/cyan]")
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")

    import schedule
    while True:
        schedule.run_pending()
        time.sleep(1)
