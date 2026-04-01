"""
Live P&L tracker and trade logger.
Records every signal and order to CSV and console.
"""
import csv
import logging
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich import box

logger = logging.getLogger(__name__)
console = Console()

LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

TRADE_FIELDS = [
    "timestamp", "instrument", "strategy", "direction",
    "entry_price", "exit_price", "quantity", "pnl", "reason",
]


class TradeTracker:
    """
    Logs all signals, entries, and exits for live and paper trading.
    Writes to a daily CSV file for post-market review.
    """

    def __init__(self):
        today = datetime.now().strftime("%Y%m%d")
        self._csv_path = LOGS_DIR / f"trades_{today}.csv"
        self._trades: list = []
        self._open_trades: dict = {}   # instrument → trade dict
        self._total_pnl: float = 0.0
        self._ensure_csv_header()

    def _ensure_csv_header(self):
        if not self._csv_path.exists():
            with open(self._csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=TRADE_FIELDS)
                writer.writeheader()

    def _append_csv(self, record: dict):
        with open(self._csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TRADE_FIELDS)
            writer.writerow({k: record.get(k, "") for k in TRADE_FIELDS})

    def record_entry(
        self,
        instrument: str,
        strategy: str,
        direction: str,
        entry_price: float,
        quantity: int,
        order_id: str = "",
    ):
        """Record a trade entry."""
        trade = {
            "timestamp": datetime.now().isoformat(),
            "instrument": instrument,
            "strategy": strategy,
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": None,
            "quantity": quantity,
            "order_id": order_id,
            "pnl": None,
            "reason": "entry",
        }
        self._open_trades[instrument] = trade
        console.print(
            f"[bold green]ENTRY[/bold green] {direction.upper()} {instrument} "
            f"@ ₹{entry_price:.2f} qty={quantity} [{strategy}]"
        )

    def record_exit(
        self,
        instrument: str,
        exit_price: float,
        reason: str = "signal",
    ):
        """Record a trade exit and compute P&L."""
        if instrument not in self._open_trades:
            logger.warning(f"No open trade found for {instrument}")
            return

        trade = self._open_trades.pop(instrument)
        qty = trade["quantity"]
        entry = trade["entry_price"]

        if trade["direction"] == "long":
            pnl = (exit_price - entry) * qty
        else:
            pnl = (entry - exit_price) * qty

        trade["exit_price"] = exit_price
        trade["pnl"] = round(pnl, 2)
        trade["reason"] = reason
        trade["timestamp"] = datetime.now().isoformat()

        self._trades.append(trade)
        self._total_pnl += pnl
        self._append_csv(trade)

        color = "green" if pnl >= 0 else "red"
        console.print(
            f"[bold {color}]EXIT[/bold {color}] {instrument} "
            f"@ ₹{exit_price:.2f} PnL=[{color}]₹{pnl:+.2f}[/{color}] ({reason})"
        )

    def record_signal(self, instrument: str, strategy: str, signal: str):
        """Log a signal that was generated (may or may not result in a trade)."""
        console.print(
            f"[yellow]SIGNAL[/yellow] {signal.upper()} {instrument} [{strategy}]"
        )

    def print_summary(self):
        """Print a rich table of today's trades and total P&L."""
        table = Table(title=f"Daily Trade Summary — {datetime.now().strftime('%Y-%m-%d')}", box=box.ROUNDED)
        for col in ["Time", "Instrument", "Strategy", "Direction", "Entry", "Exit", "Qty", "P&L"]:
            table.add_column(col, style="white")

        for t in self._trades:
            pnl = t.get("pnl", 0) or 0
            color = "green" if pnl >= 0 else "red"
            table.add_row(
                t.get("timestamp", "")[:19],
                t.get("instrument", ""),
                t.get("strategy", ""),
                t.get("direction", "").upper(),
                f"₹{t.get('entry_price', 0):.2f}",
                f"₹{t.get('exit_price', 0):.2f}" if t.get("exit_price") else "-",
                str(t.get("quantity", 0)),
                f"[{color}]₹{pnl:+.2f}[/{color}]",
            )

        console.print(table)
        color = "green" if self._total_pnl >= 0 else "red"
        console.print(
            f"\n[bold]Total P&L: [{color}]₹{self._total_pnl:+.2f}[/{color}][/bold]"
            f"  |  Trades: {len(self._trades)}"
            f"  |  Log: {self._csv_path}"
        )

    @property
    def total_pnl(self) -> float:
        return self._total_pnl

    @property
    def trades(self) -> list:
        return list(self._trades)
