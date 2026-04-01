"""
Backtest report generator.
Computes Sharpe, Calmar, win rate, and outputs to JSON + rich console table.
"""
import json
import logging
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich import box

logger = logging.getLogger(__name__)
console = Console()

REPORTS_DIR = Path(__file__).parent.parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def generate_report(backtest_result: dict, save: bool = True) -> dict:
    """
    Generate a human-readable backtest report and save to JSON.

    Args:
        backtest_result: Output dict from backtest/engine.py run_backtest()
        save: Whether to save report to disk

    Returns:
        The report dict (same as backtest_result, enriched with file path)
    """
    r = backtest_result

    # Print rich table to console
    table = Table(title=f"Backtest Report — {r['strategy']} | {r['instrument']}", box=box.ROUNDED)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="bold white")

    table.add_row("Instrument", r["instrument"])
    table.add_row("Strategy", r["strategy"])
    table.add_row("Interval", r["interval"])
    table.add_row("Period", f"{r['from_date']} → {r['to_date']}")
    table.add_row("Capital", f"₹{r['capital']:,.0f}")
    table.add_row("Final Value", f"₹{r['final_value']:,.2f}")

    returns_color = "green" if r["returns_pct"] >= 0 else "red"
    table.add_row("Total Return", f"[{returns_color}]{r['returns_pct']:+.2f}%[/{returns_color}]")

    sharpe_str = f"{r['sharpe']:.3f}" if r["sharpe"] else "N/A"
    table.add_row("Sharpe Ratio", sharpe_str)
    table.add_row("Max Drawdown", f"{r['max_drawdown_pct']:.2f}%")
    table.add_row("Total Trades", str(r["total_trades"]))
    table.add_row("Win Rate", f"{r['win_rate_pct']:.1f}% ({r['won']}W / {r['lost']}L)")

    # Calmar ratio = annualized return / max drawdown
    if r["max_drawdown_pct"] > 0:
        from_dt = datetime.strptime(r["from_date"], "%Y-%m-%d")
        to_dt = datetime.strptime(r["to_date"], "%Y-%m-%d")
        years = (to_dt - from_dt).days / 365
        ann_return = (r["returns_pct"] / 100 + 1) ** (1 / max(years, 0.1)) - 1
        calmar = ann_return / (r["max_drawdown_pct"] / 100)
        table.add_row("Calmar Ratio", f"{calmar:.3f}")
        r["calmar"] = round(calmar, 3)
    else:
        table.add_row("Calmar Ratio", "N/A")
        r["calmar"] = None

    console.print(table)

    # Save JSON
    if save:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = REPORTS_DIR / f"backtest_{r['strategy']}_{r['instrument']}_{ts}.json"
        # Remove trade_log from saved JSON (too large); save separately
        save_data = {k: v for k, v in r.items() if k != "trade_log"}
        with open(filename, "w") as f:
            json.dump(save_data, f, indent=2, default=str)
        logger.info(f"Report saved to {filename}")
        r["report_file"] = str(filename)

        if r.get("trade_log"):
            import pandas as pd
            trade_file = REPORTS_DIR / f"trades_{r['strategy']}_{r['instrument']}_{ts}.json"
            with open(trade_file, "w") as f:
                json.dump(r["trade_log"], f, indent=2, default=str)

    return r


def compare_strategies(results: list) -> None:
    """Print a comparison table for multiple backtest results."""
    table = Table(title="Strategy Comparison", box=box.ROUNDED)
    for col in ["Strategy", "Instrument", "Return %", "Sharpe", "Max DD %", "Win Rate", "Trades"]:
        table.add_column(col, style="cyan" if col in ["Strategy", "Instrument"] else "white")

    for r in sorted(results, key=lambda x: x.get("returns_pct", 0), reverse=True):
        ret = r.get("returns_pct", 0)
        color = "green" if ret >= 0 else "red"
        table.add_row(
            r.get("strategy", ""),
            r.get("instrument", ""),
            f"[{color}]{ret:+.2f}%[/{color}]",
            f"{r['sharpe']:.3f}" if r.get("sharpe") else "N/A",
            f"{r.get('max_drawdown_pct', 0):.2f}%",
            f"{r.get('win_rate_pct', 0):.1f}%",
            str(r.get("total_trades", 0)),
        )

    console.print(table)
