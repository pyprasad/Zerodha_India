"""
paper_trading.py — v12 Live Paper Trading Signal Generator
============================================================
Run once daily after NSE market close (after 15:30 IST) to get
next-session signals based on today's closing bars.

Usage:
    python paper_trading.py

What it does:
  1. Downloads latest 2 years of daily data via Yahoo Finance
  2. Loads best strategy per instrument from latest v12_backtest JSON
  3. Generates entry/exit signals for next trading session
  4. Saves signals to reports/paper_signals_{timestamp}.json
  5. Outputs a formatted signal sheet to terminal

NSE Market Hours (IST):
  Pre-open  : 09:00 – 09:15
  Session   : 09:15 – 15:30  ← signals fire at session close
  Execution : next day 09:15 ← entries/exits placed at open

Paper Trading vs Live Trading:
  This module ONLY generates signals — no orders are submitted.
  To go live with Zerodha Kite:
    pip install kiteconnect
    See: https://kite.trade/docs/pykiteconnect/
    Implement: kite_live.py  (place_order wrapper around this signal sheet)

Paper capital state: reports/paper_trading_state.json
  Tracks paper capital and open positions between runs.
  Delete this file to reset paper capital to ₹10,00,000.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from v12_backtest import run_paper_trading
from pathlib import Path

if __name__ == "__main__":
    entry_sigs, exit_sigs = run_paper_trading(
        capital_file=Path("reports/paper_trading_state.json")
    )

    total = len(entry_sigs) + len(exit_sigs)
    if total == 0:
        print("\n  No signals today. Market may be trending sideways or no setups triggered.")
    else:
        print(f"\n  Total signals: {total}  "
              f"({len(entry_sigs)} entries, {len(exit_sigs)} exits)")
        print("  All signals valid for next trading session open at 09:15 IST.")
