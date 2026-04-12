"""
tests/test_market_hours.py — Unit tests for NSE market-hours helpers
=====================================================================
Verifies that _nse_ts() correctly attaches NSE market times to dates,
and that trade records contain the correct timestamps per reason type.

Run:  cd /Users/my/mayu_solutions/Zerodha_India && .venv/bin/python -m pytest tests/ -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd

from v14_backtest import (
    _nse_ts, NSE_OPEN_TIME, NSE_CLOSE_TIME, NSE_INTRA_TIME,
    Portfolio, CAPITAL, INDEX_RISK,
)


# ─── _nse_ts helper ──────────────────────────────────────────────────────────

class TestNseTsHelper:

    def test_attaches_0915_open_time(self):
        date = pd.Timestamp("2025-04-07")
        ts = _nse_ts(date, NSE_OPEN_TIME)
        assert ts.hour == 9 and ts.minute == 15 and ts.second == 0

    def test_attaches_1530_close_time(self):
        date = pd.Timestamp("2025-04-07")
        ts = _nse_ts(date, NSE_CLOSE_TIME)
        assert ts.hour == 15 and ts.minute == 30 and ts.second == 0

    def test_attaches_1200_intra_time(self):
        date = pd.Timestamp("2025-04-07")
        ts = _nse_ts(date, NSE_INTRA_TIME)
        assert ts.hour == 12 and ts.minute == 0 and ts.second == 0

    def test_date_part_unchanged(self):
        date = pd.Timestamp("2025-04-07")
        ts = _nse_ts(date, NSE_OPEN_TIME)
        assert ts.year == 2025 and ts.month == 4 and ts.day == 7

    def test_handles_midnight_normalized_input(self):
        """Normalized (midnight) timestamps are the standard backtest input."""
        date = pd.Timestamp("2025-04-07 00:00:00")
        ts = _nse_ts(date, NSE_OPEN_TIME)
        assert ts.hour == 9 and ts.minute == 15

    def test_handles_string_input(self):
        ts = _nse_ts("2025-04-07", NSE_OPEN_TIME)
        assert ts.hour == 9 and ts.minute == 15 and ts.day == 7

    def test_handles_datetime_with_existing_time(self):
        """Input may already have a time component — should override it."""
        date = pd.Timestamp("2025-04-07 12:34:56")
        ts = _nse_ts(date, NSE_CLOSE_TIME)
        assert ts.hour == 15 and ts.minute == 30


# ─── Trade record timestamps ──────────────────────────────────────────────────

class TestTradeTimestamps:
    """
    Verify that Portfolio records the correct NSE times in trade records.
    Entries always at 09:15 (open execution).
    Stop/target exits at 12:00 (intraday approximation).
    All other exits at 15:30 (session close).
    """

    def _make_trade(self, reason, exit_price=22000.0):
        pf = Portfolio(CAPITAL)
        date_in  = pd.Timestamp("2025-01-15")
        date_out = pd.Timestamp("2025-01-16")
        pf.enter("NIFTY", "bb_reversion", date_in, 1,
                 22000.0, 21700.0, "BB_LOW", INDEX_RISK)
        pf.force_close("NIFTY", date_out, exit_price, reason)
        return pf.trades[-1]

    def test_entry_date_has_0915_time(self):
        trade = self._make_trade("target", 22600.0)
        ts = pd.Timestamp(trade["entry_date"])
        assert ts.hour == 9 and ts.minute == 15

    def test_stop_exit_has_1200_time(self):
        trade = self._make_trade("stop", 21500.0)
        ts = pd.Timestamp(trade["exit_date"])
        assert ts.hour == 12 and ts.minute == 0

    def test_target_exit_has_1200_time(self):
        trade = self._make_trade("target", 22600.0)
        ts = pd.Timestamp(trade["exit_date"])
        assert ts.hour == 12 and ts.minute == 0

    def test_signal_exit_has_1530_time(self):
        """BB_MID_EXIT, WR_EXIT and other signal exits → 15:30."""
        for reason in ("BB_MID_EXIT", "WR_EXIT", "force", "end_of_backtest"):
            trade = self._make_trade(reason, 22100.0)
            ts = pd.Timestamp(trade["exit_date"])
            assert ts.hour == 15 and ts.minute == 30, \
                f"Expected 15:30 for reason={reason}, got {ts.hour}:{ts.minute}"

    def test_no_midnight_timestamps_in_entries(self):
        """Ensure 00:00 is never stored as an entry time."""
        trade = self._make_trade("force", 22000.0)
        entry_ts = pd.Timestamp(trade["entry_date"])
        assert not (entry_ts.hour == 0 and entry_ts.minute == 0), \
            "Entry time must not be midnight 00:00"

    def test_no_midnight_timestamps_in_exits(self):
        """Ensure 00:00 is never stored as an exit time."""
        trade = self._make_trade("force", 22000.0)
        exit_ts = pd.Timestamp(trade["exit_date"])
        assert not (exit_ts.hour == 0 and exit_ts.minute == 0), \
            "Exit time must not be midnight 00:00"
