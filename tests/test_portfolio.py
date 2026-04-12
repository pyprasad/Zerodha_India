"""
tests/test_portfolio.py — Unit tests for Portfolio sizing and P&L logic
========================================================================
Tests verify position sizing, margin, SPAN calculations, P&L computation,
and portfolio constraints (max concurrent, max lots, total risk cap).

Run:  cd /Users/my/mayu_solutions/Zerodha_India && .venv/bin/python -m pytest tests/ -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from v14_backtest import (
    Portfolio, CAPITAL, LOT_SIZES, MAX_LOTS,
    INDEX_SPAN_PCT, STOCK_SPAN_PCT,
    INDEX_RISK, STOCK_RISK, MAX_CONCURRENT,
    BROKERAGE, EXCHANGE_FEE, SLIPPAGE, STT_RATE,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _date(s="2025-01-15"):
    return pd.Timestamp(s)


def _enter(pf, inst="NIFTY", direction=1, entry=22000.0, stop=21700.0,
           strategy="bb_reversion", tag="BB_LOW", risk_pct=None):
    if risk_pct is None:
        risk_pct = INDEX_RISK
    return pf.enter(inst, strategy, _date(), direction, entry, stop, tag, risk_pct)


# ─── SPAN margin ──────────────────────────────────────────────────────────────

class TestSpanMargin:

    def test_index_futures_use_10pct_span(self):
        pf = Portfolio(CAPITAL)
        assert pf._span("NIFTY")      == INDEX_SPAN_PCT  # 0.10
        assert pf._span("BANKNIFTY")  == INDEX_SPAN_PCT
        assert pf._span("FINNIFTY")   == INDEX_SPAN_PCT
        assert pf._span("NIFTYIT")    == INDEX_SPAN_PCT
        assert pf._span("MIDCPNIFTY") == INDEX_SPAN_PCT

    def test_stock_futures_use_15pct_span(self):
        pf = Portfolio(CAPITAL)
        assert pf._span("RELIANCE")   == STOCK_SPAN_PCT  # 0.15
        assert pf._span("HDFCBANK")   == STOCK_SPAN_PCT
        assert pf._span("INFY")       == STOCK_SPAN_PCT
        assert pf._span("TCS")        == STOCK_SPAN_PCT


# ─── Position sizing ──────────────────────────────────────────────────────────

class TestPositionSizing:

    def test_qty_positive_for_valid_inputs(self):
        pf = Portfolio(CAPITAL)
        qty = pf._qty("NIFTY", 22000.0, 21700.0, INDEX_RISK)
        assert qty > 0

    def test_qty_is_integer_multiple_of_lot_size(self):
        pf = Portfolio(CAPITAL)
        lot = LOT_SIZES["NIFTY"]
        qty = pf._qty("NIFTY", 22000.0, 21700.0, INDEX_RISK)
        assert qty % lot == 0

    def test_qty_capped_at_max_lots(self):
        """Even with large capital, qty cannot exceed MAX_LOTS × lot_size."""
        pf = Portfolio(CAPITAL * 1000)   # massive capital
        lot  = LOT_SIZES["NIFTY"]
        mxl  = MAX_LOTS["NIFTY"]
        qty  = pf._qty("NIFTY", 22000.0, 21700.0, INDEX_RISK)
        assert qty <= mxl * lot

    def test_qty_zero_when_capital_insufficient(self):
        """When portfolio has essentially no free capital, qty = 0."""
        pf = Portfolio(1000.0)   # tiny capital
        qty = pf._qty("NIFTY", 22000.0, 21700.0, INDEX_RISK)
        assert qty == 0

    def test_stock_qty_capped_at_lower_max_lots(self):
        pf = Portfolio(CAPITAL * 100)
        lot = LOT_SIZES["RELIANCE"]
        mxl = MAX_LOTS["RELIANCE"]    # 2 lots for RELIANCE
        qty = pf._qty("RELIANCE", 1300.0, 1200.0, STOCK_RISK)
        assert qty <= mxl * lot


# ─── Portfolio.enter() ───────────────────────────────────────────────────────

class TestEnter:

    def test_entry_returns_true_on_success(self):
        pf = Portfolio(CAPITAL)
        ok = _enter(pf)
        assert ok is True

    def test_position_recorded_after_entry(self):
        pf = Portfolio(CAPITAL)
        _enter(pf, inst="NIFTY")
        assert "NIFTY" in pf.positions

    def test_entry_time_is_0915(self):
        """Entry date stored with 09:15 NSE open time (v12 fix)."""
        pf = Portfolio(CAPITAL)
        _enter(pf, inst="NIFTY")
        pos = pf.positions["NIFTY"]
        ts = pd.Timestamp(str(pos["date"]))
        assert ts.hour == 9 and ts.minute == 15

    def test_capital_reduced_by_span_margin_and_cost(self):
        pf = Portfolio(CAPITAL)
        before = pf.C
        _enter(pf, inst="NIFTY", entry=22000.0, stop=21700.0)
        assert pf.C < before

    def test_second_entry_same_instrument_rejected(self):
        """Cannot hold two positions in the same instrument."""
        pf = Portfolio(CAPITAL)
        _enter(pf, inst="NIFTY")
        ok2 = _enter(pf, inst="NIFTY")
        assert ok2 is False
        assert len([k for k in pf.positions if k == "NIFTY"]) == 1

    def test_max_concurrent_positions_enforced(self):
        """Cannot exceed MAX_CONCURRENT open positions."""
        pf = Portfolio(CAPITAL * 10)
        instruments = [
            "NIFTY","BANKNIFTY","FINNIFTY","NIFTYIT",
            "RELIANCE","HDFCBANK","INFY","TCS","ICICIBANK","AXISBANK",
        ]
        assert len(instruments) == MAX_CONCURRENT
        for inst in instruments:
            _enter(pf, inst=inst, entry=1000.0, stop=900.0,
                   risk_pct=STOCK_RISK)
        # One more should fail
        ok = _enter(pf, inst="SBIN", entry=500.0, stop=450.0,
                    risk_pct=STOCK_RISK)
        assert ok is False

    def test_target_set_to_2x_risk_reward(self):
        """Target = entry + 2 × (entry − stop) for long positions."""
        pf = Portfolio(CAPITAL)
        entry, stop = 22000.0, 21700.0
        _enter(pf, inst="NIFTY", direction=1, entry=entry, stop=stop)
        pos = pf.positions["NIFTY"]
        expected_tgt = entry + 2.0 * abs(entry - stop)
        assert abs(pos["target"] - expected_tgt) < 0.01


# ─── Portfolio._close() / P&L ────────────────────────────────────────────────

class TestPnL:

    def test_long_winner_pnl_positive(self):
        pf = Portfolio(CAPITAL)
        _enter(pf, inst="NIFTY", direction=1, entry=22000.0, stop=21700.0)
        pos = pf.positions["NIFTY"]
        qty = pos["qty"]
        pf.force_close("NIFTY", _date("2025-01-20"), 22500.0, "target")
        assert pf.trades[-1]["pnl"] > 0

    def test_long_loser_pnl_negative(self):
        pf = Portfolio(CAPITAL)
        _enter(pf, inst="NIFTY", direction=1, entry=22000.0, stop=21700.0)
        pf.force_close("NIFTY", _date("2025-01-20"), 21500.0, "stop")
        assert pf.trades[-1]["pnl"] < 0

    def test_short_winner_pnl_positive(self):
        pf = Portfolio(CAPITAL)
        _enter(pf, inst="NIFTY", direction=-1, entry=22000.0, stop=22300.0)
        pf.force_close("NIFTY", _date("2025-01-20"), 21500.0, "target")
        assert pf.trades[-1]["pnl"] > 0

    def test_short_loser_pnl_negative(self):
        pf = Portfolio(CAPITAL)
        _enter(pf, inst="NIFTY", direction=-1, entry=22000.0, stop=22300.0)
        pf.force_close("NIFTY", _date("2025-01-20"), 22500.0, "stop")
        assert pf.trades[-1]["pnl"] < 0

    def test_exit_reason_recorded(self):
        pf = Portfolio(CAPITAL)
        _enter(pf, inst="NIFTY")
        pf.force_close("NIFTY", _date("2025-01-20"), 21800.0, "WR_EXIT")
        assert pf.trades[-1]["reason"] == "WR_EXIT"

    def test_stop_exit_time_is_noon(self):
        """Stop hits get 12:00 IST timestamp (intraday approximation)."""
        pf = Portfolio(CAPITAL)
        _enter(pf, inst="NIFTY", direction=1, entry=22000.0, stop=21700.0)
        pf.force_close("NIFTY", _date("2025-01-20"), 21500.0, "stop")
        exit_ts = pd.Timestamp(pf.trades[-1]["exit_date"])
        assert exit_ts.hour == 12 and exit_ts.minute == 0

    def test_force_close_exit_time_is_1530(self):
        """Signal exits (WR_EXIT, BB_MID_EXIT, force) get 15:30 IST."""
        pf = Portfolio(CAPITAL)
        _enter(pf, inst="NIFTY")
        pf.force_close("NIFTY", _date("2025-01-20"), 22100.0, "BB_MID_EXIT")
        exit_ts = pd.Timestamp(pf.trades[-1]["exit_date"])
        assert exit_ts.hour == 15 and exit_ts.minute == 30

    def test_position_removed_after_close(self):
        pf = Portfolio(CAPITAL)
        _enter(pf, inst="NIFTY")
        assert "NIFTY" in pf.positions
        pf.force_close("NIFTY", _date("2025-01-20"), 22000.0, "force")
        assert "NIFTY" not in pf.positions

    def test_capital_restored_after_close(self):
        """Margin released + P&L credited after closing."""
        pf = Portfolio(CAPITAL)
        cap_before_entry = pf.C
        _enter(pf, inst="NIFTY", entry=22000.0, stop=21700.0)
        cap_after_entry = pf.C
        assert cap_after_entry < cap_before_entry
        pf.force_close("NIFTY", _date("2025-01-20"), 22000.0, "force")
        # Capital should be back near original (small difference = brokerage cost)
        assert pf.C > cap_after_entry
