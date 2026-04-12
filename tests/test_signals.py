"""
tests/test_signals.py — Unit tests for v13 signal functions
=============================================================
Tests verify that each signal function fires correctly on known input data
and does NOT fire when conditions are not met.

Run:  cd /Users/my/mayu_solutions/Zerodha_India && .venv/bin/python -m pytest tests/ -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from v14_backtest import (
    sig_bb_reversion, sig_wr_vix, sig_wr_wide, sig_macd,
    sig_supertrend, sig_momentum_20d,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _series(*vals):
    """Create a pd.Series from values. i=last index."""
    return pd.Series(list(vals), dtype=float)


def make_bb_ind(pct_prev, pct_curr, rsi=42.0, e50=100.0, e200=98.0, atr=2.0,
                close=95.0):
    """Minimal indicator dict for bb_reversion tests."""
    return {
        "bb_pct":  _series(pct_prev, pct_curr),
        "rsi14":   _series(rsi, rsi),
        "e50":     _series(e50, e50),
        "e200":    _series(e200, e200),
        "atr":     _series(atr, atr),
        "close":   _series(close, close),
    }


def make_wr_ind(wr_prev, wr_curr, e50=100.0, e200=98.0, atr=2.0, close=99.0):
    """Minimal indicator dict for williams_r tests."""
    return {
        "wR14":  _series(wr_prev, wr_curr),
        "e200":  _series(e200, e200),
        "e50":   _series(e50, e50),
        "atr":   _series(atr, atr),
        "close": _series(close, close),
    }


# ─── sig_bb_reversion ────────────────────────────────────────────────────────

class TestBBReversion:

    def test_long_signal_fires_when_price_crosses_below_lower_band(self):
        """BB%B crossing from ≥0.05 to <0.05 in uptrend → Long."""
        ind = make_bb_ind(pct_prev=0.06, pct_curr=0.03, rsi=38.0,
                          e50=100.0, e200=97.0)
        result = sig_bb_reversion(1, ind)
        assert result is not None
        assert result[0] == "L"
        assert result[2] == "BB_LOW"

    def test_short_signal_fires_when_price_crosses_above_upper_band(self):
        """BB%B crossing from ≤0.95 to >0.95 in downtrend → Short."""
        ind = make_bb_ind(pct_prev=0.94, pct_curr=0.97, rsi=68.0,
                          e50=97.0, e200=100.0)
        result = sig_bb_reversion(1, ind)
        assert result is not None
        assert result[0] == "S"
        assert result[2] == "BB_HIGH"

    def test_no_signal_when_price_in_middle_of_band(self):
        """BB%B=0.5 (middle) → no signal."""
        ind = make_bb_ind(pct_prev=0.5, pct_curr=0.5)
        assert sig_bb_reversion(1, ind) is None

    def test_long_blocked_when_already_below_lower_band(self):
        """BB%B was already <0.05 (no crossover) → no signal."""
        ind = make_bb_ind(pct_prev=0.03, pct_curr=0.02)   # stayed below
        assert sig_bb_reversion(1, ind) is None

    def test_long_blocked_by_downtrend(self):
        """e50 significantly below e200 (downtrend) → Long blocked even if BB_LOW."""
        ind = make_bb_ind(pct_prev=0.06, pct_curr=0.03, rsi=38.0,
                          e50=90.0, e200=100.0)   # e50 = 90% of e200 < 0.97 threshold
        assert sig_bb_reversion(1, ind) is None

    def test_long_blocked_by_collapsed_rsi(self):
        """RSI < 25 → price in freefall, skip Long."""
        ind = make_bb_ind(pct_prev=0.06, pct_curr=0.03, rsi=20.0,
                          e50=100.0, e200=98.0)
        assert sig_bb_reversion(1, ind) is None

    def test_long_blocked_by_high_rsi(self):
        """RSI ≥ 55 for a BB_LOW entry → inconsistent, skip Long."""
        ind = make_bb_ind(pct_prev=0.06, pct_curr=0.03, rsi=60.0,
                          e50=100.0, e200=98.0)
        assert sig_bb_reversion(1, ind) is None

    def test_short_blocked_by_uptrend(self):
        """e50 > e200 (uptrend) → Short blocked even if BB_HIGH."""
        ind = make_bb_ind(pct_prev=0.94, pct_curr=0.97, rsi=68.0,
                          e50=100.0, e200=97.0)   # e50 > e200 → uptrend
        assert sig_bb_reversion(1, ind) is None

    def test_short_blocked_by_extreme_rsi(self):
        """RSI ≥ 75 (overbought momentum) → skip Short."""
        ind = make_bb_ind(pct_prev=0.94, pct_curr=0.97, rsi=78.0,
                          e50=97.0, e200=100.0)
        assert sig_bb_reversion(1, ind) is None

    def test_stop_distance_uses_atr_multiple(self):
        """Stop distance returned should be atr × 1.5."""
        atr = 5.0
        ind = make_bb_ind(pct_prev=0.06, pct_curr=0.03, rsi=38.0,
                          e50=100.0, e200=98.0, atr=atr)
        result = sig_bb_reversion(1, ind)
        assert result is not None
        assert abs(result[1] - atr * 1.5) < 1e-9

    def test_nan_pct_returns_none(self):
        """NaN BB%B → no signal (guard against missing data)."""
        ind = make_bb_ind(pct_prev=np.nan, pct_curr=0.03)
        assert sig_bb_reversion(1, ind) is None

    def test_nan_e200_returns_none(self):
        """NaN e200 → no signal."""
        ind = make_bb_ind(pct_prev=0.06, pct_curr=0.03, e200=np.nan)
        assert sig_bb_reversion(1, ind) is None


# ─── sig_wr_vix / sig_wr_wide ────────────────────────────────────────────────

class TestWilliamsR:

    def test_long_signal_on_oversold_cross(self):
        """WR crosses from below -80 to above -80 → Long."""
        ind = make_wr_ind(wr_prev=-85.0, wr_curr=-75.0,
                          e50=100.0, e200=98.0, close=99.0)
        result = sig_wr_vix(1, ind)
        assert result is not None
        assert result[0] == "L"
        assert result[2] == "WR_OB"

    def test_short_signal_on_overbought_cross(self):
        """WR crosses from above -20 to below -20 → Short."""
        ind = make_wr_ind(wr_prev=-15.0, wr_curr=-25.0,
                          e50=98.0, e200=100.0, close=101.0)
        result = sig_wr_vix(1, ind)
        assert result is not None
        assert result[0] == "S"
        assert result[2] == "WR_OS"

    def test_no_signal_when_wr_stays_below_minus80(self):
        """WR already below -80, no crossover → no signal."""
        ind = make_wr_ind(wr_prev=-85.0, wr_curr=-90.0)
        assert sig_wr_vix(1, ind) is None

    def test_long_blocked_by_downtrend_e200(self):
        """Price far below e200 → Long blocked."""
        ind = make_wr_ind(wr_prev=-85.0, wr_curr=-75.0,
                          e50=90.0, e200=100.0, close=88.0)
        assert sig_wr_vix(1, ind) is None

    def test_nan_e200_returns_none(self):
        ind = make_wr_ind(wr_prev=-85.0, wr_curr=-75.0, e200=np.nan)
        assert sig_wr_vix(1, ind) is None

    def test_wr_wide_uses_larger_atr_mult(self):
        """wr_wide uses atr×2.0 vs wr_vix uses atr×1.5."""
        atr = 4.0
        ind = make_wr_ind(wr_prev=-85.0, wr_curr=-75.0,
                          e50=100.0, e200=98.0, close=99.0)
        ind["atr"] = _series(atr, atr)
        r_vix  = sig_wr_vix(1, ind)
        r_wide = sig_wr_wide(1, ind)
        assert r_vix  is not None and abs(r_vix[1]  - atr * 1.5) < 1e-9
        assert r_wide is not None and abs(r_wide[1] - atr * 2.0) < 1e-9


# ─── THE CRITICAL BUG REGRESSION TEST ────────────────────────────────────────

class TestMeanReversionTrailingStopBug:
    """
    Regression tests for the v14 complete fix: trailing stop must NEVER be
    applied to BB_ or WR_ mean-reversion entries.

    v11/v12: trail_stop(e21 − atr×0.3) called on entry bar → fake exits.
    v13 partial fix: skipped trailing on entry day only → still fake on day 2+.
    v14 complete fix: trailing stop removed entirely from BB/WR exit handler.

    Proof of v13 residual bug: RELIANCE Apr-8 claimed exit ₹1,361.1,
    but the day's HIGH was only ₹1,350.6 — physically impossible.
    """

    def test_no_profitable_stop_exits_in_v14_json(self):
        """
        Load latest v14 backtest results and verify zero profitable stops
        for mean-reversion (BB/WR) entries.  These can only exit at a LOSS
        (stop below entry) or via signal (BB_MID_EXIT / WR_EXIT).
        """
        import json, glob
        files = sorted(glob.glob("reports/v14_backtest_*.json"))
        if not files:
            pytest.skip("No v14 backtest JSON found — run v14_backtest.py first")

        with open(files[-1]) as f:
            d = json.load(f)
        trades = d["best_portfolio"]["trades"]

        # All stop exits for BB/WR entries must be losses
        bad = [t for t in trades
               if t["reason"] == "stop"
               and t["tag"] in ("BB_LOW", "BB_HIGH", "WR_OB", "WR_OS")
               and t["pnl"] > 0]

        assert len(bad) == 0, (
            f"Found {len(bad)} profitable stop exits on mean-reversion entries — "
            f"trailing stop bug may have been re-introduced.  Examples: {bad[:3]}"
        )

    def test_no_same_day_profitable_stop_exits_in_v14_json(self):
        """Zero same-day profitable stops (entry-bar guard still active)."""
        import json, glob
        files = sorted(glob.glob("reports/v14_backtest_*.json"))
        if not files:
            pytest.skip("No v14 backtest JSON found — run v14_backtest.py first")

        with open(files[-1]) as f:
            d = json.load(f)
        trades = d["best_portfolio"]["trades"]

        bad = [t for t in trades
               if t["duration"] == 0
               and t["reason"] == "stop"
               and t["pnl"] > 0]

        assert len(bad) == 0, (
            f"Found {len(bad)} same-day profitable stops — entry-bar "
            f"bug may have been re-introduced.  Examples: {bad[:3]}"
        )

    def test_same_day_loss_stops_are_legitimate(self):
        """
        Same-day LOSS stops are fine: entry at open, price immediately
        drops through the initial ATR stop below entry = real loss.
        """
        import json, glob
        files = sorted(glob.glob("reports/v14_backtest_*.json"))
        if not files:
            pytest.skip("No v14 backtest JSON found")

        with open(files[-1]) as f:
            d = json.load(f)
        trades = d["best_portfolio"]["trades"]

        for t in trades:
            if t["duration"] == 0 and t["reason"] == "stop":
                assert t["pnl"] <= 0, (
                    f"Same-day stop exit should be a loss but found profit: {t}"
                )
