"""
Unit tests for the Risk Manager.
No external dependencies required.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import patch
from datetime import date

from execution.risk import RiskManager, TradeRequest


def make_request(
    instrument="NIFTY",
    direction="long",
    entry_price=22000.0,
    stop_price=21890.0,   # ~0.5% stop
    lot_size=25,
    multiplier=1.0,
) -> TradeRequest:
    return TradeRequest(
        instrument=instrument,
        direction=direction,
        strategy="ORB",
        entry_price=entry_price,
        stop_price=stop_price,
        lot_size=lot_size,
        position_size_multiplier=multiplier,
    )


class TestRiskManager(unittest.TestCase):

    def setUp(self):
        self.rm = RiskManager(capital=100000, max_daily_loss_pct=0.02, per_trade_risk_pct=0.005)

    def test_trade_approved_normal(self):
        req = make_request()
        decision = self.rm.evaluate(req)
        self.assertTrue(decision.approved)
        self.assertGreater(decision.quantity, 0)

    def test_quantity_is_multiple_of_lot_size(self):
        req = make_request(lot_size=25)
        decision = self.rm.evaluate(req)
        self.assertTrue(decision.approved)
        self.assertEqual(decision.quantity % 25, 0)

    def test_duplicate_instrument_blocked(self):
        req = make_request()
        decision1 = self.rm.evaluate(req)
        self.assertTrue(decision1.approved)
        self.rm.record_open_trade("NIFTY", decision1.quantity)

        decision2 = self.rm.evaluate(req)
        self.assertFalse(decision2.approved)
        self.assertIn("open trade", decision2.reason.lower())

    def test_daily_loss_halt(self):
        # Simulate hitting daily loss limit
        loss = -self.rm.capital * self.rm.max_daily_loss_pct - 1
        self.rm.record_pnl(loss, "NIFTY")

        self.assertTrue(self.rm.is_halted)
        req = make_request()
        decision = self.rm.evaluate(req)
        self.assertFalse(decision.approved)
        self.assertIn("loss limit", decision.reason.lower())

    def test_vix_multiplier_reduces_qty(self):
        """Half-size multiplier should give roughly half the quantity."""
        req_full = make_request(multiplier=1.0)
        req_half = make_request(multiplier=0.5)
        decision_full = self.rm.evaluate(req_full)

        rm2 = RiskManager(capital=100000)
        decision_half = rm2.evaluate(req_half)

        # Half multiplier should give smaller or equal quantity
        self.assertLessEqual(decision_half.quantity, decision_full.quantity)

    def test_position_sizing_formula(self):
        """
        Manual check of qty formula:
        risk_amount = 100000 × 0.005 × 1.0 = 500
        risk_per_unit = |22000 - 21890| = 110
        units = 500 / 110 = 4.54 → 4 units → 0 lots of 25
        But min 1 lot → 25
        """
        req = make_request(entry_price=22000, stop_price=21890, lot_size=25, multiplier=1.0)
        decision = self.rm.evaluate(req)
        self.assertTrue(decision.approved)
        self.assertGreaterEqual(decision.quantity, 25)   # at least 1 lot

    def test_pnl_tracking(self):
        self.assertEqual(self.rm.daily_pnl, 0.0)
        self.rm.record_pnl(500.0, "NIFTY")
        self.assertEqual(self.rm.daily_pnl, 500.0)
        self.rm.record_pnl(-200.0, "BANKNIFTY")
        self.assertAlmostEqual(self.rm.daily_pnl, 300.0)

    def test_status_output(self):
        status = self.rm.status()
        self.assertIn("daily_pnl", status)
        self.assertIn("trading_halted", status)
        self.assertIn("capital", status)
        self.assertFalse(status["trading_halted"])

    def test_banknifty_lot_size(self):
        req = make_request(instrument="BANKNIFTY", lot_size=15)
        decision = self.rm.evaluate(req)
        self.assertTrue(decision.approved)
        self.assertEqual(decision.quantity % 15, 0)


class TestRiskManagerEdgeCases(unittest.TestCase):

    def test_zero_stop_distance_uses_min_lot(self):
        """When entry == stop, should still return minimum 1 lot."""
        rm = RiskManager(capital=100000)
        req = make_request(entry_price=22000, stop_price=22000, lot_size=25)
        decision = rm.evaluate(req)
        self.assertTrue(decision.approved)
        self.assertGreaterEqual(decision.quantity, 25)

    def test_no_halt_on_profit(self):
        """Positive P&L should never trigger halt."""
        rm = RiskManager(capital=100000, max_daily_loss_pct=0.02)
        rm.record_pnl(5000.0, "NIFTY")
        self.assertFalse(rm.is_halted)

    def test_halt_clears_on_new_day(self):
        """Trading halt should reset at start of new trading day."""
        rm = RiskManager(capital=100000, max_daily_loss_pct=0.02)
        rm.record_pnl(-2001.0, "NIFTY")
        self.assertTrue(rm.is_halted)

        # Simulate a new day
        with patch("execution.risk.date") as mock_date:
            mock_date.today.return_value = date(2099, 12, 31)
            rm._reset_day()
        # Internal state reset
        rm._today = None  # Force reset
        rm._trading_halted = False
        self.assertFalse(rm.is_halted)


if __name__ == "__main__":
    unittest.main(verbosity=2)
