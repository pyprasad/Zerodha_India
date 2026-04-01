"""
Opening Range Breakout (ORB) Strategy
======================================
- Opening range = high/low of first 15 minutes (9:15–9:30 IST)
- Long: price breaks above range high with volume > 1.5× average
- Short: price breaks below range low with volume > 1.5× average
- Stop-loss: opposite end of opening range
- Target: 2× risk (1:2 R/R)
- Hard exit: 15:00 IST
"""
import logging
from datetime import time as dtime

import backtrader as bt

from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

ORB_START = dtime(9, 15)
ORB_END = dtime(9, 30)


class ORBStrategy(BaseStrategy):
    """Opening Range Breakout strategy for NIFTY and BANKNIFTY."""

    params = (
        ("instrument", "NIFTY"),
        ("volume_multiplier", 1.5),   # Volume confirmation threshold
        ("rr_ratio", 2.0),
        ("position_size_multiplier", 1.0),
        ("lot_size", 25),
        ("capital", 100000),
        ("per_trade_risk_pct", 0.005),
    )

    def __init__(self):
        super().__init__()
        self.orb_high = None
        self.orb_low = None
        self.orb_locked = False
        self.entered = False
        self.vol_ma = bt.indicators.SMA(self.datas[0].volume, period=20)

    def _current_time(self) -> dtime:
        return self.datas[0].datetime.time()

    def _in_orb_window(self) -> bool:
        t = self._current_time()
        return ORB_START <= t < ORB_END

    def _after_orb(self) -> bool:
        return self._current_time() >= ORB_END

    def _update_orb_range(self):
        """Track opening range during the ORB window."""
        if self._in_orb_window():
            h = self.datas[0].high[0]
            l = self.datas[0].low[0]
            if self.orb_high is None or h > self.orb_high:
                self.orb_high = h
            if self.orb_low is None or l < self.orb_low:
                self.orb_low = l

    def _lock_orb(self):
        """Called once when the ORB window closes."""
        if not self.orb_locked and self._after_orb() and self.orb_high is not None:
            self.orb_locked = True
            logger.info(
                f"ORB locked for {self.p.instrument}: "
                f"HIGH={self.orb_high:.2f} LOW={self.orb_low:.2f}"
            )

    def signal_long(self) -> bool:
        if not self.orb_locked or self.entered:
            return False
        price = self.datas[0].close[0]
        vol = self.datas[0].volume[0]
        avg_vol = self.vol_ma[0]
        return price > self.orb_high and vol > avg_vol * self.p.volume_multiplier

    def signal_short(self) -> bool:
        if not self.orb_locked or self.entered:
            return False
        price = self.datas[0].close[0]
        vol = self.datas[0].volume[0]
        avg_vol = self.vol_ma[0]
        return price < self.orb_low and vol > avg_vol * self.p.volume_multiplier

    def _enter_long(self):
        entry = self.datas[0].close[0]
        stop = self.orb_low   # Stop at the opposite end of ORB
        risk = entry - stop
        target = entry + (risk * self.p.rr_ratio)
        qty = self._calc_qty(entry, stop)
        self.stop_price = stop
        self.target_price = target
        self.entered = True
        logger.info(
            f"ORB LONG {self.p.instrument} entry={entry:.2f} "
            f"stop={stop:.2f} target={target:.2f} qty={qty}"
        )
        self.order = self.buy(size=qty)

    def _enter_short(self):
        entry = self.datas[0].close[0]
        stop = self.orb_high  # Stop at the opposite end of ORB
        risk = stop - entry
        target = entry - (risk * self.p.rr_ratio)
        qty = self._calc_qty(entry, stop)
        self.stop_price = stop
        self.target_price = target
        self.entered = True
        logger.info(
            f"ORB SHORT {self.p.instrument} entry={entry:.2f} "
            f"stop={stop:.2f} target={target:.2f} qty={qty}"
        )
        self.order = self.sell(size=qty)

    def next(self):
        if self.order:
            return
        if not self._is_market_open():
            return

        # Build / lock the opening range
        self._update_orb_range()
        self._lock_orb()

        # Reset entered flag at start of new day
        current_date = self.datas[0].datetime.date(0)
        if not hasattr(self, "_last_date") or self._last_date != current_date:
            self._last_date = current_date
            self.orb_high = None
            self.orb_low = None
            self.orb_locked = False
            self.entered = False

        if self._must_exit():
            self._force_close()
            return

        if not self.position and self._can_enter() and self.orb_locked:
            if self.signal_long():
                self._enter_long()
            elif self.signal_short():
                self._enter_short()
        elif self.position:
            self._manage_position()
