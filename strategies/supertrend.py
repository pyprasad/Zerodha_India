"""
SuperTrend Strategy
====================
- ATR period: 10, multiplier: 3.0
- Signal: price crosses above SuperTrend line → long; below → short
- Confirm with EMA 55 direction
- Trailing stop: SuperTrend line itself
"""
import logging

import backtrader as bt
import numpy as np

from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class SuperTrendIndicator(bt.Indicator):
    """
    SuperTrend indicator computed from ATR.
    Lines: supertrend (the dynamic support/resistance), direction (1=up, -1=down)
    """
    lines = ("supertrend", "direction")
    params = (("period", 10), ("multiplier", 3.0))

    def __init__(self):
        self.atr = bt.indicators.ATR(self.data, period=self.p.period)
        super().__init__()

    def once(self, start, end):
        high = self.data.high.array
        low = self.data.low.array
        close = self.data.close.array
        atr = self.atr.array
        m = self.p.multiplier

        st = self.lines.supertrend.array
        direction = self.lines.direction.array

        for i in range(start, end):
            if i < self.p.period:
                mid = (high[i] + low[i]) / 2
                st[i] = mid
                direction[i] = 1.0
                continue

            mid = (high[i] + low[i]) / 2
            upper_band = mid + m * atr[i]
            lower_band = mid - m * atr[i]

            prev_st = st[i - 1]
            prev_dir = direction[i - 1]

            if prev_dir == 1:  # was uptrend
                if close[i] < prev_st:
                    st[i] = upper_band
                    direction[i] = -1.0
                else:
                    st[i] = max(lower_band, prev_st)
                    direction[i] = 1.0
            else:  # was downtrend
                if close[i] > prev_st:
                    st[i] = lower_band
                    direction[i] = 1.0
                else:
                    st[i] = min(upper_band, prev_st)
                    direction[i] = -1.0


class SuperTrendStrategy(BaseStrategy):
    """SuperTrend trend-following strategy confirmed by EMA 55."""

    params = (
        ("instrument", "NIFTY"),
        ("atr_period", 10),
        ("multiplier", 3.0),
        ("trend_ema", 55),
        ("stoploss_pct", 0.005),    # Fallback fixed stop
        ("target_pct", 0.02),       # Trailing via SuperTrend, but cap at 2%
        ("position_size_multiplier", 1.0),
        ("lot_size", 25),
        ("capital", 100000),
        ("per_trade_risk_pct", 0.005),
    )

    def __init__(self):
        super().__init__()
        self.supertrend = SuperTrendIndicator(
            self.datas[0],
            period=self.p.atr_period,
            multiplier=self.p.multiplier,
        )
        self.ema_trend = bt.indicators.EMA(self.datas[0].close, period=self.p.trend_ema)
        self.prev_direction = None

    def signal_long(self) -> bool:
        """Direction flips to uptrend (1) and price is above EMA 55."""
        direction = self.supertrend.direction[0]
        prev = self.supertrend.direction[-1] if len(self.supertrend.direction) > 1 else direction
        flip_up = (direction == 1.0 and prev == -1.0)
        above_trend = self.datas[0].close[0] > self.ema_trend[0]
        return flip_up and above_trend

    def signal_short(self) -> bool:
        """Direction flips to downtrend (-1) and price is below EMA 55."""
        direction = self.supertrend.direction[0]
        prev = self.supertrend.direction[-1] if len(self.supertrend.direction) > 1 else direction
        flip_down = (direction == -1.0 and prev == 1.0)
        below_trend = self.datas[0].close[0] < self.ema_trend[0]
        return flip_down and below_trend

    def _manage_position(self):
        """Use SuperTrend line as trailing stop."""
        price = self.datas[0].close[0]
        st_line = self.supertrend.supertrend[0]
        direction = self.supertrend.direction[0]

        if self.position.size > 0:  # Long
            # Exit if SuperTrend flips to downtrend OR price < SuperTrend line
            if direction == -1.0 or price < st_line:
                logger.info(f"SuperTrend trailing stop hit (long) @ {price:.2f}")
                self.order = self.close()
        elif self.position.size < 0:  # Short
            if direction == 1.0 or price > st_line:
                logger.info(f"SuperTrend trailing stop hit (short) @ {price:.2f}")
                self.order = self.close()

    def next(self):
        if self.order:
            return
        if not self._is_market_open():
            return
        if self._must_exit():
            self._force_close()
            return

        if not self.position and self._can_enter():
            if self.signal_long():
                self._enter_long()
            elif self.signal_short():
                self._enter_short()
        elif self.position:
            self._manage_position()
