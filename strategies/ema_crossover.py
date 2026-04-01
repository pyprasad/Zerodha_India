"""
EMA Crossover + VWAP Strategy
================================
- EMA 9 crosses above EMA 21 → bullish signal (confirmed if price above VWAP and EMA 55)
- EMA 9 crosses below EMA 21 → bearish signal (confirmed if price below VWAP and EMA 55)
- Stop-loss: 0.5% fixed
- Target: 1.5% fixed
- Timeframe: 5-minute for BANKNIFTY, 15-minute for NIFTY
"""
import logging

import backtrader as bt

from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class EMACrossoverStrategy(BaseStrategy):
    """EMA 9/21/55 crossover with VWAP trend filter."""

    params = (
        ("instrument", "NIFTY"),
        ("fast_ema", 9),
        ("slow_ema", 21),
        ("trend_ema", 55),
        ("stoploss_pct", 0.005),
        ("target_pct", 0.015),
        ("position_size_multiplier", 1.0),
        ("lot_size", 25),
        ("capital", 100000),
        ("per_trade_risk_pct", 0.005),
    )

    def __init__(self):
        super().__init__()
        self.ema_fast = bt.indicators.EMA(self.datas[0].close, period=self.p.fast_ema)
        self.ema_slow = bt.indicators.EMA(self.datas[0].close, period=self.p.slow_ema)
        self.ema_trend = bt.indicators.EMA(self.datas[0].close, period=self.p.trend_ema)
        self.crossover = bt.indicators.CrossOver(self.ema_fast, self.ema_slow)
        self._vwap_cache = {}

    def _get_vwap(self) -> float:
        """Calculate VWAP for today's session from available bar data."""
        current_date = self.datas[0].datetime.date(0)
        if current_date in self._vwap_cache:
            return self._vwap_cache[current_date]

        # Compute from raw arrays
        closes = list(self.datas[0].close.array)
        highs = list(self.datas[0].high.array)
        lows = list(self.datas[0].low.array)
        volumes = list(self.datas[0].volume.array)
        datetimes = [bt.num2date(d) for d in self.datas[0].datetime.array]

        total_tp_vol = 0.0
        total_vol = 0.0
        for i, dt in enumerate(datetimes):
            if dt.date() == current_date:
                tp = (highs[i] + lows[i] + closes[i]) / 3
                total_tp_vol += tp * volumes[i]
                total_vol += volumes[i]

        vwap = (total_tp_vol / total_vol) if total_vol > 0 else closes[-1]
        self._vwap_cache[current_date] = vwap
        return vwap

    def signal_long(self) -> bool:
        """
        Long when:
        - EMA 9 crosses above EMA 21 (crossover == 1)
        - Price is above VWAP
        - Price is above EMA 55 (trend filter)
        """
        if self.crossover[0] != 1:
            return False
        price = self.datas[0].close[0]
        vwap = self._get_vwap()
        return price > vwap and price > self.ema_trend[0]

    def signal_short(self) -> bool:
        """
        Short when:
        - EMA 9 crosses below EMA 21 (crossover == -1)
        - Price is below VWAP
        - Price is below EMA 55 (trend filter)
        """
        if self.crossover[0] != -1:
            return False
        price = self.datas[0].close[0]
        vwap = self._get_vwap()
        return price < vwap and price < self.ema_trend[0]

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
