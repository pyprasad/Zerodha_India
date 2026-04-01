"""
Mean Reversion Strategy — Bollinger Bands + RSI
=================================================
- Long: price touches lower BB AND RSI < 30 (oversold)
- Short: price touches upper BB AND RSI > 70 (overbought)
- Exit: price returns to BB midline (SMA 20)
- Best for: Range-bound sideways days
"""
import logging

import backtrader as bt

from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """Bollinger Bands + RSI mean reversion strategy."""

    params = (
        ("instrument", "NIFTY"),
        ("bb_period", 20),
        ("bb_std", 2.0),
        ("rsi_period", 14),
        ("rsi_oversold", 30),
        ("rsi_overbought", 70),
        ("stoploss_pct", 0.008),    # Wider stop for mean reversion
        ("target_pct", 0.015),      # Target at midline (overridden in _manage_position)
        ("position_size_multiplier", 1.0),
        ("lot_size", 25),
        ("capital", 100000),
        ("per_trade_risk_pct", 0.005),
    )

    def __init__(self):
        super().__init__()
        self.bb = bt.indicators.BollingerBands(
            self.datas[0].close,
            period=self.p.bb_period,
            devfactor=self.p.bb_std,
        )
        self.rsi = bt.indicators.RSI(
            self.datas[0].close,
            period=self.p.rsi_period,
        )

    def signal_long(self) -> bool:
        """
        Long when price touches or breaks below lower BB AND RSI is oversold.
        """
        price = self.datas[0].close[0]
        return price <= self.bb.lines.bot[0] and self.rsi[0] < self.p.rsi_oversold

    def signal_short(self) -> bool:
        """
        Short when price touches or breaks above upper BB AND RSI is overbought.
        """
        price = self.datas[0].close[0]
        return price >= self.bb.lines.top[0] and self.rsi[0] > self.p.rsi_overbought

    def _enter_long(self):
        """For mean reversion, target is the BB midline."""
        entry = self.datas[0].close[0]
        stop = entry * (1 - self.p.stoploss_pct)
        target = self.bb.lines.mid[0]  # BB midline (SMA 20)
        qty = self._calc_qty(entry, stop)
        self.stop_price = stop
        self.target_price = target
        logger.info(
            f"MR LONG {self.p.instrument} entry={entry:.2f} "
            f"stop={stop:.2f} target={target:.2f} (BB mid) qty={qty}"
        )
        self.order = self.buy(size=qty)

    def _enter_short(self):
        entry = self.datas[0].close[0]
        stop = entry * (1 + self.p.stoploss_pct)
        target = self.bb.lines.mid[0]  # BB midline
        qty = self._calc_qty(entry, stop)
        self.stop_price = stop
        self.target_price = target
        logger.info(
            f"MR SHORT {self.p.instrument} entry={entry:.2f} "
            f"stop={stop:.2f} target={target:.2f} (BB mid) qty={qty}"
        )
        self.order = self.sell(size=qty)

    def _manage_position(self):
        """
        Exit when:
        - Price crosses BB midline (mean reversion achieved)
        - Or stop-loss is hit
        """
        price = self.datas[0].close[0]
        midline = self.bb.lines.mid[0]

        if self.position.size > 0:  # Long
            if price <= self.stop_price:
                logger.info(f"MR stop-loss hit (long) @ {price:.2f}")
                self.order = self.close()
            elif price >= midline:
                logger.info(f"MR target hit (long) — price reached midline @ {price:.2f}")
                self.order = self.close()
        elif self.position.size < 0:  # Short
            if price >= self.stop_price:
                logger.info(f"MR stop-loss hit (short) @ {price:.2f}")
                self.order = self.close()
            elif price <= midline:
                logger.info(f"MR target hit (short) — price reached midline @ {price:.2f}")
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
