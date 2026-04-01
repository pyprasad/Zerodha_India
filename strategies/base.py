"""
Abstract base class for all Backtrader-based strategies.
Provides shared utilities: VWAP calculation, trade logging, market hours check.
"""
import logging
from datetime import time as dtime

import backtrader as bt
import pandas as pd

logger = logging.getLogger(__name__)

MARKET_OPEN = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)
STRATEGY_END = dtime(15, 0)   # No new entries after 15:00


class BaseStrategy(bt.Strategy):
    """
    Common base for all NIFTY/BANKNIFTY strategies.
    Subclasses must implement: signal_long(), signal_short(), and set params.
    """

    params = (
        ("instrument", "NIFTY"),
        ("stoploss_pct", 0.005),
        ("target_pct", 0.015),
        ("rr_ratio", 2.0),
        ("position_size_multiplier", 1.0),  # Set by PCR/VIX overlay at runtime
        ("lot_size", 25),
        ("capital", 100000),
        ("per_trade_risk_pct", 0.005),
    )

    def __init__(self):
        self.order = None
        self.entry_price = None
        self.stop_price = None
        self.target_price = None
        self.trade_log = []
        self._vwap = None

    # ------------------------------------------------------------------
    # Market hours helpers
    # ------------------------------------------------------------------
    def _current_time(self) -> dtime:
        return self.datas[0].datetime.time()

    def _is_market_open(self) -> bool:
        t = self._current_time()
        return MARKET_OPEN <= t <= MARKET_CLOSE

    def _can_enter(self) -> bool:
        """No new entries after STRATEGY_END time."""
        return self._current_time() < STRATEGY_END

    def _must_exit(self) -> bool:
        """Force-close all positions at STRATEGY_END."""
        return self._current_time() >= STRATEGY_END

    # ------------------------------------------------------------------
    # VWAP (calculated as cumulative sum within the session)
    # ------------------------------------------------------------------
    def _calculate_vwap(self) -> float:
        """
        Compute session VWAP from the bar data available so far today.
        VWAP = Σ(typical_price × volume) / Σ(volume)
        """
        df = pd.DataFrame({
            "close": list(self.datas[0].close.array),
            "high": list(self.datas[0].high.array),
            "low": list(self.datas[0].low.array),
            "volume": list(self.datas[0].volume.array),
            "datetime": [bt.num2date(d) for d in self.datas[0].datetime.array],
        })
        if df.empty:
            return float(self.datas[0].close[0])

        today = df["datetime"].iloc[-1].date()
        today_df = df[df["datetime"].apply(lambda x: x.date()) == today].copy()
        if today_df.empty:
            return float(self.datas[0].close[0])

        today_df["typical"] = (today_df["high"] + today_df["low"] + today_df["close"]) / 3
        total_vol = today_df["volume"].sum()
        if total_vol == 0:
            return float(today_df["close"].iloc[-1])
        return (today_df["typical"] * today_df["volume"]).sum() / total_vol

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------
    def _calc_qty(self, entry: float, stop: float) -> int:
        """
        Risk-based position sizing.
        qty = (capital × per_trade_risk_pct) / (entry - stop) / lot_size  × lots
        Returns number of lots (minimum 1).
        """
        risk_amount = self.p.capital * self.p.per_trade_risk_pct * self.p.position_size_multiplier
        risk_per_unit = abs(entry - stop)
        if risk_per_unit == 0:
            return self.p.lot_size
        units = risk_amount / risk_per_unit
        lots = max(1, int(units / self.p.lot_size))
        return lots * self.p.lot_size

    # ------------------------------------------------------------------
    # Order lifecycle
    # ------------------------------------------------------------------
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            if order.isbuy():
                logger.info(
                    f"BUY  {self.p.instrument} @ {order.executed.price:.2f} "
                    f"qty={order.executed.size} cost={order.executed.value:.2f}"
                )
                self.entry_price = order.executed.price
            elif order.issell():
                logger.info(
                    f"SELL {self.p.instrument} @ {order.executed.price:.2f} "
                    f"qty={order.executed.size}"
                )
        elif order.status in [order.Canceled, order.Rejected, order.Margin]:
            logger.warning(f"Order {order.status}: {order.info}")

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        pnl = round(trade.pnl, 2)
        logger.info(f"TRADE CLOSED | PnL: ₹{pnl:+.2f}")
        self.trade_log.append({
            "instrument": self.p.instrument,
            "strategy": self.__class__.__name__,
            "entry": trade.price,
            "exit": trade.price + trade.pnl / (trade.size or 1),
            "pnl": pnl,
            "size": trade.size,
        })

    # ------------------------------------------------------------------
    # Force-close at end of day
    # ------------------------------------------------------------------
    def _force_close(self):
        if self.position:
            logger.info(f"Force-closing {self.p.instrument} position at market close")
            self.close()

    # ------------------------------------------------------------------
    # Abstract interface (subclasses must implement)
    # ------------------------------------------------------------------
    def signal_long(self) -> bool:
        raise NotImplementedError

    def signal_short(self) -> bool:
        raise NotImplementedError

    def next(self):
        if self.order:
            return
        if not self._is_market_open():
            return
        if self._must_exit():
            self._force_close()
            return

        if not self.position:
            if self._can_enter():
                if self.signal_long():
                    self._enter_long()
                elif self.signal_short():
                    self._enter_short()
        else:
            self._manage_position()

    def _enter_long(self):
        entry = self.datas[0].close[0]
        stop = entry * (1 - self.p.stoploss_pct)
        target = entry * (1 + self.p.target_pct)
        qty = self._calc_qty(entry, stop)
        self.stop_price = stop
        self.target_price = target
        logger.info(f"LONG signal {self.p.instrument} entry={entry:.2f} stop={stop:.2f} target={target:.2f}")
        self.order = self.buy(size=qty)

    def _enter_short(self):
        entry = self.datas[0].close[0]
        stop = entry * (1 + self.p.stoploss_pct)
        target = entry * (1 - self.p.target_pct)
        qty = self._calc_qty(entry, stop)
        self.stop_price = stop
        self.target_price = target
        logger.info(f"SHORT signal {self.p.instrument} entry={entry:.2f} stop={stop:.2f} target={target:.2f}")
        self.order = self.sell(size=qty)

    def _manage_position(self):
        """Check stop-loss and target for open positions."""
        price = self.datas[0].close[0]
        if self.position.size > 0:  # Long
            if price <= self.stop_price:
                logger.info(f"Stop-loss hit (long) @ {price:.2f}")
                self.order = self.close()
            elif price >= self.target_price:
                logger.info(f"Target hit (long) @ {price:.2f}")
                self.order = self.close()
        elif self.position.size < 0:  # Short
            if price >= self.stop_price:
                logger.info(f"Stop-loss hit (short) @ {price:.2f}")
                self.order = self.close()
            elif price <= self.target_price:
                logger.info(f"Target hit (short) @ {price:.2f}")
                self.order = self.close()
