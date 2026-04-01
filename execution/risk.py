"""
Risk Manager
=============
Controls position sizing, daily loss limits, and VIX-based scaling.
All live trades must pass through the RiskManager before execution.
"""
import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from config.settings import (
    TRADING_CAPITAL,
    MAX_DAILY_LOSS_PCT,
    PER_TRADE_RISK_PCT,
    MAX_OPEN_TRADES_PER_INSTRUMENT,
)

logger = logging.getLogger(__name__)


@dataclass
class TradeRequest:
    instrument: str
    direction: str          # "long" or "short"
    strategy: str
    entry_price: float
    stop_price: float
    lot_size: int
    position_size_multiplier: float = 1.0


@dataclass
class RiskDecision:
    approved: bool
    quantity: int
    reason: str = ""


class RiskManager:
    """
    Enforces risk rules before any order is submitted to the broker.

    Rules:
      1. Daily loss cap: halt trading if daily P&L < -2% of capital
      2. Per-trade risk: position size derived from 0.5% capital risk
      3. Max 1 open trade per instrument
      4. VIX-based position size multiplier
    """

    def __init__(
        self,
        capital: float = TRADING_CAPITAL,
        max_daily_loss_pct: float = MAX_DAILY_LOSS_PCT,
        per_trade_risk_pct: float = PER_TRADE_RISK_PCT,
    ):
        self.capital = capital
        self.max_daily_loss_pct = max_daily_loss_pct
        self.per_trade_risk_pct = per_trade_risk_pct

        self._today: Optional[date] = None
        self._daily_pnl: float = 0.0
        self._trading_halted: bool = False
        self._open_instruments: dict = {}   # instrument → quantity

    def _reset_day(self):
        today = date.today()
        if self._today != today:
            logger.info(f"New trading day: {today} — resetting daily P&L and open trades")
            self._today = today
            self._daily_pnl = 0.0
            self._trading_halted = False
            self._open_instruments = {}

    def record_pnl(self, pnl: float, instrument: str):
        """Call after each trade closes to update daily P&L."""
        self._daily_pnl += pnl
        if instrument in self._open_instruments:
            del self._open_instruments[instrument]

        loss_limit = -self.capital * self.max_daily_loss_pct
        if self._daily_pnl <= loss_limit and not self._trading_halted:
            self._trading_halted = True
            logger.warning(
                f"DAILY LOSS LIMIT HIT: P&L=₹{self._daily_pnl:.2f} "
                f"(limit=₹{loss_limit:.2f}) — TRADING HALTED FOR TODAY"
            )

    def record_open_trade(self, instrument: str, quantity: int):
        self._open_instruments[instrument] = quantity

    def evaluate(self, req: TradeRequest) -> RiskDecision:
        """
        Evaluate a trade request. Returns RiskDecision with approved=True/False.
        """
        self._reset_day()

        if self._trading_halted:
            return RiskDecision(approved=False, quantity=0, reason="Daily loss limit hit")

        # Max open trades per instrument
        if instrument_count := self._open_instruments.get(req.instrument, 0):
            return RiskDecision(
                approved=False,
                quantity=0,
                reason=f"Already have {instrument_count} open trade on {req.instrument}",
            )

        # Position sizing
        qty = self._calc_qty(req)
        if qty <= 0:
            return RiskDecision(approved=False, quantity=0, reason="Calculated quantity is 0")

        logger.info(
            f"RISK APPROVED: {req.direction.upper()} {req.instrument} "
            f"qty={qty} (mult={req.position_size_multiplier:.2f}) "
            f"daily_pnl=₹{self._daily_pnl:.2f}"
        )
        return RiskDecision(approved=True, quantity=qty)

    def _calc_qty(self, req: TradeRequest) -> int:
        """
        qty = (capital × per_trade_risk_pct × multiplier) / |entry - stop|
        Rounded down to nearest lot.
        """
        risk_amount = self.capital * self.per_trade_risk_pct * req.position_size_multiplier
        risk_per_unit = abs(req.entry_price - req.stop_price)
        if risk_per_unit < 0.01:
            return req.lot_size
        units = risk_amount / risk_per_unit
        lots = max(1, int(units / req.lot_size))
        return lots * req.lot_size

    @property
    def daily_pnl(self) -> float:
        return self._daily_pnl

    @property
    def is_halted(self) -> bool:
        self._reset_day()
        return self._trading_halted

    def status(self) -> dict:
        self._reset_day()
        return {
            "date": str(self._today),
            "daily_pnl": round(self._daily_pnl, 2),
            "trading_halted": self._trading_halted,
            "open_instruments": dict(self._open_instruments),
            "capital": self.capital,
            "daily_loss_limit": -round(self.capital * self.max_daily_loss_pct, 2),
        }
