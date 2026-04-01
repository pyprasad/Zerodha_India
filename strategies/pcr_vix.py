"""
PCR + India VIX Overlay
=========================
Not a standalone trading strategy — a signal qualifier/filter applied at session start.
Determines:
  1. Directional bias (bullish / bearish / neutral) from PCR
  2. Position size multiplier and active strategy set from VIX
"""
import logging
from dataclasses import dataclass
from typing import List

from config.settings import PCR_VIX, get_active_strategies, get_position_size_multiplier

logger = logging.getLogger(__name__)


@dataclass
class SessionContext:
    """Morning session context set before market open."""
    vix: float
    pcr: float
    directional_bias: str          # "bullish", "bearish", "neutral"
    position_size_multiplier: float
    active_strategies: List[str]
    premium_selling_mode: bool     # True when VIX > 25


def build_session_context(vix: float, pcr: float) -> SessionContext:
    """
    Build the morning session context from VIX and PCR readings.

    PCR rules:
      < 0.6  → contrarian bullish (market oversold on puts, expect bounce)
      > 1.7  → contrarian bearish (market over-hedged, likely pullback)
      else   → neutral

    VIX rules:
      < 12   → ORB only, full size
      12-20  → ORB + EMA Crossover, full size
      20-25  → Mean Reversion + SuperTrend, half size
      > 25   → Mean Reversion + SuperTrend, quarter size, premium-selling mode
    """
    # Directional bias from PCR
    if pcr < PCR_VIX["pcr_bullish_threshold"]:
        bias = "bullish"
    elif pcr > PCR_VIX["pcr_bearish_threshold"]:
        bias = "bearish"
    else:
        bias = "neutral"

    multiplier = get_position_size_multiplier(vix)
    strategies = get_active_strategies(vix)
    premium_mode = vix > PCR_VIX["vix_extreme"]

    ctx = SessionContext(
        vix=vix,
        pcr=pcr,
        directional_bias=bias,
        position_size_multiplier=multiplier,
        active_strategies=strategies,
        premium_selling_mode=premium_mode,
    )

    logger.info(
        f"Session context: VIX={vix:.2f} PCR={pcr:.3f} "
        f"bias={bias} size_mult={multiplier:.2f} "
        f"strategies={strategies} premium_mode={premium_mode}"
    )
    return ctx


def filter_signal(signal_direction: str, ctx: SessionContext) -> bool:
    """
    Check whether a trade direction is allowed given the session context.

    Args:
        signal_direction: "long" or "short"
        ctx: SessionContext from build_session_context()

    Returns:
        True if the signal is consistent with PCR/VIX bias.
    """
    if ctx.directional_bias == "neutral":
        return True  # No restriction

    if signal_direction == "long" and ctx.directional_bias == "bearish":
        logger.debug("Long signal filtered out by bearish PCR bias")
        return False

    if signal_direction == "short" and ctx.directional_bias == "bullish":
        logger.debug("Short signal filtered out by bullish PCR bias")
        return False

    return True
