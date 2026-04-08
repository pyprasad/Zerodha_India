"""
Central configuration for all strategies, instruments, and risk parameters.
All values can be overridden via .env file.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Instruments
# ---------------------------------------------------------------------------
INSTRUMENTS = {
    "NIFTY": {
        "token": 256265,
        "exchange": "NSE",
        "symbol": "NIFTY 50",
        "futures_symbol": "NFO:NIFTY",
        "lot_size": 25,
        "tick_size": 0.05,
    },
    "BANKNIFTY": {
        "token": 260105,
        "exchange": "NSE",
        "symbol": "NIFTY BANK",
        "futures_symbol": "NFO:BANKNIFTY",
        "lot_size": 15,
        "tick_size": 0.05,
    },
    "INDIA_VIX": {
        "token": 264969,
        "exchange": "NSE",
        "symbol": "INDIA VIX",
    },
    # v6 strategy instruments (added for Kite historical data fetch)
    # Tokens are NSE spot index tokens — verify after login with:
    #   kite.instruments("NSE") and search by name
    "FINNIFTY": {
        "token": 257801,          # NIFTY FIN SERVICE spot index — verify after login
        "exchange": "NSE",
        "symbol": "NIFTY FIN SERVICE",
        "futures_symbol": "NFO:FINNIFTY",
        "lot_size": 60,
        "tick_size": 0.05,
    },
    "MIDCPNIFTY": {
        "token": 288009,          # NIFTY MIDCAP SELECT spot index — verify after login
        "exchange": "NSE",
        "symbol": "NIFTY MIDCAP SELECT",
        "futures_symbol": "NFO:MIDCPNIFTY",
        "lot_size": 120,
        "tick_size": 0.05,
    },
    # INDIAVIX alias (same token as INDIA_VIX) — used by fetch_kite_daily.py
    # to store data under the key v6_backtest.py expects ("INDIAVIX")
    "INDIAVIX": {
        "token": 264969,
        "exchange": "NSE",
        "symbol": "INDIA VIX",
    },

    # ── Phase 1 expansion instrument ─────────────────────────────────────────
    # NIFTYIT: Nifty IT sector index — correlation ~0.65 with NIFTY (best diversifier)
    # F&O available since 2001, lot size 30, already in v6_backtest.py LOT_SIZES
    # Token: verify after login with the lookup command in README
    # Run: python3 -c "from data.fetcher import KiteAuth; k=KiteAuth().login();
    #       [print(i['instrument_token'],i['tradingsymbol'],i['name'])
    #        for i in k.instruments('NSE') if 'NIFTY IT' in i.get('name','')]"
    "NIFTYIT": {
        "token": 259849,              # NIFTY IT spot index — verified 2026-04-08
        "exchange": "NSE",
        "symbol": "NIFTY IT",
        "futures_symbol": "NFO:NIFTYIT",
        "lot_size": 30,
        "tick_size": 0.05,
    },

    # ── Phase 2 expansion instrument ─────────────────────────────────────────
    # NIFTYNXT50: Nifty Next 50 index — tracks rank 51-100 stocks by market cap
    # F&O launched ~2022, decent liquidity, slightly different cycle from NIFTY 50
    # Token: verify after login
    # Run: python3 -c "from data.fetcher import KiteAuth; k=KiteAuth().login();
    #       [print(i['instrument_token'],i['tradingsymbol'],i['name'])
    #        for i in k.instruments('NSE') if 'NEXT 50' in i.get('name','') or 'NIFTY NEXT' in i.get('tradingsymbol','')]"
    "NIFTYNXT50": {
        "token": None,                # TODO: fill in after running lookup above
        "exchange": "NSE",
        "symbol": "NIFTY NEXT 50",
        "futures_symbol": "NFO:NIFTYNXT50",
        "lot_size": 25,
        "tick_size": 0.05,
    },
}

# ---------------------------------------------------------------------------
# Market hours (IST)
# ---------------------------------------------------------------------------
MARKET_OPEN = "09:15"
MARKET_CLOSE = "15:30"
ORB_WINDOW_END = "09:30"       # Opening range is first 15 minutes
STRATEGY_END = "15:00"          # No new entries after this time
PRE_MARKET_LOGIN = "08:55"

# ---------------------------------------------------------------------------
# Historical data
# ---------------------------------------------------------------------------
HISTORICAL_DAYS = 1095          # 3 years
INTRADAY_INTERVAL = "5minute"   # Primary interval for intraday strategies
DAILY_INTERVAL = "day"

# ---------------------------------------------------------------------------
# Risk management
# ---------------------------------------------------------------------------
TRADING_CAPITAL = float(os.getenv("TRADING_CAPITAL", 100000))
MAX_DAILY_LOSS_PCT = float(os.getenv("MAX_DAILY_LOSS_PCT", 0.02))    # 2%
PER_TRADE_RISK_PCT = float(os.getenv("PER_TRADE_RISK_PCT", 0.005))  # 0.5%
MAX_OPEN_TRADES_PER_INSTRUMENT = 1

# Zerodha brokerage model (for backtesting cost simulation)
BROKERAGE_PER_ORDER = 20        # ₹20 flat per order
SLIPPAGE_PCT = 0.0005           # 0.05% slippage
EXCHANGE_FEES_PCT = 0.0000125   # STT + exchange charges

# ---------------------------------------------------------------------------
# Strategy: Opening Range Breakout (ORB)
# ---------------------------------------------------------------------------
ORB = {
    "volume_multiplier": 1.5,       # Entry requires volume > 1.5x average
    "rr_ratio": 2.0,                # Risk:Reward = 1:2
    "instruments": ["NIFTY", "BANKNIFTY"],
    "interval": "5minute",
}

# ---------------------------------------------------------------------------
# Strategy: EMA Crossover + VWAP
# ---------------------------------------------------------------------------
EMA_CROSSOVER = {
    "fast_ema": 9,
    "slow_ema": 21,
    "trend_ema": 55,
    "stoploss_pct": 0.005,          # 0.5%
    "target_pct": 0.015,            # 1.5%
    "intervals": {
        "NIFTY": "15minute",
        "BANKNIFTY": "5minute",
    },
}

# ---------------------------------------------------------------------------
# Strategy: SuperTrend
# ---------------------------------------------------------------------------
SUPERTREND = {
    "atr_period": 10,
    "multiplier": 3.0,
    "trend_ema": 55,
    "interval": "15minute",
}

# ---------------------------------------------------------------------------
# Strategy: Mean Reversion (Bollinger Bands + RSI)
# ---------------------------------------------------------------------------
MEAN_REVERSION = {
    "bb_period": 20,
    "bb_std": 2.0,
    "rsi_period": 14,
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "interval": "15minute",
}

# ---------------------------------------------------------------------------
# Strategy: PCR + VIX Overlay (signal filter, not standalone)
# ---------------------------------------------------------------------------
PCR_VIX = {
    "pcr_bullish_threshold": 0.6,   # PCR < 0.6 → contrarian bullish
    "pcr_bearish_threshold": 1.7,   # PCR > 1.7 → contrarian bearish
    "vix_low": 12,                  # VIX < 12: full size, ORB only
    "vix_normal_low": 15,           # VIX 15-20: full size, all strategies
    "vix_normal_high": 20,          # VIX > 20: half size, mean reversion preferred
    "vix_extreme": 25,              # VIX > 25: quarter size, premium-selling mode
}

# ---------------------------------------------------------------------------
# VIX-based strategy selection
# ---------------------------------------------------------------------------
def get_active_strategies(vix: float) -> list:
    if vix < PCR_VIX["vix_low"]:
        return ["ORB"]
    elif vix <= PCR_VIX["vix_normal_high"]:
        return ["ORB", "EMA_CROSSOVER"]
    else:
        return ["MEAN_REVERSION", "SUPERTREND"]


def get_position_size_multiplier(vix: float) -> float:
    if vix > PCR_VIX["vix_extreme"]:
        return 0.25
    elif vix > PCR_VIX["vix_normal_high"]:
        return 0.5
    else:
        return 1.0


# ---------------------------------------------------------------------------
# Trading mode
# ---------------------------------------------------------------------------
TRADING_MODE = os.getenv("TRADING_MODE", "paper")   # "paper" or "live"
