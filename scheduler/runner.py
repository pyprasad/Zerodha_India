"""
Market hours scheduler and live trading runner.
Orchestrates the full daily trading cycle for NIFTY and BANKNIFTY.

Daily schedule (IST):
  08:55 — Auto-login to Zerodha
  09:00 — Fetch India VIX + PCR, build session context
  09:15 — Market open: start signal loop
  09:30 — ORB range locked, breakout watch begins
  15:00 — Stop new entries
  15:15 — Close all open positions
  15:30 — Cancel pending orders, generate daily report
  15:35 — Send daily summary alert
"""
import logging
import threading
import time
from datetime import datetime, time as dtime

import schedule

from config import credentials as creds
from config.settings import INSTRUMENTS, TRADING_MODE, INTRADAY_INTERVAL
from data.fetcher import KiteAuth, KiteDataFetcher
from execution.broker import get_broker
from execution.risk import RiskManager, TradeRequest
from execution.tracker import TradeTracker
from alerts.notifier import Notifier
from strategies.pcr_vix import build_session_context, filter_signal

logger = logging.getLogger(__name__)

# Active strategies registry (imported on demand to avoid circular imports)
STRATEGY_MAP = {
    "ORB": ("strategies.orb", "ORBStrategy"),
    "EMA_CROSSOVER": ("strategies.ema_crossover", "EMACrossoverStrategy"),
    "SUPERTREND": ("strategies.supertrend", "SuperTrendStrategy"),
    "MEAN_REVERSION": ("strategies.mean_reversion", "MeanReversionStrategy"),
}


def _ist_now() -> dtime:
    """Current time in IST (UTC+5:30)."""
    from datetime import timezone, timedelta
    ist = timezone(timedelta(hours=5, minutes=30))
    return datetime.now(ist).time()


def _is_trading_day() -> bool:
    """Returns True if today is a weekday (Mon-Fri). Does not check NSE holidays."""
    return datetime.today().weekday() < 5


class LiveRunner:
    """
    Orchestrates the daily automated trading cycle.
    Runs in a loop during market hours, polling quotes and checking signals.
    """

    SIGNAL_POLL_INTERVAL = 60   # seconds between signal checks

    def __init__(self):
        self.kite_auth = KiteAuth()
        self.kite = None
        self.fetcher = None
        self.broker = None
        self.risk_manager = RiskManager()
        self.tracker = TradeTracker()
        self.notifier = Notifier()
        self.session_ctx = None
        self._running = False
        self._strategy_instances: dict = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def login(self):
        logger.info("Logging in to Zerodha...")
        self.kite = self.kite_auth.login()
        self.fetcher = KiteDataFetcher(self.kite)
        self.broker = get_broker(self.kite)
        logger.info("Zerodha login successful")
        self.notifier.send("Zerodha login successful — system ready")

    def build_morning_context(self):
        """Fetch VIX and PCR at 9:00 AM to configure the day's trading."""
        logger.info("Fetching India VIX and PCR for morning context...")
        vix = self.fetcher.get_india_vix()
        pcr_nifty = self.fetcher.get_pcr("NIFTY")
        pcr_banknifty = self.fetcher.get_pcr("BANKNIFTY")
        # Use NIFTY PCR as the primary signal filter
        self.session_ctx = build_session_context(vix, pcr_nifty)
        self.notifier.send(
            f"Morning context: VIX={vix:.2f} PCR-N={pcr_nifty:.3f} "
            f"PCR-BN={pcr_banknifty:.3f} | bias={self.session_ctx.directional_bias} "
            f"strategies={self.session_ctx.active_strategies}"
        )

    def start_signal_loop(self):
        """Start polling for signals in a background thread."""
        self._running = True
        thread = threading.Thread(target=self._signal_loop, daemon=True, name="SignalLoop")
        thread.start()
        logger.info("Signal loop started")

    def stop_new_entries(self):
        """Called at 15:00 — disallow new entries."""
        logger.info("15:00 IST — stopping new entries")
        self._running = False

    def close_all_positions(self):
        """Called at 15:15 — close all open positions."""
        logger.info("15:15 IST — closing all open positions")
        if TRADING_MODE == "live":
            self.broker.close_all_positions()
        else:
            logger.info("[PAPER] Simulating position close at 15:15")

    def end_of_day(self):
        """Called at 15:30 — cancel pending orders and send daily report."""
        logger.info("15:30 IST — end of day cleanup")
        self.broker.cancel_all_open_orders()
        self.tracker.print_summary()
        self.notifier.send_daily_summary(self.tracker)

    # ------------------------------------------------------------------
    # Signal loop (runs every SIGNAL_POLL_INTERVAL seconds)
    # ------------------------------------------------------------------
    def _signal_loop(self):
        from strategies.orb import ORBStrategy
        from strategies.ema_crossover import EMACrossoverStrategy
        from strategies.supertrend import SuperTrendStrategy
        from strategies.mean_reversion import MeanReversionStrategy

        # Map strategy names to live signal generators
        signal_generators = {
            "ORB": ORBLiveSignal(self.fetcher, "NIFTY"),
            "EMA_CROSSOVER": EMALiveSignal(self.fetcher),
            "SUPERTREND": SuperTrendLiveSignal(self.fetcher),
            "MEAN_REVERSION": MeanReversionLiveSignal(self.fetcher),
        }

        while self._running:
            try:
                self._check_signals(signal_generators)
            except Exception as e:
                logger.error(f"Signal loop error: {e}", exc_info=True)
            time.sleep(self.SIGNAL_POLL_INTERVAL)

    def _check_signals(self, signal_generators: dict):
        """Check each active strategy for signals on each instrument."""
        if not self.session_ctx:
            return

        for instrument in ["NIFTY", "BANKNIFTY"]:
            for strategy_name in self.session_ctx.active_strategies:
                gen = signal_generators.get(strategy_name)
                if not gen:
                    continue

                signal = gen.get_signal(instrument)
                if not signal:
                    continue

                self.tracker.record_signal(instrument, strategy_name, signal)

                # Apply PCR/VIX filter
                if not filter_signal(signal, self.session_ctx):
                    logger.debug(f"Signal {signal} for {instrument} filtered by PCR/VIX bias")
                    continue

                # Get current price
                ltp = self.fetcher.get_ltp(instrument)
                stop_distance = ltp * 0.005   # 0.5% default stop

                if signal == "long":
                    stop = ltp - stop_distance
                else:
                    stop = ltp + stop_distance

                req = TradeRequest(
                    instrument=instrument,
                    direction=signal,
                    strategy=strategy_name,
                    entry_price=ltp,
                    stop_price=stop,
                    lot_size=INSTRUMENTS[instrument]["lot_size"],
                    position_size_multiplier=self.session_ctx.position_size_multiplier,
                )

                decision = self.risk_manager.evaluate(req)
                if not decision.approved:
                    logger.info(f"Trade rejected by risk: {decision.reason}")
                    continue

                # Place order
                result = self._place_trade(instrument, signal, ltp, decision.quantity, strategy_name)
                if result and result.status != "FAILED":
                    self.tracker.record_entry(
                        instrument=instrument,
                        strategy=strategy_name,
                        direction=signal,
                        entry_price=ltp,
                        quantity=decision.quantity,
                        order_id=result.order_id,
                    )
                    self.risk_manager.record_open_trade(instrument, decision.quantity)

    def _place_trade(self, instrument: str, direction: str, price: float, qty: int, strategy: str):
        """Submit order to broker."""
        inst_info = INSTRUMENTS[instrument]
        # For live futures trading on NFO exchange
        exchange = "NFO"
        # In real deployment, tradingsymbol should be the current month's futures contract
        # e.g., "NIFTY24DECFUT" — this must be resolved dynamically
        tradingsymbol = f"{instrument}FUT"  # Placeholder; see main.py for resolution

        from kiteconnect import KiteConnect
        tx_type = (
            self.kite.TRANSACTION_TYPE_BUY if direction == "long"
            else self.kite.TRANSACTION_TYPE_SELL
        ) if self.kite else "BUY" if direction == "long" else "SELL"

        return self.broker.place_order(
            tradingsymbol=tradingsymbol,
            exchange=exchange,
            transaction_type=tx_type,
            quantity=qty,
            tag=strategy[:10],
        )


# ------------------------------------------------------------------
# Live signal generators (fetch last N candles and compute signal)
# ------------------------------------------------------------------
class _LiveSignalBase:
    """Base for live signal generators that work on recent candle data."""

    def __init__(self, fetcher: KiteDataFetcher):
        self.fetcher = fetcher

    def get_signal(self, instrument: str) -> str:
        """Returns 'long', 'short', or '' (no signal)."""
        raise NotImplementedError


class ORBLiveSignal(_LiveSignalBase):
    def __init__(self, fetcher, primary_instrument: str = "NIFTY"):
        super().__init__(fetcher)
        self._orb_ranges: dict = {}

    def get_signal(self, instrument: str) -> str:
        from datetime import timedelta
        now = datetime.now()
        start = now.replace(hour=9, minute=15, second=0, microsecond=0)
        orb_end = now.replace(hour=9, minute=30, second=0, microsecond=0)

        # Lock ORB range
        if instrument not in self._orb_ranges:
            df = self.fetcher.get_historical(
                instrument,
                from_date=start,
                to_date=orb_end,
                interval="5minute",
            )
            if df.empty:
                return ""
            self._orb_ranges[instrument] = {
                "high": df["high"].max(),
                "low": df["low"].min(),
                "avg_vol": df["volume"].mean(),
                "date": now.date(),
            }

        orb = self._orb_ranges[instrument]
        if orb.get("date") != now.date():
            del self._orb_ranges[instrument]
            return ""

        ltp = self.fetcher.get_ltp(instrument)
        quote = self.fetcher.get_live_quote(instrument)
        vol = quote.get("last_quantity", 0)

        if ltp > orb["high"] and vol > orb["avg_vol"] * 1.5:
            return "long"
        if ltp < orb["low"] and vol > orb["avg_vol"] * 1.5:
            return "short"
        return ""


class EMALiveSignal(_LiveSignalBase):
    def get_signal(self, instrument: str) -> str:
        from datetime import timedelta
        interval = "5minute" if instrument == "BANKNIFTY" else "15minute"
        df = self.fetcher.get_historical(
            instrument,
            from_date=datetime.now() - timedelta(hours=6),
            to_date=datetime.now(),
            interval=interval,
        )
        if len(df) < 55:
            return ""
        close = df["close"]
        ema9 = close.ewm(span=9, adjust=False).mean()
        ema21 = close.ewm(span=21, adjust=False).mean()
        ema55 = close.ewm(span=55, adjust=False).mean()
        vwap_val = ((df["high"] + df["low"] + df["close"]) / 3 * df["volume"]).sum() / df["volume"].sum()

        cross_up = ema9.iloc[-1] > ema21.iloc[-1] and ema9.iloc[-2] <= ema21.iloc[-2]
        cross_dn = ema9.iloc[-1] < ema21.iloc[-1] and ema9.iloc[-2] >= ema21.iloc[-2]
        price = close.iloc[-1]

        if cross_up and price > vwap_val and price > ema55.iloc[-1]:
            return "long"
        if cross_dn and price < vwap_val and price < ema55.iloc[-1]:
            return "short"
        return ""


class SuperTrendLiveSignal(_LiveSignalBase):
    def get_signal(self, instrument: str) -> str:
        from datetime import timedelta
        df = self.fetcher.get_historical(
            instrument,
            from_date=datetime.now() - timedelta(hours=8),
            to_date=datetime.now(),
            interval="15minute",
        )
        if len(df) < 20:
            return ""
        from backtest.optimizer import _calc_supertrend
        direction = _calc_supertrend(df["high"], df["low"], df["close"])
        if direction.iloc[-1] == 1.0 and direction.iloc[-2] == -1.0:
            return "long"
        if direction.iloc[-1] == -1.0 and direction.iloc[-2] == 1.0:
            return "short"
        return ""


class MeanReversionLiveSignal(_LiveSignalBase):
    def get_signal(self, instrument: str) -> str:
        from datetime import timedelta
        df = self.fetcher.get_historical(
            instrument,
            from_date=datetime.now() - timedelta(hours=8),
            to_date=datetime.now(),
            interval="15minute",
        )
        if len(df) < 20:
            return ""
        close = df["close"]
        sma = close.rolling(20).mean()
        std = close.rolling(20).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        price = close.iloc[-1]
        if price <= lower.iloc[-1] and rsi.iloc[-1] < 30:
            return "long"
        if price >= upper.iloc[-1] and rsi.iloc[-1] > 70:
            return "short"
        return ""


# ------------------------------------------------------------------
# Scheduler setup
# ------------------------------------------------------------------
def setup_schedule(runner: LiveRunner):
    """Register all daily jobs."""
    schedule.every().day.at("08:55").do(runner.login)
    schedule.every().day.at("09:00").do(runner.build_morning_context)
    schedule.every().day.at("09:15").do(runner.start_signal_loop)
    schedule.every().day.at("15:00").do(runner.stop_new_entries)
    schedule.every().day.at("15:15").do(runner.close_all_positions)
    schedule.every().day.at("15:30").do(runner.end_of_day)
    logger.info("Daily schedule registered")


def run_live():
    """Main entry point for live/paper trading."""
    creds.validate()
    runner = LiveRunner()
    setup_schedule(runner)
    logger.info("Scheduler running. Waiting for market hours...")

    while True:
        schedule.run_pending()
        time.sleep(1)
