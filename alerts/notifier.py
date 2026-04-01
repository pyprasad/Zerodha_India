"""
Alert and notification system.
Supports:
  - Rich console output (always on)
  - Telegram bot alerts (optional, configure in .env)
"""
import logging
from datetime import datetime
from typing import Optional

import requests
from rich.console import Console

from config.credentials import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)
console = Console()


class Notifier:
    """
    Multi-channel notifier for trade signals, entries, exits, and daily summaries.
    Console output is always active; Telegram is optional.
    """

    def __init__(self):
        self._telegram_enabled = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)
        if self._telegram_enabled:
            logger.info("Telegram alerts enabled")
        else:
            logger.info("Telegram alerts disabled (no BOT_TOKEN/CHAT_ID in .env)")

    def send(self, message: str, level: str = "info"):
        """Send a message to all configured channels."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        color_map = {"info": "cyan", "success": "green", "warning": "yellow", "error": "red"}
        color = color_map.get(level, "white")
        console.print(f"[{color}][{timestamp}] {message}[/{color}]")

        if self._telegram_enabled:
            self._send_telegram(f"[{timestamp}] {message}")

    def send_signal(self, instrument: str, strategy: str, direction: str, price: float):
        msg = f"SIGNAL: {direction.upper()} {instrument} @ ₹{price:.2f} [{strategy}]"
        self.send(msg, level="info")

    def send_trade_entry(self, instrument: str, direction: str, price: float, qty: int, strategy: str):
        msg = (
            f"TRADE ENTRY: {direction.upper()} {instrument}\n"
            f"  Price: ₹{price:.2f}  Qty: {qty}\n"
            f"  Strategy: {strategy}"
        )
        self.send(msg, level="success")

    def send_trade_exit(self, instrument: str, exit_price: float, pnl: float, reason: str):
        sign = "+" if pnl >= 0 else ""
        level = "success" if pnl >= 0 else "warning"
        msg = (
            f"TRADE EXIT: {instrument} @ ₹{exit_price:.2f}\n"
            f"  P&L: ₹{sign}{pnl:.2f}  Reason: {reason}"
        )
        self.send(msg, level=level)

    def send_daily_summary(self, tracker):
        """Send end-of-day summary (uses TradeTracker)."""
        trades = tracker.trades
        total_pnl = tracker.total_pnl
        n_trades = len(trades)
        wins = sum(1 for t in trades if (t.get("pnl") or 0) > 0)
        losses = n_trades - wins
        sign = "+" if total_pnl >= 0 else ""

        msg = (
            f"EOD SUMMARY — {datetime.now().strftime('%Y-%m-%d')}\n"
            f"  Total Trades: {n_trades}  (W:{wins} / L:{losses})\n"
            f"  Net P&L: ₹{sign}{total_pnl:.2f}"
        )
        level = "success" if total_pnl >= 0 else "warning"
        self.send(msg, level=level)

    def send_risk_alert(self, message: str):
        """High-priority risk alert (daily loss limit, margin call, etc.)."""
        self.send(f"RISK ALERT: {message}", level="error")

    # ------------------------------------------------------------------
    # Telegram
    # ------------------------------------------------------------------
    def _send_telegram(self, message: str):
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        try:
            resp = requests.post(
                url,
                data={"chat_id": TELEGRAM_CHAT_ID, "text": message},
                timeout=10,
            )
            if not resp.ok:
                logger.warning(f"Telegram send failed: {resp.text}")
        except Exception as e:
            logger.warning(f"Telegram error: {e}")
