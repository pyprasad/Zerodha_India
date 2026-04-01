"""
Zerodha Kite Connect order manager.
Wraps pykiteconnect for order placement, cancellation, and position management.
Supports both paper trading (simulation) and live trading modes.
"""
import logging
import time
from typing import Optional

from kiteconnect import KiteConnect

from config import credentials as creds
from config.settings import INSTRUMENTS, TRADING_MODE

logger = logging.getLogger(__name__)


class OrderResult:
    def __init__(self, order_id: str, status: str, message: str = ""):
        self.order_id = order_id
        self.status = status
        self.message = message

    def __repr__(self):
        return f"OrderResult(id={self.order_id}, status={self.status})"


class PaperBroker:
    """
    Simulated broker for paper trading mode.
    Logs all intended orders without sending them to Zerodha.
    """

    def __init__(self):
        self._order_counter = 0
        logger.info("[PAPER MODE] PaperBroker initialized — no real orders will be placed")

    def _next_id(self) -> str:
        self._order_counter += 1
        return f"PAPER_{self._order_counter:06d}"

    def place_order(self, **kwargs) -> OrderResult:
        oid = self._next_id()
        logger.info(f"[PAPER] ORDER PLACED: {kwargs} → {oid}")
        return OrderResult(order_id=oid, status="COMPLETE", message="Paper order simulated")

    def cancel_order(self, order_id: str, variety: str = "regular") -> OrderResult:
        logger.info(f"[PAPER] CANCEL ORDER: {order_id}")
        return OrderResult(order_id=order_id, status="CANCELLED")

    def get_positions(self) -> dict:
        return {"net": [], "day": []}

    def get_orders(self) -> list:
        return []

    def cancel_all_open_orders(self):
        logger.info("[PAPER] Cancel all open orders (no-op in paper mode)")


class LiveBroker:
    """
    Live Zerodha order execution via Kite Connect.
    Rate limits enforced: 10 req/sec, 200 orders/min.
    """
    _RATE_LIMIT_SLEEP = 0.12   # ~8 req/sec to stay safely under 10/sec limit

    def __init__(self, kite: KiteConnect):
        self.kite = kite

    def _rate_limited(self, fn, *args, **kwargs):
        """Wrap a kite call with rate limit sleep."""
        result = fn(*args, **kwargs)
        time.sleep(self._RATE_LIMIT_SLEEP)
        return result

    def place_order(
        self,
        tradingsymbol: str,
        exchange: str,
        transaction_type: str,   # kite.TRANSACTION_TYPE_BUY / SELL
        quantity: int,
        order_type: str = None,  # kite.ORDER_TYPE_MARKET / LIMIT
        price: float = 0,
        trigger_price: float = 0,
        variety: str = None,     # kite.VARIETY_REGULAR / CO / BO
        stoploss: float = 0,
        squareoff: float = 0,
        product: str = None,     # kite.PRODUCT_MIS / CNC / NRML
        validity: str = None,
        tag: str = "auto",
    ) -> OrderResult:
        """Place a single order on Zerodha."""
        if order_type is None:
            order_type = self.kite.ORDER_TYPE_MARKET
        if variety is None:
            variety = self.kite.VARIETY_REGULAR
        if product is None:
            product = self.kite.PRODUCT_MIS    # Intraday
        if validity is None:
            validity = self.kite.VALIDITY_DAY

        kwargs = dict(
            tradingsymbol=tradingsymbol,
            exchange=exchange,
            transaction_type=transaction_type,
            quantity=quantity,
            order_type=order_type,
            product=product,
            validity=validity,
            tag=tag,
        )
        if variety in (self.kite.VARIETY_CO, self.kite.VARIETY_BO):
            kwargs["trigger_price"] = trigger_price
            kwargs["stoploss"] = stoploss
            kwargs["squareoff"] = squareoff
        if order_type == self.kite.ORDER_TYPE_LIMIT:
            kwargs["price"] = price

        try:
            order_id = self._rate_limited(self.kite.place_order, variety=variety, **kwargs)
            logger.info(
                f"ORDER PLACED: {transaction_type} {quantity} {tradingsymbol} "
                f"type={order_type} variety={variety} → id={order_id}"
            )
            return OrderResult(order_id=str(order_id), status="PENDING")
        except Exception as e:
            logger.error(f"Order placement failed: {e}")
            return OrderResult(order_id="", status="FAILED", message=str(e))

    def place_cover_order(
        self,
        tradingsymbol: str,
        exchange: str,
        transaction_type: str,
        quantity: int,
        price: float,
        trigger_price: float,
        product: str = None,
    ) -> OrderResult:
        """Place a Cover Order (CO) — market entry with mandatory stop-loss trigger."""
        if product is None:
            product = self.kite.PRODUCT_MIS
        return self.place_order(
            tradingsymbol=tradingsymbol,
            exchange=exchange,
            transaction_type=transaction_type,
            quantity=quantity,
            order_type=self.kite.ORDER_TYPE_MARKET,
            trigger_price=trigger_price,
            variety=self.kite.VARIETY_CO,
            product=product,
        )

    def cancel_order(self, order_id: str, variety: str = "regular") -> OrderResult:
        try:
            self._rate_limited(self.kite.cancel_order, variety=variety, order_id=order_id)
            logger.info(f"Order cancelled: {order_id}")
            return OrderResult(order_id=order_id, status="CANCELLED")
        except Exception as e:
            logger.error(f"Cancel failed for {order_id}: {e}")
            return OrderResult(order_id=order_id, status="FAILED", message=str(e))

    def get_positions(self) -> dict:
        return self._rate_limited(self.kite.positions)

    def get_orders(self) -> list:
        return self._rate_limited(self.kite.orders)

    def cancel_all_open_orders(self):
        """Cancel all open orders — called at end-of-day cleanup."""
        try:
            orders = self.get_orders()
            open_orders = [o for o in orders if o["status"] in ("OPEN", "TRIGGER PENDING")]
            logger.info(f"Cancelling {len(open_orders)} open orders")
            for o in open_orders:
                self.cancel_order(o["order_id"], variety=o.get("variety", "regular"))
        except Exception as e:
            logger.error(f"cancel_all_open_orders failed: {e}")

    def close_all_positions(self):
        """Market-close all open intraday positions — EOD safety net."""
        try:
            positions = self.get_positions()
            net = positions.get("net", [])
            for pos in net:
                if pos["quantity"] == 0:
                    continue
                direction = (
                    self.kite.TRANSACTION_TYPE_SELL
                    if pos["quantity"] > 0
                    else self.kite.TRANSACTION_TYPE_BUY
                )
                self.place_order(
                    tradingsymbol=pos["tradingsymbol"],
                    exchange=pos["exchange"],
                    transaction_type=direction,
                    quantity=abs(pos["quantity"]),
                    order_type=self.kite.ORDER_TYPE_MARKET,
                    tag="eod_close",
                )
        except Exception as e:
            logger.error(f"close_all_positions failed: {e}")


def get_broker(kite: Optional[KiteConnect] = None):
    """Factory: return PaperBroker or LiveBroker depending on TRADING_MODE."""
    if TRADING_MODE == "live":
        if kite is None:
            raise ValueError("kite instance required for live mode")
        logger.info("Using LiveBroker (REAL ORDERS WILL BE PLACED)")
        return LiveBroker(kite)
    else:
        return PaperBroker()
