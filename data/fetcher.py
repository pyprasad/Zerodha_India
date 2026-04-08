"""
Zerodha Kite Connect data fetcher.
Handles authentication, historical OHLCV data, live quotes, PCR, and India VIX.
"""
import time
import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import pyotp
import requests
from kiteconnect import KiteConnect

from config import credentials as creds
from config.settings import INSTRUMENTS

logger = logging.getLogger(__name__)


class KiteAuth:
    """
    Handles automated daily login to Zerodha Kite.
    Uses pyotp for TOTP-based 2FA — no manual browser interaction required.
    """

    LOGIN_URL  = "https://kite.zerodha.com/api/login"
    TWOFA_URL  = "https://kite.zerodha.com/api/twofa"
    CONNECT_LOGIN_URL = "https://kite.zerodha.com/connect/login"

    def __init__(self):
        self.kite = KiteConnect(api_key=creds.KITE_API_KEY)
        self._access_token: Optional[str] = None

    def login(self) -> KiteConnect:
        """
        Full automated login flow:
        1. POST username+password to get request_id
        2. POST TOTP + request_id to complete 2FA and get request_token
        3. Exchange request_token for access_token via Kite SDK
        """
        session = requests.Session()
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Content-Type": "application/x-www-form-urlencoded",
        }

        # Step 1: Password login
        resp = session.post(
            self.LOGIN_URL,
            data={
                "user_id": creds.ZERODHA_USER_ID,
                "password": creds.ZERODHA_PASSWORD,
            },
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "success":
            raise RuntimeError(f"Kite login step 1 failed: {data}")
        request_id = data["data"]["request_id"]
        logger.info("Kite login step 1 successful, got request_id")

        # Step 2: TOTP 2FA
        totp_code = pyotp.TOTP(creds.ZERODHA_TOTP_SECRET).now()
        resp2 = session.post(
            self.TWOFA_URL,
            data={
                "user_id": creds.ZERODHA_USER_ID,
                "request_id": request_id,
                "twofa_value": totp_code,
                "twofa_type": "totp",
                "skip_session": "",
            },
            headers=headers,
        )
        resp2.raise_for_status()
        data2 = resp2.json()
        if data2.get("status") != "success":
            raise RuntimeError(f"Kite login step 2 (TOTP) failed: {data2}")
        logger.info("Kite TOTP 2FA successful")

        # Step 3: Extract request_token
        # First try: token may be directly in the 2FA response body
        request_token = data2["data"].get("request_token")

        # Second try: token in a redirect_url field in the response
        if not request_token:
            redirect_url_field = data2.get("data", {}).get("redirect_url", "")
            if "request_token=" in redirect_url_field:
                request_token = redirect_url_field.split("request_token=")[1].split("&")[0]

        # Third try: manually follow the OAuth redirect chain until we reach
        # redirect_url (http://127.0.0.1) which contains the request_token.
        # We stop before connecting to 127.0.0.1 — no server runs there, that's expected.
        if not request_token:
            connect_url = f"{self.CONNECT_LOGIN_URL}?v=3&api_key={creds.KITE_API_KEY}"
            next_url = connect_url
            for _ in range(10):   # follow up to 10 hops
                try:
                    r = session.get(next_url, allow_redirects=False)
                    location = r.headers.get("Location", "")
                    logger.info(f"redirect hop: status={r.status_code} Location={location[:120]}")
                    if "request_token=" in location:
                        request_token = location.split("request_token=")[1].split("&")[0]
                        logger.info("Extracted request_token from redirect Location header")
                        break
                    if not location or r.status_code not in (301, 302, 303, 307, 308):
                        break
                    next_url = location
                except Exception as hop_err:
                    # ConnectionError to 127.0.0.1 is expected — token is in the attempted URL
                    hop_str = str(hop_err)
                    logger.info(f"redirect hop exception (expected for 127.0.0.1): {hop_str[:200]}")
                    if "request_token=" in hop_str:
                        part = hop_str.split("request_token=")[1]
                        request_token = part.split("&")[0].split('"')[0].split("'")[0].split(")")[0]
                        logger.info("Extracted request_token from connection error URL")
                    break

        if not request_token:
            raise RuntimeError(
                "Could not extract request_token from Kite login flow.\n"
                "Fix: Go to developers.kite.trade → your app → set Redirect URL to "
                "http://127.0.0.1 and save."
            )

        # Step 4: Generate access_token
        session_data = self.kite.generate_session(request_token, api_secret=creds.KITE_API_SECRET)
        self._access_token = session_data["access_token"]
        self.kite.set_access_token(self._access_token)
        logger.info("Kite access_token generated successfully")
        return self.kite

    def set_access_token(self, access_token: str) -> KiteConnect:
        """Use a pre-existing access token (e.g., from manual login)."""
        self._access_token = access_token
        self.kite.set_access_token(access_token)
        return self.kite

    @property
    def access_token(self) -> Optional[str]:
        return self._access_token


class KiteDataFetcher:
    """
    Fetches market data from Zerodha Kite Connect API.
    Provides historical OHLCV, live quotes, India VIX, and PCR.
    """

    NSE_OPTION_CHAIN_URL = "https://www.nseindia.com/api/option-chain-indices"

    def __init__(self, kite: KiteConnect):
        self.kite = kite

    # ------------------------------------------------------------------
    # Historical OHLCV
    # ------------------------------------------------------------------
    def get_historical(
        self,
        instrument: str,
        from_date: datetime,
        to_date: datetime,
        interval: str = "5minute",
        continuous: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candles for an instrument.

        Args:
            instrument: One of "NIFTY", "BANKNIFTY", "INDIA_VIX"
            from_date: Start datetime
            to_date: End datetime
            interval: One of minute/3minute/5minute/10minute/15minute/30minute/60minute/day
            continuous: True for continuous futures data

        Returns:
            DataFrame with columns: date, open, high, low, close, volume
        """
        token = INSTRUMENTS[instrument]["token"]
        records = []

        # Kite API has a max window per request (~60 days for intraday)
        chunk_days = 60 if interval != "day" else 2000
        current = from_date
        while current < to_date:
            chunk_end = min(current + timedelta(days=chunk_days), to_date)
            try:
                data = self.kite.historical_data(
                    instrument_token=token,
                    from_date=current,
                    to_date=chunk_end,
                    interval=interval,
                    continuous=continuous,
                )
                records.extend(data)
                logger.debug(f"Fetched {len(data)} candles for {instrument} [{current.date()} → {chunk_end.date()}]")
            except Exception as e:
                logger.warning(f"Error fetching {instrument} chunk {current.date()}: {e}")
            current = chunk_end + timedelta(seconds=1)
            time.sleep(0.15)  # Stay well within 10 req/sec rate limit

        if not records:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        return df

    def get_historical_bulk(
        self,
        instruments: list,
        from_date: datetime,
        to_date: datetime,
        interval: str = "5minute",
    ) -> dict:
        """Fetch historical data for multiple instruments. Returns {name: DataFrame}."""
        result = {}
        for inst in instruments:
            logger.info(f"Downloading {inst} historical data ({interval}) ...")
            result[inst] = self.get_historical(inst, from_date, to_date, interval)
            logger.info(f"  → {len(result[inst])} candles")
        return result

    # ------------------------------------------------------------------
    # Live quotes
    # ------------------------------------------------------------------
    def get_live_quote(self, instrument: str) -> dict:
        """
        Returns latest OHLC + LTP for an instrument.
        Response keys: last_price, ohlc (open/high/low/close), volume, etc.
        """
        token = INSTRUMENTS[instrument]["token"]
        exchange = INSTRUMENTS[instrument]["exchange"]
        symbol = INSTRUMENTS[instrument]["symbol"]
        key = f"{exchange}:{symbol}"
        quotes = self.kite.quote([key])
        return quotes.get(key, {})

    def get_ltp(self, instrument: str) -> float:
        """Returns last traded price for an instrument."""
        quote = self.get_live_quote(instrument)
        return quote.get("last_price", 0.0)

    # ------------------------------------------------------------------
    # India VIX
    # ------------------------------------------------------------------
    def get_india_vix(self) -> float:
        """Returns current India VIX value."""
        try:
            quote = self.get_live_quote("INDIA_VIX")
            return quote.get("last_price", 0.0)
        except Exception as e:
            logger.warning(f"Could not fetch India VIX: {e}")
            return 15.0  # Return neutral default

    # ------------------------------------------------------------------
    # PCR (Put-Call Ratio) from NSE
    # ------------------------------------------------------------------
    def get_pcr(self, symbol: str = "NIFTY") -> float:
        """
        Fetches Put-Call Ratio from NSE option chain.
        PCR = Total Put OI / Total Call OI across all strikes.

        Args:
            symbol: "NIFTY" or "BANKNIFTY"

        Returns:
            PCR as float. Returns 1.0 (neutral) on failure.
        """
        try:
            nse_symbol = "NIFTY" if symbol == "NIFTY" else "BANKNIFTY"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept-Encoding": "gzip, deflate, br",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www.nseindia.com/",
            }
            session = requests.Session()
            # NSE requires a cookie from the main page first
            session.get("https://www.nseindia.com", headers=headers, timeout=10)
            resp = session.get(
                self.NSE_OPTION_CHAIN_URL,
                params={"symbol": nse_symbol},
                headers=headers,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            total_put_oi = sum(
                r.get("PE", {}).get("openInterest", 0)
                for r in data["records"]["data"]
                if r.get("PE")
            )
            total_call_oi = sum(
                r.get("CE", {}).get("openInterest", 0)
                for r in data["records"]["data"]
                if r.get("CE")
            )
            if total_call_oi == 0:
                return 1.0
            pcr = round(total_put_oi / total_call_oi, 3)
            logger.info(f"PCR ({nse_symbol}): {pcr} (Put OI: {total_put_oi:,}, Call OI: {total_call_oi:,})")
            return pcr
        except Exception as e:
            logger.warning(f"PCR fetch failed: {e}. Using neutral 1.0")
            return 1.0

    # ------------------------------------------------------------------
    # Instrument lookup
    # ------------------------------------------------------------------
    def get_instrument_token(self, exchange: str, symbol: str) -> Optional[int]:
        """Look up instrument token dynamically (useful for options strikes)."""
        try:
            instruments = self.kite.instruments(exchange)
            for inst in instruments:
                if inst["tradingsymbol"] == symbol:
                    return inst["instrument_token"]
        except Exception as e:
            logger.warning(f"Instrument lookup failed for {symbol}: {e}")
        return None
