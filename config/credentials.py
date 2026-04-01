"""
Loads Zerodha API credentials from environment variables (.env file).
Never hardcode credentials in source code.
"""
import os
from dotenv import load_dotenv

load_dotenv()

KITE_API_KEY = os.getenv("KITE_API_KEY", "")
KITE_API_SECRET = os.getenv("KITE_API_SECRET", "")
ZERODHA_USER_ID = os.getenv("ZERODHA_USER_ID", "")
ZERODHA_PASSWORD = os.getenv("ZERODHA_PASSWORD", "")
ZERODHA_TOTP_SECRET = os.getenv("ZERODHA_TOTP_SECRET", "")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


def validate():
    """Raise if any required credential is missing."""
    missing = []
    for name, val in [
        ("KITE_API_KEY", KITE_API_KEY),
        ("KITE_API_SECRET", KITE_API_SECRET),
        ("ZERODHA_USER_ID", ZERODHA_USER_ID),
        ("ZERODHA_PASSWORD", ZERODHA_PASSWORD),
        ("ZERODHA_TOTP_SECRET", ZERODHA_TOTP_SECRET),
    ]:
        if not val:
            missing.append(name)
    if missing:
        raise EnvironmentError(
            f"Missing required credentials in .env: {', '.join(missing)}\n"
            "Copy .env.example to .env and fill in your values."
        )
