"""
NSE Trading Holiday Calendar
─────────────────────────────
Loads NSE holidays from a local JSON cache (data/nse_holidays.json).
Automatically refreshes every December so the next year's holidays are
ready before the new year starts — no manual updates needed.

Refresh source: exchange_calendars library (BSE calendar = same holidays as NSE).
Falls back to last known data if the library or network is unavailable.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_CACHE_FILE = Path(__file__).parent / "nse_holidays.json"

# Last-resort hardcoded holidays (2025–2026) used only if JSON file is missing
# AND exchange_calendars is unavailable on first run.
_BOOTSTRAP_HOLIDAYS = [
    "2025-01-26", "2025-02-26", "2025-03-14", "2025-03-31",
    "2025-04-10", "2025-04-14", "2025-04-18", "2025-05-01",
    "2025-08-15", "2025-10-02", "2025-10-20", "2025-10-21",
    "2025-11-05", "2025-12-25",
    "2026-01-01", "2026-01-26", "2026-04-10", "2026-04-13",
    "2026-05-01", "2026-06-01", "2026-11-30", "2026-12-01",
    "2026-12-25",
]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_from_library(year: int) -> list[str]:
    """
    Use exchange_calendars (BSE calendar) to derive NSE holidays for `year`.
    Returns list of ISO date strings e.g. ['2026-04-10', ...].
    Raises ImportError if exchange_calendars is not installed.
    Raises ValueError if the year is beyond the library's supported range.
    """
    import exchange_calendars as xcals
    import pandas as pd

    cal = xcals.get_calendar("XBSE")

    # Clamp end date to the last session the calendar supports
    last_session = cal.last_session
    end = pd.Timestamp(f"{year}-12-31")
    if end > last_session:
        if pd.Timestamp(f"{year}-01-01") > last_session:
            raise ValueError(f"Year {year} is beyond exchange_calendars supported range")
        end = last_session

    sessions = cal.sessions_in_range(f"{year}-01-01", end.strftime("%Y-%m-%d"))
    all_weekdays = pd.bdate_range(f"{year}-01-01", end)
    holidays = [d.strftime("%Y-%m-%d") for d in all_weekdays if d not in sessions]
    return holidays


def _load_cache() -> dict:
    """Load the JSON cache file. Returns empty structure if file is missing."""
    if _CACHE_FILE.exists():
        try:
            return json.loads(_CACHE_FILE.read_text())
        except Exception:
            pass
    return {"last_updated": None, "years": {}}


def _save_cache(data: dict) -> None:
    _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_FILE.write_text(json.dumps(data, indent=2))


def _needs_refresh(cache: dict) -> bool:
    """
    Refresh logic:
      - Always refresh if file is missing (cache["years"] is empty)
      - In December: refresh to load next year's holidays
      - If next year's data is missing from cache
    """
    today = date.today()
    years_in_cache = {int(y) for y in cache["years"]}

    if not years_in_cache:
        return True

    # Always ensure current year and next year are covered
    if today.year not in years_in_cache:
        return True
    if today.year + 1 not in years_in_cache:
        return True

    # In December: re-fetch to ensure next year is up to date
    if today.month == 12:
        last_updated = cache.get("last_updated")
        if last_updated:
            lu = datetime.strptime(last_updated, "%Y-%m-%d").date()
            # Refresh once per December (if last update was before this December)
            if lu.year < today.year or (lu.year == today.year and lu.month < 12):
                return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_holidays() -> set[date]:
    """
    Return a set of NSE public holiday dates for the current and next year.
    Automatically refreshes the cache if needed (missing data or December).
    Call once at startup — result is stable for the rest of the day.
    """
    cache = _load_cache()

    if _needs_refresh(cache):
        today = date.today()
        years_to_fetch = {today.year, today.year + 1}
        refreshed_any = False

        for year in sorted(years_to_fetch):
            try:
                holidays = _fetch_from_library(year)
                cache["years"][str(year)] = holidays
                refreshed_any = True
                logger.info(
                    f"[holidays] Fetched {len(holidays)} NSE holidays for {year} "
                    f"via exchange_calendars"
                )
            except Exception as e:
                # Keep existing data for this year if available; don't overwrite
                if str(year) not in cache["years"]:
                    # No existing data — fall back to bootstrap list for this year
                    bootstrap = [d for d in _BOOTSTRAP_HOLIDAYS if d.startswith(str(year))]
                    cache["years"][str(year)] = bootstrap
                    logger.warning(
                        f"[holidays] exchange_calendars unavailable ({e}); "
                        f"using {len(bootstrap)} hardcoded holidays for {year}"
                    )
                else:
                    logger.warning(
                        f"[holidays] Could not refresh {year} holidays ({e}); "
                        f"keeping {len(cache['years'][str(year)])} cached entries"
                    )

        if refreshed_any:
            cache["last_updated"] = today.strftime("%Y-%m-%d")
            _save_cache(cache)
            logger.info(
                f"[holidays] Cache saved to {_CACHE_FILE} "
                f"(next auto-refresh: December {today.year if today.month < 12 else today.year + 1})"
            )

    # Flatten all years into a single set of date objects
    all_dates: set[date] = set()
    for year_dates in cache["years"].values():
        for d in year_dates:
            try:
                all_dates.add(datetime.strptime(d, "%Y-%m-%d").date())
            except ValueError:
                pass

    return all_dates


def is_nse_holiday(d: date | None = None) -> bool:
    """Return True if `d` (defaults to today IST) is an NSE public holiday."""
    if d is None:
        import zoneinfo
        d = datetime.now(zoneinfo.ZoneInfo("Asia/Kolkata")).date()
    return d in get_holidays()
