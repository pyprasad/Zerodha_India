"""
VectorBT-based strategy parameter optimizer.
Runs fast grid-search over strategy parameters to find optimal configs.
"""
import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from data.store import OHLCVStore

logger = logging.getLogger(__name__)


def _calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _calc_supertrend(high, low, close, atr_period=10, multiplier=3.0):
    """Compute SuperTrend direction array."""
    import pandas as pd
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=atr_period, adjust=False).mean()

    mid = (high + low) / 2
    upper = mid + multiplier * atr
    lower = mid - multiplier * atr

    st = pd.Series(index=close.index, dtype=float)
    direction = pd.Series(index=close.index, dtype=float)

    st.iloc[0] = mid.iloc[0]
    direction.iloc[0] = 1.0

    for i in range(1, len(close)):
        if direction.iloc[i - 1] == 1:
            if close.iloc[i] < st.iloc[i - 1]:
                st.iloc[i] = upper.iloc[i]
                direction.iloc[i] = -1.0
            else:
                st.iloc[i] = max(lower.iloc[i], st.iloc[i - 1])
                direction.iloc[i] = 1.0
        else:
            if close.iloc[i] > st.iloc[i - 1]:
                st.iloc[i] = lower.iloc[i]
                direction.iloc[i] = 1.0
            else:
                st.iloc[i] = min(upper.iloc[i], st.iloc[i - 1])
                direction.iloc[i] = -1.0

    return direction


def _vectorized_pnl(entries: pd.Series, exits: pd.Series, close: pd.Series) -> float:
    """
    Rough vectorized P&L simulation (long signals only).
    Returns total percentage return.
    """
    in_trade = False
    entry_price = 0.0
    pnl = 0.0
    for i in range(len(close)):
        if not in_trade and entries.iloc[i]:
            in_trade = True
            entry_price = close.iloc[i]
        elif in_trade and exits.iloc[i]:
            pnl += (close.iloc[i] - entry_price) / entry_price
            in_trade = False
    return pnl * 100


def optimize_ema_crossover(
    instrument: str,
    interval: str,
    from_date: datetime,
    to_date: datetime,
    fast_range: list = None,
    slow_range: list = None,
    trend_range: list = None,
    store: OHLCVStore = None,
) -> pd.DataFrame:
    """
    Grid-search EMA Crossover parameters.

    Returns DataFrame with columns:
        fast_ema, slow_ema, trend_ema, returns_pct, trades
    Sorted by returns_pct descending.
    """
    if store is None:
        store = OHLCVStore()

    df = store.load_parquet(instrument, interval)
    df = df[(df["date"] >= pd.Timestamp(from_date)) & (df["date"] <= pd.Timestamp(to_date))]
    close = df.set_index("date")["close"]

    if fast_range is None:
        fast_range = [5, 7, 9, 11]
    if slow_range is None:
        slow_range = [18, 21, 24, 27]
    if trend_range is None:
        trend_range = [50, 55, 60]

    results = []
    total = len(fast_range) * len(slow_range) * len(trend_range)
    logger.info(f"EMA optimizer: {total} parameter combinations to test")

    for fast in fast_range:
        for slow in slow_range:
            if fast >= slow:
                continue
            for trend in trend_range:
                ema_f = _calc_ema(close, fast)
                ema_s = _calc_ema(close, slow)
                ema_t = _calc_ema(close, trend)
                crossover_up = (ema_f > ema_s) & (ema_f.shift() <= ema_s.shift())
                crossover_dn = (ema_f < ema_s) & (ema_f.shift() >= ema_s.shift())
                above_trend = close > ema_t
                entries = crossover_up & above_trend
                exits = crossover_dn
                pnl = _vectorized_pnl(entries, exits, close)
                trade_count = entries.sum()
                results.append({
                    "fast_ema": fast,
                    "slow_ema": slow,
                    "trend_ema": trend,
                    "returns_pct": round(pnl, 2),
                    "trades": int(trade_count),
                })

    result_df = pd.DataFrame(results).sort_values("returns_pct", ascending=False)
    logger.info(f"Best EMA params: {result_df.iloc[0].to_dict()}")
    return result_df


def optimize_supertrend(
    instrument: str,
    interval: str,
    from_date: datetime,
    to_date: datetime,
    period_range: list = None,
    mult_range: list = None,
    store: OHLCVStore = None,
) -> pd.DataFrame:
    """Grid-search SuperTrend ATR period and multiplier."""
    if store is None:
        store = OHLCVStore()

    df = store.load_parquet(instrument, interval)
    df = df[(df["date"] >= pd.Timestamp(from_date)) & (df["date"] <= pd.Timestamp(to_date))]
    ohlc = df.set_index("date")
    close = ohlc["close"]
    high = ohlc["high"]
    low = ohlc["low"]

    if period_range is None:
        period_range = [7, 10, 14]
    if mult_range is None:
        mult_range = [2.0, 2.5, 3.0, 3.5]

    results = []
    for period in period_range:
        for mult in mult_range:
            direction = _calc_supertrend(high, low, close, atr_period=period, multiplier=mult)
            entries = (direction == 1.0) & (direction.shift() == -1.0)
            exits = (direction == -1.0)
            pnl = _vectorized_pnl(entries, exits, close)
            results.append({
                "atr_period": period,
                "multiplier": mult,
                "returns_pct": round(pnl, 2),
                "trades": int(entries.sum()),
            })

    result_df = pd.DataFrame(results).sort_values("returns_pct", ascending=False)
    logger.info(f"Best SuperTrend params: {result_df.iloc[0].to_dict()}")
    return result_df


def optimize_orb(
    instrument: str,
    interval: str,
    from_date: datetime,
    to_date: datetime,
    rr_range: list = None,
    vol_mult_range: list = None,
    store: OHLCVStore = None,
) -> pd.DataFrame:
    """Grid-search ORB R:R ratio and volume multiplier."""
    if store is None:
        store = OHLCVStore()

    df = store.load_parquet(instrument, interval)
    df = df[(df["date"] >= pd.Timestamp(from_date)) & (df["date"] <= pd.Timestamp(to_date))]

    if rr_range is None:
        rr_range = [1.5, 2.0, 2.5, 3.0]
    if vol_mult_range is None:
        vol_mult_range = [1.0, 1.5, 2.0]

    df["date"] = pd.to_datetime(df["date"])
    df["day"] = df["date"].dt.date
    df["time"] = df["date"].dt.time

    import datetime as dt
    orb_start = dt.time(9, 15)
    orb_end = dt.time(9, 30)

    results = []
    for rr in rr_range:
        for vol_mult in vol_mult_range:
            total_pnl = 0.0
            for day, day_df in df.groupby("day"):
                orb_df = day_df[(day_df["time"] >= orb_start) & (day_df["time"] < orb_end)]
                if orb_df.empty:
                    continue
                orb_high = orb_df["high"].max()
                orb_low = orb_df["low"].min()
                avg_vol = day_df["volume"].mean()
                rest_df = day_df[day_df["time"] >= orb_end]
                for _, row in rest_df.iterrows():
                    vol_ok = row["volume"] > avg_vol * vol_mult
                    if row["close"] > orb_high and vol_ok:
                        stop = orb_low
                        risk = row["close"] - stop
                        target = row["close"] + risk * rr
                        # Check if target or stop hit in remaining candles
                        remaining = rest_df[rest_df["date"] > row["date"]]
                        for _, r2 in remaining.iterrows():
                            if r2["high"] >= target:
                                total_pnl += (target - row["close"]) / row["close"]
                                break
                            if r2["low"] <= stop:
                                total_pnl -= (row["close"] - stop) / row["close"]
                                break
                        break  # Only one trade per day
                    elif row["close"] < orb_low and vol_ok:
                        stop = orb_high
                        risk = stop - row["close"]
                        target = row["close"] - risk * rr
                        remaining = rest_df[rest_df["date"] > row["date"]]
                        for _, r2 in remaining.iterrows():
                            if r2["low"] <= target:
                                total_pnl += (row["close"] - target) / row["close"]
                                break
                            if r2["high"] >= stop:
                                total_pnl -= (stop - row["close"]) / row["close"]
                                break
                        break

            results.append({
                "rr_ratio": rr,
                "volume_multiplier": vol_mult,
                "returns_pct": round(total_pnl * 100, 2),
            })

    result_df = pd.DataFrame(results).sort_values("returns_pct", ascending=False)
    logger.info(f"Best ORB params: {result_df.iloc[0].to_dict()}")
    return result_df


def run_all_optimizations(
    instrument: str,
    interval: str,
    from_date: datetime,
    to_date: datetime,
) -> dict:
    """Run all optimizers and return best params for each strategy."""
    store = OHLCVStore()
    results = {}

    logger.info(f"Running all optimizations for {instrument} ({interval})")

    ema_df = optimize_ema_crossover(instrument, interval, from_date, to_date, store=store)
    results["EMA_CROSSOVER"] = ema_df.iloc[0].to_dict() if not ema_df.empty else {}

    st_df = optimize_supertrend(instrument, interval, from_date, to_date, store=store)
    results["SUPERTREND"] = st_df.iloc[0].to_dict() if not st_df.empty else {}

    orb_df = optimize_orb(instrument, interval, from_date, to_date, store=store)
    results["ORB"] = orb_df.iloc[0].to_dict() if not orb_df.empty else {}

    return results
