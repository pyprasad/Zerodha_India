"""
strategies/williams_r_v6.py
============================
Thin adapter that imports the Williams%R signal logic from v6_backtest.py
and exposes it for use by the paper trading scheduler.

v6_backtest.py is NEVER modified — all signal logic is imported from it.

Signal returned per instrument:
  ("L", stop_distance, "WR_OB")  → long entry (Williams%R exits oversold)
  ("S", stop_distance, "WR_OS")  → short entry (Williams%R exits overbought)
  None                            → no signal

Entry conditions (from v6_backtest.signal_williams_r):
  LONG:  WR(14) crosses above -80 (exits oversold)
         + close > EMA200 × 0.97
         + EMA50 > EMA200 × 0.98
  SHORT: WR(14) crosses below -20 (exits overbought)
         + close < EMA200 × 1.03
         + EMA50 < EMA200 × 1.02

Stop distance:  ATR(14) × 1.5
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

# Import signal functions from v6_backtest — do NOT modify v6_backtest.py
from v6_backtest import (
    signal_williams_r,
    precompute,
    LOT_SIZES,
    INSTRUMENT_LIVE_DATE,
    WARMUP,
)


# Instruments the v6 core strategy trades
V6_INSTRUMENTS = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]

# Minimum bars required for warm-up (EMA200 needs 200+ bars, WARMUP=260 in v6)
MIN_BARS = WARMUP + 10


class WilliamsRSignalGenerator:
    """
    Computes Williams%R entry signals for all 4 v6 instruments
    from their latest daily OHLCV data.

    Usage:
        gen = WilliamsRSignalGenerator()
        signals = gen.get_signals(data_dict)
        # signals = [{"instrument": "NIFTY", "direction": "L",
        #             "stop_distance": 120.5, "tag": "WR_OB"}, ...]
    """

    def get_signals(self, data_dict: dict) -> list:
        """
        Check all 4 instruments for Williams%R entry signals on the latest bar.

        Args:
            data_dict: {instrument_name: pd.DataFrame} — daily OHLCV DataFrames.
                       Each DataFrame must have columns: open, high, low, close, volume
                       and a DatetimeIndex or 'date' column.

        Returns:
            List of signal dicts. Empty list if no signals.
        """
        signals = []
        today = pd.Timestamp.now().normalize()

        for inst in V6_INSTRUMENTS:
            df = data_dict.get(inst)
            if df is None or df.empty:
                continue

            # Ensure proper index
            df = self._normalise_df(df)
            if df is None or len(df) < MIN_BARS:
                continue

            # Enforce instrument live date (no trades before NSE F&O launch)
            live_date = INSTRUMENT_LIVE_DATE.get(inst, pd.Timestamp("2017-01-01"))
            if df.index[-1] < live_date:
                continue

            # Precompute all indicators
            try:
                ind = precompute(df)
            except Exception:
                continue

            i = len(df) - 1  # latest bar index

            # Check signal (needs at least 2 bars for cross detection)
            if i < 1:
                continue

            result = signal_williams_r(i, ind)
            if result is not None:
                direction, stop_distance, tag = result
                signals.append({
                    "instrument":    inst,
                    "direction":     direction,          # "L" or "S"
                    "stop_distance": round(float(stop_distance), 2),
                    "tag":           tag,                # "WR_OB" or "WR_OS"
                    "entry_close":   round(float(ind["close"].iloc[i]), 2),
                    "signal_date":   str(df.index[i].date()),
                })

        return signals

    def get_exit_signals(self, data_dict: dict, open_positions: dict) -> list:
        """
        Check WR mid-exit signal for open paper positions.

        WR exit rule (from v6 process_exit):
          LONG:  WR crosses above -50 (wr_prev < -50 and wr_curr >= -50) → exit
          SHORT: WR crosses below -50 (wr_prev > -50 and wr_curr <= -50) → exit

        Args:
            data_dict: {instrument_name: pd.DataFrame} — same format as get_signals()
            open_positions: {instrument: position_dict} — currently open paper positions

        Returns:
            List of {"instrument": str, "reason": str} for positions to exit.
        """
        exits = []

        for inst, pos in open_positions.items():
            tag = pos.get("tag", "")
            if not tag.startswith(("WR_", "REGIME_MR", "REGIME_MIXED")):
                continue  # Only WR-tagged positions use mid-exit logic

            df = data_dict.get(inst)
            if df is None or df.empty:
                continue

            df = self._normalise_df(df)
            if df is None or len(df) < MIN_BARS:
                continue

            try:
                ind = precompute(df)
            except Exception:
                continue

            i = len(df) - 1
            if i < 1:
                continue

            direction = pos.get("direction", "L")
            wr_curr = float(ind["wR14"].iloc[i])
            wr_prev = float(ind["wR14"].iloc[i - 1])

            if np.isnan(wr_curr) or np.isnan(wr_prev):
                continue

            if direction == "L" and wr_prev < -50 and wr_curr >= -50:
                exits.append({"instrument": inst, "reason": "WR_MID_EXIT"})
            elif direction == "S" and wr_prev > -50 and wr_curr <= -50:
                exits.append({"instrument": inst, "reason": "WR_MID_EXIT"})

        return exits

    @staticmethod
    def _normalise_df(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure df has a proper DatetimeIndex and lowercase columns."""
        try:
            df = df.copy()
            df.columns = [c.lower() for c in df.columns]

            # If 'date' is a column (not index), set it as index
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                if df["date"].dt.tz is not None:
                    df["date"] = df["date"].dt.tz_localize(None)
                df = df.set_index("date")
            else:
                df.index = pd.to_datetime(df.index)
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)

            df.index = df.index.normalize()
            df = df[df.index.dayofweek < 5]
            df = df.dropna(subset=["close"])
            df = df.sort_index()
            return df
        except Exception:
            return None
