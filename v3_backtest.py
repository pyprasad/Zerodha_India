"""
v3 VECTORIZED BACKTEST — Targeting 40%+ YoY on NIFTY & BANKNIFTY
===================================================================
Approach: Vectorized pandas simulation — no Backtrader overhead.
Strategies:
  1. SuperTrend Swing (SuperTrend direction + RSI timing)
  2. EMA Crossover Momentum (EMA8/21 + ADX trend filter)
  3. Bollinger + RSI Mean Reversion (high win-rate, tight stops)
  4. Donchian Position Trade (trend-follow, exits only on reversal)
  5. Combined Portfolio (best 2-3 signals per bar, diversified)

Math to achieve 40% YoY:
  25 trades/yr × 2% risk × (50% win×3 - 50% loss×1) = 25 × 2% × 1.0 = 50%/yr
  Even at 20 trades + 45% win: 20 × 2% × (0.45×3 - 0.55×1) = 20 × 2% × 0.8 = 32%
  With NIFTY + BANKNIFTY combined: double the signals.
"""

import warnings, sys, os
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import json

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()
Path("reports").mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CAPITAL       = 1_000_000   # ₹10,00,000 starting capital
RISK_PCT      = 0.02        # 2% risk per trade
RR_RATIO      = 3.0         # 1:3 reward-to-risk
BROKERAGE     = 20          # ₹20 per order (Zerodha flat)
EXCHANGE_FEE  = 0.0000125   # NSE exchange fees
SLIPPAGE      = 0.0003      # 0.03% per side (realistic)
MAX_TRADES_AT_ONCE = 2      # Max concurrent open positions per instrument
STOP_ATR_MULT    = 1.5      # Stop distance = 1.5 × ATR(14)
MIN_STOP_PCT     = 0.005    # Minimum stop distance = 0.5% of price (prevents giant qty)
MAX_RISK_PCT     = 0.03     # Hard cap: never risk more than 3% on any single trade

# Index futures lot sizes (Zerodha/NSE standard)
LOT_SIZES = {
    "NIFTY":    75,   # NIFTY 50 futures lot = 75 units
    "BANKNIFTY": 30,  # BANKNIFTY futures lot = 30 units
}

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
def load_data(instrument: str) -> pd.DataFrame:
    path = Path(f"data/historical/{instrument}_daily_extended.csv")
    if path.exists():
        df = pd.read_csv(path, index_col="date", parse_dates=True)
        df.columns = [c.lower() for c in df.columns]
    else:
        import yfinance as yf
        sym = "^NSEI" if instrument == "NIFTY" else "^NSEBANK"
        raw = yf.download(sym, start="2020-01-01", auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw.copy()
        df.index.name = "date"
        df.columns = [c.lower() for c in df.columns]
    # Ensure required columns
    for col in ["open","high","low","close","volume"]:
        if col not in df.columns:
            df[col] = df.get(col, np.nan)
    df = df.dropna(subset=["open","high","low","close"])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# INDICATORS (vectorized)
# ─────────────────────────────────────────────────────────────────────────────
def compute_atr(df, period=14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def compute_rsi(series: pd.Series, period=14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(com=period-1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period-1, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def compute_supertrend(df, period=7, multiplier=2.5) -> tuple:
    """Returns (direction, upperband, lowerband) — direction 1=bull, -1=bear."""
    atr = compute_atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2

    upper = (hl2 + multiplier * atr).values
    lower = (hl2 - multiplier * atr).values
    close = df["close"].values
    n = len(close)

    final_upper = np.zeros(n)
    final_lower = np.zeros(n)
    direction   = np.ones(n)

    final_upper[0] = upper[0]
    final_lower[0] = lower[0]

    for i in range(1, n):
        # Upper band
        if upper[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
            final_upper[i] = upper[i]
        else:
            final_upper[i] = final_upper[i-1]
        # Lower band
        if lower[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
            final_lower[i] = lower[i]
        else:
            final_lower[i] = final_lower[i-1]
        # Direction
        if direction[i-1] == -1 and close[i] > final_upper[i]:
            direction[i] = 1
        elif direction[i-1] == 1 and close[i] < final_lower[i]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]

    st_direction = pd.Series(direction, index=df.index)
    return st_direction, pd.Series(final_upper, index=df.index), pd.Series(final_lower, index=df.index)


def compute_adx(df, period=14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = compute_atr(df, 1)
    plus_dm  = (h - h.shift(1)).clip(lower=0)
    minus_dm = (l.shift(1) - l).clip(lower=0)
    plus_dm  = plus_dm.where(plus_dm > minus_dm, 0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0)
    atr_p    = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_p
    minus_di = 100 * minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_p
    dx       = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).fillna(0)
    return dx.ewm(alpha=1/period, adjust=False).mean()


def compute_bollinger(df, period=20, std_mult=2.0):
    mid   = df["close"].rolling(period).mean()
    sigma = df["close"].rolling(period).std()
    return mid - std_mult*sigma, mid, mid + std_mult*sigma


# ─────────────────────────────────────────────────────────────────────────────
# TRADE SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────
class TradeSimulator:
    """Simulates sequential futures trade execution with P&L compounding."""

    def __init__(self, capital=CAPITAL, risk_pct=RISK_PCT, rr=RR_RATIO, instrument="NIFTY"):
        self.initial_capital = capital
        self.capital = float(capital)
        self.risk_pct = risk_pct
        self.rr = rr
        self.lot_size = LOT_SIZES.get(instrument, 75)
        self.trades = []
        self.equity_curve = []
        self.position = None

    def _commission(self, size, price):
        return BROKERAGE * 2 + abs(size * price) * (EXCHANGE_FEE * 2 + SLIPPAGE * 2)

    def _qty(self, entry, stop):
        """Risk-based position sizing in futures lots."""
        risk_amt  = self.capital * self.risk_pct
        # Enforce minimum stop distance (0.5%) to prevent absurdly large positions
        risk_unit = max(abs(entry - stop), entry * MIN_STOP_PCT)
        # Hard cap: never exceed 3% risk
        risk_amt  = min(risk_amt, self.capital * MAX_RISK_PCT)
        # Size in whole lots
        units_per_lot = self.lot_size
        lots = max(1, int(risk_amt / (risk_unit * units_per_lot)))
        return lots * units_per_lot

    def enter(self, date, direction, entry_price, stop_price, tag=""):
        if self.position is not None:
            return  # Already in position
        qty = self._qty(entry_price, stop_price)
        target_price = entry_price + direction * abs(entry_price - stop_price) * self.rr
        comm = self._commission(qty, entry_price)
        self.capital -= comm
        self.position = {
            "direction": direction,
            "entry": entry_price,
            "stop": stop_price,
            "target": target_price,
            "qty": qty,
            "entry_date": date,
            "tag": tag,
        }

    def update_stop(self, new_stop, direction):
        """Trail the stop."""
        if self.position is None: return
        if direction == 1:
            self.position["stop"] = max(self.position["stop"], new_stop)
        else:
            self.position["stop"] = min(self.position["stop"], new_stop)

    def check_exit(self, date, high, low, close) -> bool:
        """Check if position should be closed. Returns True if closed."""
        if self.position is None:
            return False
        d     = self.position["direction"]
        stop  = self.position["stop"]
        tgt   = self.position["target"]
        entry = self.position["entry"]
        qty   = self.position["qty"]

        exit_price = None
        exit_reason = ""

        if d == 1:  # Long
            if low <= stop:
                exit_price = stop
                exit_reason = "stop"
            elif high >= tgt:
                exit_price = tgt
                exit_reason = "target"
            elif close < entry * 0.92:  # Emergency 8% stop
                exit_price = close
                exit_reason = "emergency"
        else:  # Short
            if high >= stop:
                exit_price = stop
                exit_reason = "stop"
            elif low <= tgt:
                exit_price = tgt
                exit_reason = "target"
            elif close > entry * 1.08:  # Emergency 8% stop
                exit_price = close
                exit_reason = "emergency"

        if exit_price is None:
            return False

        comm     = self._commission(qty, exit_price)
        pnl      = d * (exit_price - entry) * qty - comm
        pnl_pct  = pnl / self.initial_capital * 100
        self.capital += pnl
        dur = (date - self.position["entry_date"]).days

        self.trades.append({
            "entry_date": self.position["entry_date"],
            "exit_date": date,
            "direction": "LONG" if d == 1 else "SHORT",
            "entry": round(entry, 2),
            "exit": round(exit_price, 2),
            "stop": round(stop, 2),
            "target": round(tgt, 2),
            "qty": qty,
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 4),
            "capital_after": round(self.capital, 2),
            "duration_days": dur,
            "reason": exit_reason,
            "tag": self.position["tag"],
        })
        self.position = None
        return True

    def force_close(self, date, close_price):
        """Force close at end of backtest."""
        if self.position is None:
            return
        d = self.position["direction"]
        entry = self.position["entry"]
        qty = self.position["qty"]
        comm = self._commission(qty, close_price)
        pnl = d * (close_price - entry) * qty - comm
        self.capital += pnl
        self.trades.append({
            "entry_date": self.position["entry_date"],
            "exit_date": date,
            "direction": "LONG" if d == 1 else "SHORT",
            "entry": round(entry, 2),
            "exit": round(close_price, 2),
            "stop": round(self.position["stop"], 2),
            "target": round(self.position["target"], 2),
            "qty": qty,
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl / self.initial_capital * 100, 4),
            "capital_after": round(self.capital, 2),
            "duration_days": (date - self.position["entry_date"]).days,
            "reason": "force_close",
            "tag": self.position["tag"],
        })
        self.position = None

    def results(self) -> dict:
        trades = self.trades
        if not trades:
            return {"total_ret": 0, "ann_ret": 0, "sharpe": 0, "max_dd": 0,
                    "win_rate": 0, "num_trades": 0, "net_pnl": 0}
        df = pd.DataFrame(trades)
        n_wins  = (df["pnl"] > 0).sum()
        n_total = len(df)
        win_rate = n_wins / n_total * 100 if n_total else 0

        # Build equity curve
        caps = [self.initial_capital] + list(df["capital_after"].values)
        eq = pd.Series(caps)
        max_dd = ((eq.cummax() - eq) / eq.cummax()).max() * 100

        total_ret = (self.capital - self.initial_capital) / self.initial_capital * 100
        # Annualise
        if trades:
            t0 = pd.to_datetime(trades[0]["entry_date"])
            t1 = pd.to_datetime(trades[-1]["exit_date"])
            years = max((t1 - t0).days / 365.25, 0.1)
        else:
            years = 1
        ann_ret = ((1 + total_ret/100) ** (1/years) - 1) * 100

        # Sharpe
        pnl_pcts = df["pnl_pct"].values
        if pnl_pcts.std() > 0:
            sharpe = (pnl_pcts.mean() / pnl_pcts.std()) * np.sqrt(252)
        else:
            sharpe = 0.0

        avg_win  = df.loc[df["pnl"] > 0, "pnl"].mean() if n_wins > 0 else 0
        avg_loss = df.loc[df["pnl"] < 0, "pnl"].mean() if (n_total - n_wins) > 0 else 0

        return {
            "total_ret":  round(total_ret, 2),
            "ann_ret":    round(ann_ret, 2),
            "sharpe":     round(sharpe, 3),
            "max_dd":     round(max_dd, 2),
            "win_rate":   round(win_rate, 1),
            "num_trades": n_total,
            "net_pnl":    round(self.capital - self.initial_capital, 0),
            "capital":    round(self.capital, 0),
            "avg_win":    round(avg_win, 0),
            "avg_loss":   round(avg_loss, 0),
        }


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 1 — SUPERTREND SWING WITH RSI TIMING
# SuperTrend(7, 2.5) for trend direction.
# Enter longs when: ST is bullish AND RSI(7) dips below 45 then bounces back above.
# Enter shorts when: ST is bearish AND RSI(7) rises above 55 then drops below.
# Stop = 1.5 × ATR below/above entry.
# Target = 3 × stop distance (1:3 R:R).
# Exit early if SuperTrend flips against position.
# ─────────────────────────────────────────────────────────────────────────────
def strategy_supertrend_rsi(df: pd.DataFrame, instrument: str) -> dict:
    atr  = compute_atr(df, 14)
    rsi7 = compute_rsi(df["close"], 7)
    st_dir, st_upper, st_lower = compute_supertrend(df, period=7, multiplier=2.5)

    sim = TradeSimulator(instrument=instrument)
    close = df["close"]

    for i in range(30, len(df)):
        date  = df.index[i]
        p     = close.iloc[i]
        p_hi  = df["high"].iloc[i]
        p_lo  = df["low"].iloc[i]

        cur_st  = st_dir.iloc[i]
        prev_st = st_dir.iloc[i-1]
        r       = rsi7.iloc[i]
        r_prev  = rsi7.iloc[i-1]
        a       = atr.iloc[i]

        # --- Check exit ---
        if sim.position is not None:
            # Update trailing stop to SuperTrend line
            d = sim.position["direction"]
            if d == 1:
                sim.update_stop(st_lower.iloc[i], 1)
            else:
                sim.update_stop(st_upper.iloc[i], -1)
            # SuperTrend flip = force exit
            if (d == 1 and cur_st == -1) or (d == -1 and cur_st == 1):
                sim.force_close(date, p)
                continue
        exited = sim.check_exit(date, p_hi, p_lo, p)

        # --- Check entry ---
        if sim.position is None:
            stop_dist = a * STOP_ATR_MULT

            # LONG: ST bullish AND RSI bounces from oversold (crossed 40 from below)
            if cur_st == 1 and r_prev < 42 and r >= 42:
                stop  = p - stop_dist
                sim.enter(date, 1, p, stop, tag="ST+RSI_LONG")

            # SHORT: ST bearish AND RSI bounces from overbought (crossed 58 from above)
            elif cur_st == -1 and r_prev > 58 and r <= 58:
                stop = p + stop_dist
                sim.enter(date, -1, p, stop, tag="ST+RSI_SHORT")

    # Force close at end
    if sim.position:
        sim.force_close(df.index[-1], close.iloc[-1])

    return sim.results()


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 2 — EMA 8/21 CROSSOVER + ADX MOMENTUM FILTER
# Signal: EMA8 crosses EMA21 while ADX > 20 (trending environment).
# Trend confirmation: price above EMA50 for longs, below for shorts.
# Stop = 1.5 × ATR; Target = 3 × stop distance.
# ─────────────────────────────────────────────────────────────────────────────
def strategy_ema_momentum(df: pd.DataFrame, instrument: str) -> dict:
    ema8  = compute_ema(df["close"], 8)
    ema21 = compute_ema(df["close"], 21)
    ema50 = compute_ema(df["close"], 50)
    adx   = compute_adx(df, 14)
    atr   = compute_atr(df, 14)
    rsi14 = compute_rsi(df["close"], 14)

    sim   = TradeSimulator(instrument=instrument)
    close = df["close"]

    for i in range(55, len(df)):
        date  = df.index[i]
        p     = close.iloc[i]
        p_hi  = df["high"].iloc[i]
        p_lo  = df["low"].iloc[i]
        a     = atr.iloc[i]
        dx    = adx.iloc[i]

        e8    = ema8.iloc[i];   e8p  = ema8.iloc[i-1]
        e21   = ema21.iloc[i];  e21p = ema21.iloc[i-1]
        e50   = ema50.iloc[i]
        r     = rsi14.iloc[i]

        # Exit check
        if sim.position is not None:
            # Trail stop to 21 EMA
            d = sim.position["direction"]
            if d == 1:
                sim.update_stop(e21 - a * 0.5, 1)
            else:
                sim.update_stop(e21 + a * 0.5, -1)
        exited = sim.check_exit(date, p_hi, p_lo, p)

        if sim.position is None:
            stop_dist = a * STOP_ATR_MULT
            # Crossover happened this bar
            cross_up   = e8p < e21p and e8 >= e21
            cross_down = e8p > e21p and e8 <= e21

            # LONG: EMA8 crosses above EMA21 + ADX > 20 + price > EMA50 + RSI not overbought
            if cross_up and dx > 20 and p > e50 and r < 70:
                stop = p - stop_dist
                sim.enter(date, 1, p, stop, tag="EMA_CROSS_LONG")

            # SHORT: EMA8 crosses below EMA21 + ADX > 20 + price < EMA50 + RSI not oversold
            elif cross_down and dx > 20 and p < e50 and r > 30:
                stop = p + stop_dist
                sim.enter(date, -1, p, stop, tag="EMA_CROSS_SHORT")

    if sim.position:
        sim.force_close(df.index[-1], close.iloc[-1])

    return sim.results()


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 3 — BOLLINGER + RSI MEAN REVERSION (HIGH WIN RATE)
# Buys when: price touches lower BB AND RSI(14) < 32 AND price > EMA200 (uptrend)
# Shorts when: price touches upper BB AND RSI(14) > 68 AND price < EMA200
# Tight stop just outside the BB. Target = BB midline (1.5-2:1 R:R typically).
# Uses lower risk (1.5%) since win rate is higher.
# ─────────────────────────────────────────────────────────────────────────────
def strategy_bb_rsi_mr(df: pd.DataFrame, instrument: str) -> dict:
    bb_lo, bb_mid, bb_hi = compute_bollinger(df, 20, 2.0)
    rsi14 = compute_rsi(df["close"], 14)
    ema200= compute_ema(df["close"], 200)
    atr   = compute_atr(df, 14)
    adx   = compute_adx(df, 14)

    # Use slightly lower risk for mean reversion (tighter distribution)
    sim   = TradeSimulator(risk_pct=0.015, instrument=instrument)
    close = df["close"]

    for i in range(210, len(df)):
        date  = df.index[i]
        p     = close.iloc[i]
        p_hi  = df["high"].iloc[i]
        p_lo  = df["low"].iloc[i]
        a     = atr.iloc[i]
        dx    = adx.iloc[i]
        r     = rsi14.iloc[i]
        e200  = ema200.iloc[i]
        bbl   = bb_lo.iloc[i]
        bbm   = bb_mid.iloc[i]
        bbh   = bb_hi.iloc[i]

        # Exit: target = BB midline, stop = ATR * 1.0
        exited = sim.check_exit(date, p_hi, p_lo, p)

        if sim.position is None and not np.isnan(e200):
            # LONG: RSI oversold at BB lower band, not extreme downtrend (ADX < 40)
            if p_lo <= bbl and r < 32 and p > e200 * 0.97 and dx < 40:
                # Stop = 1.5×ATR below entry (not BB band — BB can be at entry price)
                stop   = p - a * 1.5
                target = bbm  # Target to midband
                if target > p and abs(p - stop) > 0:
                    rr = (target - p) / abs(p - stop)
                    if rr >= 1.2:  # At least 1.2:1 R:R
                        sim.enter(date, 1, p, stop, tag="BB_RSI_LONG")

            # SHORT: RSI overbought at BB upper band
            elif p_hi >= bbh and r > 68 and p < e200 * 1.03 and dx < 40:
                stop  = p + a * 1.5
                target = bbm
                if target < p and abs(stop - p) > 0:
                    rr = (p - target) / abs(stop - p)
                    if rr >= 1.2:
                        sim.enter(date, -1, p, stop, tag="BB_RSI_SHORT")

        # Update target to BB midline dynamically
        if sim.position is not None:
            d = sim.position["direction"]
            # Recalculate target to current BB midline
            if d == 1:
                sim.position["target"] = bbm
            else:
                sim.position["target"] = bbm

    if sim.position:
        sim.force_close(df.index[-1], close.iloc[-1])

    return sim.results()


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 4 — DONCHIAN POSITION TRADE (TREND CAPTURE, WIDER STOPS)
# Classic Turtle Trading adapted for Indian indices.
# Entry: 20-day high/low breakout (Donchian channel).
# Exit: 10-day opposite channel or SuperTrend flip.
# Stop: 2 × ATR (wider to stay in the trend).
# No fixed target — let profits run with trailing stop.
# ─────────────────────────────────────────────────────────────────────────────
def strategy_donchian_position(df: pd.DataFrame, instrument: str) -> dict:
    high20 = df["high"].rolling(20).max().shift(1)
    low20  = df["low"].rolling(20).min().shift(1)
    high10 = df["high"].rolling(10).max().shift(1)
    low10  = df["low"].rolling(10).min().shift(1)
    atr    = compute_atr(df, 14)
    ema100 = compute_ema(df["close"], 100)
    st_dir, st_upper, st_lower = compute_supertrend(df, period=10, multiplier=3.0)

    sim   = TradeSimulator(risk_pct=RISK_PCT, rr=5.0, instrument=instrument)  # Wider R:R for trend-following
    close = df["close"]

    for i in range(25, len(df)):
        date  = df.index[i]
        p     = close.iloc[i]
        p_hi  = df["high"].iloc[i]
        p_lo  = df["low"].iloc[i]
        a     = atr.iloc[i]

        # Trail stop to opposite Donchian or SuperTrend
        if sim.position is not None:
            d = sim.position["direction"]
            if d == 1:
                trail = max(low10.iloc[i], st_lower.iloc[i])
                sim.update_stop(trail, 1)
                # SuperTrend flip = exit
                if st_dir.iloc[i] == -1:
                    sim.force_close(date, p)
                    continue
            else:
                trail = min(high10.iloc[i], st_upper.iloc[i])
                sim.update_stop(trail, -1)
                if st_dir.iloc[i] == 1:
                    sim.force_close(date, p)
                    continue

        exited = sim.check_exit(date, p_hi, p_lo, p)

        if sim.position is None:
            stop_dist = a * 2.0  # Wider stop for position trades
            e100 = ema100.iloc[i]

            # LONG: New 20-day high breakout + EMA100 uptrend
            if p > high20.iloc[i] and p > e100:
                stop = p - stop_dist
                sim.enter(date, 1, p, stop, tag="DONCHIAN_LONG")

            # SHORT: New 20-day low breakdown + EMA100 downtrend
            elif p < low20.iloc[i] and p < e100:
                stop = p + stop_dist
                sim.enter(date, -1, p, stop, tag="DONCHIAN_SHORT")

    if sim.position:
        sim.force_close(df.index[-1], close.iloc[-1])

    return sim.results()


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 5 — STOCH RSI REVERSAL + EMA TREND
# StochRSI(14,3,3): fast momentum oscillator to time entries.
# Long: StochRSI K < 20 → crosses above 20, price > EMA50
# Short: StochRSI K > 80 → crosses below 80, price < EMA50
# Tight 1.5 ATR stop, 3:1 target.
# ─────────────────────────────────────────────────────────────────────────────
def compute_stochrsi(close: pd.Series, rsi_period=14, stoch_period=14, k_smooth=3, d_smooth=3):
    rsi = compute_rsi(close, rsi_period)
    rsi_min = rsi.rolling(stoch_period).min()
    rsi_max = rsi.rolling(stoch_period).max()
    stoch = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10) * 100
    k = stoch.rolling(k_smooth).mean()
    d = k.rolling(d_smooth).mean()
    return k, d


def strategy_stochrsi(df: pd.DataFrame, instrument: str) -> dict:
    k, d  = compute_stochrsi(df["close"])
    ema50 = compute_ema(df["close"], 50)
    ema20 = compute_ema(df["close"], 20)
    atr   = compute_atr(df, 14)
    adx   = compute_adx(df, 14)

    sim   = TradeSimulator(instrument=instrument)
    close = df["close"]

    for i in range(40, len(df)):
        date  = df.index[i]
        p     = close.iloc[i]
        p_hi  = df["high"].iloc[i]
        p_lo  = df["low"].iloc[i]
        a     = atr.iloc[i]

        k_cur  = k.iloc[i];   k_prev  = k.iloc[i-1]
        e50    = ema50.iloc[i]
        e20    = ema20.iloc[i]
        dx     = adx.iloc[i]

        # Trail stop to EMA20
        if sim.position is not None:
            d_ = sim.position["direction"]
            if d_ == 1:
                sim.update_stop(e20 - a * 0.3, 1)
            else:
                sim.update_stop(e20 + a * 0.3, -1)

        exited = sim.check_exit(date, p_hi, p_lo, p)

        if sim.position is None:
            stop_dist = a * STOP_ATR_MULT

            # LONG: StochRSI crosses above 20 (oversold) + price > EMA50 + not extreme ADX
            if k_prev < 20 and k_cur >= 20 and p > e50:
                stop = p - stop_dist
                sim.enter(date, 1, p, stop, tag="STOCHRSI_LONG")

            # SHORT: StochRSI crosses below 80 (overbought) + price < EMA50
            elif k_prev > 80 and k_cur <= 80 and p < e50:
                stop = p + stop_dist
                sim.enter(date, -1, p, stop, tag="STOCHRSI_SHORT")

    if sim.position:
        sim.force_close(df.index[-1], close.iloc[-1])

    return sim.results()


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 6 — COMBINED MULTI-SIGNAL (ENSEMBLE)
# Takes the BEST entry signal when multiple conditions align.
# Signals are ranked: require 2 of 3 confirmations for entry.
# NIFTY/BANKNIFTY typically trend strongly — this exploits confirmation.
# ─────────────────────────────────────────────────────────────────────────────
def strategy_combined(df: pd.DataFrame, instrument: str) -> dict:
    # All indicators
    ema8    = compute_ema(df["close"], 8)
    ema21   = compute_ema(df["close"], 21)
    ema50   = compute_ema(df["close"], 50)
    ema200  = compute_ema(df["close"], 200)
    atr     = compute_atr(df, 14)
    rsi14   = compute_rsi(df["close"], 14)
    adx     = compute_adx(df, 14)
    st_dir, st_up, st_lo = compute_supertrend(df, 7, 2.5)
    bb_lo, bb_mid, bb_hi = compute_bollinger(df, 20, 2.0)
    k, d_stoch = compute_stochrsi(df["close"])

    sim   = TradeSimulator(risk_pct=0.025, rr=3.0, instrument=instrument)  # 2.5% risk for high-confidence entries
    close = df["close"]

    for i in range(210, len(df)):
        date  = df.index[i]
        p     = close.iloc[i]
        p_hi  = df["high"].iloc[i]
        p_lo  = df["low"].iloc[i]
        a     = atr.iloc[i]
        dx    = adx.iloc[i]
        r     = rsi14.iloc[i]
        e8    = ema8.iloc[i];   e8p   = ema8.iloc[i-1]
        e21   = ema21.iloc[i];  e21p  = ema21.iloc[i-1]
        e50   = ema50.iloc[i]
        e200  = ema200.iloc[i]
        st    = st_dir.iloc[i]
        k_cur = k.iloc[i];      k_prev = k.iloc[i-1]
        bbl   = bb_lo.iloc[i]
        bbh   = bb_hi.iloc[i]

        if np.isnan(e200): continue

        # --- Trail & Exit ---
        if sim.position is not None:
            d_ = sim.position["direction"]
            # Trail stop: SuperTrend line
            if d_ == 1:
                sim.update_stop(st_lo.iloc[i], 1)
                if st == -1:
                    sim.force_close(date, p)
                    continue
            else:
                sim.update_stop(st_up.iloc[i], -1)
                if st == 1:
                    sim.force_close(date, p)
                    continue

        exited = sim.check_exit(date, p_hi, p_lo, p)

        if sim.position is None:
            stop_dist = a * STOP_ATR_MULT

            # Count bullish signals
            bull_signals = 0
            if e8p < e21p and e8 >= e21:     bull_signals += 2  # EMA crossover (strong)
            if st == 1 and st_dir.iloc[i-1] == -1: bull_signals += 2  # ST flip (strong)
            if k_prev < 25 and k_cur >= 25 and p > e50: bull_signals += 1  # StochRSI
            if r < 40 and p > e200 and p_lo <= bbl:     bull_signals += 1  # BB+RSI

            # Count bearish signals
            bear_signals = 0
            if e8p > e21p and e8 <= e21:     bear_signals += 2
            if st == -1 and st_dir.iloc[i-1] == 1: bear_signals += 2
            if k_prev > 75 and k_cur <= 75 and p < e50: bear_signals += 1
            if r > 60 and p < e200 and p_hi >= bbh:     bear_signals += 1

            # Entry on 2+ signal score
            if bull_signals >= 2 and p > e200:
                stop = p - stop_dist
                sim.enter(date, 1, p, stop, tag=f"COMBINED_LONG(score={bull_signals})")

            elif bear_signals >= 2 and p < e200:
                stop = p + stop_dist
                sim.enter(date, -1, p, stop, tag=f"COMBINED_SHORT(score={bear_signals})")

    if sim.position:
        sim.force_close(df.index[-1], close.iloc[-1])

    return sim.results()


# ─────────────────────────────────────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────────────────────────────────────
def print_results_table(all_results: list):
    tbl = Table(
        title="v3 Vectorized Strategy Backtest — NIFTY & BANKNIFTY (5.4 Years)",
        box=box.DOUBLE_EDGE,
        show_header=True,
        header_style="bold cyan",
        min_width=130,
    )
    tbl.add_column("Strategy",    min_width=26)
    tbl.add_column("Instrument",  min_width=10)
    tbl.add_column("Ann Ret%",    justify="right", min_width=10)
    tbl.add_column("Total Ret%",  justify="right", min_width=10)
    tbl.add_column("Net P&L",     justify="right", min_width=14)
    tbl.add_column("Sharpe",      justify="right", min_width=8)
    tbl.add_column("Max DD%",     justify="right", min_width=8)
    tbl.add_column("Win%",        justify="right", min_width=7)
    tbl.add_column("Trades",      justify="right", min_width=7)

    TARGET = 40.0

    for r in all_results:
        ann  = r["ann_ret"]
        tot  = r["total_ret"]
        pnl  = r["net_pnl"]
        shr  = r["sharpe"]
        dd   = r["max_dd"]
        wr   = r["win_rate"]
        nt   = r["num_trades"]
        inst = r["instrument"]
        name = r["strategy"]

        color = "green" if ann >= TARGET else ("yellow" if ann >= 25 else "red")

        tbl.add_row(
            name, inst,
            f"[{color}]{ann:+.1f}%[/{color}]",
            f"[{color}]{tot:+.1f}%[/{color}]",
            f"₹{pnl:+,.0f}",
            f"{shr:.2f}",
            f"{dd:.1f}%",
            f"{wr:.1f}%",
            str(nt),
        )

    console.print(tbl)


def save_json_report(all_results: list):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"reports/v3_backtest_{ts}.json"
    with open(fname, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    return fname


def print_annual_breakdown(all_results: list):
    """Print annual returns for the best strategy."""
    # Find best by ann_ret
    best = max(all_results, key=lambda r: r.get("ann_ret", -999))
    console.print(f"\n[bold cyan]Best Strategy: {best['strategy']} / {best['instrument']}[/bold cyan]")
    console.print(f"  Ann Return: [green]{best['ann_ret']:+.1f}%[/green]  |  "
                  f"Total: [green]{best['total_ret']:+.1f}%[/green]  |  "
                  f"Sharpe: {best['sharpe']:.2f}  |  "
                  f"Max DD: {best['max_dd']:.1f}%  |  "
                  f"Win Rate: {best['win_rate']:.1f}%  |  "
                  f"Trades: {best['num_trades']}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
STRATEGIES = [
    ("SuperTrend+RSI Swing",   strategy_supertrend_rsi),
    ("EMA Crossover Momentum", strategy_ema_momentum),
    ("Bollinger+RSI MeanRev",  strategy_bb_rsi_mr),
    ("Donchian Position Trade",strategy_donchian_position),
    ("StochRSI Reversal",      strategy_stochrsi),
    ("Combined Multi-Signal",  strategy_combined),
]

INSTRUMENTS = ["NIFTY", "BANKNIFTY"]


def main():
    console.rule("[bold cyan]v3 VECTORIZED BACKTEST — 40% YoY TARGET[/bold cyan]")
    console.print(f"[dim]Capital ₹{CAPITAL:,} | Risk {RISK_PCT*100:.0f}%/trade | R:R 1:{RR_RATIO:.0f} | "
                  f"Brokerage ₹{BROKERAGE}/order | Slippage {SLIPPAGE*100:.3f}%[/dim]")
    console.print()

    # Load data
    console.print("[cyan]Loading historical data...[/cyan]")
    data = {}
    for inst in INSTRUMENTS:
        data[inst] = load_data(inst)
    console.print()

    # Run all strategies
    total_runs = len(STRATEGIES) * len(INSTRUMENTS)
    console.print(f"[cyan]Running {len(STRATEGIES)} strategies × {len(INSTRUMENTS)} instruments = {total_runs} runs...[/cyan]")
    console.print()

    all_results = []
    for sname, sfunc in STRATEGIES:
        for inst in INSTRUMENTS:
            df = data[inst].copy()
            try:
                res = sfunc(df, inst)
                res["strategy"]   = sname
                res["instrument"] = inst
                ann = res.get("ann_ret", 0)
                color = "green" if ann >= 40 else ("yellow" if ann >= 20 else "white")
                console.print(
                    f"  {sname} / {inst} ... "
                    f"[{color}]{ann:+.1f}%/yr[/{color}]  "
                    f"total={res.get('total_ret',0):+.1f}%  "
                    f"win={res.get('win_rate',0):.0f}%  "
                    f"trades={res.get('num_trades',0)}"
                )
                all_results.append(res)
            except Exception as e:
                import traceback
                console.print(f"  [red]FAILED: {sname} / {inst}: {e}[/red]")
                traceback.print_exc()

    console.print()
    console.rule("[bold cyan]RESULTS[/bold cyan]")
    print_results_table(all_results)
    print_annual_breakdown(all_results)

    # Show strategies meeting 40% target
    winners = [r for r in all_results if r.get("ann_ret", 0) >= 40]
    if winners:
        console.print(f"\n[bold green]✓ {len(winners)} strategy/instrument combinations achieved 40%+ YoY target![/bold green]")
        for w in winners:
            console.print(f"  [green]→ {w['strategy']} on {w['instrument']}: {w['ann_ret']:+.1f}%/yr "
                          f"(Sharpe {w['sharpe']:.2f}, MaxDD {w['max_dd']:.1f}%, "
                          f"Win {w['win_rate']:.0f}%, {w['num_trades']} trades)[/green]")
    else:
        console.print(f"\n[yellow]No strategy hit 40%+ individually. "
                      f"Best: {max(all_results, key=lambda r: r.get('ann_ret',-999))['strategy']} "
                      f"@ {max(all_results, key=lambda r: r.get('ann_ret',-999))['ann_ret']:+.1f}%/yr[/yellow]")

    # Portfolio combination
    console.rule("[bold cyan]PORTFOLIO COMBINATION[/bold cyan]")
    total_pnl  = sum(r.get("net_pnl", 0) for r in all_results)
    avg_ann    = sum(r.get("ann_ret", 0) for r in all_results) / len(all_results) if all_results else 0
    console.print(f"[cyan]All-strategy average annualised return: [{('green' if avg_ann >= 40 else 'yellow')}]{avg_ann:+.1f}%[/][/cyan]")
    console.print(f"[cyan]Combined net P&L (sum all strategies × instruments): ₹{total_pnl:+,.0f}[/cyan]")
    console.print()

    fname = save_json_report(all_results)
    console.print(f"[dim]Results saved to {fname}[/dim]")
    console.print()

    # Trade statistics
    console.rule("[bold cyan]TRADE FREQUENCY ANALYSIS[/bold cyan]")
    tbl2 = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold")
    tbl2.add_column("Strategy+Instrument")
    tbl2.add_column("Total Trades", justify="right")
    tbl2.add_column("Trades/Year",  justify="right")
    tbl2.add_column("Avg Win",      justify="right")
    tbl2.add_column("Avg Loss",     justify="right")
    tbl2.add_column("Profit Factor",justify="right")

    for r in all_results:
        nt   = r.get("num_trades", 0)
        tpy  = round(nt / 5.4, 1)
        awl  = r.get("avg_win", 0)
        alss = r.get("avg_loss", 0)
        if alss != 0:
            pf = round(abs(awl / alss) * (r.get("win_rate",0)/100) / (1 - r.get("win_rate",0)/100 + 1e-9), 2)
        else:
            pf = 999.0
        tbl2.add_row(
            f"{r['strategy']} / {r['instrument']}",
            str(nt), f"{tpy:.1f}",
            f"₹{awl:,.0f}" if awl else "—",
            f"₹{alss:,.0f}" if alss else "—",
            f"{pf:.2f}",
        )
    console.print(tbl2)


if __name__ == "__main__":
    main()
