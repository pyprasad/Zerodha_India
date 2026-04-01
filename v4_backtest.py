"""
v4 HIGH-FREQUENCY BACKTEST — Targeting 40%+ YoY
=================================================
Focus: High win-rate mean-reversion + momentum burst strategies
that generate 25-50 trades/year on daily NIFTY/BANKNIFTY data.

Math to hit 40% YoY:
  30 trades × 2% risk × 0.8 EV_ratio = 48% raw
  where EV_ratio = win_rate × RR - (1 - win_rate) × 1

Strategies:
  1. RSI(2) Extreme MR       — Connors-style, 70%+ win rate
  2. Pullback-to-EMA         — Buy dips to EMA21 in uptrend (55-65% win)
  3. Gap-Reversal            — NIFTY gap fade: proven mean-reversion
  4. Momentum Burst          — 3 consecutive down-closes → bounce buy
  5. Williams%R + EMA Filter — High win-rate oscillator timing
  6. Combined Best Signals   — Ensemble of top 3 methods
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
CAPITAL       = 1_000_000   # ₹10,00,000
RISK_PCT      = 0.02        # 2% risk per trade
BROKERAGE     = 20          # ₹20 Zerodha flat rate
EXCHANGE_FEE  = 0.0000125
SLIPPAGE      = 0.0003
MIN_STOP_PCT  = 0.004       # Min 0.4% stop distance
MAX_RISK_PCT  = 0.03        # Hard cap 3% risk per trade

LOT_SIZES = {"NIFTY": 75, "BANKNIFTY": 30}

# ─────────────────────────────────────────────────────────────────────────────
# DATA
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
    df = df.dropna(subset=["open","high","low","close"])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────────────────────────────────────
def ema(series, period): return series.ewm(span=period, adjust=False).mean()
def sma(series, period): return series.rolling(period).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(com=period-1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period-1, adjust=False).mean()
    return 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

def atr(df, period=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def adx(df, period=14):
    h, l = df["high"], df["low"]
    tr = atr(df, 1)
    pdm = (h - h.shift()).clip(lower=0)
    ndm = (l.shift() - l).clip(lower=0)
    pdm = pdm.where(pdm > ndm, 0)
    ndm = ndm.where(ndm > pdm, 0)
    atr_ = tr.ewm(alpha=1/period, adjust=False).mean()
    pdi  = 100 * pdm.ewm(alpha=1/period, adjust=False).mean() / atr_
    ndi  = 100 * ndm.ewm(alpha=1/period, adjust=False).mean() / atr_
    dx   = (100 * (pdi - ndi).abs() / (pdi + ndi)).fillna(0)
    return dx.ewm(alpha=1/period, adjust=False).mean()

def williams_r(df, period=14):
    hh = df["high"].rolling(period).max()
    ll = df["low"].rolling(period).min()
    return -100 * (hh - df["close"]) / (hh - ll).replace(0, np.nan)

def bollinger(df, period=20, std=2.0):
    m = df["close"].rolling(period).mean()
    s = df["close"].rolling(period).std()
    return m - std*s, m, m + std*s


# ─────────────────────────────────────────────────────────────────────────────
# TRADE SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────
class Sim:
    def __init__(self, capital=CAPITAL, risk_pct=RISK_PCT, rr=2.0, instrument="NIFTY"):
        self.C0   = float(capital)
        self.C    = float(capital)
        self.rp   = risk_pct
        self.rr   = rr
        self.lot  = LOT_SIZES.get(instrument, 75)
        self.pos  = None
        self.trades = []

    def _comm(self, qty, price):
        return BROKERAGE*2 + qty * price * (EXCHANGE_FEE + SLIPPAGE) * 2

    def _qty(self, entry, stop):
        risk_amt  = min(self.C * self.rp, self.C * MAX_RISK_PCT)
        risk_unit = max(abs(entry - stop), entry * MIN_STOP_PCT)
        lots = max(1, int(risk_amt / (risk_unit * self.lot)))
        return lots * self.lot

    def enter(self, date, d, entry, stop, tag=""):
        if self.pos: return
        qty = self._qty(entry, stop)
        tgt = entry + d * abs(entry - stop) * self.rr
        self.C -= self._comm(qty, entry)
        self.pos = dict(d=d, entry=entry, stop=stop, target=tgt,
                        qty=qty, date=date, tag=tag)

    def trail_stop(self, new_stop):
        if not self.pos: return
        d = self.pos["d"]
        if d == 1:
            self.pos["stop"] = max(self.pos["stop"], new_stop)
        else:
            self.pos["stop"] = min(self.pos["stop"], new_stop)

    def set_target(self, new_tgt):
        if self.pos: self.pos["target"] = new_tgt

    def check_exit(self, date, high, low, close) -> bool:
        if not self.pos: return False
        d = self.pos["d"]
        stop, tgt, qty, entry = self.pos["stop"], self.pos["target"], self.pos["qty"], self.pos["entry"]
        xp = reason = None
        if d == 1:
            if low  <= stop: xp, reason = stop, "stop"
            elif high >= tgt: xp, reason = tgt, "target"
        else:
            if high >= stop: xp, reason = stop, "stop"
            elif low  <= tgt: xp, reason = tgt, "target"
        if xp is None: return False
        pnl = d * (xp - entry) * qty - self._comm(qty, xp)
        self.C += pnl
        dur = (date - self.pos["date"]).days
        self.trades.append(dict(
            entry_date=self.pos["date"], exit_date=date,
            direction="L" if d==1 else "S",
            entry=round(entry,1), exit=round(xp,1),
            qty=qty, pnl=round(pnl,1),
            pnl_pct=round(pnl/self.C0*100, 4),
            capital=round(self.C,1),
            duration=dur, reason=reason, tag=self.pos["tag"]
        ))
        self.pos = None
        return True

    def force_close(self, date, price):
        if not self.pos: return
        d, entry, qty = self.pos["d"], self.pos["entry"], self.pos["qty"]
        pnl = d * (price - entry) * qty - self._comm(qty, price)
        self.C += pnl
        self.trades.append(dict(
            entry_date=self.pos["date"], exit_date=date,
            direction="L" if d==1 else "S",
            entry=round(entry,1), exit=round(price,1),
            qty=qty, pnl=round(pnl,1),
            pnl_pct=round(pnl/self.C0*100, 4),
            capital=round(self.C,1),
            duration=(date-self.pos["date"]).days,
            reason="force_close", tag=self.pos["tag"]
        ))
        self.pos = None

    def stats(self):
        T = self.trades
        if not T: return {k:0 for k in ["total_ret","ann_ret","sharpe","max_dd","win_rate","num_trades","net_pnl"]}
        df = pd.DataFrame(T)
        wins   = (df.pnl > 0).sum()
        losses = (df.pnl < 0).sum()
        wr     = wins / len(df) * 100
        caps   = pd.Series([self.C0] + list(df.capital))
        dd     = ((caps.cummax() - caps) / caps.cummax()).max() * 100
        total  = (self.C - self.C0) / self.C0 * 100
        t0, t1 = pd.to_datetime(T[0]["entry_date"]), pd.to_datetime(T[-1]["exit_date"])
        yrs    = max((t1-t0).days/365.25, 0.1)
        ann    = ((1+total/100)**(1/yrs)-1)*100
        pps    = df.pnl_pct.values
        sharpe = pps.mean()/pps.std()*np.sqrt(252) if pps.std()>0 else 0
        avgw   = df.loc[df.pnl>0,"pnl"].mean() if wins else 0
        avgl   = df.loc[df.pnl<0,"pnl"].mean() if losses else 0
        return dict(
            total_ret=round(total,2), ann_ret=round(ann,2),
            sharpe=round(sharpe,3), max_dd=round(dd,2),
            win_rate=round(wr,1), num_trades=len(T),
            net_pnl=round(self.C-self.C0), capital=round(self.C),
            avg_win=round(avgw), avg_loss=round(avgl)
        )


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 1 — RSI(2) EXTREME MEAN REVERSION (Connors-style)
#
# Larry Connors' research shows RSI(2) < 10 in an uptrend (price > EMA200)
# produces 70%+ win rate on major indices.
#
# Rules:
#   LONG: Close > EMA200 AND RSI(2) < 10
#   Exit: RSI(2) > 65 OR RSI(2-day) > 65 (quick profit take) OR 2% stop
#   No short side (NIFTY predominantly trends up)
#
# Expected: 30-50 trades/year, 65-75% win, R:R ~1:1 to 1.5:1 → 20-30%/yr
# ─────────────────────────────────────────────────────────────────────────────
def strat_rsi2(df, instrument):
    rsi2  = rsi(df["close"], 2)
    rsi5  = rsi(df["close"], 5)
    e200  = ema(df["close"], 200)
    e20   = ema(df["close"], 20)
    a     = atr(df, 10)
    close = df["close"]

    sim = Sim(rr=1.5, instrument=instrument)

    for i in range(205, len(df)):
        date = df.index[i]
        p    = close.iloc[i]
        ph   = df["high"].iloc[i]
        pl   = df["low"].iloc[i]
        r2   = rsi2.iloc[i]
        r2p  = rsi2.iloc[i-1]
        r5   = rsi5.iloc[i]
        at   = a.iloc[i]

        # Exit: RSI2 crosses above 65 → profit take
        if sim.pos and sim.pos["d"] == 1:
            if r2p < 65 and r2 >= 65:
                sim.force_close(date, p)
                continue
            # Also exit if back above EMA20
            if p > e20.iloc[i] and r5 > 50:
                sim.force_close(date, p)
                continue

        exited = sim.check_exit(date, ph, pl, p)

        if not sim.pos:
            # LONG: price > EMA200 (uptrend) AND RSI2 extremely oversold
            if p > e200.iloc[i] * 0.98 and r2 < 10:
                stop = p - at * 1.5
                sim.enter(date, 1, p, stop, tag="RSI2_OB")

            # SHORT: price < EMA200 (downtrend) AND RSI2 extremely overbought
            elif p < e200.iloc[i] * 1.02 and r2 > 90:
                stop = p + at * 1.5
                sim.enter(date, -1, p, stop, tag="RSI2_OS")

    if sim.pos: sim.force_close(df.index[-1], close.iloc[-1])
    return sim.stats()


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 2 — PULLBACK TO EMA21 IN TREND
#
# In a clear uptrend (EMA21 > EMA55 > EMA200), wait for:
#   - Price pulls back to EMA21 (within 0.5%)
#   - RSI(14) between 40-55 (not oversold/overbought — middle of range)
#   - ADX > 20 (trending)
# Enter long, stop below EMA55 - ATR. Target: 2× risk.
#
# Expected: 20-30 trades/year, 55-65% win rate, 2:1 R:R → 25-35%/yr
# ─────────────────────────────────────────────────────────────────────────────
def strat_ema_pullback(df, instrument):
    e21  = ema(df["close"], 21)
    e55  = ema(df["close"], 55)
    e200 = ema(df["close"], 200)
    r14  = rsi(df["close"], 14)
    dx   = adx(df, 14)
    a    = atr(df, 14)
    close = df["close"]

    sim  = Sim(rr=2.5, instrument=instrument)

    for i in range(210, len(df)):
        date = df.index[i]
        p    = close.iloc[i]
        ph   = df["high"].iloc[i]
        pl   = df["low"].iloc[i]
        at   = a.iloc[i]
        e21v = e21.iloc[i]
        e55v = e55.iloc[i]
        e200v= e200.iloc[i]
        r    = r14.iloc[i]
        dxv  = dx.iloc[i]

        # Trail stop to EMA21 once profitable
        if sim.pos and sim.pos["d"] == 1:
            if p > sim.pos["entry"] * 1.01:
                sim.trail_stop(e21v - at * 0.5)
        if sim.pos and sim.pos["d"] == -1:
            if p < sim.pos["entry"] * 0.99:
                sim.trail_stop(e21v + at * 0.5)

        exited = sim.check_exit(date, ph, pl, p)

        if not sim.pos and not np.isnan(e200v):
            bull_trend = e21v > e55v > e200v * 0.98
            bear_trend = e21v < e55v < e200v * 1.02
            near_e21   = abs(p - e21v) / e21v < 0.006  # within 0.6% of EMA21

            # LONG pullback
            if bull_trend and near_e21 and 38 < r < 58 and dxv > 18:
                stop = e55v - at * 1.0
                if stop < p:
                    sim.enter(date, 1, p, stop, tag="PULLBACK_EMA21_L")

            # SHORT pullback (price rallies to EMA21 in downtrend)
            elif bear_trend and near_e21 and 42 < r < 62 and dxv > 18:
                stop = e55v + at * 1.0
                if stop > p:
                    sim.enter(date, -1, p, stop, tag="PULLBACK_EMA21_S")

    if sim.pos: sim.force_close(df.index[-1], close.iloc[-1])
    return sim.stats()


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 3 — GAP REVERSAL (NIFTY/BANKNIFTY gap-fade)
#
# Indian indices gap up/down significantly at open.
# Large gaps (>0.5%) tend to FILL within 2-3 days → mean reversion trade.
#
# Rules:
#   Gap Down Long: today's open < prev_close × 0.995 AND price > EMA50
#   Gap Up Short:  today's open > prev_close × 1.005 AND price < EMA50
#   Stop: 1.5 × ATR. Target: gap fill (prev close level) OR 1.5:1 R:R.
#
# Expected: 40-60 triggers/year (not all tradeable), 55-65% win
# ─────────────────────────────────────────────────────────────────────────────
def strat_gap_reversal(df, instrument):
    e50  = ema(df["close"], 50)
    e200 = ema(df["close"], 200)
    a    = atr(df, 10)
    r14  = rsi(df["close"], 14)
    close = df["close"]
    open_ = df["open"]
    prev_close = close.shift(1)

    sim  = Sim(rr=1.5, instrument=instrument)

    for i in range(55, len(df)):
        date = df.index[i]
        p    = close.iloc[i]
        ph   = df["high"].iloc[i]
        pl   = df["low"].iloc[i]
        op   = open_.iloc[i]
        pc   = prev_close.iloc[i]
        at   = a.iloc[i]
        e50v = e50.iloc[i]
        e200v= e200.iloc[i]
        r    = r14.iloc[i]

        # Update target to gap-fill level (prev close)
        if sim.pos:
            sim.set_target(pc)  # Target = prev close (gap fill)

        exited = sim.check_exit(date, ph, pl, p)

        if not sim.pos and not np.isnan(e200v):
            gap_pct = (op - pc) / pc * 100

            # Gap DOWN → buy (fade the gap down)
            if gap_pct < -0.6 and p > e200v * 0.97 and r < 60:
                stop = p - at * 1.5
                tgt  = pc  # Fill the gap
                if tgt > p and (tgt - p) / max(p - stop, 0.01) >= 1.3:
                    sim.enter(date, 1, p, stop, tag=f"GAP_DN_{gap_pct:.1f}%")

            # Gap UP → short (fade the gap up)
            elif gap_pct > 0.6 and p < e200v * 1.03 and r > 40:
                stop = p + at * 1.5
                tgt  = pc  # Fill the gap
                if tgt < p and (p - tgt) / max(stop - p, 0.01) >= 1.3:
                    sim.enter(date, -1, p, stop, tag=f"GAP_UP_{gap_pct:.1f}%")

    if sim.pos: sim.force_close(df.index[-1], close.iloc[-1])
    return sim.stats()


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 4 — MOMENTUM BURST (3-BAR PATTERN)
#
# After 3 consecutive lower closes, probability of reversal is high.
# Entry: 3rd consecutive down-close AND price > EMA100 (in uptrend)
# Exit: 1 bar close above entry OR 2% stop
# Very short-term: 1-4 day holds.
#
# Expected: 35-50 trades/year, 60-70% win, R:R ~1.2:1 → 20-30%/yr
# ─────────────────────────────────────────────────────────────────────────────
def strat_momentum_burst(df, instrument):
    e50  = ema(df["close"], 50)
    e100 = ema(df["close"], 100)
    e200 = ema(df["close"], 200)
    a    = atr(df, 10)
    r14  = rsi(df["close"], 14)
    close = df["close"]
    high  = df["high"]

    sim  = Sim(rr=1.5, instrument=instrument, risk_pct=0.015)

    for i in range(210, len(df)):
        date = df.index[i]
        p    = close.iloc[i]
        ph   = df["high"].iloc[i]
        pl   = df["low"].iloc[i]
        at   = a.iloc[i]
        e100v= e100.iloc[i]
        e200v= e200.iloc[i]
        r    = r14.iloc[i]

        # Exit: close above previous high (momentum resumed)
        if sim.pos and sim.pos["d"] == 1:
            if p > high.iloc[i-1]:
                sim.force_close(date, p)
                continue
        if sim.pos and sim.pos["d"] == -1:
            if pl < df["low"].iloc[i-1]:
                sim.force_close(date, p)
                continue

        exited = sim.check_exit(date, ph, pl, p)

        if not sim.pos and not np.isnan(e200v):
            c0 = close.iloc[i]
            c1 = close.iloc[i-1]
            c2 = close.iloc[i-2]
            c3 = close.iloc[i-3]

            # 3 consecutive lower closes in uptrend → bounce buy
            three_down = c0 < c1 < c2 < c3
            if three_down and p > e100v * 0.97 and r > 25 and r < 55:
                stop = p - at * 1.5
                sim.enter(date, 1, p, stop, tag="BURST_3DOWN_L")

            # 3 consecutive higher closes in downtrend → short
            three_up = c0 > c1 > c2 > c3
            if three_up and p < e100v * 1.03 and r < 75 and r > 45:
                stop = p + at * 1.5
                sim.enter(date, -1, p, stop, tag="BURST_3UP_S")

    if sim.pos: sim.force_close(df.index[-1], close.iloc[-1])
    return sim.stats()


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 5 — WILLIAMS %R + EMA TREND FILTER
#
# Williams %R is a fast-moving oscillator good for timing entries in trends.
# LONG: %R < -80 (oversold) AND price > EMA50 AND EMA50 > EMA200
# SHORT: %R > -20 (overbought) AND price < EMA50 AND EMA50 < EMA200
# Stop: 1.5 × ATR. Target: %R reaches -50 (midpoint) OR 2:1 R:R.
#
# Expected: 25-40 trades/year, 55-65% win, 2:1 R:R → 25-35%/yr
# ─────────────────────────────────────────────────────────────────────────────
def strat_williams_r(df, instrument):
    wR   = williams_r(df, 14)
    e21  = ema(df["close"], 21)
    e50  = ema(df["close"], 50)
    e200 = ema(df["close"], 200)
    a    = atr(df, 14)
    r14  = rsi(df["close"], 14)
    close = df["close"]

    sim = Sim(rr=2.0, instrument=instrument)

    for i in range(215, len(df)):
        date  = df.index[i]
        p     = close.iloc[i]
        ph    = df["high"].iloc[i]
        pl    = df["low"].iloc[i]
        wr    = wR.iloc[i]
        wr_p  = wR.iloc[i-1]
        at    = a.iloc[i]
        e21v  = e21.iloc[i]
        e50v  = e50.iloc[i]
        e200v = e200.iloc[i]
        r     = r14.iloc[i]

        # Exit when %R reaches -50 (midpoint) = profit take
        if sim.pos:
            d = sim.pos["d"]
            if d == 1 and wr_p < -50 and wr >= -50:
                sim.force_close(date, p)
                continue
            if d == -1 and wr_p > -50 and wr <= -50:
                sim.force_close(date, p)
                continue
            # Trail stop to EMA21
            if d == 1: sim.trail_stop(e21v - at * 0.3)
            else:       sim.trail_stop(e21v + at * 0.3)

        exited = sim.check_exit(date, ph, pl, p)

        if not sim.pos and not np.isnan(e200v):
            bull = e50v > e200v * 0.99 and p > e200v * 0.97
            bear = e50v < e200v * 1.01 and p < e200v * 1.03

            # LONG: %R crosses above -80 (exits oversold) in bull trend
            if bull and wr_p < -80 and wr >= -80 and r < 65:
                stop = p - at * 1.5
                sim.enter(date, 1, p, stop, tag="WR_OB_L")

            # SHORT: %R crosses below -20 (exits overbought) in bear trend
            elif bear and wr_p > -20 and wr <= -20 and r > 35:
                stop = p + at * 1.5
                sim.enter(date, -1, p, stop, tag="WR_OS_S")

    if sim.pos: sim.force_close(df.index[-1], close.iloc[-1])
    return sim.stats()


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 6 — DUAL TIMEFRAME MOMENTUM + MEAN REVERSION HYBRID
#
# Weekly trend (5-day EMA direction) combined with daily RSI timing.
# When weekly trend is UP and daily is oversold → strong buy signal.
# When weekly trend is DOWN and daily is overbought → strong sell signal.
#
# Rules:
#   LONG: EMA5(weekly) > EMA5(weekly prev) [uptrend] AND daily RSI(14) < 35
#         AND price > 20DMA
#   SHORT: EMA5(weekly) < EMA5(weekly prev) [downtrend] AND daily RSI(14) > 65
#          AND price < 20DMA
# Stop: 2 × ATR. Target: RSI back to 50 OR 2.5:1 R:R
#
# Expected: 25-35 trades/year, 60%+ win, 2.5:1 R:R → 30-40%/yr
# ─────────────────────────────────────────────────────────────────────────────
def strat_dual_tf(df, instrument):
    # Simulate weekly trend using 5-bar MA of closes
    e5w   = df["close"].rolling(5).mean()  # Proxy for 5-day "weekly" EMA
    e20   = ema(df["close"], 20)
    e50   = ema(df["close"], 50)
    e200  = ema(df["close"], 200)
    r14   = rsi(df["close"], 14)
    r5    = rsi(df["close"], 5)
    a     = atr(df, 14)
    bb_lo, bb_mid, bb_hi = bollinger(df, 20, 2.0)
    close = df["close"]

    sim = Sim(rr=2.5, instrument=instrument)

    for i in range(215, len(df)):
        date = df.index[i]
        p    = close.iloc[i]
        ph   = df["high"].iloc[i]
        pl   = df["low"].iloc[i]
        at   = a.iloc[i]
        e5w_v  = e5w.iloc[i]
        e5w_p  = e5w.iloc[i-1]
        e20v   = e20.iloc[i]
        e200v  = e200.iloc[i]
        r      = r14.iloc[i]
        r5v    = r5.iloc[i]
        bbm    = bb_mid.iloc[i]
        bbl    = bb_lo.iloc[i]
        bbh    = bb_hi.iloc[i]

        if np.isnan(e200v): continue

        # Trail to BB midline and EMA20
        if sim.pos:
            d = sim.pos["d"]
            if d == 1:
                sim.trail_stop(e20v - at * 0.5)
                # Exit when RSI recovers to 55+
                if r > 55 and p > bbm:
                    sim.force_close(date, p)
                    continue
            else:
                sim.trail_stop(e20v + at * 0.5)
                if r < 45 and p < bbm:
                    sim.force_close(date, p)
                    continue

        exited = sim.check_exit(date, ph, pl, p)

        if not sim.pos:
            weekly_up   = e5w_v > e5w_p
            weekly_down = e5w_v < e5w_p

            # LONG: Weekly uptrend + daily oversold + not in extreme downmove
            if weekly_up and r < 38 and p > e200v * 0.96 and pl <= bbl:
                stop = min(bbl - at * 0.8, p - at * 2.0)
                sim.enter(date, 1, p, stop, tag="DTF_OB_L")

            # SHORT: Weekly downtrend + daily overbought + not in extreme upmove
            elif weekly_down and r > 62 and p < e200v * 1.04 and ph >= bbh:
                stop = max(bbh + at * 0.8, p + at * 2.0)
                sim.enter(date, -1, p, stop, tag="DTF_OB_S")

    if sim.pos: sim.force_close(df.index[-1], close.iloc[-1])
    return sim.stats()


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 7 — SUPERTREND MOMENTUM (BETTER PARAMETERS)
# SuperTrend(10, 2.0) for trend + RSI(14) dip timing.
# Long: ST bullish AND RSI(14) < 50 (mild dip in uptrend)
# Short: ST bearish AND RSI(14) > 50 (mild rally in downtrend)
# Tighter RSI filter means more signals with good timing.
# ─────────────────────────────────────────────────────────────────────────────
def compute_supertrend(df, period=10, mult=2.0):
    a_   = atr(df, period)
    hl2  = (df["high"] + df["low"]) / 2
    upper= (hl2 + mult * a_).values
    lower= (hl2 - mult * a_).values
    cl   = df["close"].values
    n    = len(cl)
    fu   = np.zeros(n); fl = np.zeros(n); dr = np.ones(n)
    fu[0] = upper[0]; fl[0] = lower[0]
    for i in range(1, n):
        fu[i] = upper[i] if upper[i] < fu[i-1] or cl[i-1] > fu[i-1] else fu[i-1]
        fl[i] = lower[i] if lower[i] > fl[i-1] or cl[i-1] < fl[i-1] else fl[i-1]
        if dr[i-1] == -1 and cl[i] > fu[i]:   dr[i] = 1
        elif dr[i-1] == 1 and cl[i] < fl[i]:  dr[i] = -1
        else:                                   dr[i] = dr[i-1]
    return pd.Series(dr, index=df.index), pd.Series(fu, index=df.index), pd.Series(fl, index=df.index)


def strat_supertrend_v2(df, instrument):
    st_d, st_u, st_l = compute_supertrend(df, 10, 2.0)
    r14   = rsi(df["close"], 14)
    e50   = ema(df["close"], 50)
    e200  = ema(df["close"], 200)
    a     = atr(df, 14)
    close = df["close"]

    sim = Sim(rr=3.0, instrument=instrument)

    for i in range(215, len(df)):
        date  = df.index[i]
        p     = close.iloc[i]
        ph    = df["high"].iloc[i]
        pl    = df["low"].iloc[i]
        at    = a.iloc[i]
        d_cur = st_d.iloc[i]
        d_prv = st_d.iloc[i-1]
        r     = r14.iloc[i]
        r_p   = r14.iloc[i-1]
        e200v = e200.iloc[i]

        if np.isnan(e200v): continue

        # Trail stop to SuperTrend line
        if sim.pos:
            d = sim.pos["d"]
            if d == 1:
                sim.trail_stop(st_l.iloc[i])
                if d_cur == -1:  # ST flipped
                    sim.force_close(date, p); continue
            else:
                sim.trail_stop(st_u.iloc[i])
                if d_cur == 1:
                    sim.force_close(date, p); continue

        exited = sim.check_exit(date, ph, pl, p)

        if not sim.pos:
            stop_d = at * 1.5

            # LONG: ST bullish direction + RSI in 35-55 range (dip in uptrend)
            if d_cur == 1 and 30 < r < 55 and r_p < r and p > e200v * 0.97:
                stop = st_l.iloc[i] - at * 0.3
                sim.enter(date, 1, p, stop, tag="ST2_LONG")

            # SHORT: ST bearish + RSI in 45-65 range (bounce in downtrend)
            elif d_cur == -1 and 45 < r < 70 and r_p > r and p < e200v * 1.03:
                stop = st_u.iloc[i] + at * 0.3
                sim.enter(date, -1, p, stop, tag="ST2_SHORT")

    if sim.pos: sim.force_close(df.index[-1], close.iloc[-1])
    return sim.stats()


# ─────────────────────────────────────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────────────────────────────────────
def print_table(results):
    tbl = Table(
        title="v4 High-Frequency Strategy Backtest — NIFTY & BANKNIFTY (5.4 Years)",
        box=box.DOUBLE_EDGE, show_header=True, header_style="bold cyan", min_width=140
    )
    for col, just, w in [
        ("Strategy","left",28), ("Inst","left",10),
        ("Ann%","right",10), ("Total%","right",10), ("Net P&L","right",14),
        ("Sharpe","right",8), ("MaxDD%","right",8),
        ("Win%","right",7), ("Trades","right",7), ("Trades/yr","right",10)
    ]:
        tbl.add_column(col, justify=just, min_width=w)

    for r in sorted(results, key=lambda x: x["ann_ret"], reverse=True):
        ann  = r["ann_ret"]
        c    = "green" if ann >= 40 else ("yellow" if ann >= 20 else "red")
        tpy  = round(r["num_trades"] / 5.4, 1)
        tbl.add_row(
            r["strategy"], r["instrument"],
            f"[{c}]{ann:+.1f}%[/{c}]",
            f"[{c}]{r['total_ret']:+.1f}%[/{c}]",
            f"₹{r['net_pnl']:+,.0f}",
            f"{r['sharpe']:.2f}",
            f"{r['max_dd']:.1f}%",
            f"{r['win_rate']:.1f}%",
            str(r["num_trades"]),
            str(tpy),
        )
    console.print(tbl)


def print_top_trades(results, top_n=3):
    """Show per-year breakdown for best strategies."""
    best = sorted(results, key=lambda x: x["ann_ret"], reverse=True)[:top_n]
    for b in best:
        console.print(f"\n[bold cyan]{b['strategy']} / {b['instrument']}[/bold cyan]")
        console.print(
            f"  Ann Return: [green]{b['ann_ret']:+.1f}%[/green]  |  "
            f"Total: [green]{b['total_ret']:+.1f}%[/green]  |  "
            f"Sharpe: {b['sharpe']:.2f}  |  "
            f"MaxDD: {b['max_dd']:.1f}%  |  "
            f"Win: {b['win_rate']:.1f}%  |  "
            f"Trades: {b['num_trades']}  |  "
            f"Avg Win: ₹{b.get('avg_win',0):,.0f}  |  "
            f"Avg Loss: ₹{b.get('avg_loss',0):,.0f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
STRATEGIES = [
    ("RSI(2) Extreme MR",        strat_rsi2),
    ("EMA21 Pullback",           strat_ema_pullback),
    ("Gap Reversal",             strat_gap_reversal),
    ("Momentum Burst (3-bar)",   strat_momentum_burst),
    ("Williams%R + EMA",         strat_williams_r),
    ("Dual-TF BB+RSI Hybrid",    strat_dual_tf),
    ("SuperTrend v2 (10,2.0)",   strat_supertrend_v2),
]
INSTRUMENTS = ["NIFTY", "BANKNIFTY"]


def main():
    console.rule("[bold cyan]v4 HIGH-FREQUENCY BACKTEST — 40% YoY TARGET[/bold cyan]")
    console.print(f"[dim]Capital ₹{CAPITAL:,} | Futures lot sizes: NIFTY×75, BANKNIFTY×30 | "
                  f"Risk 2%/trade | Brokerage ₹{BROKERAGE}/order[/dim]\n")

    console.print("[cyan]Loading data...[/cyan]")
    data = {inst: load_data(inst) for inst in INSTRUMENTS}
    console.print()

    all_results = []
    runs = len(STRATEGIES) * len(INSTRUMENTS)
    console.print(f"[cyan]Running {len(STRATEGIES)} strategies × {len(INSTRUMENTS)} instruments = {runs} combinations...[/cyan]\n")

    for sname, sfunc in STRATEGIES:
        for inst in INSTRUMENTS:
            try:
                r = sfunc(data[inst].copy(), inst)
                r["strategy"]   = sname
                r["instrument"] = inst
                ann = r["ann_ret"]
                c   = "green" if ann >= 40 else ("yellow" if ann >= 20 else "white")
                console.print(
                    f"  {sname}/{inst}: [{c}]{ann:+.1f}%/yr[/{c}]  "
                    f"total={r['total_ret']:+.1f}%  win={r['win_rate']:.0f}%  "
                    f"trades={r['num_trades']}  sharpe={r['sharpe']:.2f}"
                )
                all_results.append(r)
            except Exception as e:
                import traceback
                console.print(f"  [red]FAILED {sname}/{inst}: {e}[/red]")
                traceback.print_exc()

    console.print()
    console.rule("[bold cyan]RANKED RESULTS[/bold cyan]")
    print_table(all_results)
    print_top_trades(all_results, 5)

    winners = [r for r in all_results if r["ann_ret"] >= 40]
    if winners:
        console.print(f"\n[bold green]✓ {len(winners)} combinations hit 40%+ YoY target![/bold green]")
        for w in winners:
            console.print(f"  [green]→ {w['strategy']} / {w['instrument']}: "
                          f"{w['ann_ret']:+.1f}%/yr | Sharpe {w['sharpe']:.2f} | "
                          f"MaxDD {w['max_dd']:.1f}% | Win {w['win_rate']:.0f}% | "
                          f"{w['num_trades']} trades[/green]")
    else:
        best = max(all_results, key=lambda r: r["ann_ret"])
        console.print(f"\n[yellow]Best individual: {best['strategy']}/{best['instrument']} "
                      f"@ {best['ann_ret']:+.1f}%/yr[/yellow]")

    # Portfolio aggregate
    console.rule("[bold cyan]PORTFOLIO AGGREGATE[/bold cyan]")
    total_pnl = sum(r["net_pnl"] for r in all_results)
    best_n    = sorted(all_results, key=lambda r: r["ann_ret"], reverse=True)[:4]
    avg_best4 = sum(r["ann_ret"] for r in best_n) / 4
    console.print(f"[cyan]Best-4 strategy average: [{'green' if avg_best4>=40 else 'yellow'}]{avg_best4:+.1f}%/yr[/][/cyan]")
    console.print(f"[cyan]All-strategy total P&L: ₹{total_pnl:+,.0f}[/cyan]")

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"reports/v4_backtest_{ts}.json"
    with open(fname, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    console.print(f"\n[dim]Saved to {fname}[/dim]")


if __name__ == "__main__":
    main()
