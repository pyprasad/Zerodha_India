"""
v10_backtest.py — Per-Instrument Strategy Optimization
=======================================================
Core insight: NIFTY, BANKNIFTY, NIFTYIT, FINNIFTY, MIDCPNIFTY are fundamentally
different instruments. Running the same strategy on all of them is suboptimal.

PHASE 1 — Strategy Discovery Grid
  Run every strategy on every instrument independently (5×7 = 35 combinations).
  Score each: composite = CAGR×0.5 + Sharpe×10 − MaxDD×0.3
  Print a matrix showing which strategy wins per instrument.

PHASE 2 — Per-Instrument Assignment
  Each instrument gets its own best strategy (no forced uniformity).
  All 5 instruments still trade simultaneously in one portfolio.

PHASE 3 — Final Portfolio Benchmark
  Compare per-instrument optimal vs best v9 config (+28.4% CAGR).

STRATEGIES TESTED (7 total):
  1. williams_r_vix   — WR(14) oversold/overbought mean reversion + VIX sizing
  2. vix_spike        — VIX 1-day jump >15% panic mean reversion
  3. supertrend       — SuperTrend(10,2) flip + ADX > 22 trend following
  4. momentum_20d     — 20-day high breakout + volume + ADX > 18
  5. rsi_trend        — RSI(14) cross above/below 55/45 + EMA alignment (NEW)
                        Good for instruments with sustained directional moves
  6. ema_cross        — EMA8 crosses EMA21 + EMA21 > EMA50 + ADX > 20 (NEW)
                        Pure trend-following, works on trending instruments
  7. wr_wide          — WR(14) same signal but ATR×2.0 stops (NEW)
                        Suited for high-volatility instruments like BANKNIFTY

Instrument characteristics:
  NIFTY       : Broad market, macro-driven, mean-reversion + VIX signals proven
  BANKNIFTY   : Banking sector, high vol (2× NIFTY ATR%), banking events key
  NIFTYIT     : IT/Tech, tracks USD/INR + global tech, long trends persist
  FINNIFTY    : Financial sector (NBFCs+insurance+banking), banking-adjacent
  MIDCPNIFTY  : Midcap index, less liquid, trends persist longer than large caps
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
CAPITAL        = 1_000_000
BROKERAGE      = 20
EXCHANGE_FEE   = 0.0000125
SLIPPAGE       = 0.0003
STT_RATE       = 0.0001
MIN_STOP_PCT   = 0.004
ROLL_COST_PCT  = 0.0020

LOT_SIZES = {
    "NIFTY":      65,
    "BANKNIFTY":  30,
    "MIDCPNIFTY": 120,
    "FINNIFTY":   60,
    "NIFTYIT":    30,
}

INSTRUMENT_LIVE_DATE = {
    "NIFTY":      pd.Timestamp("2008-01-01"),
    "BANKNIFTY":  pd.Timestamp("2008-01-01"),
    "FINNIFTY":   pd.Timestamp("2021-07-28"),
    "MIDCPNIFTY": pd.Timestamp("2023-10-03"),
    "NIFTYIT":    pd.Timestamp("2012-01-01"),
}

MAX_LOTS_PER_INST = {
    "NIFTY":      20,
    "BANKNIFTY":  20,
    "MIDCPNIFTY": 15,
    "FINNIFTY":   15,
    "NIFTYIT":    20,
}

RISK_PER_TRADE     = 0.04
MAX_CONCURRENT     = 5
MAX_TOTAL_RISK     = 0.10
WARMUP             = 260
SPAN_MARGIN_PCT    = 0.10
MAINT_MARGIN_RATIO = 0.75

INSTRUMENTS_5 = ["NIFTY", "BANKNIFTY", "MIDCPNIFTY", "FINNIFTY", "NIFTYIT"]

ALL_STRATEGIES = ["williams_r_vix", "vix_spike", "supertrend",
                  "momentum_20d", "rsi_trend", "ema_cross", "wr_wide"]


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────────────────────────────────────────
def load_data(instrument):
    path = Path(f"data/historical/{instrument}_daily_extended.csv")
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    df.columns = [c.lower() for c in df.columns]
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df["volume"] = df["volume"].fillna(0.0)
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[df.index.dayofweek < 5]
    no_move = ((df["open"] == df["close"]) & (df["high"] == df["close"])
               & (df["low"]  == df["close"]))
    df = df[~no_move]
    df.index = df.index.normalize()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────────────────────────────────────
def ema(s, p):
    return s.ewm(span=p, adjust=False).mean()

def atr_series(df, p=14):
    h, l, c = df.high, df.low, df.close
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/p, adjust=False).mean()

def rsi_series(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(com=p-1, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(com=p-1, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def wR_series(df, p=14):
    hh = df.high.rolling(p).max()
    ll = df.low.rolling(p).min()
    return -100 * (hh - df.close) / (hh - ll).replace(0, np.nan)

def adx_series(df, p=14):
    h, l = df.high, df.low
    pdm  = (h - h.shift()).clip(lower=0)
    mdm  = (l.shift() - l).clip(lower=0)
    pdm  = pdm.where(pdm  >= mdm, 0.0)
    mdm  = mdm.where(mdm  > pdm,  0.0)
    atr14 = atr_series(df, p)
    pdi  = 100 * (pdm.ewm(alpha=1/p, adjust=False).mean() / atr14)
    mdi  = 100 * (mdm.ewm(alpha=1/p, adjust=False).mean() / atr14)
    dx   = (abs(pdi - mdi) / (pdi + mdi).replace(0, np.nan)) * 100
    return dx.ewm(alpha=1/p, adjust=False).mean(), pdi, mdi

def supertrend(df, p=10, m=2.0):
    a_  = atr_series(df, p)
    hl2 = (df.high + df.low) / 2
    up  = (hl2 + m * a_).values
    lo  = (hl2 - m * a_).values
    cl  = df.close.values
    n   = len(cl)
    fu  = np.zeros(n); fl = np.zeros(n); dr = np.ones(n)
    fu[0] = up[0]; fl[0] = lo[0]
    for i in range(1, n):
        fu[i] = up[i] if up[i] < fu[i-1] or cl[i-1] > fu[i-1] else fu[i-1]
        fl[i] = lo[i] if lo[i] > fl[i-1] or cl[i-1] < fl[i-1] else fl[i-1]
        if   dr[i-1] == -1 and cl[i] > fu[i]: dr[i] =  1
        elif dr[i-1] ==  1 and cl[i] < fl[i]: dr[i] = -1
        else: dr[i] = dr[i-1]
    return (pd.Series(dr, index=df.index),
            pd.Series(fu, index=df.index),
            pd.Series(fl, index=df.index))

def precompute(df):
    d = {}
    d["close"]     = df.close
    d["high"]      = df.high
    d["low"]       = df.low
    d["open"]      = df.open
    d["volume"]    = df["volume"]
    d["atr"]       = atr_series(df, 14)
    d["rsi14"]     = rsi_series(df.close, 14)
    d["wR14"]      = wR_series(df, 14)
    d["e8"]        = ema(df.close, 8)
    d["e21"]       = ema(df.close, 21)
    d["e50"]       = ema(df.close, 50)
    d["e200"]      = ema(df.close, 200)
    d["adx14"], d["plus_di"], d["minus_di"] = adx_series(df, 14)
    d["high_20d"]  = df.high.rolling(20).max().shift(1)
    d["roc1"]      = df.close.pct_change(21) * 100
    d["roc3"]      = df.close.pct_change(63) * 100
    d["vol_sma20"] = df["volume"].rolling(20).mean()
    d["st_d"], d["st_u"], d["st_l"] = supertrend(df, 10, 2.0)
    return d


# ─────────────────────────────────────────────────────────────────────────────
# VIX SIZING
# ─────────────────────────────────────────────────────────────────────────────
def vix_dynamic_risk(vix_value):
    if vix_value is None or np.isnan(float(vix_value)): return 0.04
    v = float(vix_value)
    if v > 30:  return 0.03   # Extreme panic: reduce (data proves these lose)
    if v > 20:  return 0.05   # Elevated vol: best mean-reversion zone
    if v >= 15: return 0.04
    return 0.03


# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────
class Portfolio:
    def __init__(self, capital=CAPITAL):
        self.C0 = float(capital)
        self.C  = float(capital)
        self.positions       = {}
        self.trades          = []
        self.daily_equity    = []
        self.margin_reserved = {}
        self.margin_calls    = 0
        self.roll_cost_total = 0.0

    def _qty(self, instrument, entry, stop, risk_pct=RISK_PER_TRADE):
        avail_cash   = self.C - sum(self.margin_reserved.values())
        current_risk = sum(abs(p["entry"] - p["stop"]) * p["qty"] / self.C
                          for p in self.positions.values())
        avail_risk = min(risk_pct, MAX_TOTAL_RISK - current_risk)
        if avail_risk <= 0: return 0
        risk_amt  = avail_cash * avail_risk
        lot       = LOT_SIZES.get(instrument, 65)
        risk_unit = max(abs(entry - stop), entry * MIN_STOP_PCT)
        lots      = int(risk_amt / (risk_unit * lot))
        if lots <= 0: return 0
        return min(lots, MAX_LOTS_PER_INST.get(instrument, 20)) * lot

    def enter(self, instrument, strategy, date, direction, entry, stop,
              tag="", risk_pct=RISK_PER_TRADE):
        if instrument in self.positions:            return False
        if len(self.positions) >= MAX_CONCURRENT:   return False
        qty = self._qty(instrument, entry, stop, risk_pct)
        if qty == 0:                                return False
        span_margin = qty * entry * SPAN_MARGIN_PCT
        if self.C - sum(self.margin_reserved.values()) < span_margin:
            return False
        tgt = entry + direction * abs(entry - stop) * 2.0
        entry_cost = BROKERAGE + qty * entry * (EXCHANGE_FEE + SLIPPAGE)
        if direction == -1:
            entry_cost += qty * entry * STT_RATE
        self.C -= entry_cost + span_margin
        self.margin_reserved[instrument] = span_margin
        self.positions[instrument] = dict(
            strategy=strategy, d=direction, entry=entry, stop=stop,
            target=tgt, qty=qty, date=date, tag=tag,
            last_mtm_price=entry
        )
        return True

    def trail_stop(self, instrument, new_stop):
        if instrument not in self.positions: return
        p = self.positions[instrument]
        p["stop"] = max(p["stop"], new_stop) if p["d"] == 1 else min(p["stop"], new_stop)

    def check_exits(self, date, instrument, high, low, close):
        if instrument not in self.positions: return False
        pos = self.positions[instrument]
        d, stop, tgt = pos["d"], pos["stop"], pos["target"]
        if d == 1:
            if low  <= stop: return self._close(instrument, date, stop,  "stop")
            if high >= tgt:  return self._close(instrument, date, tgt,   "target")
        else:
            if high >= stop: return self._close(instrument, date, stop,  "stop")
            if low  <= tgt:  return self._close(instrument, date, tgt,   "target")
        return False

    def force_close(self, instrument, date, price, reason="force"):
        if instrument not in self.positions: return False
        return self._close(instrument, date, price, reason)

    def _close(self, instrument, date, exit_price, reason):
        pos  = self.positions.pop(instrument)
        d, entry, qty = pos["d"], pos["entry"], pos["qty"]
        last_price = pos.get("last_mtm_price", entry)
        self.C += d * (exit_price - last_price) * qty
        self.C += self.margin_reserved.pop(instrument, 0)
        exit_cost = BROKERAGE + qty * exit_price * (EXCHANGE_FEE + SLIPPAGE)
        if d == 1:
            exit_cost += qty * exit_price * STT_RATE
        self.C -= exit_cost
        entry_cost = BROKERAGE + qty * entry * (EXCHANGE_FEE + SLIPPAGE)
        if d == -1:
            entry_cost += qty * entry * STT_RATE
        pnl = d * (exit_price - entry) * qty - entry_cost - exit_cost
        self.trades.append(dict(
            instrument=instrument, strategy=pos["strategy"],
            entry_date=pos["date"], exit_date=date,
            direction="L" if d == 1 else "S",
            entry=round(entry, 1), exit=round(exit_price, 1),
            qty=qty, pnl=round(pnl, 1),
            pnl_pct=round(pnl / self.C0 * 100, 4),
            capital=round(self.C, 1),
            duration=(date - pos["date"]).days,
            reason=reason, tag=pos["tag"]
        ))
        return True

    def record_equity(self, date):
        self.daily_equity.append((date, round(self.C, 1)))

    def stats(self, label=""):
        T = self.trades
        if not T:
            return dict(label=label, ann_ret=0, total_ret=0, sharpe=0,
                        max_dd=0, win_rate=0, num_trades=0, net_pnl=0,
                        capital=round(self.C), score=0, roll_cost_total=0,
                        margin_calls=0)
        df   = pd.DataFrame(T)
        wr   = (df.pnl > 0).sum() / len(df) * 100
        tot  = (self.C - self.C0) / self.C0 * 100
        t0   = pd.to_datetime(T[0]["entry_date"])
        t1   = pd.to_datetime(T[-1]["exit_date"])
        yrs  = max((t1 - t0).days / 365.25, 0.1)
        ann  = ((1 + tot / 100) ** (1 / yrs) - 1) * 100

        if len(self.daily_equity) >= 20:
            eq_vals = pd.Series([v for _, v in self.daily_equity])
            daily_r = eq_vals.pct_change().dropna()
            shr = (daily_r.mean() / daily_r.std() * np.sqrt(252)
                   if daily_r.std() > 0 else 0)
            peak = eq_vals.cummax()
            dd   = ((peak - eq_vals) / peak).max() * 100
        else:
            pps = df.pnl_pct.values
            shr = pps.mean() / pps.std() * np.sqrt(252) if pps.std() > 0 else 0
            dd  = 0

        # Composite score: reward CAGR and Sharpe, penalise drawdown
        score = ann * 0.5 + shr * 10 - dd * 0.3

        return dict(label=label, total_ret=round(tot, 2), ann_ret=round(ann, 2),
                    sharpe=round(shr, 3), max_dd=round(dd, 2), win_rate=round(wr, 1),
                    num_trades=len(T), net_pnl=round(self.C - self.C0),
                    capital=round(self.C), score=round(score, 2),
                    roll_cost_total=round(self.roll_cost_total),
                    margin_calls=self.margin_calls)


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL FUNCTIONS  (7 strategies)
# ─────────────────────────────────────────────────────────────────────────────
def signal_williams_r(i, ind, atr_mult=1.5):
    """WR(14) oversold/overbought mean reversion."""
    wr   = ind["wR14"].iloc[i];   wr_p = ind["wR14"].iloc[i-1]
    e200 = ind["e200"].iloc[i];   e50  = ind["e50"].iloc[i]
    atr_ = ind["atr"].iloc[i];    p    = ind["close"].iloc[i]
    if np.isnan(e200): return None
    if wr_p < -80 and wr >= -80 and p > e200 * 0.97 and e50 > e200 * 0.98:
        return ("L", atr_ * atr_mult, "WR_OB")
    if wr_p > -20 and wr <= -20 and p < e200 * 1.03 and e50 < e200 * 1.02:
        return ("S", atr_ * atr_mult, "WR_OS")
    return None


def signal_wr_wide(i, ind):
    """WR with wider 2.0× ATR stops — suited for high-vol instruments (BANKNIFTY)."""
    return signal_williams_r(i, ind, atr_mult=2.0)


def signal_vix_spike(i, ind, vix_chg_today):
    """VIX 1-day jump/drop >15% — panic mean reversion."""
    if vix_chg_today is None or np.isnan(float(vix_chg_today)): return None
    chg  = float(vix_chg_today)
    atr_ = ind["atr"].iloc[i];  e200 = ind["e200"].iloc[i]
    if np.isnan(e200): return None
    if chg >  0.15: return ("L", atr_ * 1.5, "VIX_SPIKE")
    if chg < -0.15: return ("S", atr_ * 1.5, "VIX_DROP")
    return None


def signal_supertrend(i, ind):
    """SuperTrend direction flip + ADX > 22."""
    st_d  = ind["st_d"].iloc[i];  st_dp = ind["st_d"].iloc[i-1]
    adx   = ind["adx14"].iloc[i]
    e200  = ind["e200"].iloc[i];  atr_  = ind["atr"].iloc[i]
    p     = ind["close"].iloc[i]
    pdi   = ind["plus_di"].iloc[i]; mdi = ind["minus_di"].iloc[i]
    if np.isnan(adx) or np.isnan(e200): return None
    if adx < 22: return None
    if st_dp == -1 and st_d == 1 and p > e200 * 0.97 and pdi > mdi:
        return ("L", atr_ * 2.0, "ST_TREND_L")
    if st_dp == 1 and st_d == -1 and p < e200 * 1.03 and mdi > pdi:
        return ("S", atr_ * 2.0, "ST_TREND_S")
    return None


def signal_momentum_20d(i, ind):
    """20-day high breakout + volume + ADX > 18."""
    h20  = ind["high_20d"].iloc[i]
    p    = ind["close"].iloc[i]
    e50  = ind["e50"].iloc[i];   e200 = ind["e200"].iloc[i]
    adx  = ind["adx14"].iloc[i]; atr_ = ind["atr"].iloc[i]
    vol  = ind["volume"].iloc[i]; vsma = ind["vol_sma20"].iloc[i]
    if np.isnan(h20) or np.isnan(e200) or h20 <= 0: return None
    if adx < 18: return None
    vol_ok = (vsma <= 0) or np.isnan(vsma) or (vol > vsma * 1.2)
    if p > h20 and p > e50 and p > e200 * 0.98 and vol_ok:
        return ("L", atr_ * 2.0, "MOM_20D_L")
    return None


def signal_rsi_trend(i, ind):
    """
    RSI(14) crosses above 55 (long) or below 45 (short) with EMA trend alignment.
    Designed for instruments with sustained directional moves (NIFTYIT, MIDCPNIFTY).
    RSI crossing 55 from below = momentum building in up-trend.
    RSI crossing 45 from above = momentum building in down-trend.
    Requires EMA50 > EMA200 (long) or < EMA200 (short) — only trade with the trend.
    """
    rsi  = ind["rsi14"].iloc[i];  rsi_p = ind["rsi14"].iloc[i-1]
    e8   = ind["e8"].iloc[i];     e21   = ind["e21"].iloc[i]
    e50  = ind["e50"].iloc[i];    e200  = ind["e200"].iloc[i]
    adx  = ind["adx14"].iloc[i];  atr_  = ind["atr"].iloc[i]
    if np.isnan(rsi) or np.isnan(e200) or np.isnan(adx): return None
    if adx < 20: return None   # Only enter with some directional strength
    # Long: RSI crosses above 55, EMA stack bullish (8 > 21 > 50 > 200)
    if (rsi_p < 55 and rsi >= 55 and e50 > e200 * 0.99
            and e21 > e50 * 0.99 and e8 > e21 * 0.99):
        return ("L", atr_ * 2.0, "RSI_BULL")
    # Short: RSI crosses below 45, EMA stack bearish (8 < 21 < 50 < 200)
    if (rsi_p > 45 and rsi <= 45 and e50 < e200 * 1.01
            and e21 < e50 * 1.01 and e8 < e21 * 1.01):
        return ("S", atr_ * 2.0, "RSI_BEAR")
    return None


def signal_ema_cross(i, ind):
    """
    EMA8 crosses above/below EMA21, with EMA21 > EMA50 and ADX > 20.
    Pure trend entry — catches early momentum in trending instruments.
    The EMA21 > EMA50 filter prevents trading into major headwinds.
    Works best on instruments that trend for weeks (NIFTYIT, MIDCPNIFTY).
    """
    e8    = ind["e8"].iloc[i];    e8p   = ind["e8"].iloc[i-1]
    e21   = ind["e21"].iloc[i];   e21p  = ind["e21"].iloc[i-1]
    e50   = ind["e50"].iloc[i];   e200  = ind["e200"].iloc[i]
    adx   = ind["adx14"].iloc[i]; atr_  = ind["atr"].iloc[i]
    pdi   = ind["plus_di"].iloc[i]; mdi = ind["minus_di"].iloc[i]
    if np.isnan(e200) or np.isnan(adx): return None
    if adx < 20: return None
    # Long: EMA8 crossed above EMA21, EMA21 > EMA50, price above EMA200
    if (e8p <= e21p and e8 > e21 and e21 > e50 * 0.99
            and ind["close"].iloc[i] > e200 * 0.97 and pdi > mdi):
        return ("L", atr_ * 2.0, "EMA_BULL")
    # Short: EMA8 crossed below EMA21, EMA21 < EMA50, price below EMA200
    if (e8p >= e21p and e8 < e21 and e21 < e50 * 1.01
            and ind["close"].iloc[i] < e200 * 1.03 and mdi > pdi):
        return ("S", atr_ * 2.0, "EMA_BEAR")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# EXIT ROUTER
# ─────────────────────────────────────────────────────────────────────────────
def process_exit(pf, inst, date, ind, ii, vix_series):
    if inst not in pf.positions: return
    pos  = pf.positions[inst]
    tag  = pos["tag"]
    d    = pos["d"]
    p    = ind["close"].iloc[ii]
    ph   = ind["high"].iloc[ii]
    pl   = ind["low"].iloc[ii]
    e21  = ind["e21"].iloc[ii]
    e50  = ind["e50"].iloc[ii]
    atr_ = ind["atr"].iloc[ii]

    # WR mean-reversion: exit at WR -50 crossover, trail EMA21
    if tag.startswith("WR_"):
        wr14  = ind["wR14"].iloc[ii]; wr14p = ind["wR14"].iloc[ii-1]
        if d == 1 and wr14p < -50 and wr14 >= -50:
            pf.force_close(inst, date, p, "WR_MID_EXIT"); return
        if d == -1 and wr14p > -50 and wr14 <= -50:
            pf.force_close(inst, date, p, "WR_MID_EXIT"); return
        pf.trail_stop(inst, e21 - d * atr_ * 0.3)

    # SuperTrend: exit when ST direction flips against position
    elif tag.startswith("ST_TREND"):
        st_d = ind["st_d"].iloc[ii]
        if d == 1 and st_d == -1:
            pf.force_close(inst, date, p, "ST_FLIP_EXIT"); return
        if d == -1 and st_d == 1:
            pf.force_close(inst, date, p, "ST_FLIP_EXIT"); return
        st_line = ind["st_l"].iloc[ii] if d == 1 else ind["st_u"].iloc[ii]
        pf.trail_stop(inst, st_line)

    # 20-day momentum: exit when price drops below EMA50 or ST flips
    elif tag.startswith("MOM_20D"):
        st_d = ind["st_d"].iloc[ii]
        if d == 1 and (st_d == -1 or p < e50 * 0.99):
            pf.force_close(inst, date, p, "MOM_EXIT"); return
        pf.trail_stop(inst, e21 - d * atr_ * 0.5)

    # RSI trend: exit when RSI crosses back through 50
    elif tag.startswith("RSI_"):
        rsi  = ind["rsi14"].iloc[ii]; rsi_p = ind["rsi14"].iloc[ii-1]
        if d == 1 and rsi_p > 50 and rsi <= 50:
            pf.force_close(inst, date, p, "RSI_EXIT"); return
        if d == -1 and rsi_p < 50 and rsi >= 50:
            pf.force_close(inst, date, p, "RSI_EXIT"); return
        pf.trail_stop(inst, e50 - d * atr_ * 0.5)   # Trail EMA50

    # EMA cross: exit when EMA8 crosses back against us
    elif tag.startswith("EMA_"):
        e8    = ind["e8"].iloc[ii]; e8p   = ind["e8"].iloc[ii-1]
        e21c  = ind["e21"].iloc[ii]; e21p  = ind["e21"].iloc[ii-1]
        if d == 1 and e8p >= e21p and e8 < e21c:
            pf.force_close(inst, date, p, "EMA_CROSS_EXIT"); return
        if d == -1 and e8p <= e21p and e8 > e21c:
            pf.force_close(inst, date, p, "EMA_CROSS_EXIT"); return
        pf.trail_stop(inst, e21c - d * atr_ * 0.3)

    # VIX spike: exit when VIX normalises to 5-bar mean
    elif tag.startswith("VIX_"):
        if date in vix_series.index:
            vi = vix_series.index.get_loc(date)
            if vi >= 5:
                vix_sma5 = vix_series.iloc[vi-5:vi].mean()
                vix_now  = vix_series.iloc[vi]
                if (not np.isnan(vix_sma5) and not np.isnan(vix_now)
                        and vix_sma5 > 0
                        and abs(vix_now - vix_sma5) / vix_sma5 < 0.05):
                    pf.force_close(inst, date, p, "VIX_NORM"); return

    # Monthly rotation
    elif tag.startswith("MONTHLY"):
        e200 = ind["e200"].iloc[ii]
        if d == 1 and not np.isnan(e200) and p < e200:
            pf.force_close(inst, date, p, "MONTHLY_EMA_EXIT"); return
        if d == 1:
            pf.trail_stop(inst, e50 - atr_ * 0.5)

    if inst in pf.positions:
        pf.check_exits(date, inst, ph, pl, p)


# ─────────────────────────────────────────────────────────────────────────────
# CORE RUNNER  (single instrument OR multi-instrument)
# ─────────────────────────────────────────────────────────────────────────────
def run_backtest(instrument_strategy_map, capital=CAPITAL):
    """
    instrument_strategy_map: dict of {instrument: [strategy_names]}
    Each instrument only receives signals from its assigned strategies.
    """
    instruments = list(instrument_strategy_map.keys())
    data = {}; inds = {}
    for inst in instruments:
        df = load_data(inst)
        data[inst] = df
        inds[inst] = precompute(df)

    vix_df = load_data("INDIAVIX")
    union_set = set.union(*[set(data[inst].index) for inst in instruments])
    all_dates = pd.DatetimeIndex(sorted(union_set))

    vix_chg_raw = vix_df.close.pct_change()
    vix_series  = vix_df.close.reindex(all_dates, method="ffill")
    vix_chg     = vix_chg_raw.reindex(all_dates)

    pf = Portfolio(capital)
    pending_entries = {}
    pending_exits   = {}
    last_roll_month = (-1, -1)

    for idx, date in enumerate(all_dates):
        if idx < WARMUP: continue

        vix_today     = vix_series.iloc[idx] if idx < len(vix_series) else None
        vix_chg_today = vix_chg.iloc[idx]    if idx < len(vix_chg)    else None
        if vix_today     is not None and (pd.isna(vix_today)     or np.isnan(float(vix_today))):
            vix_today = None
        if vix_chg_today is not None and (pd.isna(vix_chg_today) or np.isnan(float(vix_chg_today))):
            vix_chg_today = None

        # Execute pending exits at open
        for inst, reason in list(pending_exits.items()):
            del pending_exits[inst]
            if inst not in pf.positions: continue
            if inst not in data or date not in data[inst].index: continue
            ii = data[inst].index.get_loc(date)
            pf.force_close(inst, date, float(inds[inst]["open"].iloc[ii]), reason)

        # Execute pending entries at open
        for inst, (strat, d_val, stop_dist, tag, risk) in list(pending_entries.items()):
            del pending_entries[inst]
            if inst in pf.positions: continue
            if inst not in data or date not in data[inst].index: continue
            ii    = data[inst].index.get_loc(date)
            entry = float(inds[inst]["open"].iloc[ii])
            stop  = entry - d_val * stop_dist
            pf.enter(inst, strat, date, d_val, entry, stop, tag, risk)

        # Exits
        for inst in list(pf.positions.keys()):
            if inst not in data or date not in data[inst].index: continue
            ii = data[inst].index.get_loc(date)
            if ii < 1: continue
            process_exit(pf, inst, date, inds[inst], ii, vix_series)

        # Daily MTM
        for inst, pos in list(pf.positions.items()):
            if inst not in data or date not in data[inst].index: continue
            ii          = data[inst].index.get_loc(date)
            today_close = float(inds[inst]["close"].iloc[ii])
            last_price  = pos.get("last_mtm_price", pos["entry"])
            pf.C       += pos["d"] * (today_close - last_price) * pos["qty"]
            pos["last_mtm_price"] = today_close

        # Margin call
        if pf.margin_reserved:
            total_maint = sum(pf.margin_reserved.values()) * MAINT_MARGIN_RATIO
            if pf.C < total_maint:
                pf.margin_calls += 1
                for inst in list(pf.positions.keys()):
                    if inst in data and date in data[inst].index:
                        ii = data[inst].index.get_loc(date)
                        pf.force_close(inst, date,
                                       float(inds[inst]["close"].iloc[ii]), "margin_call")

        # Roll cost
        cur_month = (date.year, date.month)
        if cur_month != last_roll_month and pf.positions:
            if idx + 1 >= len(all_dates) or all_dates[idx + 1].month != date.month:
                last_roll_month = cur_month
                for inst, pos in list(pf.positions.items()):
                    if inst not in data or date not in data[inst].index: continue
                    ii = data[inst].index.get_loc(date)
                    close_price = float(inds[inst]["close"].iloc[ii])
                    roll_cost   = pos["qty"] * close_price * ROLL_COST_PCT
                    pf.C -= roll_cost
                    pf.roll_cost_total += roll_cost

        # Entry signals → pending (per-instrument strategy routing)
        for inst in instruments:
            if inst in pf.positions or inst in pending_entries: continue
            if date < INSTRUMENT_LIVE_DATE.get(inst, pd.Timestamp("2008-01-01")): continue
            if inst not in data or date not in data[inst].index: continue
            ii = data[inst].index.get_loc(date)
            if ii < WARMUP: continue
            if idx + 1 >= len(all_dates): continue

            ind   = inds[inst]
            strats = instrument_strategy_map[inst]
            sig   = None

            # Fire signals in priority order for this instrument's assigned strategies
            if "vix_spike" in strats and inst == "NIFTY":
                sig = signal_vix_spike(ii, ind, vix_chg_today)
                if sig:
                    pending_entries[inst] = (sig[2], 1 if sig[0]=="L" else -1,
                                             sig[1], sig[2], RISK_PER_TRADE)
                    continue

            if sig is None and "williams_r_vix" in strats:
                s = signal_williams_r(ii, ind)
                if s:
                    risk = vix_dynamic_risk(vix_today)
                    pending_entries[inst] = (f"WR_VIX_{s[2]}", 1 if s[0]=="L" else -1,
                                             s[1], s[2], risk)
                    continue

            if sig is None and "wr_wide" in strats:
                s = signal_wr_wide(ii, ind)
                if s:
                    risk = vix_dynamic_risk(vix_today)
                    pending_entries[inst] = (f"WR_VIX_{s[2]}", 1 if s[0]=="L" else -1,
                                             s[1], s[2], risk)
                    continue

            if sig is None and "rsi_trend" in strats:
                sig = signal_rsi_trend(ii, ind)

            if sig is None and "ema_cross" in strats:
                sig = signal_ema_cross(ii, ind)

            if sig is None and "supertrend" in strats:
                sig = signal_supertrend(ii, ind)

            if sig is None and "momentum_20d" in strats:
                sig = signal_momentum_20d(ii, ind)

            if sig is None: continue
            d_val = 1 if sig[0] == "L" else -1
            pending_entries[inst] = (sig[2], d_val, sig[1], sig[2], RISK_PER_TRADE)

        pf.record_equity(date)

    # Force-close at end
    for inst in list(pf.positions.keys()):
        last_p = float(data[inst].close.iloc[-1])
        pf.force_close(inst, all_dates[-1], last_p, "end_of_backtest")

    return pf


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def annual_breakdown(pf):
    if not pf.daily_equity: return []
    eq_df = pd.DataFrame(pf.daily_equity, columns=["date", "equity"])
    eq_df["date"] = pd.to_datetime(eq_df["date"])
    eq_df = eq_df.set_index("date")
    rows = []
    for yr in sorted(eq_df.index.year.unique()):
        yr_data  = eq_df[eq_df.index.year == yr]
        if yr_data.empty: continue
        prev     = eq_df[eq_df.index.year < yr]
        start_v  = prev["equity"].iloc[-1] if not prev.empty else pf.C0
        end_v    = yr_data["equity"].iloc[-1]
        pnl      = end_v - start_v
        ret      = pnl / start_v * 100
        yr_trades = [t for t in pf.trades
                     if pd.to_datetime(t["exit_date"]).year == yr]
        wr = ((sum(1 for t in yr_trades if t["pnl"] > 0) / len(yr_trades) * 100)
              if yr_trades else 0.0)
        rows.append(dict(year=yr, trades=len(yr_trades), win_rate=round(wr, 1),
                         pnl=round(pnl), ret_pct=round(ret, 2)))
    return rows


def instrument_stats(pf):
    df = pd.DataFrame(pf.trades)
    if df.empty: return []
    results = []
    for inst, grp in df.groupby("instrument"):
        wr = (grp.pnl > 0).sum() / len(grp) * 100
        aw = grp.loc[grp.pnl > 0, "pnl"].mean() if (grp.pnl > 0).any() else 0
        results.append(dict(instrument=inst, trades=len(grp),
                            win_rate=round(wr, 1), net_pnl=round(grp.pnl.sum()),
                            avg_win=round(aw)))
    return sorted(results, key=lambda x: x["net_pnl"], reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: STRATEGY DISCOVERY GRID
# ─────────────────────────────────────────────────────────────────────────────
def run_discovery_grid():
    """
    For each instrument × strategy: run single-instrument backtest.
    Returns dict: {instrument: {strategy: stats}}
    """
    console.rule("[bold yellow]PHASE 1 — Strategy Discovery Grid (5 instruments × 7 strategies)[/bold yellow]")
    console.print("[dim]Running each strategy on each instrument independently...[/dim]\n")

    grid = {}   # {inst: {strat: stats_dict}}

    for inst in INSTRUMENTS_5:
        grid[inst] = {}
        live_date = INSTRUMENT_LIVE_DATE[inst]
        console.print(f"  [cyan]{inst:12s}[/cyan] (live from {live_date.date()})")

        for strat in ALL_STRATEGIES:
            # vix_spike only applies to NIFTY (it's a market-wide signal)
            if strat == "vix_spike" and inst != "NIFTY":
                grid[inst][strat] = None
                continue

            try:
                pf = run_backtest({inst: [strat]}, capital=CAPITAL)
                s  = pf.stats(f"{inst}_{strat}")
                grid[inst][strat] = s
                color = "green" if s["ann_ret"] >= 25 else ("yellow" if s["ann_ret"] >= 10 else "red")
                console.print(
                    f"    {strat:20s} [{color}]{s['ann_ret']:+.1f}%[/{color}] ann | "
                    f"Sharpe {s['sharpe']:.2f} | DD {s['max_dd']:.1f}% | "
                    f"{s['num_trades']:3d} trades | score {s['score']:+.1f}"
                )
            except Exception as e:
                console.print(f"    {strat:20s} [red]FAILED: {e}[/red]")
                grid[inst][strat] = None

        console.print()

    return grid


def best_strategy_per_instrument(grid):
    """Pick the highest-score strategy per instrument."""
    assignment = {}
    for inst in INSTRUMENTS_5:
        best_strat = None; best_score = -999
        for strat, s in grid[inst].items():
            if s is None: continue
            if s["num_trades"] < 5: continue   # Need enough trades to be meaningful
            if s["score"] > best_score:
                best_score = s["score"]
                best_strat = strat
        assignment[inst] = best_strat or "williams_r_vix"   # Fallback
    return assignment


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    console.rule("[bold cyan]v10 — Per-Instrument Strategy Optimization[/bold cyan]")
    console.print(
        "[dim]The same strategy running on all instruments wastes alpha.\n"
        "BANKNIFTY ≠ NIFTYIT ≠ MIDCPNIFTY — each has unique volatility and trend profile.\n"
        "This backtest finds the optimal strategy per instrument, then combines them.[/dim]\n"
    )

    # ── PHASE 1: Discovery ───────────────────────────────────────────────────
    grid = run_discovery_grid()

    # Print discovery grid summary
    console.rule("[bold yellow]Discovery Grid Summary[/bold yellow]")
    gtbl = Table(box=box.DOUBLE_EDGE, show_header=True, header_style="bold yellow",
                 title="Score = CAGR×0.5 + Sharpe×10 − MaxDD×0.3  |  ★ = best for instrument")
    gtbl.add_column("Strategy",       min_width=18)
    for inst in INSTRUMENTS_5:
        gtbl.add_column(inst, justify="right", min_width=12)

    # Find best per instrument for marking
    best_per_inst = {}
    for inst in INSTRUMENTS_5:
        best_s = None; best_sc = -999
        for strat, s in grid[inst].items():
            if s and s["num_trades"] >= 5 and s["score"] > best_sc:
                best_sc = s["score"]; best_s = strat
        best_per_inst[inst] = best_s

    for strat in ALL_STRATEGIES:
        row = [strat]
        for inst in INSTRUMENTS_5:
            s = grid[inst].get(strat)
            if s is None or s["num_trades"] < 5:
                row.append("[dim]n/a[/dim]")
            else:
                is_best = (best_per_inst.get(inst) == strat)
                color = "bold green" if is_best else ("green" if s["ann_ret"] >= 20 else
                        ("yellow" if s["ann_ret"] >= 10 else "red"))
                marker = "★" if is_best else " "
                row.append(f"[{color}]{marker}{s['ann_ret']:+.1f}%[/{color}]\n"
                            f"[dim]sh:{s['sharpe']:.2f} dd:{s['max_dd']:.0f}%[/dim]")
        gtbl.add_row(*row)
    console.print(gtbl)

    # ── PHASE 2: Per-instrument assignment ──────────────────────────────────
    console.rule("[bold yellow]PHASE 2 — Per-Instrument Strategy Assignment[/bold yellow]")
    assignment = best_strategy_per_instrument(grid)

    atbl = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold",
                 title="Per-Instrument Optimal Strategy")
    atbl.add_column("Instrument",   min_width=14)
    atbl.add_column("Best Strategy",min_width=20)
    atbl.add_column("CAGR",         justify="right")
    atbl.add_column("Sharpe",       justify="right")
    atbl.add_column("MaxDD",        justify="right")
    atbl.add_column("Score",        justify="right")
    atbl.add_column("Rationale",    min_width=40)

    inst_notes = {
        "NIFTY":      "Macro index — mean-reversion + VIX events",
        "BANKNIFTY":  "Banking sector — high-vol mean-reversion",
        "NIFTYIT":    "IT/Tech — long trends, USD/INR driven",
        "FINNIFTY":   "Financial sector — banking-adjacent",
        "MIDCPNIFTY": "Midcap — trends persist, breakouts common",
    }
    for inst in INSTRUMENTS_5:
        strat = assignment[inst]
        s = grid[inst].get(strat, {}) or {}
        c = "green" if s.get("ann_ret", 0) >= 20 else ("yellow" if s.get("ann_ret", 0) >= 10 else "red")
        atbl.add_row(
            inst, strat,
            f"[{c}]{s.get('ann_ret',0):+.1f}%[/{c}]",
            f"{s.get('sharpe',0):.2f}",
            f"{s.get('max_dd',0):.1f}%",
            f"{s.get('score',0):+.1f}",
            inst_notes.get(inst, "")
        )
        console.print(f"  [bold]{inst:12s}[/bold] → [cyan]{strat}[/cyan]  "
                      f"({s.get('ann_ret',0):+.1f}% CAGR, Sharpe {s.get('sharpe',0):.2f})")
    console.print(atbl)

    # ── PHASE 3: Combined per-instrument portfolio ───────────────────────────
    console.rule("[bold cyan]PHASE 3 — Combined Per-Instrument Portfolio[/bold cyan]")

    # Map each instrument to a list (each gets its own optimal strategy)
    inst_strat_map = {inst: [assignment[inst]] for inst in INSTRUMENTS_5}

    console.print("  Running combined portfolio with per-instrument optimal strategies...")
    pf_optimal = run_backtest(inst_strat_map, capital=CAPITAL)
    s_opt = pf_optimal.stats("PER_INST_OPTIMAL")

    # Also run the v9 baseline for comparison
    console.print("  Running v9 baseline (WR_VIX on all 5 instruments)...")
    pf_base = run_backtest({inst: ["williams_r_vix"] for inst in INSTRUMENTS_5}, capital=CAPITAL)
    s_base = pf_base.stats("WR_VIX_ALL")

    # Also run: each instrument gets TOP-2 strategies (primary + secondary)
    console.print("  Running top-2 strategies per instrument...")
    top2_map = {}
    for inst in INSTRUMENTS_5:
        scored = [(st, grid[inst][st]["score"])
                  for st in ALL_STRATEGIES
                  if grid[inst].get(st) and grid[inst][st]["num_trades"] >= 5]
        scored.sort(key=lambda x: x[1], reverse=True)
        top2_map[inst] = [s[0] for s in scored[:2]] or ["williams_r_vix"]
    pf_top2 = run_backtest(top2_map, capital=CAPITAL)
    s_top2  = pf_top2.stats("TOP2_PER_INST")

    console.print()
    console.rule("[bold cyan]Final Comparison[/bold cyan]")

    ctbl = Table(box=box.DOUBLE_EDGE, show_header=True, header_style="bold cyan")
    ctbl.add_column("Portfolio",     min_width=22)
    ctbl.add_column("Ann Ret%",      justify="right", min_width=10)
    ctbl.add_column("Total%",        justify="right", min_width=11)
    ctbl.add_column("Sharpe",        justify="right", min_width=8)
    ctbl.add_column("MaxDD%",        justify="right", min_width=8)
    ctbl.add_column("Win%",          justify="right", min_width=7)
    ctbl.add_column("Trades",        justify="right", min_width=7)
    ctbl.add_column("Net P&L",       justify="right", min_width=16)
    ctbl.add_column("Final Capital", justify="right", min_width=16)

    all_results = [
        ("v9 Baseline (WR all)", s_base),
        ("Per-Inst Optimal",     s_opt),
        ("Top-2 Per-Inst",       s_top2),
    ]

    best_ann = max(s["ann_ret"] for _, s in all_results)
    for name, s in sorted(all_results, key=lambda x: x[1]["ann_ret"], reverse=True):
        is_best = abs(s["ann_ret"] - best_ann) < 0.1
        c = "bold green" if is_best else ("green" if s["ann_ret"] >= 25 else
            ("yellow" if s["ann_ret"] >= 15 else "red"))
        ctbl.add_row(
            f"[{c}]{'★ ' if is_best else '  '}{name}[/{c}]",
            f"[{c}]{s['ann_ret']:+.1f}%[/{c}]",
            f"{s['total_ret']:+.1f}%",
            f"{s['sharpe']:.2f}",
            f"{s['max_dd']:.1f}%",
            f"{s['win_rate']:.1f}%",
            str(s["num_trades"]),
            f"₹{s['net_pnl']:+,.0f}",
            f"₹{s['capital']:,.0f}",
        )
    console.print(ctbl)

    # Year-by-year for best result
    best_pf = max([(pf_optimal, s_opt), (pf_base, s_base), (pf_top2, s_top2)],
                  key=lambda x: x[1]["ann_ret"])[0]
    best_s  = max([s_opt, s_base, s_top2], key=lambda x: x["ann_ret"])

    ybl  = annual_breakdown(best_pf)
    ytbl = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold",
                 title=f"Year-by-Year — {best_s['label']}")
    ytbl.add_column("Year"); ytbl.add_column("Trades", justify="right")
    ytbl.add_column("Win%", justify="right"); ytbl.add_column("P&L", justify="right")
    ytbl.add_column("Return%", justify="right")
    for row in ybl:
        c = "green" if row["ret_pct"] >= 20 else ("yellow" if row["ret_pct"] >= 5 else "red")
        ytbl.add_row(str(row["year"]), str(row["trades"]),
                     f"{row['win_rate']:.1f}%", f"₹{row['pnl']:+,.0f}",
                     f"[{c}]{row['ret_pct']:+.2f}%[/{c}]")
    console.print(ytbl)

    # Instrument breakdown for best portfolio
    inst_attr = instrument_stats(best_pf)
    if inst_attr:
        itbl = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold",
                     title=f"Instrument Breakdown — {best_s['label']}")
        itbl.add_column("Instrument", min_width=14)
        itbl.add_column("Strategy",   min_width=18)
        itbl.add_column("Trades",     justify="right")
        itbl.add_column("Win%",       justify="right")
        itbl.add_column("Net P&L",    justify="right")
        itbl.add_column("Avg Win",    justify="right")
        for a in inst_attr:
            c = "green" if a["net_pnl"] > 0 else "red"
            itbl.add_row(
                a["instrument"],
                assignment.get(a["instrument"], "?"),
                str(a["trades"]),
                f"{a['win_rate']:.1f}%",
                f"[{c}]₹{a['net_pnl']:+,.0f}[/{c}]",
                f"₹{a['avg_win']:,.0f}"
            )
        console.print(itbl)

    # Save results
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"reports/v10_backtest_{ts}.json"
    all_data = {
        "discovery_grid": {
            inst: {strat: s for strat, s in strats.items() if s}
            for inst, strats in grid.items()
        },
        "assignment": assignment,
        "top2_assignment": top2_map,
        "results": {
            "per_inst_optimal": {"stats": s_opt,  "annual": annual_breakdown(pf_optimal)},
            "v9_baseline":      {"stats": s_base, "annual": annual_breakdown(pf_base)},
            "top2_per_inst":    {"stats": s_top2, "annual": annual_breakdown(pf_top2)},
        },
    }
    with open(fname, "w") as f:
        json.dump(all_data, f, indent=2, default=str)
    console.print(f"\n[dim]Results saved → {fname}[/dim]")

    # Final panel
    ann    = best_s["ann_ret"]
    v9_ann = 28.4
    eq_dates = [d for d, _ in best_pf.daily_equity]
    period   = (f"{eq_dates[0].date()} → {eq_dates[-1].date()}"
                f" (~{(eq_dates[-1]-eq_dates[0]).days/365.25:.1f} yrs)"
                if eq_dates else "n/a")

    strat_lines = "\n".join(
        f"    {inst:12s} → {assignment[inst]}"
        for inst in INSTRUMENTS_5
    )
    console.print(Panel(
        f"[bold {'green' if ann >= 28 else 'yellow'}]"
        f"v10 RESULT — {ann:+.1f}% annualised ({best_s['label']})"
        f"[/bold {'green' if ann >= 28 else 'yellow'}]\n\n"
        f"  Period         : {period}\n"
        f"  vs v9 baseline : {ann:+.1f}% vs {v9_ann:+.1f}%  "
        f"({'▲ IMPROVED' if ann > v9_ann else '▼ WORSE'} by {abs(ann-v9_ann):.1f}%)\n"
        f"  Sharpe         : {best_s['sharpe']:.2f}\n"
        f"  Max Drawdown   : {best_s['max_dd']:.1f}%\n"
        f"  Win Rate       : {best_s['win_rate']:.1f}%\n"
        f"  Net P&L        : ₹{best_s['net_pnl']:+,.0f}\n"
        f"  Final Capital  : ₹{best_s['capital']:,.0f}\n\n"
        f"  Per-instrument optimal strategy:\n{strat_lines}",
        title="[bold]RESULT — v10 PER-INSTRUMENT OPTIMIZATION[/bold]",
        border_style="green" if ann >= 28 else "yellow"
    ))

    return grid, assignment


if __name__ == "__main__":
    main()
