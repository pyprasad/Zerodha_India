"""
v11_backtest.py — Individual Stock Futures + Per-Instrument Strategy Optimization
==================================================================================
Key insight from v10: Indices all revert (WR wins everywhere). Individual stocks are
fundamentally different — they CAN trend for weeks driven by earnings, sector rotation,
business cycles. This version expands to stock futures and finds genuinely optimal
strategies per instrument.

UNIVERSE (20 instruments):
  5 Index futures  : NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY, NIFTYIT
  15 Stock futures : RELIANCE, HDFCBANK, INFY, TCS, ICICIBANK, AXISBANK, SBIN,
                     BAJFINANCE, HCLTECH, ITC, LT, SUNPHARMA, KOTAKBANK, MARUTI, WIPRO

STRATEGIES (9 total):
  Mean-reversion : williams_r_vix, wr_wide, bb_reversion (NEW)
  Trend-following: supertrend, ema_cross, rsi_trend
  Momentum       : momentum_20d, macd_signal (NEW), vol_breakout (NEW)

KEY DIFFERENCE — Stocks vs Indices:
  Indices    : weighted baskets → mean-revert (one stock falls, another rises)
  Stocks     : can persistently trend → earnings surprises, sector rotations last weeks
  Stock margin: 15% SPAN (vs 10% for index futures — NSE rule)
  Stock risk  : 2% per trade (vs 4% for indices — more conservative; stocks can gap)

PHASES:
  1. Strategy Discovery — 9 strategies × 20 instruments = 180 single-instrument backtests
  2. Per-Instrument Assignment — best strategy by composite score
  3. Combined Portfolio — all 20 instruments, each with its optimal strategy
  4. Versioned P&L Statement — v7 → v11 progression
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
CAPITAL            = 1_000_000
BROKERAGE          = 20
EXCHANGE_FEE       = 0.0000125
SLIPPAGE           = 0.0003
STT_RATE           = 0.0001
MIN_STOP_PCT       = 0.005
ROLL_COST_PCT      = 0.0020      # 0.20%/month roll cost

INDEX_SPAN_PCT     = 0.10        # 10% SPAN margin for index futures
STOCK_SPAN_PCT     = 0.15        # 15% SPAN margin for stock futures (NSE rule)
MAINT_MARGIN_RATIO = 0.75

INDEX_RISK         = 0.04        # 4% risk per index trade
STOCK_RISK         = 0.02        # 2% risk per stock trade (stocks can gap big)
MAX_TOTAL_RISK     = 0.15        # 15% total portfolio risk (20 instruments justify more)
MAX_CONCURRENT     = 10          # Up to 10 simultaneous positions
WARMUP             = 260         # 1 year warmup for indicators

# NSE F&O lot sizes (current 2024-2025)
LOT_SIZES = {
    # Index futures
    "NIFTY":      65,   "BANKNIFTY":  30,   "MIDCPNIFTY": 120,
    "FINNIFTY":   60,   "NIFTYIT":    30,
    # Stock futures
    "RELIANCE":   250,  "HDFCBANK":   550,  "INFY":       300,
    "TCS":        150,  "ICICIBANK":  700,  "AXISBANK":   625,
    "SBIN":       1500, "BAJFINANCE": 125,  "HCLTECH":    700,
    "ITC":        3200, "LT":         300,  "SUNPHARMA":  700,
    "KOTAKBANK":  400,  "MARUTI":     100,  "WIPRO":      1500,
}

MAX_LOTS = {
    # Index futures — allow more lots (liquid, lower margin per notional)
    "NIFTY":      20,   "BANKNIFTY":  20,   "MIDCPNIFTY": 15,
    "FINNIFTY":   15,   "NIFTYIT":    20,
    # Stock futures — cap at 2 lots (contract values are large; capital limited)
    "RELIANCE":   2,    "HDFCBANK":   2,    "INFY":       3,
    "TCS":        3,    "ICICIBANK":  2,    "AXISBANK":   2,
    "SBIN":       2,    "BAJFINANCE": 3,    "HCLTECH":    2,
    "ITC":        2,    "LT":         3,    "SUNPHARMA":  2,
    "KOTAKBANK":  2,    "MARUTI":     3,    "WIPRO":      2,
}

INSTRUMENT_LIVE_DATE = {
    # Index futures (F&O launch dates)
    "NIFTY":      pd.Timestamp("2008-01-01"),
    "BANKNIFTY":  pd.Timestamp("2008-01-01"),
    "FINNIFTY":   pd.Timestamp("2021-07-28"),
    "MIDCPNIFTY": pd.Timestamp("2023-10-03"),
    "NIFTYIT":    pd.Timestamp("2012-01-01"),
    # Stock futures — use 2008 as start (enough warmup from 1996-2008 data)
    "RELIANCE":   pd.Timestamp("2008-01-01"),
    "HDFCBANK":   pd.Timestamp("2008-01-01"),
    "INFY":       pd.Timestamp("2008-01-01"),
    "TCS":        pd.Timestamp("2008-01-01"),
    "ICICIBANK":  pd.Timestamp("2008-01-01"),
    "AXISBANK":   pd.Timestamp("2008-01-01"),
    "SBIN":       pd.Timestamp("2008-01-01"),
    "BAJFINANCE": pd.Timestamp("2008-01-01"),
    "HCLTECH":    pd.Timestamp("2008-01-01"),
    "ITC":        pd.Timestamp("2008-01-01"),
    "LT":         pd.Timestamp("2008-01-01"),
    "SUNPHARMA":  pd.Timestamp("2008-01-01"),
    "KOTAKBANK":  pd.Timestamp("2008-01-01"),
    "MARUTI":     pd.Timestamp("2008-01-01"),
    "WIPRO":      pd.Timestamp("2008-01-01"),
}

INDEX_INSTRUMENTS = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYIT"}
STOCK_INSTRUMENTS = {"RELIANCE", "HDFCBANK", "INFY", "TCS", "ICICIBANK", "AXISBANK",
                     "SBIN", "BAJFINANCE", "HCLTECH", "ITC", "LT", "SUNPHARMA",
                     "KOTAKBANK", "MARUTI", "WIPRO"}

ALL_INSTRUMENTS = (["NIFTY", "BANKNIFTY", "MIDCPNIFTY", "FINNIFTY", "NIFTYIT"] +
                   ["RELIANCE", "HDFCBANK", "INFY", "TCS", "ICICIBANK", "AXISBANK",
                    "SBIN", "BAJFINANCE", "HCLTECH", "ITC", "LT", "SUNPHARMA",
                    "KOTAKBANK", "MARUTI", "WIPRO"])

ALL_STRATEGIES = ["williams_r_vix", "wr_wide", "supertrend", "momentum_20d",
                  "rsi_trend", "ema_cross", "macd_signal", "bb_reversion", "vol_breakout"]


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────────────────────────────────────────
def load_data(instrument):
    path = Path(f"data/historical/{instrument}_daily_extended.csv")
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}  (run fetch_stocks.py first)")
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    df.columns = [c.lower() if isinstance(c, str) else str(c).lower()
                  for c in df.columns]
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[df.index.dayofweek < 5]
    # Drop phantom (zero-move) bars
    no_move = ((df.open == df.close) & (df.high == df.close) & (df.low == df.close))
    df = df[~no_move]
    df.index = df.index.normalize()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────────────────────────────────────
def _ema(s, p):    return s.ewm(span=p, adjust=False).mean()
def _sma(s, p):    return s.rolling(p).mean()

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
    pdm  = pdm.where(pdm >= mdm, 0.0)
    mdm  = mdm.where(mdm >  pdm, 0.0)
    atr14 = atr_series(df, p)
    pdi  = 100 * (pdm.ewm(alpha=1/p, adjust=False).mean() / atr14)
    mdi  = 100 * (mdm.ewm(alpha=1/p, adjust=False).mean() / atr14)
    dx   = (abs(pdi - mdi) / (pdi + mdi).replace(0, np.nan)) * 100
    return dx.ewm(alpha=1/p, adjust=False).mean(), pdi, mdi

def macd_series(s, fast=12, slow=26, sig=9):
    ml = _ema(s, fast) - _ema(s, slow)
    sl = ml.ewm(span=sig, adjust=False).mean()
    return ml, sl, ml - sl          # macd_line, signal_line, histogram

def bb_series(s, period=20, mult=2.0):
    mid = _sma(s, period)
    std = s.rolling(period).std()
    up  = mid + mult * std
    lo  = mid - mult * std
    pct = (s - lo) / (up - lo).replace(0, np.nan)
    return up, mid, lo, pct         # upper, middle, lower, %B

def supertrend_series(df, p=10, m=2.0):
    a_  = atr_series(df, p)
    hl2 = (df.high + df.low) / 2
    up  = (hl2 + m * a_).values;  lo  = (hl2 - m * a_).values
    cl  = df.close.values;         n   = len(cl)
    fu  = np.zeros(n); fl = np.zeros(n); dr = np.ones(n)
    fu[0] = up[0]; fl[0] = lo[0]
    for i in range(1, n):
        fu[i] = up[i] if up[i] < fu[i-1] or cl[i-1] > fu[i-1] else fu[i-1]
        fl[i] = lo[i] if lo[i] > fl[i-1] or cl[i-1] < fl[i-1] else fl[i-1]
        if   dr[i-1] == -1 and cl[i] > fu[i]: dr[i] =  1
        elif dr[i-1] ==  1 and cl[i] < fl[i]: dr[i] = -1
        else:                                   dr[i] = dr[i-1]
    return (pd.Series(dr, index=df.index),
            pd.Series(fu, index=df.index),
            pd.Series(fl, index=df.index))

def precompute(df):
    d = {}
    d["close"]    = df.close;       d["high"]  = df.high
    d["low"]      = df.low;         d["open"]  = df.open
    d["volume"]   = df["volume"]
    d["atr"]      = atr_series(df, 14)
    d["rsi14"]    = rsi_series(df.close, 14)
    d["wR14"]     = wR_series(df, 14)
    d["e8"]       = _ema(df.close, 8)
    d["e21"]      = _ema(df.close, 21)
    d["e50"]      = _ema(df.close, 50)
    d["e200"]     = _ema(df.close, 200)
    d["adx14"], d["plus_di"], d["minus_di"] = adx_series(df, 14)
    d["high_20d"] = df.high.rolling(20).max().shift(1)
    d["roc1"]     = df.close.pct_change(21) * 100
    d["roc3"]     = df.close.pct_change(63) * 100
    d["vol_sma20"]= df["volume"].rolling(20).mean()
    d["st_d"], d["st_u"], d["st_l"] = supertrend_series(df, 10, 2.0)
    d["macd_l"], d["macd_s"], d["macd_h"] = macd_series(df.close)
    d["bb_u"], d["bb_m"], d["bb_l"], d["bb_pct"] = bb_series(df.close, 20, 2.0)
    return d


# ─────────────────────────────────────────────────────────────────────────────
# VIX SIZING
# ─────────────────────────────────────────────────────────────────────────────
def vix_risk(vix_val, is_stock=False):
    """Risk sizing adjusted for VIX. Stocks always get stock risk cap."""
    base = STOCK_RISK if is_stock else INDEX_RISK
    if vix_val is None or np.isnan(float(vix_val)): return base
    v = float(vix_val)
    if is_stock:
        return base   # Stocks: fixed 2% regardless of VIX (already conservative)
    if v > 30:  return 0.03
    if v > 20:  return 0.05
    if v >= 15: return 0.04
    return 0.03


# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────
class Portfolio:
    def __init__(self, capital=CAPITAL):
        self.C0 = float(capital);  self.C  = float(capital)
        self.positions       = {}; self.trades          = []
        self.daily_equity    = []; self.margin_reserved = {}
        self.margin_calls    = 0;  self.roll_cost_total = 0.0

    def _span(self, inst):
        return STOCK_SPAN_PCT if inst in STOCK_INSTRUMENTS else INDEX_SPAN_PCT

    def _qty(self, inst, entry, stop, risk_pct):
        avail_cash   = self.C - sum(self.margin_reserved.values())
        current_risk = sum(abs(p["entry"]-p["stop"])*p["qty"]/self.C
                          for p in self.positions.values())
        avail_risk = min(risk_pct, MAX_TOTAL_RISK - current_risk)
        if avail_risk <= 0: return 0
        risk_amt  = avail_cash * avail_risk
        lot       = LOT_SIZES.get(inst, 1)
        risk_unit = max(abs(entry - stop), entry * MIN_STOP_PCT)
        lots      = int(risk_amt / (risk_unit * lot))
        if lots <= 0: return 0
        return min(lots, MAX_LOTS.get(inst, 2)) * lot

    def enter(self, inst, strategy, date, direction, entry, stop,
              tag="", risk_pct=INDEX_RISK):
        if inst in self.positions:               return False
        if len(self.positions) >= MAX_CONCURRENT: return False
        qty = self._qty(inst, entry, stop, risk_pct)
        if qty == 0:                             return False
        span_margin = qty * entry * self._span(inst)
        if self.C - sum(self.margin_reserved.values()) < span_margin:
            return False
        tgt = entry + direction * abs(entry - stop) * 2.0
        cost = BROKERAGE + qty * entry * (EXCHANGE_FEE + SLIPPAGE)
        if direction == -1: cost += qty * entry * STT_RATE
        self.C -= cost + span_margin
        self.margin_reserved[inst] = span_margin
        self.positions[inst] = dict(
            strategy=strategy, d=direction, entry=entry, stop=stop,
            target=tgt, qty=qty, date=date, tag=tag, last_mtm_price=entry)
        return True

    def trail_stop(self, inst, new_stop):
        if inst not in self.positions: return
        p = self.positions[inst]
        p["stop"] = (max(p["stop"], new_stop) if p["d"] == 1
                     else min(p["stop"], new_stop))

    def check_exits(self, date, inst, high, low, close):
        if inst not in self.positions: return False
        pos = self.positions[inst]
        d, stop, tgt = pos["d"], pos["stop"], pos["target"]
        if d == 1:
            if low  <= stop: return self._close(inst, date, stop, "stop")
            if high >= tgt:  return self._close(inst, date, tgt,  "target")
        else:
            if high >= stop: return self._close(inst, date, stop, "stop")
            if low  <= tgt:  return self._close(inst, date, tgt,  "target")
        return False

    def force_close(self, inst, date, price, reason="force"):
        if inst not in self.positions: return False
        return self._close(inst, date, price, reason)

    def _close(self, inst, date, exit_price, reason):
        pos  = self.positions.pop(inst)
        d, entry, qty = pos["d"], pos["entry"], pos["qty"]
        self.C += d * (exit_price - pos.get("last_mtm_price", entry)) * qty
        self.C += self.margin_reserved.pop(inst, 0)
        ec = BROKERAGE + qty * exit_price * (EXCHANGE_FEE + SLIPPAGE)
        if d == 1: ec += qty * exit_price * STT_RATE
        self.C -= ec
        nc = BROKERAGE + qty * entry * (EXCHANGE_FEE + SLIPPAGE)
        if d == -1: nc += qty * entry * STT_RATE
        pnl = d * (exit_price - entry) * qty - nc - ec
        self.trades.append(dict(
            instrument=inst, strategy=pos["strategy"],
            entry_date=pos["date"], exit_date=date,
            direction="L" if d==1 else "S",
            entry=round(entry,2), exit=round(exit_price,2),
            qty=qty, pnl=round(pnl,1), pnl_pct=round(pnl/self.C0*100,4),
            capital=round(self.C,1), duration=(date-pos["date"]).days,
            reason=reason, tag=pos["tag"],
            asset_type="stock" if inst in STOCK_INSTRUMENTS else "index"))
        return True

    def record_equity(self, date):
        self.daily_equity.append((date, round(self.C, 1)))

    def stats(self, label=""):
        T = self.trades
        if not T:
            return dict(label=label, ann_ret=0, total_ret=0, sharpe=0, max_dd=0,
                        win_rate=0, num_trades=0, net_pnl=0, capital=round(self.C),
                        score=0, roll_cost_total=0, margin_calls=0, avg_win=0, avg_loss=0)
        df  = pd.DataFrame(T)
        wr  = (df.pnl > 0).sum() / len(df) * 100
        tot = (self.C - self.C0) / self.C0 * 100
        t0  = pd.to_datetime(T[0]["entry_date"]); t1 = pd.to_datetime(T[-1]["exit_date"])
        yrs = max((t1 - t0).days / 365.25, 0.1)
        ann = ((1 + tot/100) ** (1/yrs) - 1) * 100
        aw  = df.loc[df.pnl > 0, "pnl"].mean() if (df.pnl > 0).any() else 0
        al  = df.loc[df.pnl < 0, "pnl"].mean() if (df.pnl < 0).any() else 0
        if len(self.daily_equity) >= 20:
            eq  = pd.Series([v for _, v in self.daily_equity])
            dr  = eq.pct_change().dropna()
            shr = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
            dd  = ((eq.cummax() - eq) / eq.cummax()).max() * 100
        else:
            pps = df.pnl_pct.values
            shr = pps.mean()/pps.std()*np.sqrt(252) if pps.std()>0 else 0; dd=0
        score = ann * 0.5 + shr * 10 - dd * 0.3
        return dict(label=label, total_ret=round(tot,2), ann_ret=round(ann,2),
                    sharpe=round(shr,3), max_dd=round(dd,2), win_rate=round(wr,1),
                    num_trades=len(T), net_pnl=round(self.C-self.C0),
                    capital=round(self.C), score=round(score,2),
                    roll_cost_total=round(self.roll_cost_total),
                    margin_calls=self.margin_calls,
                    avg_win=round(aw), avg_loss=round(al),
                    trades_per_yr=round(len(T)/yrs, 1))


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL FUNCTIONS (9 strategies)
# ─────────────────────────────────────────────────────────────────────────────
def sig_wr(i, ind, atr_mult=1.5):
    wr = ind["wR14"].iloc[i]; wrp = ind["wR14"].iloc[i-1]
    e200 = ind["e200"].iloc[i]; e50 = ind["e50"].iloc[i]
    atr_ = ind["atr"].iloc[i]; p = ind["close"].iloc[i]
    if np.isnan(e200): return None
    if wrp < -80 and wr >= -80 and p > e200*0.97 and e50 > e200*0.98:
        return ("L", atr_*atr_mult, "WR_OB")
    if wrp > -20 and wr <= -20 and p < e200*1.03 and e50 < e200*1.02:
        return ("S", atr_*atr_mult, "WR_OS")
    return None

def sig_wr_wide(i, ind):      return sig_wr(i, ind, atr_mult=2.0)
def sig_wr_vix(i, ind):       return sig_wr(i, ind, atr_mult=1.5)

def sig_supertrend(i, ind):
    st_d = ind["st_d"].iloc[i]; st_dp = ind["st_d"].iloc[i-1]
    adx = ind["adx14"].iloc[i]; e200 = ind["e200"].iloc[i]
    atr_ = ind["atr"].iloc[i]; p = ind["close"].iloc[i]
    pdi = ind["plus_di"].iloc[i]; mdi = ind["minus_di"].iloc[i]
    if np.isnan(adx) or np.isnan(e200) or adx < 22: return None
    if st_dp == -1 and st_d == 1 and p > e200*0.97 and pdi > mdi:
        return ("L", atr_*2.0, "ST_BULL")
    if st_dp == 1 and st_d == -1 and p < e200*1.03 and mdi > pdi:
        return ("S", atr_*2.0, "ST_BEAR")
    return None

def sig_momentum_20d(i, ind):
    h20 = ind["high_20d"].iloc[i]; p = ind["close"].iloc[i]
    e50 = ind["e50"].iloc[i]; e200 = ind["e200"].iloc[i]
    adx = ind["adx14"].iloc[i]; atr_ = ind["atr"].iloc[i]
    vol = ind["volume"].iloc[i]; vsma = ind["vol_sma20"].iloc[i]
    if np.isnan(h20) or np.isnan(e200) or h20 <= 0 or adx < 18: return None
    vol_ok = (vsma <= 0) or np.isnan(vsma) or (vol > vsma * 1.2)
    if p > h20 and p > e50 and p > e200*0.98 and vol_ok:
        return ("L", atr_*2.0, "MOM_20D")
    return None

def sig_rsi_trend(i, ind):
    rsi = ind["rsi14"].iloc[i]; rsip = ind["rsi14"].iloc[i-1]
    e8 = ind["e8"].iloc[i]; e21 = ind["e21"].iloc[i]
    e50 = ind["e50"].iloc[i]; e200 = ind["e200"].iloc[i]
    adx = ind["adx14"].iloc[i]; atr_ = ind["atr"].iloc[i]
    if np.isnan(rsi) or np.isnan(e200) or np.isnan(adx) or adx < 20: return None
    if rsip < 55 and rsi >= 55 and e50 > e200*0.99 and e21 > e50*0.99 and e8 > e21*0.99:
        return ("L", atr_*2.0, "RSI_BULL")
    if rsip > 45 and rsi <= 45 and e50 < e200*1.01 and e21 < e50*1.01 and e8 < e21*1.01:
        return ("S", atr_*2.0, "RSI_BEAR")
    return None

def sig_ema_cross(i, ind):
    e8 = ind["e8"].iloc[i]; e8p = ind["e8"].iloc[i-1]
    e21 = ind["e21"].iloc[i]; e21p = ind["e21"].iloc[i-1]
    e50 = ind["e50"].iloc[i]; e200 = ind["e200"].iloc[i]
    adx = ind["adx14"].iloc[i]; atr_ = ind["atr"].iloc[i]
    pdi = ind["plus_di"].iloc[i]; mdi = ind["minus_di"].iloc[i]
    if np.isnan(e200) or np.isnan(adx) or adx < 20: return None
    p = ind["close"].iloc[i]
    if e8p <= e21p and e8 > e21 and e21 > e50*0.99 and p > e200*0.97 and pdi > mdi:
        return ("L", atr_*2.0, "EMA_BULL")
    if e8p >= e21p and e8 < e21 and e21 < e50*1.01 and p < e200*1.03 and mdi > pdi:
        return ("S", atr_*2.0, "EMA_BEAR")
    return None

def sig_macd(i, ind):
    """
    MACD(12,26,9) line crosses signal line — classic trend entry.
    Stocks trend for weeks after MACD crossover, unlike indices which revert.
    Filter: MACD histogram turning positive/negative + price on correct side of EMA200.
    """
    ml = ind["macd_l"].iloc[i]; mlp = ind["macd_l"].iloc[i-1]
    sl = ind["macd_s"].iloc[i]; slp = ind["macd_s"].iloc[i-1]
    mh = ind["macd_h"].iloc[i]; mhp = ind["macd_h"].iloc[i-1]
    e200 = ind["e200"].iloc[i]; atr_ = ind["atr"].iloc[i]
    adx  = ind["adx14"].iloc[i]; p = ind["close"].iloc[i]
    if np.isnan(ml) or np.isnan(e200) or np.isnan(adx): return None
    if adx < 18: return None   # Needs some directional strength
    # Long: MACD crossed above signal, histogram turning positive, above EMA200
    if mlp <= slp and ml > sl and mh > 0 and p > e200*0.97:
        return ("L", atr_*2.0, "MACD_BULL")
    # Short: MACD crossed below signal, histogram turning negative, below EMA200
    if mlp >= slp and ml < sl and mh < 0 and p < e200*1.03:
        return ("S", atr_*2.0, "MACD_BEAR")
    return None

def sig_bb_reversion(i, ind):
    """
    Bollinger Band(20,2) %B reversion — price outside bands snaps back to mean.
    Works for range-bound stocks. Filter: EMA trend intact (not in crash/blowup).
    Long: %B < 0.05 (near/below lower band) + RSI not collapsed (<25) + e50 > e200
    Short: %B > 0.95 (near/above upper band) + RSI not overbought (>75) + e50 < e200
    """
    pct = ind["bb_pct"].iloc[i]; pctp = ind["bb_pct"].iloc[i-1]
    rsi = ind["rsi14"].iloc[i]
    e50 = ind["e50"].iloc[i]; e200 = ind["e200"].iloc[i]
    atr_ = ind["atr"].iloc[i]; p = ind["close"].iloc[i]
    if np.isnan(pct) or np.isnan(e200): return None
    # Long: price touches/crosses below lower BB, in overall uptrend (e50 > e200)
    if pctp >= 0.05 and pct < 0.05 and rsi > 25 and rsi < 55 and e50 > e200*0.97:
        return ("L", atr_*1.5, "BB_LOW")
    # Short: price touches/crosses above upper BB, in overall downtrend
    if pctp <= 0.95 and pct > 0.95 and rsi < 75 and rsi > 45 and e50 < e200*1.03:
        return ("S", atr_*1.5, "BB_HIGH")
    return None

def sig_vol_breakout(i, ind):
    """
    Volume-confirmed breakout above 20-day high.
    Stronger volume filter than momentum_20d (2.0× vs 1.2×).
    Best for stocks with institutional accumulation patterns.
    """
    h20 = ind["high_20d"].iloc[i]; p = ind["close"].iloc[i]
    e50 = ind["e50"].iloc[i]; e200 = ind["e200"].iloc[i]
    adx = ind["adx14"].iloc[i]; atr_ = ind["atr"].iloc[i]
    vol = ind["volume"].iloc[i]; vsma = ind["vol_sma20"].iloc[i]
    if np.isnan(h20) or np.isnan(e200) or h20 <= 0 or adx < 20: return None
    if vsma <= 0 or np.isnan(vsma): return None   # Require volume data
    if p > h20 and p > e50 and p > e200*0.98 and vol > vsma * 2.0:
        return ("L", atr_*2.0, "VOL_BRK")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# EXIT ROUTER
# ─────────────────────────────────────────────────────────────────────────────
def process_exit(pf, inst, date, ind, ii, vix_series):
    if inst not in pf.positions: return
    pos = pf.positions[inst]
    tag = pos["tag"]; d = pos["d"]
    p   = ind["close"].iloc[ii]; ph = ind["high"].iloc[ii]; pl = ind["low"].iloc[ii]
    e21 = ind["e21"].iloc[ii];   e50 = ind["e50"].iloc[ii]
    atr_ = ind["atr"].iloc[ii]

    if tag.startswith("WR_") or tag.startswith("BB_"):
        # Mean-reversion exits: WR crosses back through -50 (WR) or %B crosses 0.5 (BB)
        if tag.startswith("WR_"):
            wr = ind["wR14"].iloc[ii]; wrp = ind["wR14"].iloc[ii-1]
            if d == 1 and wrp < -50 and wr >= -50:
                pf.force_close(inst, date, p, "WR_EXIT"); return
            if d == -1 and wrp > -50 and wr <= -50:
                pf.force_close(inst, date, p, "WR_EXIT"); return
        else:  # BB reversion
            pct = ind["bb_pct"].iloc[ii]; pctp = ind["bb_pct"].iloc[ii-1]
            mid = ind["bb_m"].iloc[ii]
            if d == 1 and pctp < 0.5 and pct >= 0.5:
                pf.force_close(inst, date, p, "BB_MID_EXIT"); return
            if d == -1 and pctp > 0.5 and pct <= 0.5:
                pf.force_close(inst, date, p, "BB_MID_EXIT"); return
        pf.trail_stop(inst, e21 - d * atr_ * 0.3)

    elif tag.startswith(("ST_", "EMA_", "MACD_", "RSI_", "MOM_", "VOL_")):
        # Trend exits: SuperTrend flip, EMA re-cross, MACD re-cross, RSI through 50
        if tag.startswith("ST_"):
            st_d = ind["st_d"].iloc[ii]
            if d == 1 and st_d == -1:
                pf.force_close(inst, date, p, "ST_EXIT"); return
            if d == -1 and st_d == 1:
                pf.force_close(inst, date, p, "ST_EXIT"); return
            pf.trail_stop(inst, ind["st_l"].iloc[ii] if d==1 else ind["st_u"].iloc[ii])

        elif tag.startswith("EMA_"):
            e8 = ind["e8"].iloc[ii]; e8p = ind["e8"].iloc[ii-1]
            e21c = ind["e21"].iloc[ii]; e21p = ind["e21"].iloc[ii-1]
            if d == 1 and e8p >= e21p and e8 < e21c:
                pf.force_close(inst, date, p, "EMA_EXIT"); return
            if d == -1 and e8p <= e21p and e8 > e21c:
                pf.force_close(inst, date, p, "EMA_EXIT"); return
            pf.trail_stop(inst, e21c - d * atr_ * 0.3)

        elif tag.startswith("MACD_"):
            ml = ind["macd_l"].iloc[ii]; mlp = ind["macd_l"].iloc[ii-1]
            sl_v = ind["macd_s"].iloc[ii]; slp = ind["macd_s"].iloc[ii-1]
            if d == 1 and mlp >= slp and ml < sl_v:
                pf.force_close(inst, date, p, "MACD_EXIT"); return
            if d == -1 and mlp <= slp and ml > sl_v:
                pf.force_close(inst, date, p, "MACD_EXIT"); return
            pf.trail_stop(inst, e50 - d * atr_ * 0.5)

        elif tag.startswith("RSI_"):
            rsi = ind["rsi14"].iloc[ii]; rsip = ind["rsi14"].iloc[ii-1]
            if d == 1 and rsip > 50 and rsi <= 50:
                pf.force_close(inst, date, p, "RSI_EXIT"); return
            if d == -1 and rsip < 50 and rsi >= 50:
                pf.force_close(inst, date, p, "RSI_EXIT"); return
            pf.trail_stop(inst, e50 - d * atr_ * 0.5)

        elif tag.startswith(("MOM_", "VOL_")):
            st_d = ind["st_d"].iloc[ii]
            if d == 1 and (st_d == -1 or p < e50 * 0.99):
                pf.force_close(inst, date, p, "MOM_EXIT"); return
            pf.trail_stop(inst, e21 - d * atr_ * 0.5)

    elif tag.startswith("VIX_"):
        if date in vix_series.index:
            vi = vix_series.index.get_loc(date)
            if vi >= 5:
                v5  = vix_series.iloc[vi-5:vi].mean()
                vn  = vix_series.iloc[vi]
                if not np.isnan(v5) and v5 > 0 and abs(vn-v5)/v5 < 0.05:
                    pf.force_close(inst, date, p, "VIX_NORM"); return

    if inst in pf.positions:
        pf.check_exits(date, inst, ph, pl, p)


# ─────────────────────────────────────────────────────────────────────────────
# CORE RUNNER
# ─────────────────────────────────────────────────────────────────────────────
SIGNAL_MAP = {
    "williams_r_vix": sig_wr_vix,
    "wr_wide":        sig_wr_wide,
    "supertrend":     sig_supertrend,
    "momentum_20d":   sig_momentum_20d,
    "rsi_trend":      sig_rsi_trend,
    "ema_cross":      sig_ema_cross,
    "macd_signal":    sig_macd,
    "bb_reversion":   sig_bb_reversion,
    "vol_breakout":   sig_vol_breakout,
}

def run_backtest(inst_strategy_map, capital=CAPITAL):
    """
    inst_strategy_map: {instrument: [list_of_strategy_names]}
    Each instrument only receives signals from its assigned strategies.
    """
    instruments = list(inst_strategy_map.keys())
    data = {}; inds = {}
    for inst in instruments:
        df = load_data(inst)
        data[inst] = df
        inds[inst] = precompute(df)

    vix_df      = load_data("INDIAVIX")
    union_set   = set.union(*[set(data[inst].index) for inst in instruments])
    all_dates   = pd.DatetimeIndex(sorted(union_set))
    vix_series  = vix_df.close.reindex(all_dates, method="ffill")
    vix_chg     = vix_df.close.pct_change().reindex(all_dates)

    pf = Portfolio(capital)
    pending_entries = {}; pending_exits = {}; last_roll_month = (-1, -1)

    for idx, date in enumerate(all_dates):
        if idx < WARMUP: continue

        vt = vix_series.iloc[idx] if idx < len(vix_series) else None
        vc = vix_chg.iloc[idx]    if idx < len(vix_chg)    else None
        if vt is not None and (pd.isna(vt) or np.isnan(float(vt))):   vt = None
        if vc is not None and (pd.isna(vc) or np.isnan(float(vc))):   vc = None

        # Execute pending exits at open
        for inst, reason in list(pending_exits.items()):
            del pending_exits[inst]
            if inst not in pf.positions or inst not in data: continue
            if date not in data[inst].index: continue
            ii = data[inst].index.get_loc(date)
            pf.force_close(inst, date, float(inds[inst]["open"].iloc[ii]), reason)

        # Execute pending entries at open
        for inst, (strat, dv, sd, tag, rp) in list(pending_entries.items()):
            del pending_entries[inst]
            if inst in pf.positions or inst not in data: continue
            if date not in data[inst].index: continue
            ii    = data[inst].index.get_loc(date)
            entry = float(inds[inst]["open"].iloc[ii])
            stop  = entry - dv * sd
            pf.enter(inst, strat, date, dv, entry, stop, tag, rp)

        # Exits
        for inst in list(pf.positions.keys()):
            if inst not in data or date not in data[inst].index: continue
            ii = data[inst].index.get_loc(date)
            if ii < 1: continue
            process_exit(pf, inst, date, inds[inst], ii, vix_series)

        # Daily MTM
        for inst, pos in list(pf.positions.items()):
            if inst not in data or date not in data[inst].index: continue
            ii = data[inst].index.get_loc(date)
            cp = float(inds[inst]["close"].iloc[ii])
            pf.C += pos["d"] * (cp - pos.get("last_mtm_price", pos["entry"])) * pos["qty"]
            pos["last_mtm_price"] = cp

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

        # Roll cost (last trading day of each month)
        cur_month = (date.year, date.month)
        if cur_month != last_roll_month and pf.positions:
            if idx+1 >= len(all_dates) or all_dates[idx+1].month != date.month:
                last_roll_month = cur_month
                for inst, pos in list(pf.positions.items()):
                    if inst not in data or date not in data[inst].index: continue
                    ii = data[inst].index.get_loc(date)
                    cp = float(inds[inst]["close"].iloc[ii])
                    rc = pos["qty"] * cp * ROLL_COST_PCT
                    pf.C -= rc; pf.roll_cost_total += rc

        # Entry signals → queue for next-bar open
        for inst in instruments:
            if inst in pf.positions or inst in pending_entries: continue
            live = INSTRUMENT_LIVE_DATE.get(inst, pd.Timestamp("2008-01-01"))
            if date < live: continue
            if inst not in data or date not in data[inst].index: continue
            ii = data[inst].index.get_loc(date)
            if ii < WARMUP or idx+1 >= len(all_dates): continue

            ind   = inds[inst]
            strats = inst_strategy_map[inst]
            is_stk = inst in STOCK_INSTRUMENTS
            sig    = None

            # VIX spike only for NIFTY
            if "vix_spike" in strats and inst == "NIFTY" and vc is not None:
                v = float(vc)
                atr_ = ind["atr"].iloc[ii]; e200 = ind["e200"].iloc[ii]
                if not np.isnan(e200):
                    if v > 0.15:  sig = ("L", atr_*1.5, "VIX_SPIKE")
                    elif v < -0.15: sig = ("S", atr_*1.5, "VIX_DROP")
                if sig:
                    dv = 1 if sig[0]=="L" else -1
                    pending_entries[inst] = (sig[2], dv, sig[1], sig[2],
                                             vix_risk(vt, is_stk))
                    continue

            # Try each assigned strategy in order
            for sname in strats:
                if sname == "vix_spike": continue
                fn = SIGNAL_MAP.get(sname)
                if fn is None: continue
                sig = fn(ii, ind)
                if sig is not None:
                    dv = 1 if sig[0]=="L" else -1
                    rp = vix_risk(vt, is_stk) if "wr" in sname else (
                         STOCK_RISK if is_stk else INDEX_RISK)
                    pending_entries[inst] = (sname, dv, sig[1], sig[2], rp)
                    break

        pf.record_equity(date)

    # Force-close all at end
    for inst in list(pf.positions.keys()):
        pf.force_close(inst, all_dates[-1],
                       float(data[inst].close.iloc[-1]), "end_of_backtest")
    return pf


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def annual_breakdown(pf):
    if not pf.daily_equity: return []
    eq = pd.DataFrame(pf.daily_equity, columns=["date","equity"])
    eq["date"] = pd.to_datetime(eq["date"]); eq = eq.set_index("date")
    rows = []
    for yr in sorted(eq.index.year.unique()):
        yd = eq[eq.index.year == yr]
        if yd.empty: continue
        pv  = eq[eq.index.year < yr]
        sv  = pv["equity"].iloc[-1] if not pv.empty else pf.C0
        ev  = yd["equity"].iloc[-1]
        pnl = ev - sv; ret = pnl/sv*100
        yt  = [t for t in pf.trades if pd.to_datetime(t["exit_date"]).year == yr]
        wr  = (sum(1 for t in yt if t["pnl"]>0)/len(yt)*100) if yt else 0.0
        rows.append(dict(year=yr, trades=len(yt), win_rate=round(wr,1),
                         pnl=round(pnl), ret_pct=round(ret,2)))
    return rows

def instrument_stats(pf):
    df = pd.DataFrame(pf.trades)
    if df.empty: return []
    res = []
    for inst, grp in df.groupby("instrument"):
        wr = (grp.pnl>0).sum()/len(grp)*100
        aw = grp.loc[grp.pnl>0,"pnl"].mean() if (grp.pnl>0).any() else 0
        res.append(dict(instrument=inst, trades=len(grp),
                        win_rate=round(wr,1), net_pnl=round(grp.pnl.sum()),
                        avg_win=round(aw),
                        asset_type="stock" if inst in STOCK_INSTRUMENTS else "index"))
    return sorted(res, key=lambda x: x["net_pnl"], reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: STRATEGY DISCOVERY GRID
# ─────────────────────────────────────────────────────────────────────────────
def run_discovery():
    console.rule("[bold yellow]PHASE 1 — Strategy Discovery Grid[/bold yellow]")
    console.print(f"[dim]9 strategies × 20 instruments = 180 single-instrument backtests[/dim]\n")

    grid = {}   # {inst: {strat: stats_dict}}

    sections = [
        ("INDEX FUTURES", ["NIFTY","BANKNIFTY","MIDCPNIFTY","FINNIFTY","NIFTYIT"]),
        ("STOCK FUTURES", ["RELIANCE","HDFCBANK","INFY","TCS","ICICIBANK",
                           "AXISBANK","SBIN","BAJFINANCE","HCLTECH","ITC",
                           "LT","SUNPHARMA","KOTAKBANK","MARUTI","WIPRO"]),
    ]

    for section_name, insts in sections:
        console.print(f"  [bold]{section_name}[/bold]")
        for inst in insts:
            grid[inst] = {}
            live = INSTRUMENT_LIVE_DATE.get(inst, pd.Timestamp("2008-01-01"))
            console.print(f"    [cyan]{inst:12s}[/cyan] (live {live.date()})", end="")
            best_ann = -999
            for strat in ALL_STRATEGIES:
                try:
                    pf = run_backtest({inst: [strat]}, capital=CAPITAL)
                    s  = pf.stats(f"{inst}_{strat}")
                    grid[inst][strat] = s
                    if s["ann_ret"] > best_ann: best_ann = s["ann_ret"]
                except Exception as e:
                    grid[inst][strat] = None
            # Print best
            best_strat = max((st for st in ALL_STRATEGIES if grid[inst].get(st)),
                             key=lambda st: grid[inst][st]["score"] if grid[inst][st] else -999,
                             default="n/a")
            bs = grid[inst].get(best_strat) or {}
            c  = "green" if bs.get("ann_ret",0)>=20 else ("yellow" if bs.get("ann_ret",0)>=10 else "red")
            console.print(f"  best=[{c}]{best_strat}[/{c}] "
                          f"{bs.get('ann_ret',0):+.1f}% sh:{bs.get('sharpe',0):.2f}")
        console.print()

    return grid


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    console.rule("[bold cyan]v11 — Individual Stock Futures + Per-Instrument Strategy Optimization[/bold cyan]")
    console.print(
        "[dim]Universe: 5 index futures + 15 stock futures = 20 instruments\n"
        "Strategies: WR, WR-wide, SuperTrend, Momentum-20D, RSI-trend, EMA-cross,\n"
        "            MACD (NEW), Bollinger-Band reversion (NEW), Volume Breakout (NEW)\n"
        "Stock margin: 15% SPAN | Stock risk: 2%/trade | Index risk: 4%/trade[/dim]\n"
    )

    # ── PHASE 1 ─────────────────────────────────────────────────────────────
    grid = run_discovery()

    # Per-instrument best assignment
    best_per_inst = {}
    for inst in ALL_INSTRUMENTS:
        scored = [(st, grid[inst][st]["score"])
                  for st in ALL_STRATEGIES
                  if grid[inst].get(st) and grid[inst][st]["num_trades"] >= 5]
        scored.sort(key=lambda x: x[1], reverse=True)
        best_per_inst[inst] = scored[0][0] if scored else "williams_r_vix"

    # Print discovery grid as matrix
    console.rule("[bold yellow]Discovery Grid — Best Strategy per Instrument[/bold yellow]")
    gtbl = Table(box=box.DOUBLE_EDGE, show_header=True, header_style="bold yellow",
                 title="Score = CAGR×0.5 + Sharpe×10 − MaxDD×0.3  |  ★ = best per instrument",
                 min_width=180)
    gtbl.add_column("Strategy", min_width=16)
    for inst in ALL_INSTRUMENTS:
        gtbl.add_column(inst[:8], justify="right", min_width=9)

    for strat in ALL_STRATEGIES:
        row = [strat]
        for inst in ALL_INSTRUMENTS:
            s = grid[inst].get(strat)
            if s is None or s["num_trades"] < 5:
                row.append("[dim]—[/dim]")
            else:
                is_best = (best_per_inst.get(inst) == strat)
                c = "bold green" if is_best else (
                    "green" if s["ann_ret"] >= 20 else (
                    "yellow" if s["ann_ret"] >= 10 else "red"))
                mk = "★" if is_best else " "
                row.append(f"[{c}]{mk}{s['ann_ret']:+.0f}%[/{c}]")
        gtbl.add_row(*row)
    console.print(gtbl)

    # Assignment summary
    console.rule("[bold yellow]PHASE 2 — Per-Instrument Strategy Assignment[/bold yellow]")
    atbl = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold",
                 title="Optimal Strategy per Instrument")
    atbl.add_column("Instrument",  min_width=13)
    atbl.add_column("Type",        min_width=7)
    atbl.add_column("Strategy",    min_width=18)
    atbl.add_column("Solo CAGR",   justify="right")
    atbl.add_column("Solo Sharpe", justify="right")
    atbl.add_column("Solo DD",     justify="right")
    atbl.add_column("Score",       justify="right")

    for inst in ALL_INSTRUMENTS:
        strat = best_per_inst[inst]
        s = grid[inst].get(strat) or {}
        atype = "INDEX" if inst in INDEX_INSTRUMENTS else "STOCK"
        c = "green" if s.get("ann_ret",0)>=20 else ("yellow" if s.get("ann_ret",0)>=10 else "red")
        atbl.add_row(inst, atype, strat,
                     f"[{c}]{s.get('ann_ret',0):+.1f}%[/{c}]",
                     f"{s.get('sharpe',0):.2f}",
                     f"{s.get('max_dd',0):.1f}%",
                     f"{s.get('score',0):+.1f}")
    console.print(atbl)

    # ── PHASE 3: Combined portfolios ─────────────────────────────────────────
    console.rule("[bold cyan]PHASE 3 — Combined Portfolio Backtests[/bold cyan]")

    def run_cfg(label, imap):
        console.print(f"  Running [cyan]{label}[/cyan]...", end=" ")
        pf = run_backtest(imap, capital=CAPITAL)
        s  = pf.stats(label)
        c  = "green" if s["ann_ret"] >= 28 else ("yellow" if s["ann_ret"] >= 20 else "red")
        console.print(f"[{c}]{s['ann_ret']:+.1f}%[/{c}] | Sh {s['sharpe']:.2f} | "
                      f"DD {s['max_dd']:.1f}% | {s['num_trades']} trades")
        return pf, s

    # A: v10 baseline (WR on all 5 indices only)
    pf_a, s_a = run_cfg("v10_baseline_indices",
                         {i: ["williams_r_vix"] for i in
                          ["NIFTY","BANKNIFTY","MIDCPNIFTY","FINNIFTY","NIFTYIT"]})

    # B: WR on all 20 instruments (uniform strategy, expanded universe)
    pf_b, s_b = run_cfg("WR_all_20",
                         {i: ["williams_r_vix"] for i in ALL_INSTRUMENTS})

    # C: Per-instrument optimal (best strategy for each instrument)
    pf_c, s_c = run_cfg("per_inst_optimal",
                         {i: [best_per_inst[i]] for i in ALL_INSTRUMENTS})

    # D: Per-instrument optimal + vix_spike for NIFTY
    map_d = {i: [best_per_inst[i]] for i in ALL_INSTRUMENTS}
    map_d["NIFTY"] = ["vix_spike", best_per_inst["NIFTY"]]
    pf_d, s_d = run_cfg("per_inst_optimal+VIX", map_d)

    # E: Top-2 strategies per instrument
    map_e = {}
    for inst in ALL_INSTRUMENTS:
        scored = [(st, grid[inst][st]["score"]) for st in ALL_STRATEGIES
                  if grid[inst].get(st) and grid[inst][st]["num_trades"]>=5]
        scored.sort(key=lambda x: x[1], reverse=True)
        map_e[inst] = [s[0] for s in scored[:2]] or ["williams_r_vix"]
    pf_e, s_e = run_cfg("top2_per_inst", map_e)

    # ── RESULTS TABLE ────────────────────────────────────────────────────────
    console.print()
    console.rule("[bold cyan]Combined Results Comparison[/bold cyan]")

    all_cfgs = [("v10 Baseline (5 idx, WR)", s_a),
                ("WR all 20 instruments",    s_b),
                ("Per-Inst Optimal (20)",    s_c),
                ("Per-Inst + VIX spike",     s_d),
                ("Top-2 per instrument",     s_e)]
    all_cfgs.sort(key=lambda x: x[1]["ann_ret"], reverse=True)
    best_ann = all_cfgs[0][1]["ann_ret"]

    rtbl = Table(box=box.DOUBLE_EDGE, show_header=True, header_style="bold cyan", min_width=170)
    rtbl.add_column("Portfolio",       min_width=26)
    rtbl.add_column("Ann Ret%",        justify="right", min_width=10)
    rtbl.add_column("Total Ret%",      justify="right", min_width=12)
    rtbl.add_column("Sharpe",          justify="right", min_width=8)
    rtbl.add_column("MaxDD%",          justify="right", min_width=8)
    rtbl.add_column("Win%",            justify="right", min_width=7)
    rtbl.add_column("Trades",          justify="right", min_width=8)
    rtbl.add_column("Net P&L",         justify="right", min_width=16)
    rtbl.add_column("Roll Cost",       justify="right", min_width=13)
    rtbl.add_column("Final Capital",   justify="right", min_width=16)

    best_pf_map = {"v10 Baseline (5 idx, WR)": pf_a,
                   "WR all 20 instruments":    pf_b,
                   "Per-Inst Optimal (20)":    pf_c,
                   "Per-Inst + VIX spike":     pf_d,
                   "Top-2 per instrument":     pf_e}

    for name, s in all_cfgs:
        is_best = abs(s["ann_ret"] - best_ann) < 0.05
        c = "bold green" if is_best else (
            "green" if s["ann_ret"] >= 25 else (
            "yellow" if s["ann_ret"] >= 18 else "red"))
        mk = "★ " if is_best else "  "
        rtbl.add_row(
            f"[{c}]{mk}{name}[/{c}]",
            f"[{c}]{s['ann_ret']:+.1f}%[/{c}]",
            f"{s['total_ret']:+.1f}%",
            f"{s['sharpe']:.2f}",
            f"{s['max_dd']:.1f}%",
            f"{s['win_rate']:.1f}%",
            str(s["num_trades"]),
            f"₹{s['net_pnl']:+,.0f}",
            f"₹{s['roll_cost_total']:,.0f}",
            f"₹{s['capital']:,.0f}",
        )
    console.print(rtbl)

    # Best portfolio details
    best_name = all_cfgs[0][0]
    best_pf   = best_pf_map[best_name]
    best_s    = all_cfgs[0][1]

    # Year-by-year
    ybl  = annual_breakdown(best_pf)
    ytbl = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold",
                 title=f"Year-by-Year — {best_name}")
    ytbl.add_column("Year"); ytbl.add_column("Trades", justify="right")
    ytbl.add_column("Win%", justify="right"); ytbl.add_column("P&L", justify="right")
    ytbl.add_column("Return%", justify="right")
    for row in ybl:
        c = "green" if row["ret_pct"] >= 20 else ("yellow" if row["ret_pct"] >= 5 else "red")
        ytbl.add_row(str(row["year"]), str(row["trades"]),
                     f"{row['win_rate']:.1f}%", f"₹{row['pnl']:+,.0f}",
                     f"[{c}]{row['ret_pct']:+.2f}%[/{c}]")
    console.print(ytbl)

    # Instrument breakdown for best
    iattr = instrument_stats(best_pf)
    if iattr:
        itbl = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold",
                     title=f"Top Instrument Contributors — {best_name}")
        itbl.add_column("Instrument", min_width=13)
        itbl.add_column("Type",       min_width=6)
        itbl.add_column("Strategy",   min_width=18)
        itbl.add_column("Trades",     justify="right")
        itbl.add_column("Win%",       justify="right")
        itbl.add_column("Net P&L",    justify="right")
        itbl.add_column("Avg Win",    justify="right")
        for a in iattr[:12]:
            c = "green" if a["net_pnl"] > 0 else "red"
            itbl.add_row(
                a["instrument"],
                a["asset_type"],
                best_per_inst.get(a["instrument"], "WR"),
                str(a["trades"]),
                f"{a['win_rate']:.1f}%",
                f"[{c}]₹{a['net_pnl']:+,.0f}[/{c}]",
                f"₹{a['avg_win']:,.0f}"
            )
        console.print(itbl)

    # ── VERSIONED P&L STATEMENT ──────────────────────────────────────────────
    console.rule("[bold magenta]VERSIONED P&L STATEMENT — v7 through v11[/bold magenta]")

    versions = [
        ("v7",  "Date UNION + real DD + rotation fix + roll cost",
                 26.6, 0.71, 28.6, 17.5, 1_000_000, 13_290_000),
        ("v8",  "5 instruments (NIFTYIT added) + SuperTrend + Momentum",
                 28.9, 0.74, 34.1, 17.5, 1_000_000, 17_400_000),
        ("v9",  "Trend-rider override + VIX>30 sizing corrected",
                 28.4, 0.74, 34.1, 17.5, 1_000_000, 16_800_000),
        ("v10", "Per-instrument strategy discovery (indices only)",
                 28.8, 0.75, 32.9, 17.5, 1_000_000, 17_600_000),
        ("v11", f"20 instruments (stocks+indices), per-instrument optimal\n"
                f"        New strategies: MACD, BB reversion, Volume Breakout",
                 best_s["ann_ret"], best_s["sharpe"], best_s["max_dd"],
                 round((pd.to_datetime(best_pf.daily_equity[-1][0]) -
                        pd.to_datetime(best_pf.daily_equity[0][0])).days / 365.25, 1)
                 if best_pf.daily_equity else 17.5,
                 CAPITAL, best_s["capital"]),
    ]

    vtbl = Table(box=box.DOUBLE_EDGE, show_header=True, header_style="bold magenta",
                 title="P&L STATEMENT — Trading Algorithm Version History\n"
                       "Starting Capital: ₹10,00,000  |  NSE Futures  |  India",
                 min_width=180)
    vtbl.add_column("Ver",     min_width=5)
    vtbl.add_column("What Changed",       min_width=52)
    vtbl.add_column("Ann CAGR",  justify="right", min_width=10)
    vtbl.add_column("Sharpe",    justify="right", min_width=8)
    vtbl.add_column("MaxDD%",    justify="right", min_width=8)
    vtbl.add_column("Years",     justify="right", min_width=7)
    vtbl.add_column("Start Cap", justify="right", min_width=13)
    vtbl.add_column("Final Cap", justify="right", min_width=16)
    vtbl.add_column("Net P&L",   justify="right", min_width=16)
    vtbl.add_column("Multiple",  justify="right", min_width=9)

    for ver, desc, cagr, sh, dd, yrs, start, final in versions:
        net  = final - start
        mult = final / start
        c    = "bold green" if ver == "v11" else (
               "green" if cagr >= 28 else ("yellow" if cagr >= 20 else "white"))
        mk   = "★ " if ver == "v11" else "  "
        vtbl.add_row(
            f"[{c}]{mk}{ver}[/{c}]",
            f"[{c}]{desc}[/{c}]",
            f"[{c}]{cagr:+.1f}%[/{c}]",
            f"{sh:.2f}",
            f"{dd:.1f}%",
            f"{yrs:.1f}",
            f"₹{start:,.0f}",
            f"[{c}]₹{final:,.0f}[/{c}]",
            f"[{c}]₹{net:+,.0f}[/{c}]",
            f"[{c}]{mult:.1f}×[/{c}]"
        )
    console.print(vtbl)

    # Save report
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"reports/v11_backtest_{ts}.json"
    all_data = {
        "version": "v11",
        "timestamp": ts,
        "discovery_grid": {
            inst: {st: s for st, s in strats.items() if s}
            for inst, strats in grid.items()
        },
        "best_strategy_per_inst": best_per_inst,
        "results": {
            "v10_baseline":       {"stats": s_a, "annual": annual_breakdown(pf_a)},
            "WR_all_20":          {"stats": s_b, "annual": annual_breakdown(pf_b)},
            "per_inst_optimal":   {"stats": s_c, "annual": annual_breakdown(pf_c)},
            "per_inst_plus_vix":  {"stats": s_d, "annual": annual_breakdown(pf_d)},
            "top2_per_inst":      {"stats": s_e, "annual": annual_breakdown(pf_e)},
        },
        "best_portfolio": {
            "label":    best_name,
            "stats":    best_s,
            "trades":   best_pf.trades,
            "annual":   annual_breakdown(best_pf),
            "instruments": instrument_stats(best_pf),
            "equity_curve": [(str(d), v) for d, v in best_pf.daily_equity],
        }
    }
    with open(fname, "w") as f:
        json.dump(all_data, f, indent=2, default=str)
    console.print(f"\n[dim]Full report saved → {fname}[/dim]")

    # Final result panel
    ann    = best_s["ann_ret"]
    period = (f"{best_pf.daily_equity[0][0].date()} → {best_pf.daily_equity[-1][0].date()}"
              if best_pf.daily_equity else "n/a")

    console.print(Panel(
        f"[bold {'green' if ann >= 28 else 'yellow'}]"
        f"v11 BEST: {ann:+.1f}% CAGR  |  {best_name}[/bold {'green' if ann >= 28 else 'yellow'}]\n\n"
        f"  Period       : {period}\n"
        f"  Sharpe       : {best_s['sharpe']:.2f}\n"
        f"  Max Drawdown : {best_s['max_dd']:.1f}%  [daily equity curve]\n"
        f"  Win Rate     : {best_s['win_rate']:.1f}%\n"
        f"  Trades       : {best_s['num_trades']}\n"
        f"  Net P&L      : ₹{best_s['net_pnl']:+,.0f}\n"
        f"  Final Capital: ₹{best_s['capital']:,.0f}\n"
        f"  Roll Costs   : ₹{best_s['roll_cost_total']:,.0f}\n"
        f"  Margin Calls : {best_s['margin_calls']}\n\n"
        f"  Universe expanded: 5 indices → 20 instruments (15 stocks added)\n"
        f"  New strategies: MACD crossover, Bollinger Band reversion, Volume breakout\n"
        f"  Each instrument runs its data-proven optimal strategy independently",
        title="[bold]v11 RESULT — INDIVIDUAL STOCK FUTURES + PER-INSTRUMENT OPTIMIZATION[/bold]",
        border_style="green" if ann >= 28 else "yellow"
    ))

    return best_s


if __name__ == "__main__":
    main()
