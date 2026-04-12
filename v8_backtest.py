"""
v8_backtest.py — Enhanced Multi-Strategy, 5-Instrument Backtest
================================================================
Builds on all v7 structural fixes (date union, real drawdown, rotation exit at
next-day open, roll cost) and adds genuinely new alpha sources:

RESEARCH FINDINGS that drove this version:
  1. NIFTYIT (Nifty IT Index futures) is only 0.31 correlated with BANKNIFTY
     and 0.56 correlated with NIFTY — adds real diversification.  IT often
     outperforms when banking underperforms and vice versa.
  2. WR alone is a mean-reversion strategy. It misses strong trending years
     (e.g. 2017 was +1.76% — very weak).  Adding trend-following fills this gap.
  3. 20-day momentum breakouts reliably catch continuation after consolidations
     (e.g. 2013-14 bull run, 2020 recovery, 2023 breakout) that WR never enters.
  4. Monthly rotation across 5 instruments (picking top 3) is more diversified
     than picking top 2 from 4.

NEW IN v8:
  Strategy A  — WR + VIX sizing (v7 best, kept unchanged)
  Strategy B  — SuperTrend + ADX trend following (NEW)
                Catches big directional moves: 2020 recovery, 2021 bull run,
                2013-14 strong uptrend that WR largely missed.
  Strategy C  — 20-Day High Momentum Breakout (NEW)
                Buys when price breaks above 20-day high with volume + trend
                confirmation. Riding momentum, not fading it.
  Strategy D  — VIX Spike mean reversion (v7, kept)
  Strategy E  — Monthly rotation top-3 from 5 instruments (IMPROVED)

INSTRUMENT UNIVERSE (5 instruments):
  NIFTY      : from Jan 2008 (full 18yr history)
  BANKNIFTY  : from Jan 2008 (full 18yr history)
  FINNIFTY   : from Jul 2021 (F&O launch date)
  MIDCPNIFTY : from Oct 2023 (F&O launch date)
  NIFTYIT    : from Jan 2012 (IT index futures — liquid from 2012)
               Correlation with BANKNIFTY: 0.31 — genuine diversification!

All v7 structural fixes retained:
  Fix A: Date UNION (no more intersection collapse)
  Fix B: Max drawdown from daily equity curve
  Fix C: Rotation exits at next-day open
  Fix D: 0.20%/month futures roll cost
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
ROLL_COST_PCT  = 0.0020          # 0.20% per monthly roll

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
    "NIFTYIT":    pd.Timestamp("2012-01-01"),  # IT futures liquid from 2012
}

MAX_LOTS_PER_INST = {
    "NIFTY":      20,
    "BANKNIFTY":  20,
    "MIDCPNIFTY": 15,
    "FINNIFTY":   15,
    "NIFTYIT":    20,
}

RISK_PER_TRADE = 0.04
MAX_CONCURRENT = 5          # One per instrument (5 now)
MAX_TOTAL_RISK = 0.10       # 10% total (was 8% in v7; 5 instruments justify more room)
WARMUP         = 260

SPAN_MARGIN_PCT    = 0.10
MAINT_MARGIN_RATIO = 0.75

INSTRUMENTS_5 = ["NIFTY", "BANKNIFTY", "MIDCPNIFTY", "FINNIFTY", "NIFTYIT"]


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
    d["high_52wk"] = df.close.rolling(252).max().shift(1)
    d["high_20d"]  = df.high.rolling(20).max().shift(1)   # 20-day high (shift=no lookahead)
    d["roc1"]      = df.close.pct_change(21) * 100
    d["roc3"]      = df.close.pct_change(63) * 100
    d["vol_sma20"] = df["volume"].rolling(20).mean()
    d["st_d"], d["st_u"], d["st_l"] = supertrend(df, 10, 2.0)
    return d


# ─────────────────────────────────────────────────────────────────────────────
# VIX UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def vix_dynamic_risk(vix_value):
    if vix_value is None or np.isnan(float(vix_value)): return 0.04
    v = float(vix_value)
    if v > 20:  return 0.05
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
            if low  <= stop: return self._close(instrument, date, stop, "stop")
            if high >= tgt:  return self._close(instrument, date, tgt,  "target")
        else:
            if high >= stop: return self._close(instrument, date, stop, "stop")
            if low  <= tgt:  return self._close(instrument, date, tgt,  "target")
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

    def stats(self, label="Portfolio"):
        T = self.trades
        if not T:
            return {k: 0 for k in ["ann_ret","total_ret","sharpe","max_dd",
                                    "win_rate","num_trades","net_pnl","capital"]}
        df   = pd.DataFrame(T)
        wr   = (df.pnl > 0).sum() / len(df) * 100
        tot  = (self.C - self.C0) / self.C0 * 100
        t0   = pd.to_datetime(T[0]["entry_date"])
        t1   = pd.to_datetime(T[-1]["exit_date"])
        yrs  = max((t1 - t0).days / 365.25, 0.1)
        ann  = ((1 + tot / 100) ** (1 / yrs) - 1) * 100
        aw   = df.loc[df.pnl > 0, "pnl"].mean() if (df.pnl > 0).any() else 0
        al   = df.loc[df.pnl < 0, "pnl"].mean() if (df.pnl < 0).any() else 0

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

        return dict(label=label, total_ret=round(tot, 2), ann_ret=round(ann, 2),
                    sharpe=round(shr, 3), max_dd=round(dd, 2), win_rate=round(wr, 1),
                    num_trades=len(T), net_pnl=round(self.C - self.C0),
                    capital=round(self.C), avg_win=round(aw), avg_loss=round(al),
                    trades_per_yr=round(len(T) / yrs, 1),
                    margin_calls=self.margin_calls,
                    roll_cost_total=round(self.roll_cost_total))


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def signal_williams_r(i, ind):
    """Williams%R(14) mean-reversion — proven core signal."""
    wr   = ind["wR14"].iloc[i];   wr_p = ind["wR14"].iloc[i-1]
    e200 = ind["e200"].iloc[i];   e50  = ind["e50"].iloc[i]
    atr_ = ind["atr"].iloc[i];    p    = ind["close"].iloc[i]
    if np.isnan(e200): return None
    if wr_p < -80 and wr >= -80 and p > e200 * 0.97 and e50 > e200 * 0.98:
        return ("L", atr_ * 1.5, "WR_OB")
    if wr_p > -20 and wr <= -20 and p < e200 * 1.03 and e50 < e200 * 1.02:
        return ("S", atr_ * 1.5, "WR_OS")
    return None


def signal_supertrend(i, ind):
    """
    SuperTrend direction flip = trend entry.
    Catches big trending moves that mean-reversion (WR) misses.
    Only enters when ADX > 22 (confirmed trend, not just noise).
    """
    st_d  = ind["st_d"].iloc[i];  st_dp = ind["st_d"].iloc[i-1]
    adx   = ind["adx14"].iloc[i]
    e200  = ind["e200"].iloc[i];  e50   = ind["e50"].iloc[i]
    atr_  = ind["atr"].iloc[i];   p     = ind["close"].iloc[i]
    pdi   = ind["plus_di"].iloc[i]; mdi = ind["minus_di"].iloc[i]
    if np.isnan(adx) or np.isnan(e200): return None
    if adx < 22: return None   # Only trade confirmed trends
    # Long: ST flipped up + price above EMA200 + +DI > -DI (bulls in control)
    if st_dp == -1 and st_d == 1 and p > e200 * 0.97 and pdi > mdi:
        return ("L", atr_ * 2.0, "ST_TREND_L")
    # Short: ST flipped down + price below EMA200 + -DI > +DI (bears in control)
    if st_dp == 1 and st_d == -1 and p < e200 * 1.03 and mdi > pdi:
        return ("S", atr_ * 2.0, "ST_TREND_S")
    return None


def signal_momentum_20d(i, ind):
    """
    20-Day High Breakout Momentum.
    Buys when index closes above its 20-day high — a classic breakout that
    signals momentum continuation. Requires EMA trend and moderate ADX.
    Volume confirmation applied where available.
    """
    h20  = ind["high_20d"].iloc[i]   # yesterday's 20-day high (no lookahead)
    p    = ind["close"].iloc[i]
    e50  = ind["e50"].iloc[i];   e200 = ind["e200"].iloc[i]
    adx  = ind["adx14"].iloc[i]
    atr_ = ind["atr"].iloc[i]
    vol  = ind["volume"].iloc[i]; vsma = ind["vol_sma20"].iloc[i]
    if np.isnan(h20) or np.isnan(e200) or h20 <= 0: return None
    if adx < 18: return None   # Only enter when there's some directional strength
    # Volume confirmation (relax if no volume data)
    vol_ok = (vsma <= 0) or np.isnan(vsma) or (vol > vsma * 1.2)
    # Long: close > 20-day high, price trending above EMA50 and EMA200
    if p > h20 and p > e50 and p > e200 * 0.98 and vol_ok:
        return ("L", atr_ * 2.0, "MOM_20D_L")
    return None


def signal_vix_spike(i, ind, vix_chg_today):
    """VIX 1-day jump/drop >15% — NIFTY only, mean reversion after panic."""
    if vix_chg_today is None or np.isnan(float(vix_chg_today)): return None
    chg  = float(vix_chg_today)
    atr_ = ind["atr"].iloc[i];  e200 = ind["e200"].iloc[i]
    if np.isnan(e200): return None
    if chg >  0.15: return ("L", atr_ * 1.5, "VIX_SPIKE")
    if chg < -0.15: return ("S", atr_ * 1.5, "VIX_DROP")
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

    # WR mean-reversion exits
    if tag.startswith(("WR_",)):
        wr14  = ind["wR14"].iloc[ii]; wr14p = ind["wR14"].iloc[ii-1]
        if d == 1 and wr14p < -50 and wr14 >= -50:
            pf.force_close(inst, date, p, "WR_MID_EXIT"); return
        if d == -1 and wr14p > -50 and wr14 <= -50:
            pf.force_close(inst, date, p, "WR_MID_EXIT"); return
        e21  = ind["e21"].iloc[ii]; atr_ = ind["atr"].iloc[ii]
        pf.trail_stop(inst, e21 - d * atr_ * 0.3)

    # SuperTrend trend exits — exit when SuperTrend direction flips against us
    elif tag.startswith("ST_TREND"):
        st_d = ind["st_d"].iloc[ii]
        atr_ = ind["atr"].iloc[ii]
        if d == 1 and st_d == -1:
            pf.force_close(inst, date, p, "ST_FLIP_EXIT"); return
        if d == -1 and st_d == 1:
            pf.force_close(inst, date, p, "ST_FLIP_EXIT"); return
        # Trail using the SuperTrend line itself
        st_line = ind["st_l"].iloc[ii] if d == 1 else ind["st_u"].iloc[ii]
        pf.trail_stop(inst, st_line)

    # 20-Day momentum exits — exit when price drops below EMA21 or ST flips
    elif tag.startswith("MOM_20D"):
        st_d = ind["st_d"].iloc[ii]
        e21  = ind["e21"].iloc[ii]; e50  = ind["e50"].iloc[ii]
        atr_ = ind["atr"].iloc[ii]
        # Exit if SuperTrend turns against us or price drops below EMA50
        if d == 1 and (st_d == -1 or p < e50 * 0.99):
            pf.force_close(inst, date, p, "MOM_20D_EXIT"); return
        # Trail using EMA21 - 0.5 ATR
        pf.trail_stop(inst, e21 - d * atr_ * 0.5)

    # VIX spike exits — normalise when VIX reverts to 5-bar mean
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

    # Monthly rotation exits
    elif tag.startswith("MONTHLY"):
        e200 = ind["e200"].iloc[ii]; e50  = ind["e50"].iloc[ii]
        atr_ = ind["atr"].iloc[ii]
        if d == 1 and not np.isnan(e200) and p < e200:
            pf.force_close(inst, date, p, "MONTHLY_EMA_EXIT"); return
        if d == 1:
            pf.trail_stop(inst, e50 - atr_ * 0.5)

    if inst in pf.positions:
        pf.check_exits(date, inst, ph, pl, p)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PORTFOLIO RUNNER
# ─────────────────────────────────────────────────────────────────────────────
def run_v8(instruments, strategy_names, capital=CAPITAL):
    data = {}; inds = {}
    for inst in instruments:
        df = load_data(inst)
        data[inst] = df
        inds[inst] = precompute(df)

    vix_df = load_data("INDIAVIX")

    # Date UNION — all 5 instruments, each trades when data + live date allow
    union_set = set.union(*[set(data[inst].index) for inst in instruments])
    all_dates = pd.DatetimeIndex(sorted(union_set))

    vix_chg_raw = vix_df.close.pct_change()
    vix_series  = vix_df.close.reindex(all_dates, method="ffill")
    vix_chg     = vix_chg_raw.reindex(all_dates)

    pf = Portfolio(capital)

    use_wr_vix  = "williams_r_vix"    in strategy_names
    use_st      = "supertrend"         in strategy_names
    use_mom20   = "momentum_20d"       in strategy_names
    use_vix_spk = "vix_spike"          in strategy_names
    use_monthly = "monthly_rotation"   in strategy_names

    monthly_allowed = set(instruments)
    last_rot_month  = (-1, -1)
    pending_entries = {}
    pending_exits   = {}
    last_roll_month = (-1, -1)

    for idx, date in enumerate(all_dates):
        if idx < WARMUP: continue

        vix_today     = vix_series.iloc[idx] if idx < len(vix_series) else None
        vix_chg_today = vix_chg.iloc[idx]    if idx < len(vix_chg)    else None
        if vix_today is not None and (pd.isna(vix_today) or np.isnan(float(vix_today))):
            vix_today = None
        if vix_chg_today is not None and (pd.isna(vix_chg_today)
                                          or np.isnan(float(vix_chg_today))):
            vix_chg_today = None

        # Execute pending rotation EXITS at today's open (Fix C)
        for inst, reason in list(pending_exits.items()):
            del pending_exits[inst]
            if inst not in pf.positions: continue
            if inst not in data or date not in data[inst].index: continue
            ii = data[inst].index.get_loc(date)
            pf.force_close(inst, date, float(inds[inst]["open"].iloc[ii]), reason)

        # Execute pending ENTRIES at today's open
        for inst, (strat, d_val, stop_dist, tag, risk) in list(pending_entries.items()):
            del pending_entries[inst]
            if inst in pf.positions: continue
            if inst not in data or date not in data[inst].index: continue
            ii    = data[inst].index.get_loc(date)
            entry = float(inds[inst]["open"].iloc[ii])
            stop  = entry - d_val * stop_dist
            pf.enter(inst, strat, date, d_val, entry, stop, tag, risk)

        # EXITS — strategy-based, checked at today's bar
        for inst in list(pf.positions.keys()):
            if inst not in data or date not in data[inst].index: continue
            ii = data[inst].index.get_loc(date)
            if ii < 1: continue
            process_exit(pf, inst, date, inds[inst], ii, vix_series)

        # MONTHLY ROTATION — top 3 from 5 instruments
        if use_monthly:
            cur_month = (date.year, date.month)
            if cur_month != last_rot_month:
                last_rot_month = cur_month
                scores = {}
                for inst in instruments:
                    if date < INSTRUMENT_LIVE_DATE.get(inst, pd.Timestamp("2008-01-01")):
                        continue
                    if date not in data[inst].index: continue
                    ii = data[inst].index.get_loc(date)
                    if ii < WARMUP: continue
                    ind = inds[inst]
                    p_  = ind["close"].iloc[ii]; e200_ = ind["e200"].iloc[ii]
                    if np.isnan(e200_) or p_ < e200_: continue
                    r1 = ind["roc1"].iloc[ii]; r3 = ind["roc3"].iloc[ii]
                    if np.isnan(r1) or np.isnan(r3): continue
                    scores[inst] = r1 * 0.4 + r3 * 0.6

                # Top 3 (was top 2 in v7)
                new_allowed = (set(sorted(scores, key=lambda x: scores[x],
                                         reverse=True)[:3])
                               if scores else set(instruments))

                # Queue exits for instruments no longer in top-3 (Fix C: next-day open)
                for inst in list(pf.positions.keys()):
                    if (pf.positions[inst]["tag"].startswith("MONTHLY")
                            and inst not in new_allowed):
                        pending_exits[inst] = "MONTHLY_ROTATE_OUT"

                monthly_allowed = new_allowed
                for inst in monthly_allowed:
                    if inst in pf.positions or inst in pending_entries: continue
                    if date < INSTRUMENT_LIVE_DATE.get(inst, pd.Timestamp("2008-01-01")): continue
                    if date not in data[inst].index: continue
                    ii = data[inst].index.get_loc(date)
                    ind = inds[inst]
                    p_  = ind["close"].iloc[ii]; e200_ = ind["e200"].iloc[ii]
                    if np.isnan(e200_) or p_ < e200_: continue
                    atr_ = ind["atr"].iloc[ii]
                    pending_entries[inst] = ("MONTHLY_ROT", 1, atr_ * 1.5,
                                             "MONTHLY_ROT", RISK_PER_TRADE)

        # DAILY MTM SETTLEMENT
        for inst, pos in list(pf.positions.items()):
            if inst not in data or date not in data[inst].index: continue
            ii          = data[inst].index.get_loc(date)
            today_close = float(inds[inst]["close"].iloc[ii])
            last_price  = pos.get("last_mtm_price", pos["entry"])
            pf.C       += pos["d"] * (today_close - last_price) * pos["qty"]
            pos["last_mtm_price"] = today_close

        # MARGIN CALL CHECK
        if pf.margin_reserved:
            total_maint = sum(pf.margin_reserved.values()) * MAINT_MARGIN_RATIO
            if pf.C < total_maint:
                pf.margin_calls += 1
                for inst in list(pf.positions.keys()):
                    if inst in data and date in data[inst].index:
                        ii = data[inst].index.get_loc(date)
                        pf.force_close(inst, date,
                                       float(inds[inst]["close"].iloc[ii]), "margin_call")

        # MONTHLY ROLL COST — deduct on last trading day of each month
        cur_month = (date.year, date.month)
        if cur_month != last_roll_month and pf.positions:
            if idx + 1 >= len(all_dates) or all_dates[idx + 1].month != date.month:
                last_roll_month = cur_month
                for inst, pos in list(pf.positions.items()):
                    if inst not in data or date not in data[inst].index: continue
                    ii          = data[inst].index.get_loc(date)
                    close_price = float(inds[inst]["close"].iloc[ii])
                    roll_cost   = pos["qty"] * close_price * ROLL_COST_PCT
                    pf.C -= roll_cost
                    pf.roll_cost_total += roll_cost

        # ENTRY SIGNALS → pending (no same-bar execution)
        for inst in instruments:
            if inst in pf.positions or inst in pending_entries: continue
            if date < INSTRUMENT_LIVE_DATE.get(inst, pd.Timestamp("2008-01-01")): continue
            if inst not in data or date not in data[inst].index: continue
            ii = data[inst].index.get_loc(date)
            if ii < WARMUP: continue
            if idx + 1 >= len(all_dates): continue

            ind = inds[inst]
            sig = None

            # Priority 1: VIX Spike (NIFTY only)
            if use_vix_spk and inst == "NIFTY":
                sig = signal_vix_spike(ii, ind, vix_chg_today)

            # Priority 2: Williams%R + VIX dynamic sizing
            if sig is None and use_wr_vix:
                s = signal_williams_r(ii, ind)
                if s is not None:
                    risk = vix_dynamic_risk(vix_today)
                    pending_entries[inst] = (f"WR_VIX_{s[2]}", 1 if s[0]=="L" else -1,
                                             s[1], s[2], risk)
                    continue

            # Priority 3: SuperTrend trend-following
            if sig is None and use_st:
                sig = signal_supertrend(ii, ind)

            # Priority 4: 20-Day momentum breakout
            if sig is None and use_mom20:
                sig = signal_momentum_20d(ii, ind)

            if sig is None: continue
            direction_str, stop_dist, tag = sig
            d_val = 1 if direction_str == "L" else -1
            pending_entries[inst] = (tag, d_val, stop_dist, tag, RISK_PER_TRADE)

        pf.record_equity(date)

    # Force-close all open positions at end of backtest
    for inst in list(pf.positions.keys()):
        last_p = float(data[inst].close.iloc[-1])
        pf.force_close(inst, all_dates[-1], last_p, "end_of_backtest")

    return pf


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def attribution_stats(pf):
    df = pd.DataFrame(pf.trades)
    if df.empty: return []
    results = []
    for strat, grp in df.groupby("strategy"):
        wr = (grp.pnl > 0).sum() / len(grp) * 100
        aw = grp.loc[grp.pnl > 0, "pnl"].mean() if (grp.pnl > 0).any() else 0
        al = grp.loc[grp.pnl < 0, "pnl"].mean() if (grp.pnl < 0).any() else 0
        results.append(dict(strategy=strat, trades=len(grp), win_rate=round(wr, 1),
                            net_pnl=round(grp.pnl.sum()), avg_win=round(aw),
                            avg_loss=round(al)))
    return sorted(results, key=lambda x: x["net_pnl"], reverse=True)


def instrument_stats(pf):
    df = pd.DataFrame(pf.trades)
    if df.empty: return []
    results = []
    for inst, grp in df.groupby("instrument"):
        wr = (grp.pnl > 0).sum() / len(grp) * 100
        results.append(dict(instrument=inst, trades=len(grp),
                            win_rate=round(wr, 1), net_pnl=round(grp.pnl.sum())))
    return sorted(results, key=lambda x: x["net_pnl"], reverse=True)


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


# ─────────────────────────────────────────────────────────────────────────────
# NAMED CONFIGS
# ─────────────────────────────────────────────────────────────────────────────
CONFIGS = {
    # Individual strategies (benchmarks)
    "WR_VIX_5Inst":     (INSTRUMENTS_5, ["williams_r_vix"]),
    "SuperTrend_5Inst": (INSTRUMENTS_5, ["supertrend"]),
    "Momentum_20D":     (INSTRUMENTS_5, ["momentum_20d"]),
    "VIX_Spike+WR":     (INSTRUMENTS_5, ["vix_spike", "williams_r_vix"]),
    "Monthly_Rot5":     (INSTRUMENTS_5, ["monthly_rotation"]),
    # Combined — the flagship
    "COMBINED_V8":      (INSTRUMENTS_5, ["vix_spike", "williams_r_vix",
                                          "supertrend", "momentum_20d",
                                          "monthly_rotation"]),
}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    console.rule("[bold cyan]v8 BACKTEST — Enhanced 5-Instrument Multi-Strategy[/bold cyan]")
    console.print(
        "[dim]Instruments: NIFTY · BANKNIFTY · FINNIFTY · MIDCPNIFTY · NIFTYIT (NEW)\n"
        "NIFTYIT correlation: 0.56 vs NIFTY, 0.31 vs BANKNIFTY — genuine diversification\n"
        "New strategies: SuperTrend trend-following · 20-Day Momentum Breakout\n"
        "Monthly rotation: top 3 from 5 (was top 2 from 4)\n"
        "Max concurrent: 5 | Total risk cap: 10% | All v7 structural fixes retained[/dim]\n"
    )

    results = {}; portfolios = {}

    for cfg_name, (insts, strats) in CONFIGS.items():
        console.print(f"  [cyan]Running {cfg_name:20s}[/cyan]", end=" ")
        try:
            pf = run_v8(insts, strats, CAPITAL)
            s  = pf.stats(cfg_name)
            results[cfg_name]    = s
            portfolios[cfg_name] = pf
            color = "green" if s["ann_ret"] >= 25 else ("yellow" if s["ann_ret"] >= 15 else "red")
            console.print(
                f"[{color}]{s['ann_ret']:+.1f}%[/{color}] ann | "
                f"Sharpe {s['sharpe']:.2f} | MaxDD {s['max_dd']:.1f}% | "
                f"{s['num_trades']} trades | roll ₹{s['roll_cost_total']:,.0f}"
            )
        except Exception as e:
            import traceback
            console.print(f"[red]FAILED: {e}[/red]")
            traceback.print_exc()
            results[cfg_name] = None

    console.print()
    console.rule("[bold cyan]Config Comparison[/bold cyan]")

    ranked = sorted([(k, v) for k, v in results.items() if v is not None],
                    key=lambda x: x[1]["ann_ret"], reverse=True)

    tbl = Table(box=box.DOUBLE_EDGE, show_header=True, header_style="bold cyan", min_width=150)
    tbl.add_column("Rank",     justify="right", min_width=5)
    tbl.add_column("Config",   min_width=20)
    tbl.add_column("Ann Ret%", justify="right", min_width=9)
    tbl.add_column("Total%",   justify="right", min_width=11)
    tbl.add_column("Sharpe",   justify="right", min_width=8)
    tbl.add_column("MaxDD%",   justify="right", min_width=8)
    tbl.add_column("Win%",     justify="right", min_width=7)
    tbl.add_column("Trades",   justify="right", min_width=7)
    tbl.add_column("Net P&L",  justify="right", min_width=14)
    tbl.add_column("Roll Cost",justify="right", min_width=12)
    tbl.add_column("Final Cap",justify="right", min_width=15)

    for rank, (name, s) in enumerate(ranked, 1):
        beats  = s["ann_ret"] >= 25
        color  = "bold green" if beats else ("yellow" if s["ann_ret"] >= 15 else "red")
        marker = "★" if name == "COMBINED_V8" else ("✓" if beats else " ")
        tbl.add_row(
            str(rank),
            f"[{color}]{marker} {name}[/{color}]",
            f"[{color}]{s['ann_ret']:+.1f}%[/{color}]",
            f"{s['total_ret']:+.1f}%",
            f"{s['sharpe']:.2f}",
            f"{s['max_dd']:.1f}%",
            f"{s['win_rate']:.1f}%",
            str(s["num_trades"]),
            f"₹{s['net_pnl']:+,.0f}",
            f"₹{s['roll_cost_total']:,.0f}",
            f"₹{s['capital']:,.0f}",
        )
    console.print(tbl)

    if not ranked: return {}

    best_name, best_s = ranked[0]
    best_pf = portfolios[best_name]

    console.print(f"\n[bold cyan]Best Config: {best_name}[/bold cyan]")

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

    # Strategy attribution
    attr = attribution_stats(best_pf)
    if attr:
        atbl = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold",
                     title="Strategy Attribution")
        atbl.add_column("Strategy", min_width=20)
        atbl.add_column("Trades",   justify="right")
        atbl.add_column("Win%",     justify="right")
        atbl.add_column("Net P&L",  justify="right")
        atbl.add_column("Avg Win",  justify="right")
        atbl.add_column("Avg Loss", justify="right")
        for a in attr:
            c = "green" if a["net_pnl"] > 0 else "red"
            atbl.add_row(a["strategy"], str(a["trades"]),
                         f"{a['win_rate']:.1f}%",
                         f"[{c}]₹{a['net_pnl']:+,.0f}[/{c}]",
                         f"₹{a['avg_win']:,.0f}", f"₹{a['avg_loss']:,.0f}")
        console.print(atbl)

    # Instrument attribution
    inst_attr = instrument_stats(best_pf)
    if inst_attr:
        itbl = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold",
                     title="Instrument Attribution")
        itbl.add_column("Instrument", min_width=14)
        itbl.add_column("Trades",     justify="right")
        itbl.add_column("Win%",       justify="right")
        itbl.add_column("Net P&L",    justify="right")
        itbl.add_column("Notes",      min_width=40)
        notes = {
            "NIFTY":      "Full 18yr history | Corr vs BANKNIFTY: 0.88",
            "BANKNIFTY":  "Full 18yr history | Corr vs NIFTY: 0.88",
            "NIFTYIT":    "From Jan-2012 | Corr vs BANKNIFTY: 0.31 ← diversifier",
            "FINNIFTY":   "From Jul-2021 (F&O launch)",
            "MIDCPNIFTY": "From Oct-2023 (F&O launch)",
        }
        for a in inst_attr:
            c = "green" if a["net_pnl"] > 0 else "red"
            itbl.add_row(a["instrument"], str(a["trades"]),
                         f"{a['win_rate']:.1f}%",
                         f"[{c}]₹{a['net_pnl']:+,.0f}[/{c}]",
                         notes.get(a["instrument"], ""))
        console.print(itbl)

    # Save JSON
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"reports/v8_backtest_{ts}.json"
    all_data = {}
    for name, s in results.items():
        if s is None: continue
        pf2 = portfolios[name]
        all_data[name] = {
            "stats": s, "trades": pf2.trades,
            "annual": annual_breakdown(pf2),
            "attribution": attribution_stats(pf2),
            "instrument_stats": instrument_stats(pf2),
            "equity_curve": [(str(d), v) for d, v in pf2.daily_equity],
        }
    with open(fname, "w") as f:
        json.dump(all_data, f, indent=2, default=str)
    console.print(f"\n[dim]Results saved → {fname}[/dim]")

    ann      = best_s["ann_ret"]
    eq_dates = [d for d, _ in best_pf.daily_equity]
    period   = (f"{eq_dates[0].date()} → {eq_dates[-1].date()}"
                f" (~{(eq_dates[-1]-eq_dates[0]).days/365.25:.1f} yrs)"
                if eq_dates else "n/a")
    v7_ann   = 26.6   # benchmark from v7

    console.print(Panel(
        f"[bold {'green' if ann >= 25 else 'yellow'}]"
        f"v8 RESULT — {ann:+.1f}% annualised ({best_name})[/bold {'green' if ann >= 25 else 'yellow'}]\n\n"
        f"  Period tested    : {period}\n"
        f"  vs v7 benchmark  : {ann:+.1f}% vs {v7_ann:+.1f}%  "
        f"({'▲ IMPROVED' if ann > v7_ann else '▼ WORSE'} by {abs(ann-v7_ann):.1f}%)\n"
        f"  Sharpe           : {best_s['sharpe']:.2f}\n"
        f"  Max Drawdown     : {best_s['max_dd']:.1f}%  [daily equity curve]\n"
        f"  Win Rate         : {best_s['win_rate']:.1f}%\n"
        f"  Net P&L          : ₹{best_s['net_pnl']:+,.0f}\n"
        f"  Final Capital    : ₹{best_s['capital']:,.0f}\n"
        f"  Roll Costs       : ₹{best_s['roll_cost_total']:,.0f}\n"
        f"  Margin Calls     : {best_s['margin_calls']}\n\n"
        f"  New alpha sources vs v7:\n"
        f"    + NIFTYIT (5th instrument, 0.31 corr with BANKNIFTY)\n"
        f"    + SuperTrend ADX trend-following (catches big directional moves)\n"
        f"    + 20-Day Momentum Breakout (rides continuation after breakouts)\n"
        f"    + Monthly rotation: top 3 from 5 (was top 2 from 4)\n"
        f"    + MAX_TOTAL_RISK 10% (was 8%) — justified by 5-instrument diversification",
        title="[bold]RESULT — v8 ENHANCED BACKTEST[/bold]",
        border_style="green" if ann >= 25 else "yellow"
    ))

    return results


if __name__ == "__main__":
    main()
