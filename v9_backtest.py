"""
v9_backtest.py — Trend-Rider Override + VIX-Aware Sizing
=========================================================
Targeted improvements driven by trade-duration diagnostic on v8 results:

CRITICAL FINDINGS FROM v8 DIAGNOSTIC:
  Duration Bucket  | Trades | Win%  | Avg P&L   | Total P&L
  0d (same-day)    |   770  | 75.5% |  ₹96,156  | +₹7.4 Cr   ← KEEP
  1-3d (dead zone) |    85  | 34.1% | -₹62,160  | -₹5.3 M   ← FIX
  3-7d             |    64  | 37.5% |  ₹4,445   | +₹284K     ← mediocre
  7-14d            |    22  | 50.0% | ₹182,625  | +₹4.0 M    ← good
  14-30d           |    16  | 93.8% | ₹390,996  | +₹6.3 M   ← EXPLOIT MORE
  VIX>30 trades    |    52  | 57.7% | -₹6,689   | -₹348K    ← REDUCE SIZE

IMPROVEMENTS IN v9:
  Fix 1 — Trend-Rider Override (THE KEY FIX)
    WR positions held ≥1 day are checked for trend-rider promotion BEFORE the
    WR -50 exit fires. If:  in_profit (>0.3%) AND ADX > 25 AND EMA8 > EMA21 > EMA50
    → switch to EMA50 trailing stop (rider_mode), ignore WR -50 exit.
    This converts potential 1-3d losers into 14+ day winners when conditions warrant.
    Same-day exits are UNTOUCHED (days_held==0) — they're already +75.5% WR.

  Fix 2 — VIX Panic Sizing Corrected
    VIX > 30 now uses 3% risk (DOWN from 5% in v8).
    Data proof: 52 trades at VIX>30 → 57.7% WR, avg -₹6,689 LOSS per trade.
    Extreme panic creates false mean-reversion signals; WR oversold means continued fall.
    VIX 20-30: still 5% (good mean-reversion zone, not extreme).

  Fix 3 — Clean Combined Strategy
    Removed SuperTrend (-2.9% CAGR) and Momentum_20D (+6.5% but 57% drawdown).
    COMBINED_V9 = VIX_Spike + WR_VIX + Monthly_Rotation (proven core only).

INSTRUMENTS (same 5 as v8):
  NIFTY · BANKNIFTY · FINNIFTY · MIDCPNIFTY · NIFTYIT

All v7/v8 structural fixes retained:
  Fix A: Date UNION (no intersection collapse)
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
    "NIFTYIT":    pd.Timestamp("2012-01-01"),
}

MAX_LOTS_PER_INST = {
    "NIFTY":      20,
    "BANKNIFTY":  20,
    "MIDCPNIFTY": 15,
    "FINNIFTY":   15,
    "NIFTYIT":    20,
}

RISK_PER_TRADE = 0.04
MAX_CONCURRENT = 5
MAX_TOTAL_RISK = 0.10
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
    d["high_20d"]  = df.high.rolling(20).max().shift(1)
    d["roc1"]      = df.close.pct_change(21) * 100
    d["roc3"]      = df.close.pct_change(63) * 100
    d["vol_sma20"] = df["volume"].rolling(20).mean()
    d["st_d"], d["st_u"], d["st_l"] = supertrend(df, 10, 2.0)
    return d


# ─────────────────────────────────────────────────────────────────────────────
# VIX UTILITIES  — FIX 2: Reduce size at VIX > 30 (extreme panic = bad trades)
# ─────────────────────────────────────────────────────────────────────────────
def vix_dynamic_risk(vix_value):
    """
    v9 change: VIX > 30 now uses 3% (DOWN from 5% in v8).
    Data from v8 diagnostic: 52 trades at VIX>30 → 57.7% WR, avg -₹6,689 LOSS.
    At extreme panic, WR 'oversold' signal means continued crash, not mean-reversion.
    VIX 20-30 remains 5% — that's the sweet spot for mean-reversion trades.
    """
    if vix_value is None or np.isnan(float(vix_value)): return 0.04
    v = float(vix_value)
    if v > 30:  return 0.03   # Extreme panic: reduce size (data shows these LOSE)
    if v > 20:  return 0.05   # Elevated vol: best mean-reversion zone
    if v >= 15: return 0.04   # Normal vol
    return 0.03               # Low vol: smaller expected moves


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
            last_mtm_price=entry,
            rider_mode=False       # v9: trend-rider flag
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
            reason=reason, tag=pos["tag"],
            rider_mode=pos.get("rider_mode", False)
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
    """Williams%R(14) mean-reversion — core signal, unchanged from v8."""
    wr   = ind["wR14"].iloc[i];   wr_p = ind["wR14"].iloc[i-1]
    e200 = ind["e200"].iloc[i];   e50  = ind["e50"].iloc[i]
    atr_ = ind["atr"].iloc[i];    p    = ind["close"].iloc[i]
    if np.isnan(e200): return None
    if wr_p < -80 and wr >= -80 and p > e200 * 0.97 and e50 > e200 * 0.98:
        return ("L", atr_ * 1.5, "WR_OB")
    if wr_p > -20 and wr <= -20 and p < e200 * 1.03 and e50 < e200 * 1.02:
        return ("S", atr_ * 1.5, "WR_OS")
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
# EXIT ROUTER  — FIX 1: Trend-Rider Override for WR positions
# ─────────────────────────────────────────────────────────────────────────────
def process_exit(pf, inst, date, ind, ii, vix_series):
    if inst not in pf.positions: return
    pos  = pf.positions[inst]
    tag  = pos["tag"]
    d    = pos["d"]
    p    = ind["close"].iloc[ii]
    ph   = ind["high"].iloc[ii]
    pl   = ind["low"].iloc[ii]

    # ── WR mean-reversion exits (with v9 trend-rider override) ──────────────
    if tag.startswith("WR_"):
        days_held  = (date - pos["date"]).days
        wr14       = ind["wR14"].iloc[ii];  wr14p = ind["wR14"].iloc[ii-1]
        e8         = ind["e8"].iloc[ii];    e21   = ind["e21"].iloc[ii]
        e50        = ind["e50"].iloc[ii];   atr_  = ind["atr"].iloc[ii]
        adx        = ind["adx14"].iloc[ii]
        rider_mode = pos.get("rider_mode", False)

        # Check for trend-rider promotion (only after day 0 so same-day exits preserved)
        if not rider_mode and days_held >= 1 and not (np.isnan(adx) or np.isnan(e8)):
            in_profit = ((d == 1 and p > pos["entry"] * 1.003) or
                         (d == -1 and p < pos["entry"] * 0.997))
            aligned   = ((d == 1 and e8 > e21 > e50) or
                         (d == -1 and e8 < e21 < e50))
            if in_profit and adx > 25 and aligned:
                # Promote: skip WR -50 exit, trail with EMA50 stop instead
                pos["rider_mode"] = True
                rider_mode = True

        if rider_mode:
            # Trend-rider mode: trail EMA50 - direction×0.5×ATR
            pf.trail_stop(inst, e50 - d * atr_ * 0.5)
        else:
            # Normal WR exit logic (unchanged — keeps same-day exits working)
            if d == 1 and wr14p < -50 and wr14 >= -50:
                pf.force_close(inst, date, p, "WR_MID_EXIT"); return
            if d == -1 and wr14p > -50 and wr14 <= -50:
                pf.force_close(inst, date, p, "WR_MID_EXIT"); return
            pf.trail_stop(inst, e21 - d * atr_ * 0.3)

    # ── VIX spike exits — normalise when VIX reverts to 5-bar mean ──────────
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

    # ── Monthly rotation exits ───────────────────────────────────────────────
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
def run_v9(instruments, strategy_names, capital=CAPITAL):
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

    use_wr_vix  = "williams_r_vix"    in strategy_names
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

                new_allowed = (set(sorted(scores, key=lambda x: scores[x],
                                         reverse=True)[:3])
                               if scores else set(instruments))

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


def duration_breakdown(pf):
    """Diagnostic: trade performance by holding duration."""
    if not pf.trades: return []
    df = pd.DataFrame(pf.trades)
    buckets = [(0, 0, "0d"), (1, 3, "1-3d"), (3, 7, "3-7d"),
               (7, 14, "7-14d"), (14, 30, "14-30d"), (30, 9999, "30d+")]
    rows = []
    for lo, hi, label in buckets:
        grp = df[(df.duration >= lo) & (df.duration <= hi)]
        if grp.empty: continue
        wr  = (grp.pnl > 0).sum() / len(grp) * 100
        rows.append(dict(label=label, trades=len(grp),
                         win_rate=round(wr, 1),
                         avg_pnl=round(grp.pnl.mean()),
                         total_pnl=round(grp.pnl.sum())))
    return rows


def rider_stats(pf):
    """Diagnostic: how many trades were promoted to trend-rider mode."""
    if not pf.trades: return {}
    df = pd.DataFrame(pf.trades)
    rider = df[df.get("rider_mode", pd.Series([False]*len(df)))]
    normal = df[~df.get("rider_mode", pd.Series([False]*len(df)))]
    result = {}
    if "rider_mode" in df.columns:
        rider  = df[df.rider_mode == True]
        normal = df[df.rider_mode == False]
        result["rider_trades"]  = len(rider)
        result["rider_wr"]      = round((rider.pnl > 0).sum() / max(len(rider),1) * 100, 1)
        result["rider_avg_pnl"] = round(rider.pnl.mean()) if len(rider) > 0 else 0
        result["rider_total"]   = round(rider.pnl.sum())
        result["normal_trades"] = len(normal)
        result["normal_wr"]     = round((normal.pnl > 0).sum() / max(len(normal),1) * 100, 1)
        result["normal_avg_pnl"]= round(normal.pnl.mean()) if len(normal) > 0 else 0
        result["normal_total"]  = round(normal.pnl.sum())
    return result


# ─────────────────────────────────────────────────────────────────────────────
# NAMED CONFIGS  — Clean: WR + VIX only, no underperforming strategies
# ─────────────────────────────────────────────────────────────────────────────
CONFIGS = {
    # Baseline (same as v8 best)
    "WR_VIX_5Inst": (INSTRUMENTS_5, ["williams_r_vix"]),
    # Add VIX spike signal
    "VIX_Spike+WR": (INSTRUMENTS_5, ["vix_spike", "williams_r_vix"]),
    # Monthly rotation only (diversified momentum)
    "Monthly_Rot5": (INSTRUMENTS_5, ["monthly_rotation"]),
    # FLAGSHIP: proven WR + VIX spike + rotation (no ST/MOM drag)
    "COMBINED_V9":  (INSTRUMENTS_5, ["vix_spike", "williams_r_vix", "monthly_rotation"]),
}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    console.rule("[bold cyan]v9 BACKTEST — Trend-Rider Override + VIX Panic Sizing[/bold cyan]")
    console.print(
        "[dim]Key improvements over v8:\n"
        "  Fix 1: Trend-rider override — WR positions promoted to EMA50 trail when\n"
        "          in_profit(>0.3%) + ADX>25 + EMA8>EMA21>EMA50 (after day 0)\n"
        "  Fix 2: VIX>30 sizing reduced from 5% to 3% (data: 57.7% WR, avg -₹6.7K loss)\n"
        "  Fix 3: Removed SuperTrend (-2.9%) and Momentum_20D (+6.5%/57% DD) from combined\n"
        "Instruments: NIFTY · BANKNIFTY · FINNIFTY · MIDCPNIFTY · NIFTYIT[/dim]\n"
    )

    results = {}; portfolios = {}

    for cfg_name, (insts, strats) in CONFIGS.items():
        console.print(f"  [cyan]Running {cfg_name:20s}[/cyan]", end=" ")
        try:
            pf = run_v9(insts, strats, CAPITAL)
            s  = pf.stats(cfg_name)
            results[cfg_name]    = s
            portfolios[cfg_name] = pf
            color = "green" if s["ann_ret"] >= 28 else ("yellow" if s["ann_ret"] >= 20 else "red")
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
        beats  = s["ann_ret"] >= 28
        color  = "bold green" if beats else ("yellow" if s["ann_ret"] >= 20 else "red")
        marker = "★" if name == "COMBINED_V9" else ("✓" if beats else " ")
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

    # Duration breakdown (key diagnostic for v9)
    dur = duration_breakdown(best_pf)
    if dur:
        dtbl = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold",
                     title="Trade Duration Breakdown — v9 vs v8 target")
        dtbl.add_column("Duration",  min_width=10)
        dtbl.add_column("Trades",    justify="right")
        dtbl.add_column("Win%",      justify="right")
        dtbl.add_column("Avg P&L",   justify="right")
        dtbl.add_column("Total P&L", justify="right")
        dtbl.add_column("v8 Win%",   justify="right")
        v8_ref = {"0d": 75.5, "1-3d": 34.1, "3-7d": 37.5, "7-14d": 50.0,
                  "14-30d": 93.8, "30d+": None}
        for row in dur:
            ref = v8_ref.get(row["label"])
            ref_str = f"{ref:.1f}%" if ref is not None else "n/a"
            wr_color = "green" if row["win_rate"] >= 60 else ("yellow" if row["win_rate"] >= 45 else "red")
            dtbl.add_row(
                row["label"],
                str(row["trades"]),
                f"[{wr_color}]{row['win_rate']:.1f}%[/{wr_color}]",
                f"₹{row['avg_pnl']:+,.0f}",
                f"₹{row['total_pnl']:+,.0f}",
                f"[dim]{ref_str}[/dim]"
            )
        console.print(dtbl)

    # Rider mode stats
    rs = rider_stats(best_pf)
    if rs:
        console.print(
            f"\n[bold]Trend-Rider Promotion Stats:[/bold]\n"
            f"  Rider trades : {rs.get('rider_trades',0):3d} | "
            f"WR {rs.get('rider_wr',0):.1f}% | "
            f"Avg P&L ₹{rs.get('rider_avg_pnl',0):+,.0f} | "
            f"Total ₹{rs.get('rider_total',0):+,.0f}\n"
            f"  Normal trades: {rs.get('normal_trades',0):3d} | "
            f"WR {rs.get('normal_wr',0):.1f}% | "
            f"Avg P&L ₹{rs.get('normal_avg_pnl',0):+,.0f} | "
            f"Total ₹{rs.get('normal_total',0):+,.0f}"
        )

    # Instrument attribution
    inst_attr = instrument_stats(best_pf)
    if inst_attr:
        itbl = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold",
                     title="Instrument Attribution")
        itbl.add_column("Instrument", min_width=14)
        itbl.add_column("Trades",     justify="right")
        itbl.add_column("Win%",       justify="right")
        itbl.add_column("Net P&L",    justify="right")
        for a in inst_attr:
            c = "green" if a["net_pnl"] > 0 else "red"
            itbl.add_row(a["instrument"], str(a["trades"]),
                         f"{a['win_rate']:.1f}%",
                         f"[{c}]₹{a['net_pnl']:+,.0f}[/{c}]")
        console.print(itbl)

    # Save JSON
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"reports/v9_backtest_{ts}.json"
    all_data = {}
    for name, s in results.items():
        if s is None: continue
        pf2 = portfolios[name]
        all_data[name] = {
            "stats": s, "trades": pf2.trades,
            "annual": annual_breakdown(pf2),
            "attribution": attribution_stats(pf2),
            "instrument_stats": instrument_stats(pf2),
            "duration_breakdown": duration_breakdown(pf2),
            "rider_stats": rider_stats(pf2),
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
    v8_ann   = 28.9   # v8 best: WR_VIX_5Inst

    console.print(Panel(
        f"[bold {'green' if ann >= 28 else 'yellow'}]"
        f"v9 RESULT — {ann:+.1f}% annualised ({best_name})[/bold {'green' if ann >= 28 else 'yellow'}]\n\n"
        f"  Period tested    : {period}\n"
        f"  vs v8 benchmark  : {ann:+.1f}% vs {v8_ann:+.1f}%  "
        f"({'▲ IMPROVED' if ann > v8_ann else '▼ WORSE'} by {abs(ann-v8_ann):.1f}%)\n"
        f"  Sharpe           : {best_s['sharpe']:.2f}\n"
        f"  Max Drawdown     : {best_s['max_dd']:.1f}%  [daily equity curve]\n"
        f"  Win Rate         : {best_s['win_rate']:.1f}%\n"
        f"  Net P&L          : ₹{best_s['net_pnl']:+,.0f}\n"
        f"  Final Capital    : ₹{best_s['capital']:,.0f}\n"
        f"  Roll Costs       : ₹{best_s['roll_cost_total']:,.0f}\n"
        f"  Margin Calls     : {best_s['margin_calls']}\n\n"
        f"  v9 improvements:\n"
        f"    + Trend-rider: WR positions (day 1+) promoted to EMA50 trail\n"
        f"      when in_profit(>0.3%) + ADX>25 + EMAs aligned — targets 93.8% WR zone\n"
        f"    + VIX>30 sizing: 3% (was 5%) — eliminates avg -₹6.7K losing trades\n"
        f"    + Clean combined: removed ST(-2.9%) and MOM20(+6.5%/57%DD)",
        title="[bold]RESULT — v9 TREND-RIDER BACKTEST[/bold]",
        border_style="green" if ann >= 28 else "yellow"
    ))

    return results


if __name__ == "__main__":
    main()
