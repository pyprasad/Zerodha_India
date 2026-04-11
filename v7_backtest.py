"""
v7_backtest.py  — Full-History Honest Backtest (4 structural fixes applied)
=============================================================================
Built on top of v6 (which fixed look-ahead bias, STT, daily MTM, SPAN margin).
This version adds 4 additional correctness fixes identified in the v6 audit:

  FIX A — Date UNION instead of intersection
    v6 used the date intersection of all 4 instruments. Because MIDCPNIFTY data
    file had 1,241 phantom bars from 2017-2022, the effective intersection was
    only ~3 years (2023-2026 bull market). This version uses the DATE UNION so
    NIFTY/BANKNIFTY trade from their full 18-year history (2007-2026), stress-
    tested through 2008 crisis, 2013 taper tantrum, 2020 COVID crash, 2022 bear
    market. Each instrument is traded only when it has data AND has passed its
    F&O live date.

  FIX B — Max Drawdown from daily equity curve (not trade-close snapshots)
    v6 computed max drawdown from df.capital — a snapshot recorded only at trade
    close. Between two trade closes, daily MTM losses were invisible, causing the
    reported drawdown to be understated. This version uses self.daily_equity
    (recorded every bar) to compute the true high-water-mark drawdown.

  FIX C — Monthly rotation EXIT queued to next-day open
    v6 made the rotation decision at today's close AND executed the exit at
    today's close — you can't trade at a price you've just observed. This version
    queues rotation exits in pending_exits (same as how entries are queued) and
    executes them at the next bar's open, consistent with the entry logic.

  FIX D — Futures roll cost (~0.20 % per monthly roll)
    Futures contracts expire monthly. Rolling from expiring to next contract costs
    ~0.15-0.25% of notional. v6 used spot index prices, ignoring this drag. We
    deduct ROLL_COST_PCT on the last trading day of each month for every open
    position, approximating the real cost paid by any futures trader.

Data: Yahoo Finance — maximum available history fetched fresh:
  NIFTY:      2007-09-17 → 2026-04-10  (18.5 years)
  BANKNIFTY:  2007-09-17 → 2026-04-10  (18.5 years)
  FINNIFTY:   2012-02-27 → 2026-04-10  (14 years; signals only from Jul-2021)
  MIDCPNIFTY: 2007-09-17 → 2026-04-10  (18.5 years; signals only from Oct-2023)
  INDIAVIX:   2008-03-03 → 2026-04-10  (18+ years)

Capital: ₹10 lakh | Risk: 4%/trade | Max lots: NIFTY/BN=20, MID/FIN=15
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
CAPITAL        = 1_000_000       # ₹10 lakh starting capital
BROKERAGE      = 20              # ₹20 flat per order (Zerodha)
EXCHANGE_FEE   = 0.0000125       # 0.00125% NSE exchange transaction charge
SLIPPAGE       = 0.0003          # 0.03% per side bid-ask slippage
STT_RATE       = 0.0001          # 0.01% STT on SELL side of futures
MIN_STOP_PCT   = 0.004           # 0.4% minimum stop distance

# FIX D: Monthly futures roll cost (~0.20% of notional per roll)
# Based on NSE index futures average basis at roll: 0.15-0.25% is typical.
# Deducted on the last trading day of each calendar month.
ROLL_COST_PCT  = 0.0020          # 0.20% of open notional per monthly roll

LOT_SIZES = {
    "NIFTY":      65,
    "BANKNIFTY":  30,
    "MIDCPNIFTY": 120,
    "FINNIFTY":   60,
    "NIFTYIT":    30,
}

# Actual NSE F&O launch / data availability dates
# NIFTY/BANKNIFTY: full history from 2007 — no restriction
# FINNIFTY futures: listed 2021-07-28 (before that: index data for indicators only)
# MIDCPNIFTY futures: listed 2023-10-03 (before that: index data for indicators only)
INSTRUMENT_LIVE_DATE = {
    "NIFTY":      pd.Timestamp("2008-01-01"),   # Allow ~3m warmup after data start
    "BANKNIFTY":  pd.Timestamp("2008-01-01"),
    "FINNIFTY":   pd.Timestamp("2021-07-28"),   # F&O launch date
    "MIDCPNIFTY": pd.Timestamp("2023-10-03"),   # F&O launch date
    "NIFTYIT":    pd.Timestamp("2008-01-01"),
}

MAX_LOTS_PER_INST = {
    "NIFTY":      20,
    "BANKNIFTY":  20,
    "MIDCPNIFTY": 15,
    "FINNIFTY":   15,
}

RISK_PER_TRADE = 0.04       # 4% of portfolio per trade
MAX_CONCURRENT = 4          # Max 4 open positions (one per instrument)
MAX_TOTAL_RISK = 0.08       # Total portfolio risk cap: 8%
WARMUP         = 260        # Bars to skip for indicator warm-up

SPAN_MARGIN_PCT    = 0.10   # 10% of notional locked as initial margin
MAINT_MARGIN_RATIO = 0.75   # maintenance margin = 75% of SPAN (NSE rule)

INSTRUMENTS_4 = ["NIFTY", "BANKNIFTY", "MIDCPNIFTY", "FINNIFTY"]


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────────────────────────────────────────
def load_data(instrument):
    path = Path(f"data/historical/{instrument}_daily_extended.csv")
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}. Run fetch_all_instruments.py first.")
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    df.columns = [c.lower() for c in df.columns]
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df["volume"] = df["volume"].fillna(0.0)
    df = df.dropna(subset=["open", "high", "low", "close"])

    # Drop weekends
    df = df[df.index.dayofweek < 5]

    # Drop phantom no-move bars (Yahoo Finance artefacts on inactive instruments)
    no_move = ((df["open"] == df["close"]) &
               (df["high"] == df["close"]) &
               (df["low"]  == df["close"]))
    df = df[~no_move]

    # Normalise index to midnight timestamps (timezone-naive)
    df.index = df.index.normalize()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────────────────────────────────────
def ema(s, p):
    return s.ewm(span=p, adjust=False).mean()

def atr_series(df, p=14):
    h, l, c = df.high, df.low, df.close
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
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
    plus_dm  = (h - h.shift()).clip(lower=0)
    minus_dm = (l.shift() - l).clip(lower=0)
    plus_dm  = plus_dm.where(plus_dm  >= minus_dm, 0.0)
    minus_dm = minus_dm.where(minus_dm > plus_dm,   0.0)
    atr14    = atr_series(df, p)
    plus_di  = 100 * (plus_dm.ewm(alpha=1/p, adjust=False).mean()  / atr14)
    minus_di = 100 * (minus_dm.ewm(alpha=1/p, adjust=False).mean() / atr14)
    dx  = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx = dx.ewm(alpha=1/p, adjust=False).mean()
    return adx, plus_di, minus_di

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
    d["high_52wk"] = df.close.rolling(252).max().shift(1)   # shift(1) prevents lookahead
    d["roc1"]      = df.close.pct_change(21)  * 100
    d["roc3"]      = df.close.pct_change(63)  * 100
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
        self.roll_cost_total = 0.0   # FIX D: track total roll costs paid

    def _comm(self, qty, entry_price, exit_price, direction):
        notional_entry = qty * entry_price
        notional_exit  = qty * exit_price
        brokerage      = BROKERAGE * 2
        exchange       = (notional_entry + notional_exit) * EXCHANGE_FEE
        slippage       = (notional_entry + notional_exit) * SLIPPAGE
        stt_notional   = notional_exit if direction == 1 else notional_entry
        stt            = stt_notional * STT_RATE
        return brokerage + exchange + slippage + stt

    def _qty(self, instrument, entry, stop, risk_pct=RISK_PER_TRADE):
        avail_cash   = self.C - sum(self.margin_reserved.values())
        current_risk = sum(
            abs(p["entry"] - p["stop"]) * p["qty"] / self.C
            for p in self.positions.values()
        )
        avail_risk = min(risk_pct, MAX_TOTAL_RISK - current_risk)
        if avail_risk <= 0:
            return 0

        risk_amt  = avail_cash * avail_risk
        lot       = LOT_SIZES.get(instrument, 65)
        risk_unit = max(abs(entry - stop), entry * MIN_STOP_PCT)
        lots      = int(risk_amt / (risk_unit * lot))

        if lots <= 0:
            return 0

        max_lots = MAX_LOTS_PER_INST.get(instrument, 20)
        lots = min(lots, max_lots)
        return lots * lot

    def enter(self, instrument, strategy, date, direction, entry, stop, tag="", risk_pct=RISK_PER_TRADE):
        if instrument in self.positions:             return False
        if len(self.positions) >= MAX_CONCURRENT:    return False
        qty = self._qty(instrument, entry, stop, risk_pct)
        if qty == 0:                                  return False

        span_margin = qty * entry * SPAN_MARGIN_PCT
        avail_cash  = self.C - sum(self.margin_reserved.values())
        if avail_cash < span_margin:
            return False

        tgt = entry + direction * abs(entry - stop) * 2.0
        entry_cost = BROKERAGE + qty * entry * (EXCHANGE_FEE + SLIPPAGE)
        if direction == -1:
            entry_cost += qty * entry * STT_RATE
        self.C -= entry_cost
        self.C -= span_margin
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
        if p["d"] == 1: p["stop"] = max(p["stop"], new_stop)
        else:            p["stop"] = min(p["stop"], new_stop)

    def check_exits(self, date, instrument, high, low, close):
        if instrument not in self.positions: return False
        pos = self.positions[instrument]
        d, stop, tgt = pos["d"], pos["stop"], pos["target"]
        xp = reason = None
        if d == 1:
            if low  <= stop: xp, reason = stop, "stop"
            elif high >= tgt: xp, reason = tgt,  "target"
        else:
            if high >= stop: xp, reason = stop, "stop"
            elif low  <= tgt: xp, reason = tgt,  "target"
        if xp is None: return False
        return self._close(instrument, date, xp, reason)

    def force_close(self, instrument, date, price, reason="force"):
        if instrument not in self.positions: return False
        return self._close(instrument, date, price, reason)

    def _close(self, instrument, date, exit_price, reason):
        pos  = self.positions.pop(instrument)
        d, entry, qty = pos["d"], pos["entry"], pos["qty"]

        last_price = pos.get("last_mtm_price", entry)
        final_mtm  = d * (exit_price - last_price) * qty
        self.C += final_mtm

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
        t0, t1 = pd.to_datetime(T[0]["entry_date"]), pd.to_datetime(T[-1]["exit_date"])
        yrs  = max((t1 - t0).days / 365.25, 0.1)
        ann  = ((1 + tot / 100) ** (1 / yrs) - 1) * 100
        aw   = df.loc[df.pnl > 0, "pnl"].mean() if (df.pnl > 0).any() else 0
        al   = df.loc[df.pnl < 0, "pnl"].mean() if (df.pnl < 0).any() else 0

        # Sharpe from daily equity curve
        if len(self.daily_equity) >= 20:
            eq_vals  = pd.Series([v for _, v in self.daily_equity])
            daily_r  = eq_vals.pct_change().dropna()
            shr      = (daily_r.mean() / daily_r.std() * np.sqrt(252)
                        if daily_r.std() > 0 else 0)
        else:
            pps = df.pnl_pct.values
            shr = pps.mean() / pps.std() * np.sqrt(252) if pps.std() > 0 else 0

        # FIX B: Max drawdown from DAILY EQUITY CURVE (not trade-close snapshots)
        # This captures MTM losses between trade closes — the true worst-case pain.
        if len(self.daily_equity) >= 2:
            eq_vals  = pd.Series([v for _, v in self.daily_equity])
            peak     = eq_vals.cummax()
            dd       = ((peak - eq_vals) / peak).max() * 100
        else:
            caps = pd.Series([self.C0] + list(df.capital))
            dd   = ((caps.cummax() - caps) / caps.cummax()).max() * 100

        return dict(label=label, total_ret=round(tot, 2), ann_ret=round(ann, 2),
                    sharpe=round(shr, 3), max_dd=round(dd, 2), win_rate=round(wr, 1),
                    num_trades=len(T), net_pnl=round(self.C - self.C0),
                    capital=round(self.C), avg_win=round(aw), avg_loss=round(al),
                    trades_per_yr=round(len(T) / yrs, 1),
                    margin_calls=self.margin_calls,
                    roll_cost_total=round(self.roll_cost_total))


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL FUNCTIONS  (read bar i data only — no lookahead)
# ─────────────────────────────────────────────────────────────────────────────
def signal_williams_r(i, ind):
    wr   = ind["wR14"].iloc[i];  wr_p = ind["wR14"].iloc[i-1]
    e200 = ind["e200"].iloc[i];  e50  = ind["e50"].iloc[i]
    atr_ = ind["atr"].iloc[i];   p    = ind["close"].iloc[i]
    if np.isnan(e200): return None
    if wr_p < -80 and wr >= -80 and p > e200 * 0.97 and e50 > e200 * 0.98:
        return ("L", atr_ * 1.5, "WR_OB")
    if wr_p > -20 and wr <= -20 and p < e200 * 1.03 and e50 < e200 * 1.02:
        return ("S", atr_ * 1.5, "WR_OS")
    return None


def signal_regime_adaptive(i, ind):
    adx  = ind["adx14"].iloc[i]
    e8   = ind["e8"].iloc[i];   e8p  = ind["e8"].iloc[i-1]
    e21  = ind["e21"].iloc[i];  e21p = ind["e21"].iloc[i-1]
    e50  = ind["e50"].iloc[i];  e200 = ind["e200"].iloc[i]
    atr_ = ind["atr"].iloc[i];  p    = ind["close"].iloc[i]
    wr   = ind["wR14"].iloc[i]; wr_p = ind["wR14"].iloc[i-1]
    if np.isnan(adx) or np.isnan(e200): return None
    if adx > 25:
        if e8p < e21p and e8 >= e21 and p > e200 * 0.97:
            return ("L", atr_ * 2.0, "REGIME_TREND_L")
        if e8p > e21p and e8 <= e21 and p < e200 * 1.03:
            return ("S", atr_ * 2.0, "REGIME_TREND_S")
    elif adx < 20:
        if wr_p < -80 and wr >= -80 and p > e200 * 0.97 and e50 > e200 * 0.98:
            return ("L", atr_ * 1.5, "REGIME_MR_L")
        if wr_p > -20 and wr <= -20 and p < e200 * 1.03 and e50 < e200 * 1.02:
            return ("S", atr_ * 1.5, "REGIME_MR_S")
    else:
        ema_bull = e8 > e21 and e21 > e200 * 0.98
        ema_bear = e8 < e21 and e21 < e200 * 1.02
        if wr_p < -80 and wr >= -80 and p > e200 * 0.97 and ema_bull:
            return ("L", atr_ * 1.5, "REGIME_MIXED_L")
        if wr_p > -20 and wr <= -20 and p < e200 * 1.03 and ema_bear:
            return ("S", atr_ * 1.5, "REGIME_MIXED_S")
    return None


def signal_vix_spike(i, ind, vix_chg_today):
    if vix_chg_today is None or np.isnan(float(vix_chg_today)): return None
    chg  = float(vix_chg_today)
    atr_ = ind["atr"].iloc[i];  e200 = ind["e200"].iloc[i]
    if np.isnan(e200): return None
    if chg >  0.15: return ("L", atr_ * 1.5, "VIX_SPIKE")
    if chg < -0.15: return ("S", atr_ * 1.5, "VIX_DROP")
    return None


def signal_momentum_52wk(i, ind):
    h52  = ind["high_52wk"].iloc[i];  p    = ind["close"].iloc[i]
    e200 = ind["e200"].iloc[i];       vol  = ind["volume"].iloc[i]
    vsma = ind["vol_sma20"].iloc[i];  atr_ = ind["atr"].iloc[i]
    if np.isnan(h52) or np.isnan(e200) or h52 <= 0: return None
    vol_ok = (vsma == 0) or np.isnan(vsma) or (vol > vsma * 1.5)
    if p > h52 and vol_ok and p > e200:
        return ("L", atr_ * 2.0, "MOM_52WK")
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

    if tag.startswith(("WR_", "REGIME_MR", "REGIME_MIXED")):
        wr14  = ind["wR14"].iloc[ii];  wr14p = ind["wR14"].iloc[ii-1]
        if d == 1 and wr14p < -50 and wr14 >= -50:
            pf.force_close(inst, date, p, "WR_MID_EXIT"); return
        if d == -1 and wr14p > -50 and wr14 <= -50:
            pf.force_close(inst, date, p, "WR_MID_EXIT"); return
        e21  = ind["e21"].iloc[ii];  atr_ = ind["atr"].iloc[ii]
        pf.trail_stop(inst, e21 - d * atr_ * 0.3)

    elif tag.startswith("REGIME_TREND"):
        e8   = ind["e8"].iloc[ii];   e8p  = ind["e8"].iloc[ii-1]
        e21  = ind["e21"].iloc[ii];  e21p = ind["e21"].iloc[ii-1]
        atr_ = ind["atr"].iloc[ii]
        if d == 1 and e8p > e21p and e8 <= e21:
            pf.force_close(inst, date, p, "EMA_CROSS_EXIT"); return
        if d == -1 and e8p < e21p and e8 >= e21:
            pf.force_close(inst, date, p, "EMA_CROSS_EXIT"); return
        pf.trail_stop(inst, e21 - d * atr_ * 0.5)

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

    elif tag == "MOM_52WK":
        if ii >= 10:
            lo10 = ind["low"].iloc[ii-10:ii].min()
            pf.trail_stop(inst, lo10)

    elif tag.startswith("MONTHLY"):
        e200 = ind["e200"].iloc[ii];  e50  = ind["e50"].iloc[ii]
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
def run_v7(instruments, strategy_names, capital=CAPITAL):
    """
    Portfolio simulation with all v6 fixes PLUS the 4 structural fixes (A-D).

    FIX A — DATE UNION:
      Uses the union of all instrument trading dates so NIFTY/BANKNIFTY keep their
      full 18-year history. On each date, instruments without data are skipped.

    FIX B — DRAWDOWN from equity curve (in Portfolio.stats).

    FIX C — MONTHLY ROTATION EXIT at next-day open:
      Rotation exits are stored in pending_exits dict. On the next bar, they
      execute at that bar's open price (same pattern as pending_entries for buys).

    FIX D — ROLL COST deducted on last trading day of each month:
      Every open position pays ROLL_COST_PCT of notional (0.20%) to simulate
      the cost of rolling from expiring to next-month futures contract.
    """
    data = {}; inds = {}
    for inst in instruments:
        df = load_data(inst)
        data[inst] = df
        inds[inst] = precompute(df)

    vix_df = load_data("INDIAVIX")

    # FIX A: Use DATE UNION — each instrument trades when it has data and is live
    union_set  = set.union(*[set(data[inst].index) for inst in instruments])
    all_dates  = pd.DatetimeIndex(sorted(union_set))
    console.print(f"  [dim]Date union: {len(all_dates)} trading days "
                  f"({all_dates[0].date()} → {all_dates[-1].date()})[/dim]")

    # VIX: pct_change on raw series before ffill (prevents ffill masking real moves)
    vix_chg_raw = vix_df.close.pct_change()
    vix_series  = vix_df.close.reindex(all_dates, method="ffill")
    vix_chg     = vix_chg_raw.reindex(all_dates)

    pf = Portfolio(capital)

    use_wr      = "williams_r"       in strategy_names
    use_wr_vix  = "williams_r_vix"   in strategy_names
    use_regime  = "regime_adaptive"  in strategy_names
    use_vix_spk = "vix_spike"        in strategy_names
    use_52wk    = "momentum_52wk"    in strategy_names
    use_monthly = "monthly_rotation" in strategy_names

    monthly_allowed = set(instruments)
    last_rot_month  = (-1, -1)

    # Pending entries: {inst: (strategy, direction, stop_dist, tag, risk_pct)}
    # Signals fire at bar-i close → execute at bar i+1 open
    pending_entries = {}

    # FIX C: Pending exits: {inst: reason}
    # Rotation exit decision at bar-i close → execute at bar i+1 open
    pending_exits = {}

    # FIX D: track last month we charged roll cost
    last_roll_month = (-1, -1)

    for idx, date in enumerate(all_dates):
        if idx < WARMUP: continue

        vix_today     = vix_series.iloc[idx] if idx < len(vix_series) else None
        vix_chg_today = vix_chg.iloc[idx]    if idx < len(vix_chg)    else None
        if vix_today is not None and (pd.isna(vix_today) or np.isnan(float(vix_today))):
            vix_today = None
        if vix_chg_today is not None and (pd.isna(vix_chg_today) or np.isnan(float(vix_chg_today))):
            vix_chg_today = None

        # ── FIX C: Execute pending rotation EXITS at today's open ────────────
        for inst, reason in list(pending_exits.items()):
            del pending_exits[inst]
            if inst not in pf.positions: continue
            if inst not in data or date not in data[inst].index: continue
            ii    = data[inst].index.get_loc(date)
            price = float(inds[inst]["open"].iloc[ii])
            pf.force_close(inst, date, price, reason)

        # ── Execute pending ENTRIES at today's open ───────────────────────────
        for inst, (strat, d_val, stop_dist, tag, risk) in list(pending_entries.items()):
            del pending_entries[inst]
            if inst in pf.positions: continue
            if inst not in data or date not in data[inst].index: continue
            ii    = data[inst].index.get_loc(date)
            entry = float(inds[inst]["open"].iloc[ii])
            stop  = entry - d_val * stop_dist
            pf.enter(inst, strat, date, d_val, entry, stop, tag, risk)

        # ── EXITS (strategy-based, checked at bar close) ──────────────────────
        for inst in list(pf.positions.keys()):
            if inst not in data or date not in data[inst].index: continue
            ii = data[inst].index.get_loc(date)
            if ii < 1: continue
            process_exit(pf, inst, date, inds[inst], ii, vix_series)

        # ── MONTHLY ROTATION ──────────────────────────────────────────────────
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
                                         reverse=True)[:2])
                               if scores else set(instruments))

                # FIX C: queue rotation exits for next-day open (not today's close)
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

        # ── DAILY MTM SETTLEMENT ──────────────────────────────────────────────
        for inst, pos in list(pf.positions.items()):
            if inst not in data or date not in data[inst].index: continue
            ii          = data[inst].index.get_loc(date)
            today_close = float(inds[inst]["close"].iloc[ii])
            last_price  = pos.get("last_mtm_price", pos["entry"])
            pf.C       += pos["d"] * (today_close - last_price) * pos["qty"]
            pos["last_mtm_price"] = today_close

        # ── MARGIN CALL CHECK ─────────────────────────────────────────────────
        if pf.margin_reserved:
            total_maint = sum(pf.margin_reserved.values()) * MAINT_MARGIN_RATIO
            if pf.C < total_maint:
                pf.margin_calls += 1
                for inst in list(pf.positions.keys()):
                    if inst in data and date in data[inst].index:
                        ii = data[inst].index.get_loc(date)
                        cp = float(inds[inst]["close"].iloc[ii])
                        pf.force_close(inst, date, cp, "margin_call")

        # ── FIX D: MONTHLY ROLL COST ──────────────────────────────────────────
        # Deduct on the last trading day of each month for all open positions.
        # Approximates cost of rolling expiring futures contract to next month.
        cur_month = (date.year, date.month)
        if cur_month != last_roll_month and pf.positions:
            # Check if today is the last bar of the current month in all_dates
            if idx + 1 >= len(all_dates) or all_dates[idx + 1].month != date.month:
                last_roll_month = cur_month
                for inst, pos in list(pf.positions.items()):
                    if inst not in data or date not in data[inst].index: continue
                    ii          = data[inst].index.get_loc(date)
                    close_price = float(inds[inst]["close"].iloc[ii])
                    roll_cost   = pos["qty"] * close_price * ROLL_COST_PCT
                    pf.C -= roll_cost
                    pf.roll_cost_total += roll_cost

        # ── ENTRY SIGNALS → pending (no same-bar execution) ───────────────────
        for inst in instruments:
            if inst in pf.positions or inst in pending_entries: continue

            if date < INSTRUMENT_LIVE_DATE.get(inst, pd.Timestamp("2008-01-01")): continue

            if inst not in data or date not in data[inst].index: continue
            ii = data[inst].index.get_loc(date)
            if ii < WARMUP: continue

            if idx + 1 >= len(all_dates): continue

            ind = inds[inst]
            sig = None

            if use_vix_spk and inst == "NIFTY":
                sig = signal_vix_spike(ii, ind, vix_chg_today)

            if sig is None and use_wr_vix:
                s = signal_williams_r(ii, ind)
                if s is not None:
                    risk = vix_dynamic_risk(vix_today)
                    pending_entries[inst] = (f"WR_VIX_{s[2]}", 1 if s[0]=="L" else -1,
                                             s[1], s[2], risk)
                    continue

            if sig is None and use_wr:
                sig = signal_williams_r(ii, ind)

            if sig is None and use_regime:
                sig = signal_regime_adaptive(ii, ind)

            if sig is None and use_52wk:
                sig = signal_momentum_52wk(ii, ind)

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
        wr   = (grp.pnl > 0).sum() / len(grp) * 100
        aw   = grp.loc[grp.pnl > 0, "pnl"].mean() if (grp.pnl > 0).any() else 0
        al   = grp.loc[grp.pnl < 0, "pnl"].mean() if (grp.pnl < 0).any() else 0
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
    """Year-by-year returns computed from the DAILY EQUITY CURVE (not trade closes)."""
    if not pf.daily_equity:
        return []
    eq_df = pd.DataFrame(pf.daily_equity, columns=["date", "equity"])
    eq_df["date"] = pd.to_datetime(eq_df["date"])
    eq_df = eq_df.set_index("date")

    # Get year-start and year-end equity values
    rows = []
    for yr in sorted(eq_df.index.year.unique()):
        yr_data = eq_df[eq_df.index.year == yr]
        if yr_data.empty: continue
        start_val = eq_df[eq_df.index.year  < yr]["equity"].iloc[-1] if (eq_df.index.year < yr).any() else pf.C0
        end_val   = yr_data["equity"].iloc[-1]
        pnl       = end_val - start_val
        ret       = pnl / start_val * 100
        # Trade count for the year
        yr_trades = [t for t in pf.trades
                     if pd.to_datetime(t["exit_date"]).year == yr]
        wr = (sum(1 for t in yr_trades if t["pnl"] > 0) / len(yr_trades) * 100
              if yr_trades else 0.0)
        rows.append(dict(year=yr, trades=len(yr_trades), win_rate=round(wr, 1),
                         pnl=round(pnl), ret_pct=round(ret, 2)))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# NAMED CONFIGS
# ─────────────────────────────────────────────────────────────────────────────
CONFIGS = {
    "WR_4Inst":         (INSTRUMENTS_4, ["williams_r"]),
    "WR_VIX_Sizing":    (INSTRUMENTS_4, ["williams_r_vix"]),
    "Regime_Adaptive":  (INSTRUMENTS_4, ["regime_adaptive"]),
    "VIX_Spike+WR":     (INSTRUMENTS_4, ["vix_spike", "williams_r"]),
    "Momentum_52Wk":    (INSTRUMENTS_4, ["momentum_52wk"]),
    "Monthly_Rotation": (INSTRUMENTS_4, ["monthly_rotation"]),
    "COMBINED_ALL":     (INSTRUMENTS_4, ["vix_spike", "williams_r_vix",
                                          "regime_adaptive", "momentum_52wk",
                                          "monthly_rotation"]),
}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    console.rule("[bold cyan]v7 PORTFOLIO BACKTEST — FULL HISTORY + 4 STRUCTURAL FIXES[/bold cyan]")
    console.print(
        "[dim]Instruments: NIFTY · BANKNIFTY · FINNIFTY (signals from Jul-2021) "
        "· MIDCPNIFTY (signals from Oct-2023)\n"
        "Data history: NIFTY/BANKNIFTY/MIDCPNIFTY from Sep-2007 · FINNIFTY from Feb-2012 "
        "· INDIAVIX from Mar-2008\n"
        "Fix A: Date UNION (full 18yr history for NIFTY/BN)\n"
        "Fix B: Max drawdown from daily equity curve (not trade snapshots)\n"
        "Fix C: Monthly rotation exits at next-day open\n"
        "Fix D: 0.20%/month futures roll cost\n"
        f"Capital: ₹{CAPITAL:,} | Risk: 4%/trade | Max lots: NIFTY/BN=20, MID/FIN=15[/dim]\n"
    )

    results    = {}
    portfolios = {}

    for cfg_name, (insts, strats) in CONFIGS.items():
        console.print(f"  [cyan]Running {cfg_name:22s}[/cyan]", end=" ")
        try:
            pf = run_v7(insts, strats, CAPITAL)
            s  = pf.stats(cfg_name)
            results[cfg_name]    = s
            portfolios[cfg_name] = pf
            color = "green" if s["ann_ret"] >= 20 else ("yellow" if s["ann_ret"] >= 10 else "red")
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

    # ── Ranked comparison table ──────────────────────────────────────────────
    console.print()
    console.rule("[bold cyan]Config Comparison — Ranked by Annualised Return[/bold cyan]")

    ranked = sorted(
        [(k, v) for k, v in results.items() if v is not None],
        key=lambda x: x[1]["ann_ret"], reverse=True
    )

    tbl = Table(box=box.DOUBLE_EDGE, show_header=True, header_style="bold cyan", min_width=140)
    tbl.add_column("Rank",     justify="right", min_width=5)
    tbl.add_column("Config",   min_width=22)
    tbl.add_column("Ann Ret%", justify="right", min_width=9)
    tbl.add_column("Total%",   justify="right", min_width=9)
    tbl.add_column("Sharpe",   justify="right", min_width=8)
    tbl.add_column("MaxDD%",   justify="right", min_width=8)
    tbl.add_column("Win%",     justify="right", min_width=7)
    tbl.add_column("Trades",   justify="right", min_width=7)
    tbl.add_column("Net P&L",  justify="right", min_width=14)
    tbl.add_column("Roll Cost",justify="right", min_width=12)
    tbl.add_column("Final Cap",justify="right", min_width=14)

    for rank, (name, s) in enumerate(ranked, 1):
        beats  = s["ann_ret"] >= 20
        color  = "bold green" if beats else ("yellow" if s["ann_ret"] >= 10 else "red")
        marker = "★" if name == "COMBINED_ALL" else ("✓" if beats else " ")
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

    if not ranked:
        console.print("[red]All configs failed.[/red]")
        return {}

    # ── Best config detail ───────────────────────────────────────────────────
    best_name, best_s = ranked[0]
    best_pf = portfolios[best_name]

    console.print(f"\n[bold cyan]Best Config: {best_name}[/bold cyan]\n")

    ybl = annual_breakdown(best_pf)
    ytbl = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold",
                 title=f"Year-by-Year — {best_name}")
    ytbl.add_column("Year"); ytbl.add_column("Trades", justify="right")
    ytbl.add_column("Win%",    justify="right")
    ytbl.add_column("P&L",     justify="right")
    ytbl.add_column("Return%", justify="right")
    for row in ybl:
        c = "green" if row["ret_pct"] >= 20 else ("yellow" if row["ret_pct"] >= 5 else "red")
        ytbl.add_row(str(row["year"]), str(row["trades"]),
                     f"{row['win_rate']:.1f}%", f"₹{row['pnl']:+,.0f}",
                     f"[{c}]{row['ret_pct']:+.2f}%[/{c}]")
    console.print(ytbl)

    attr = attribution_stats(best_pf)
    if attr:
        atbl = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold",
                     title="Strategy Attribution")
        atbl.add_column("Strategy/Signal", min_width=22)
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

    inst_attr = instrument_stats(best_pf)
    if inst_attr:
        itbl = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold",
                     title="Instrument Attribution")
        itbl.add_column("Instrument", min_width=14)
        itbl.add_column("Trades",     justify="right")
        itbl.add_column("Win%",       justify="right")
        itbl.add_column("Net P&L",    justify="right")
        itbl.add_column("Notes",      min_width=35)
        notes = {
            "NIFTY":      "Full history from Jan-2008",
            "BANKNIFTY":  "Full history from Jan-2008",
            "FINNIFTY":   "Signals from Jul-2021 (F&O launch)",
            "MIDCPNIFTY": "Signals from Oct-2023 (F&O launch)",
        }
        for a in inst_attr:
            c = "green" if a["net_pnl"] > 0 else "red"
            itbl.add_row(a["instrument"], str(a["trades"]),
                         f"{a['win_rate']:.1f}%",
                         f"[{c}]₹{a['net_pnl']:+,.0f}[/{c}]",
                         notes.get(a["instrument"], ""))
        console.print(itbl)

    # ── Save JSON ────────────────────────────────────────────────────────────
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"reports/v7_backtest_{ts}.json"
    all_data = {}
    for name, s in results.items():
        if s is None: continue
        pf2 = portfolios[name]
        all_data[name] = {
            "stats":            s,
            "trades":           pf2.trades,
            "annual":           annual_breakdown(pf2),
            "attribution":      attribution_stats(pf2),
            "instrument_stats": instrument_stats(pf2),
            "equity_curve":     [(str(d), v) for d, v in pf2.daily_equity],
        }
    with open(fname, "w") as f:
        json.dump(all_data, f, indent=2, default=str)
    console.print(f"\n[dim]Results saved → {fname}[/dim]")

    ann = best_s["ann_ret"]
    eq_dates   = [d for d, _ in best_pf.daily_equity]
    period_str = (f"{eq_dates[0].date()} → {eq_dates[-1].date()}"
                  f"  (~{(eq_dates[-1]-eq_dates[0]).days/365.25:.1f} years)"
                  if eq_dates else "n/a")
    console.print(Panel(
        f"[bold {'green' if ann >= 15 else 'yellow'}]"
        f"HONEST BACKTEST RESULT — {ann:+.1f}% annualised ({best_name})[/bold {'green' if ann >= 15 else 'yellow'}]\n\n"
        f"  Period tested    : {period_str}\n"
        f"  Sharpe           : {best_s['sharpe']:.2f}\n"
        f"  Max Drawdown     : {best_s['max_dd']:.1f}%  [daily equity curve — real figure]\n"
        f"  Win Rate         : {best_s['win_rate']:.1f}%\n"
        f"  Net P&L          : ₹{best_s['net_pnl']:+,.0f}\n"
        f"  Final Capital    : ₹{best_s['capital']:,.0f}\n"
        f"  Roll Costs Paid  : ₹{best_s['roll_cost_total']:,.0f}\n"
        f"  Margin Calls     : {best_s['margin_calls']}\n\n"
        f"  Structural fixes applied:\n"
        f"    ✓ Fix A: Date UNION — full 18yr history (2008-2026) incl. 2008/2020/2022 crashes\n"
        f"    ✓ Fix B: Drawdown from daily equity curve (no longer understated)\n"
        f"    ✓ Fix C: Monthly rotation exits at next-day open (not same-bar close)\n"
        f"    ✓ Fix D: 0.20%/month futures roll cost deducted",
        title="[bold]RESULT — v7 CORRECTED BACKTEST[/bold]",
        border_style="green" if ann >= 15 else "yellow"
    ))

    return results


if __name__ == "__main__":
    main()
