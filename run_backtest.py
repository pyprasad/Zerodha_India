"""
Standalone Backtest Runner — No Zerodha credentials required.
Uses yfinance to fetch 3 years of daily OHLCV data for NIFTY and BANKNIFTY.
Runs all 4 strategies and generates a comprehensive P&L report.

Run: python3 run_backtest.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings("ignore")

import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import backtrader as bt
import numpy as np
import pandas as pd
import yfinance as yf
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

CAPITAL = 1_000_000          # ₹10 lakh starting capital
BROKERAGE_PER_ORDER = 20     # ₹20 flat Zerodha
SLIPPAGE_PCT = 0.0005        # 0.05%
EXCHANGE_FEE_PCT = 0.0000125

# ─────────────────────────────────────────────────────────────────────────────
# 1. FETCH HISTORICAL DATA
# ─────────────────────────────────────────────────────────────────────────────
def fetch_yfinance(symbol: str, name: str) -> pd.DataFrame:
    console.print(f"  [cyan]Downloading {name} ({symbol}) from Yahoo Finance...[/cyan]")
    end = datetime.today()
    start = end - timedelta(days=1200)   # ~3.3 years
    raw = yf.download(symbol, start=start.strftime("%Y-%m-%d"),
                      end=end.strftime("%Y-%m-%d"), auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError(f"No data returned for {symbol}")

    # Flatten MultiIndex columns if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df = df.dropna()
    console.print(f"  [green]✓ {name}: {len(df)} daily candles "
                  f"({df.index[0].date()} → {df.index[-1].date()})[/green]")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. BACKTRADER DATA FEED
# ─────────────────────────────────────────────────────────────────────────────
class PandasFeed(bt.feeds.PandasData):
    params = (
        ("datetime", None),
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", "volume"),
        ("openinterest", -1),
    )


def df_to_feed(df: pd.DataFrame) -> bt.feeds.PandasData:
    return PandasFeed(dataname=df)


# ─────────────────────────────────────────────────────────────────────────────
# 3. COMMISSION MODEL
# ─────────────────────────────────────────────────────────────────────────────
class ZerodhaCommission(bt.CommInfoBase):
    params = (
        ("commission", BROKERAGE_PER_ORDER),
        ("exchange_fee_pct", EXCHANGE_FEE_PCT),
        ("stocklike", False),
        ("commtype", bt.CommInfoBase.COMM_FIXED),
    )
    def getcommission(self, size, price):
        return self.p.commission + abs(size * price * self.p.exchange_fee_pct)


# ─────────────────────────────────────────────────────────────────────────────
# 4. STRATEGIES
# ─────────────────────────────────────────────────────────────────────────────

class _Base(bt.Strategy):
    params = (
        ("instrument", "NIFTY"),
        ("stoploss_pct", 0.02),
        ("target_pct", 0.04),
        ("lot_size", 1),
        ("capital", CAPITAL),
        ("risk_pct", 0.01),
    )

    def __init__(self):
        self.order = None
        self.stop_price = None
        self.target_price = None
        self.trade_records = []

    def _calc_qty(self, entry, stop):
        risk_amt = self.p.capital * self.p.risk_pct
        risk_unit = abs(entry - stop)
        if risk_unit < 0.01:
            return self.p.lot_size
        return max(self.p.lot_size, int(risk_amt / risk_unit / self.p.lot_size) * self.p.lot_size)

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.trade_records.append({
            "date": bt.num2date(trade.dtopen).strftime("%Y-%m-%d"),
            "exit_date": bt.num2date(trade.dtclose).strftime("%Y-%m-%d"),
            "entry": round(trade.price, 2),
            "exit": round(trade.price + trade.pnl / max(abs(trade.size), 1), 2),
            "size": trade.size,
            "pnl": round(trade.pnl, 2),
            "pnlcomm": round(trade.pnlcomm, 2),
            "won": trade.pnl > 0,
        })

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.entry_price = order.executed.price
        if order.status in [order.Completed, order.Canceled, order.Rejected, order.Margin]:
            self.order = None

    def _enter_long(self, stop_offset=None):
        price = self.datas[0].close[0]
        stop = price * (1 - self.p.stoploss_pct) if stop_offset is None else stop_offset
        target = price * (1 + self.p.target_pct)
        qty = self._calc_qty(price, stop)
        self.stop_price = stop
        self.target_price = target
        self.order = self.buy(size=qty)

    def _enter_short(self, stop_offset=None):
        price = self.datas[0].close[0]
        stop = price * (1 + self.p.stoploss_pct) if stop_offset is None else stop_offset
        target = price * (1 - self.p.target_pct)
        qty = self._calc_qty(price, stop)
        self.stop_price = stop
        self.target_price = target
        self.order = self.sell(size=qty)

    def _check_exit(self):
        price = self.datas[0].close[0]
        if self.position.size > 0:
            if price <= self.stop_price or price >= self.target_price:
                self.order = self.close()
        elif self.position.size < 0:
            if price >= self.stop_price or price <= self.target_price:
                self.order = self.close()


# ── Strategy 1: Opening Range Breakout (adapted to daily: prev day high/low) ──
class ORBStrategy(_Base):
    """
    Daily ORB equivalent: previous day's high/low as the 'range'.
    Long if today opens and breaks above yesterday's high.
    Short if breaks below yesterday's low.
    Volume confirmation: today's volume > 1.5× 20-day average.
    """
    params = (
        ("instrument", "NIFTY"),
        ("rr_ratio", 2.0),
        ("vol_mult", 1.5),
        ("stoploss_pct", 0.015),
        ("target_pct", 0.03),
        ("lot_size", 1),
        ("capital", CAPITAL),
        ("risk_pct", 0.01),
    )

    def __init__(self):
        super().__init__()
        self.vol_ma = bt.indicators.SMA(self.datas[0].volume, period=20)

    def next(self):
        if self.order:
            return
        if len(self.datas[0]) < 2:
            return

        prev_high = self.datas[0].high[-1]
        prev_low  = self.datas[0].low[-1]
        price     = self.datas[0].close[0]
        vol       = self.datas[0].volume[0]
        avg_vol   = self.vol_ma[0]

        if not self.position:
            if price > prev_high and vol > avg_vol * self.p.vol_mult:
                risk = price - prev_low
                self.stop_price  = prev_low
                self.target_price = price + risk * self.p.rr_ratio
                qty = self._calc_qty(price, prev_low)
                self.order = self.buy(size=qty)
            elif price < prev_low and vol > avg_vol * self.p.vol_mult:
                risk = prev_high - price
                self.stop_price  = prev_high
                self.target_price = price - risk * self.p.rr_ratio
                qty = self._calc_qty(price, prev_high)
                self.order = self.sell(size=qty)
        else:
            self._check_exit()


# ── Strategy 2: EMA Crossover + Trend Filter ────────────────────────────────
class EMACrossoverStrategy(_Base):
    params = (
        ("instrument", "NIFTY"),
        ("fast", 9), ("slow", 21), ("trend", 55),
        ("stoploss_pct", 0.02),
        ("target_pct", 0.04),
        ("lot_size", 1),
        ("capital", CAPITAL),
        ("risk_pct", 0.01),
    )

    def __init__(self):
        super().__init__()
        self.ema_f = bt.indicators.EMA(self.datas[0].close, period=self.p.fast)
        self.ema_s = bt.indicators.EMA(self.datas[0].close, period=self.p.slow)
        self.ema_t = bt.indicators.EMA(self.datas[0].close, period=self.p.trend)
        self.cross = bt.indicators.CrossOver(self.ema_f, self.ema_s)

    def next(self):
        if self.order:
            return
        price = self.datas[0].close[0]

        if not self.position:
            if self.cross[0] == 1.0 and price > self.ema_t[0]:
                self._enter_long()
            elif self.cross[0] == -1.0 and price < self.ema_t[0]:
                self._enter_short()
        else:
            self._check_exit()


# ── Strategy 3: SuperTrend ───────────────────────────────────────────────────
class SuperTrendStrategy(_Base):
    params = (
        ("instrument", "NIFTY"),
        ("atr_period", 10), ("multiplier", 3.0), ("trend_ema", 55),
        ("stoploss_pct", 0.02),
        ("target_pct", 0.05),
        ("lot_size", 1),
        ("capital", CAPITAL),
        ("risk_pct", 0.01),
    )

    def __init__(self):
        super().__init__()
        self.atr   = bt.indicators.ATR(self.datas[0], period=self.p.atr_period)
        self.ema_t = bt.indicators.EMA(self.datas[0].close, period=self.p.trend_ema)
        # Track previous direction for flip detection
        self._prev_dir = None
        self._cur_dir  = None
        self._cur_st   = None

    def _supertrend(self):
        """Compute SuperTrend incrementally for the current bar only."""
        price = self.datas[0].close[0]
        high  = self.datas[0].high[0]
        low   = self.datas[0].low[0]
        atr_v = self.atr[0]
        m     = self.p.multiplier

        mid   = (high + low) / 2
        upper = mid + m * atr_v
        lower = mid - m * atr_v

        if self._cur_dir is None:
            # Initialise
            self._cur_st  = lower
            self._cur_dir = 1
        else:
            prev_st  = self._cur_st
            prev_dir = self._cur_dir
            if prev_dir == 1:
                if price < prev_st:
                    self._cur_st  = upper
                    self._cur_dir = -1
                else:
                    self._cur_st  = max(lower, prev_st)
                    self._cur_dir = 1
            else:
                if price > prev_st:
                    self._cur_st  = lower
                    self._cur_dir = 1
                else:
                    self._cur_st  = min(upper, prev_st)
                    self._cur_dir = -1

        return self._cur_st, self._cur_dir

    def next(self):
        if self.order:
            return
        if len(self.datas[0]) < self.p.atr_period + 2:
            return

        prev_dir  = self._cur_dir
        st, direction = self._supertrend()
        price = self.datas[0].close[0]

        if not self.position:
            flip_up   = (direction ==  1 and prev_dir == -1 and price > self.ema_t[0])
            flip_down = (direction == -1 and prev_dir ==  1 and price < self.ema_t[0])
            if flip_up:
                self.stop_price  = st
                self.target_price = price * (1 + self.p.target_pct)
                qty = self._calc_qty(price, st)
                self.order = self.buy(size=qty)
            elif flip_down:
                self.stop_price  = st
                self.target_price = price * (1 - self.p.target_pct)
                qty = self._calc_qty(price, st)
                self.order = self.sell(size=qty)
        else:
            if self.position.size > 0 and (direction == -1 or price < st):
                self.order = self.close()
            elif self.position.size < 0 and (direction == 1 or price > st):
                self.order = self.close()


# ── Strategy 4: Bollinger + RSI Mean Reversion ──────────────────────────────
class MeanReversionStrategy(_Base):
    params = (
        ("instrument", "NIFTY"),
        ("bb_period", 20), ("bb_std", 2.0),
        ("rsi_period", 14), ("rsi_lo", 30), ("rsi_hi", 70),
        ("stoploss_pct", 0.02),
        ("target_pct", 0.03),
        ("lot_size", 1),
        ("capital", CAPITAL),
        ("risk_pct", 0.01),
    )

    def __init__(self):
        super().__init__()
        self.bb  = bt.indicators.BollingerBands(
            self.datas[0].close, period=self.p.bb_period, devfactor=self.p.bb_std)
        self.rsi = bt.indicators.RSI(self.datas[0].close, period=self.p.rsi_period)

    def next(self):
        if self.order:
            return
        price = self.datas[0].close[0]

        if not self.position:
            if price <= self.bb.lines.bot[0] and self.rsi[0] < self.p.rsi_lo:
                self.stop_price  = price * (1 - self.p.stoploss_pct)
                self.target_price = self.bb.lines.mid[0]
                qty = self._calc_qty(price, self.stop_price)
                self.order = self.buy(size=qty)
            elif price >= self.bb.lines.top[0] and self.rsi[0] > self.p.rsi_hi:
                self.stop_price  = price * (1 + self.p.stoploss_pct)
                self.target_price = self.bb.lines.mid[0]
                qty = self._calc_qty(price, self.stop_price)
                self.order = self.sell(size=qty)
        else:
            price = self.datas[0].close[0]
            mid   = self.bb.lines.mid[0]
            if self.position.size > 0:
                if price <= self.stop_price or price >= mid:
                    self.order = self.close()
            elif self.position.size < 0:
                if price >= self.stop_price or price <= mid:
                    self.order = self.close()


# ─────────────────────────────────────────────────────────────────────────────
# 5. RUN SINGLE BACKTEST
# ─────────────────────────────────────────────────────────────────────────────
def run_bt(strategy_cls, df: pd.DataFrame, instrument: str, strategy_name: str) -> dict:
    cerebro = bt.Cerebro()   # stdstats=True required for AnnualReturn analyzer
    cerebro.adddata(df_to_feed(df))
    cerebro.addstrategy(strategy_cls, instrument=instrument)
    cerebro.broker.setcash(CAPITAL)
    cerebro.broker.addcommissioninfo(ZerodhaCommission())
    cerebro.broker.set_slippage_perc(SLIPPAGE_PCT)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe",
                        riskfreerate=0.065, annualize=True, timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name="annual")

    results = cerebro.run()
    strat   = results[0]

    final   = cerebro.broker.getvalue()
    net_pnl = final - CAPITAL
    ret_pct = net_pnl / CAPITAL * 100

    sharpe_raw = strat.analyzers.sharpe.get_analysis().get("sharperatio")
    sharpe     = round(sharpe_raw, 3) if sharpe_raw and not math.isnan(sharpe_raw) else None

    dd_data  = strat.analyzers.dd.get_analysis()
    max_dd   = round(dd_data.get("max", {}).get("drawdown", 0), 2)

    ta       = strat.analyzers.trades.get_analysis()
    total    = ta.get("total", {}).get("closed", 0)
    won      = ta.get("won",   {}).get("total", 0)
    lost     = ta.get("lost",  {}).get("total", 0)
    win_rate = round(won / total * 100, 1) if total > 0 else 0

    avg_win  = round(ta.get("won",  {}).get("pnl", {}).get("average", 0), 2)
    avg_loss = round(ta.get("lost", {}).get("pnl", {}).get("average", 0), 2)
    profit_factor = round(abs(avg_win * won / (avg_loss * lost)), 3) if lost > 0 and avg_loss != 0 else None

    annual   = strat.analyzers.annual.get_analysis()

    return {
        "strategy":      strategy_name,
        "instrument":    instrument,
        "capital":       CAPITAL,
        "final_value":   round(final, 2),
        "net_pnl":       round(net_pnl, 2),
        "returns_pct":   round(ret_pct, 2),
        "sharpe":        sharpe,
        "max_drawdown":  max_dd,
        "total_trades":  total,
        "won":           won,
        "lost":          lost,
        "win_rate":      win_rate,
        "avg_win":       avg_win,
        "avg_loss":      avg_loss,
        "profit_factor": profit_factor,
        "annual":        dict(annual),
        "trade_log":     strat.trade_records,
        "data_from":     str(df.index[0].date()),
        "data_to":       str(df.index[-1].date()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. REPORTS
# ─────────────────────────────────────────────────────────────────────────────
def colour(val, good=True):
    if good:
        return "green" if val >= 0 else "red"
    else:
        return "red" if val >= 0 else "green"


def print_summary_table(all_results: list):
    t = Table(title="Strategy Performance Summary — NIFTY & BANKNIFTY",
              box=box.DOUBLE_EDGE, show_lines=True)

    cols = ["Strategy", "Instrument", "Capital (₹)", "Final Value (₹)",
            "Net P&L (₹)", "Return %", "Sharpe", "Max DD %",
            "Trades", "Win Rate", "Profit Factor"]
    for c in cols:
        t.add_column(c, justify="right" if c not in ("Strategy","Instrument") else "left")

    for r in sorted(all_results, key=lambda x: x["returns_pct"], reverse=True):
        ret   = r["returns_pct"]
        pnl   = r["net_pnl"]
        c_ret = "green" if ret >= 0 else "red"
        t.add_row(
            r["strategy"],
            r["instrument"],
            f"₹{r['capital']:,.0f}",
            f"₹{r['final_value']:,.2f}",
            f"[{c_ret}]₹{pnl:+,.2f}[/{c_ret}]",
            f"[{c_ret}]{ret:+.2f}%[/{c_ret}]",
            str(r["sharpe"]) if r["sharpe"] else "N/A",
            f"{r['max_drawdown']:.2f}%",
            str(r["total_trades"]),
            f"{r['win_rate']:.1f}% ({r['won']}W/{r['lost']}L)",
            str(r["profit_factor"]) if r["profit_factor"] else "N/A",
        )

    console.print(t)


def print_annual_table(all_results: list):
    # Collect all years
    years = set()
    for r in all_results:
        years.update(r["annual"].keys())
    years = sorted(years)

    t = Table(title="Annual Returns by Strategy & Instrument",
              box=box.ROUNDED, show_lines=True)
    t.add_column("Strategy", style="cyan")
    t.add_column("Instrument", style="cyan")
    for yr in years:
        t.add_column(str(yr), justify="right")

    for r in all_results:
        row = [r["strategy"], r["instrument"]]
        for yr in years:
            val = r["annual"].get(yr)
            if val is not None:
                pct = round(val * 100, 1)
                c = "green" if pct >= 0 else "red"
                row.append(f"[{c}]{pct:+.1f}%[/{c}]")
            else:
                row.append("-")
        t.add_row(*row)

    console.print(t)


def print_trade_details(result: dict, max_trades: int = 20):
    trades = result["trade_log"]
    if not trades:
        return
    label = f"Trade Log — {result['strategy']} / {result['instrument']} (first {min(max_trades, len(trades))} of {len(trades)})"
    t = Table(title=label, box=box.SIMPLE, show_lines=False)
    for c in ["#", "Entry Date", "Exit Date", "Entry ₹", "Exit ₹", "Size", "P&L ₹", "Net P&L ₹", "W/L"]:
        t.add_column(c, justify="right" if c not in ("#", "Entry Date", "Exit Date", "W/L") else "left")

    for i, trade in enumerate(trades[:max_trades], 1):
        pnl  = trade["pnl"]
        npnl = trade["pnlcomm"]
        c    = "green" if pnl >= 0 else "red"
        wl   = "[green]W[/green]" if trade["won"] else "[red]L[/red]"
        t.add_row(
            str(i),
            trade["date"],
            trade["exit_date"],
            f"₹{trade['entry']:,.2f}",
            f"₹{trade['exit']:,.2f}",
            str(trade["size"]),
            f"[{c}]₹{pnl:+,.2f}[/{c}]",
            f"[{c}]₹{npnl:+,.2f}[/{c}]",
            wl,
        )
    console.print(t)


def build_monthly_pnl(result: dict) -> pd.DataFrame:
    trades = result["trade_log"]
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame(trades)
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    df["month"] = df["exit_date"].dt.to_period("M")
    monthly = df.groupby("month")["pnlcomm"].sum().reset_index()
    monthly.columns = ["Month", "Net P&L (₹)"]
    monthly["Net P&L (₹)"] = monthly["Net P&L (₹)"].round(2)
    return monthly


def print_monthly_pnl(result: dict):
    df = build_monthly_pnl(result)
    if df.empty:
        return
    t = Table(title=f"Monthly P&L — {result['strategy']} / {result['instrument']}",
              box=box.SIMPLE, show_lines=False)
    t.add_column("Month")
    t.add_column("Net P&L (₹)", justify="right")
    t.add_column("Bar", justify="left")

    max_abs = df["Net P&L (₹)"].abs().max()
    for _, row in df.iterrows():
        pnl = row["Net P&L (₹)"]
        c   = "green" if pnl >= 0 else "red"
        bar_len = int(abs(pnl) / max_abs * 20) if max_abs > 0 else 0
        bar = ("█" * bar_len)
        t.add_row(str(row["Month"]), f"[{c}]₹{pnl:+,.2f}[/{c}]", f"[{c}]{bar}[/{c}]")

    console.print(t)


def save_json_report(all_results: list):
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = REPORTS_DIR / f"backtest_full_{ts}.json"
    save = []
    for r in all_results:
        d = {k: v for k, v in r.items() if k != "trade_log"}
        d["trades_sample"] = r["trade_log"][:10]
        save.append(d)
    with open(out_file, "w") as f:
        json.dump(save, f, indent=2, default=str)
    return out_file


def save_csv_trades(all_results: list):
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = []
    for r in all_results:
        for t in r["trade_log"]:
            t2 = dict(t)
            t2["strategy"]   = r["strategy"]
            t2["instrument"] = r["instrument"]
            rows.append(t2)
    if rows:
        out_file = REPORTS_DIR / f"all_trades_{ts}.csv"
        pd.DataFrame(rows).to_csv(out_file, index=False)
        return out_file
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    console.rule("[bold cyan]NIFTY / BANKNIFTY — Autonomous Strategy Backtest[/bold cyan]")
    console.print(f"[dim]Capital: ₹{CAPITAL:,.0f} | Brokerage: ₹{BROKERAGE_PER_ORDER}/order | "
                  f"Slippage: {SLIPPAGE_PCT*100:.3f}%[/dim]\n")

    # ── Fetch data ────────────────────────────────────────────────────────────
    console.print("[bold]Step 1: Fetching Historical Data[/bold]")
    data = {
        "NIFTY":     fetch_yfinance("^NSEI",    "NIFTY 50"),
        "BANKNIFTY": fetch_yfinance("^NSEBANK", "NIFTY BANK"),
    }
    console.print()

    # ── Run backtests ─────────────────────────────────────────────────────────
    console.print("[bold]Step 2: Running Backtests[/bold]")
    strategies = [
        ("ORB",            ORBStrategy,          "Opening Range Breakout"),
        ("EMA Crossover",  EMACrossoverStrategy,  "EMA 9/21/55 + Trend Filter"),
        ("SuperTrend",     SuperTrendStrategy,    "SuperTrend ATR(10,3)"),
        ("Mean Reversion", MeanReversionStrategy, "Bollinger(20,2) + RSI(14)"),
    ]

    all_results = []
    for sname, scls, desc in strategies:
        for inst, df in data.items():
            console.print(f"  [cyan]Running {desc} on {inst}...[/cyan]", end=" ")
            try:
                result = run_bt(scls, df.copy(), inst, sname)
                all_results.append(result)
                ret = result["returns_pct"]
                c   = "green" if ret >= 0 else "red"
                console.print(f"[{c}]{ret:+.2f}%[/{c}] | "
                               f"Trades: {result['total_trades']} | "
                               f"Win: {result['win_rate']}%")
            except Exception as e:
                console.print(f"[red]FAILED: {e}[/red]")

    if not all_results:
        console.print("[red]No backtest results to display.[/red]")
        return

    console.print()
    console.rule("[bold cyan]BACKTEST RESULTS[/bold cyan]")

    # ── Summary Table ─────────────────────────────────────────────────────────
    console.print("\n[bold]Performance Summary[/bold]")
    print_summary_table(all_results)

    # ── Annual Returns ────────────────────────────────────────────────────────
    console.print("\n[bold]Annual Returns[/bold]")
    print_annual_table(all_results)

    # ── Monthly P&L for each strategy/instrument ──────────────────────────────
    console.print("\n[bold]Monthly P&L Breakdown[/bold]")
    for r in all_results:
        print_monthly_pnl(r)

    # ── Trade Logs (top strategies) ───────────────────────────────────────────
    console.print("\n[bold]Trade Log Detail[/bold]")
    for r in sorted(all_results, key=lambda x: x["returns_pct"], reverse=True)[:4]:
        print_trade_details(r, max_trades=15)

    # ── Best Strategy Per Instrument ──────────────────────────────────────────
    console.print()
    console.rule("[bold green]Best Strategy Per Instrument[/bold green]")
    for inst in ["NIFTY", "BANKNIFTY"]:
        inst_results = [r for r in all_results if r["instrument"] == inst]
        if not inst_results:
            continue
        best = max(inst_results, key=lambda x: x["returns_pct"])
        console.print(Panel(
            f"[bold]{best['strategy']}[/bold]\n"
            f"Return:      [{colour(best['returns_pct'])}]{best['returns_pct']:+.2f}%[/{colour(best['returns_pct'])}]\n"
            f"Net P&L:     [{colour(best['net_pnl'])}]₹{best['net_pnl']:+,.2f}[/{colour(best['net_pnl'])}]\n"
            f"Sharpe:      {best['sharpe']}\n"
            f"Max Drawdown:{best['max_drawdown']}%\n"
            f"Win Rate:    {best['win_rate']}%  ({best['won']}W / {best['lost']}L)\n"
            f"Trades:      {best['total_trades']}\n"
            f"Profit Factor: {best['profit_factor']}",
            title=f"[bold cyan]Best for {inst}[/bold cyan]",
            border_style="green",
        ))

    # ── Save reports ──────────────────────────────────────────────────────────
    json_file = save_json_report(all_results)
    csv_file  = save_csv_trades(all_results)
    console.print(f"\n[dim]Reports saved:[/dim]")
    console.print(f"  [dim]JSON: {json_file}[/dim]")
    if csv_file:
        console.print(f"  [dim]CSV:  {csv_file}[/dim]")

    # ── Final Verdict ─────────────────────────────────────────────────────────
    total_across_all  = sum(r["net_pnl"] for r in all_results)
    best_overall      = max(all_results, key=lambda x: x["returns_pct"])
    console.print()
    console.rule("[bold cyan]Final Verdict[/bold cyan]")
    console.print(
        f"Best Overall: [bold green]{best_overall['strategy']}[/bold green] on "
        f"[bold]{best_overall['instrument']}[/bold] — "
        f"[green]{best_overall['returns_pct']:+.2f}% return[/green] | "
        f"Sharpe {best_overall['sharpe']} | "
        f"Max DD {best_overall['max_drawdown']}%"
    )


if __name__ == "__main__":
    main()
