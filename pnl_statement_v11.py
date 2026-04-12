"""
pnl_statement_v11.py
====================
Standalone P&L statement for Algorithm Version 11.
Reads from the saved backtest JSON and prints a full auditable report.

Run:  python pnl_statement_v11.py
"""

import json, sys
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text
from rich import box

console = Console(width=160)

# ── Load report ──────────────────────────────────────────────────────────────
REPORT_FILE = sorted(Path("reports").glob("v11_backtest_*.json"))[-1]
with open(REPORT_FILE) as f:
    D = json.load(f)

S      = D["results"]["top2_per_inst"]["stats"]          # best config stats
ANNUAL = [r for r in D["best_portfolio"]["annual"] if r["trades"] > 0]
INSTS  = D["best_portfolio"]["instruments"]
TRADES = D["best_portfolio"]["trades"]
ASSIGN = D["best_strategy_per_inst"]

STOCK_SET = {"RELIANCE","HDFCBANK","INFY","TCS","ICICIBANK","AXISBANK","SBIN",
             "BAJFINANCE","HCLTECH","ITC","LT","SUNPHARMA","KOTAKBANK","MARUTI","WIPRO"}

# Derived
idx_trades  = [t for t in TRADES if t.get("asset_type") == "index"]
stk_trades  = [t for t in TRADES if t.get("asset_type") == "stock"]
idx_pnl     = sum(t["pnl"] for t in idx_trades)
stk_pnl     = sum(t["pnl"] for t in stk_trades)
idx_wr      = sum(1 for t in idx_trades if t["pnl"] > 0) / max(len(idx_trades), 1) * 100
stk_wr      = sum(1 for t in stk_trades if t["pnl"] > 0) / max(len(stk_trades), 1) * 100
win_trades  = sum(1 for t in TRADES if t["pnl"] > 0)
loss_trades = sum(1 for t in TRADES if t["pnl"] <= 0)
gross_profit= sum(t["pnl"] for t in TRADES if t["pnl"] > 0)
gross_loss  = sum(t["pnl"] for t in TRADES if t["pnl"] < 0)
profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else 999
avg_win     = gross_profit / max(win_trades, 1)
avg_loss    = gross_loss   / max(loss_trades, 1)
expectancy  = (S["win_rate"]/100 * avg_win + (1 - S["win_rate"]/100) * avg_loss)

START_CAP   = 1_000_000
FINAL_CAP   = S["capital"]
NET_PNL     = S["net_pnl"]
MULTIPLE    = FINAL_CAP / START_CAP

generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
console.print()
console.print(Panel(
    Text.assemble(
        ("ALGORITHMIC TRADING STRATEGY  ·  P&L STATEMENT\n", "bold white"),
        ("VERSION 11  ·  NSE Futures (India)  ·  Zerodha Platform\n\n", "bold cyan"),
        (f"Report Generated : {generated_at}\n", "dim"),
        (f"Source File      : {REPORT_FILE.name}\n", "dim"),
        (f"Backtest Config  : Top-2 strategies per instrument\n", "dim"),
        (f"Universe         : 20 instruments (5 index + 15 stock futures)\n", "dim"),
    ),
    border_style="cyan", padding=(0, 2)
))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — EXECUTIVE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
console.print(Rule("[bold cyan]SECTION 1 — EXECUTIVE SUMMARY[/bold cyan]"))

period_start = ANNUAL[0]["year"] if ANNUAL else 2008
period_end   = ANNUAL[-1]["year"] if ANNUAL else 2026
years        = period_end - period_start + 1

sum_tbl = Table(box=box.SIMPLE, show_header=False, padding=(0, 2), min_width=155)
sum_tbl.add_column("Metric",   min_width=32, style="bold")
sum_tbl.add_column("Value",    min_width=22)
sum_tbl.add_column("Metric",   min_width=32, style="bold")
sum_tbl.add_column("Value",    min_width=22)

rows = [
    ("Starting Capital",         f"₹{START_CAP:>15,.0f}",   "Strategy",              "Bollinger Band Reversion"),
    ("Final Capital",            f"[bold green]₹{FINAL_CAP:>15,.0f}[/bold green]",
                                                              "Universe",              "20 instruments (5 index + 15 stocks)"),
    ("Net Profit",               f"[bold green]₹{NET_PNL:>15,.0f}[/bold green]",
                                                              "Exchange",              "NSE Futures (F&O Segment)"),
    ("Return on Capital",        f"[bold green]{S['total_ret']:>+14.1f}%[/bold green]",
                                                              "Broker",                "Zerodha"),
    ("CAGR (Annualised)",        f"[bold green]{S['ann_ret']:>+14.1f}%[/bold green]",
                                                              "Brokerage per trade",   "₹20 flat"),
    ("Capital Multiple",         f"[bold green]{MULTIPLE:>14.0f}×[/bold green]",
                                                              "Slippage assumption",   "0.03% per side"),
    ("Backtest Period",          f"{period_start} – {period_end}  ({years} yrs)",
                                                              "Roll Cost",             "0.20% / month / position"),
    ("Sharpe Ratio",             f"[bold green]{S['sharpe']:>14.2f}[/bold green]",
                                                              "SPAN Margin (Index)",   "10% of notional"),
    ("Max Drawdown",             f"[yellow]{S['max_dd']:>13.1f}%[/yellow]",
                                                              "SPAN Margin (Stock)",   "15% of notional"),
    ("Win Rate",                 f"[bold green]{S['win_rate']:>13.1f}%[/bold green]",
                                                              "Risk / Index Trade",    "4% of portfolio"),
    ("Total Trades",             f"{S['num_trades']:>14,}",   "Risk / Stock Trade",    "2% of portfolio"),
    ("Profit Factor",            f"[bold green]{profit_factor:>14.2f}[/bold green]",
                                                              "Max Concurrent Pos.",   "10"),
    ("Avg Win",                  f"₹{avg_win:>15,.0f}",       "Margin Calls",          str(S["margin_calls"])),
    ("Avg Loss",                 f"₹{avg_loss:>15,.0f}",       "Roll Costs Paid",      f"₹{S['roll_cost_total']:,.0f}"),
    ("Expectancy / Trade",       f"₹{expectancy:>15,.0f}",    "Trades / Year (avg)",   f"{S['trades_per_yr']:.0f}"),
]

for r in rows:
    sum_tbl.add_row(*r)
console.print(sum_tbl)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — YEAR-BY-YEAR P&L
# ─────────────────────────────────────────────────────────────────────────────
console.print(Rule("[bold cyan]SECTION 2 — YEAR-BY-YEAR P&L[/bold cyan]"))

ytbl = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold",
             min_width=155,
             title="Annual Returns  |  Starting Capital ₹10,00,000")
ytbl.add_column("Year",       justify="center", min_width=6)
ytbl.add_column("Trades",     justify="right",  min_width=8)
ytbl.add_column("Win Rate",   justify="right",  min_width=10)
ytbl.add_column("P&L (₹)",    justify="right",  min_width=20)
ytbl.add_column("Return %",   justify="right",  min_width=11)
ytbl.add_column("Equity (₹)", justify="right",  min_width=22)
ytbl.add_column("Bar",        min_width=40)

equity = START_CAP
for row in ANNUAL:
    equity += row["pnl"]
    r = row["ret_pct"]
    c = "bold green" if r >= 20 else ("green" if r >= 10 else ("yellow" if r >= 0 else "red"))
    bar_len = max(0, min(38, int(abs(r) / 5)))
    bar_char = "█" if r >= 0 else "░"
    bar = bar_char * bar_len
    bar_col = "green" if r >= 0 else "red"
    ytbl.add_row(
        str(row["year"]),
        f"{row['trades']:,}",
        f"{row['win_rate']:.1f}%",
        f"₹{row['pnl']:+,.0f}",
        f"[{c}]{r:+.2f}%[/{c}]",
        f"₹{equity:,.0f}",
        f"[{bar_col}]{bar}[/{bar_col}]"
    )

ytbl.add_section()
ytbl.add_row(
    "TOTAL", f"{S['num_trades']:,}", f"{S['win_rate']:.1f}%",
    f"[bold green]₹{NET_PNL:+,.0f}[/bold green]",
    f"[bold green]{S['total_ret']:+.1f}%[/bold green]",
    f"[bold green]₹{FINAL_CAP:,.0f}[/bold green]",
    f"[bold green]CAGR {S['ann_ret']:+.1f}% / year[/bold green]"
)
console.print(ytbl)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — INSTRUMENT BREAKDOWN
# ─────────────────────────────────────────────────────────────────────────────
console.print(Rule("[bold cyan]SECTION 3 — INSTRUMENT P&L BREAKDOWN[/bold cyan]"))

# Totals by type
type_tbl = Table(box=box.SIMPLE, show_header=True, header_style="bold",
                 title="Index Futures vs Stock Futures Contribution")
type_tbl.add_column("Asset Type",  min_width=18)
type_tbl.add_column("Instruments", justify="right", min_width=13)
type_tbl.add_column("Trades",      justify="right", min_width=10)
type_tbl.add_column("Win Rate",    justify="right", min_width=10)
type_tbl.add_column("Net P&L",     justify="right", min_width=20)
type_tbl.add_column("% of Total",  justify="right", min_width=12)
type_tbl.add_row("Index Futures",  "5",
                 f"{len(idx_trades):,}", f"{idx_wr:.1f}%",
                 f"[green]₹{idx_pnl:+,.0f}[/green]",
                 f"{idx_pnl/NET_PNL*100:.1f}%")
type_tbl.add_row("Stock Futures",  "15",
                 f"{len(stk_trades):,}", f"{stk_wr:.1f}%",
                 f"[green]₹{stk_pnl:+,.0f}[/green]",
                 f"{stk_pnl/NET_PNL*100:.1f}%")
type_tbl.add_section()
type_tbl.add_row("[bold]TOTAL[/bold]", "[bold]20[/bold]",
                 f"[bold]{S['num_trades']:,}[/bold]", f"[bold]{S['win_rate']:.1f}%[/bold]",
                 f"[bold green]₹{NET_PNL:+,.0f}[/bold green]", "[bold]100.0%[/bold]")
console.print(type_tbl)

console.print()
itbl = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold",
             title="Per-Instrument P&L  (sorted by Net P&L, highest first)")
itbl.add_column("#",          justify="right", min_width=3)
itbl.add_column("Instrument", min_width=13)
itbl.add_column("Type",       min_width=7)
itbl.add_column("Strategy",   min_width=16)
itbl.add_column("Trades",     justify="right", min_width=8)
itbl.add_column("Win Rate",   justify="right", min_width=9)
itbl.add_column("Avg Win",    justify="right", min_width=13)
itbl.add_column("Avg Loss",   justify="right", min_width=13)
itbl.add_column("Net P&L",    justify="right", min_width=18)
itbl.add_column("% of Total", justify="right", min_width=11)

inst_trades_map = {}
for t in TRADES:
    inst_trades_map.setdefault(t["instrument"], []).append(t)

for rank, row in enumerate(INSTS, 1):
    inst  = row["instrument"]
    atype = "STOCK" if inst in STOCK_SET else "INDEX"
    strat = ASSIGN.get(inst, "bb_reversion")
    ts    = inst_trades_map.get(inst, [])
    al    = sum(t["pnl"] for t in ts if t["pnl"] < 0) / max(sum(1 for t in ts if t["pnl"] < 0), 1)
    pct   = row["net_pnl"] / NET_PNL * 100
    c     = "green" if row["net_pnl"] >= 0 else "red"
    type_c = "cyan" if atype == "INDEX" else "white"
    itbl.add_row(
        str(rank),
        f"[bold]{inst}[/bold]",
        f"[{type_c}]{atype}[/{type_c}]",
        strat,
        f"{row['trades']:,}",
        f"{row['win_rate']:.1f}%",
        f"₹{row['avg_win']:,.0f}",
        f"₹{al:,.0f}",
        f"[{c}]₹{row['net_pnl']:+,.0f}[/{c}]",
        f"{pct:.1f}%"
    )
console.print(itbl)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — RISK METRICS
# ─────────────────────────────────────────────────────────────────────────────
console.print(Rule("[bold cyan]SECTION 4 — RISK METRICS[/bold cyan]"))

risk_tbl = Table(box=box.SIMPLE, show_header=False, padding=(0, 3), min_width=155)
risk_tbl.add_column("Metric",  min_width=35, style="bold")
risk_tbl.add_column("Value",   min_width=22)
risk_tbl.add_column("Metric",  min_width=35, style="bold")
risk_tbl.add_column("Value",   min_width=22)

risk_rows = [
    ("Sharpe Ratio",                f"[bold green]{S['sharpe']:.2f}[/bold green]",
     "Win Rate",                    f"[bold green]{S['win_rate']:.1f}%[/bold green]"),
    ("Max Drawdown (equity curve)", f"[yellow]{S['max_dd']:.1f}%[/yellow]",
     "Profit Factor",               f"[bold green]{profit_factor:.2f}[/bold green]"),
    ("Gross Profit",                f"₹{gross_profit:,.0f}",
     "Avg Win / Avg Loss Ratio",    f"{abs(avg_win/avg_loss):.2f}×"),
    ("Gross Loss",                  f"₹{gross_loss:,.0f}",
     "Expectancy per Trade",        f"₹{expectancy:,.0f}"),
    ("Total Trades",                f"{S['num_trades']:,}",
     "Avg Trades per Year",         f"{S['trades_per_yr']:.0f}"),
    ("Winning Trades",              f"{win_trades:,}  ({S['win_rate']:.1f}%)",
     "Margin Calls",                f"{S['margin_calls']}"),
    ("Losing Trades",               f"{loss_trades:,}  ({100-S['win_rate']:.1f}%)",
     "Roll Costs Paid",             f"₹{S['roll_cost_total']:,.0f}"),
]
for r in risk_rows:
    risk_tbl.add_row(*r)
console.print(risk_tbl)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — STRATEGY CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
console.print(Rule("[bold cyan]SECTION 5 — STRATEGY CONFIGURATION[/bold cyan]"))

console.print("""
  [bold]Primary Strategy : Bollinger Band %B Reversion[/bold]
  ─────────────────────────────────────────────────────────────────────────
  Signal Logic   : Price closes below lower Bollinger Band (BB%B < 0.05) in
                   an uptrend (EMA50 > EMA200) → Long entry at next-day open.
                   Price closes above upper Bollinger Band (BB%B > 0.95) in a
                   downtrend → Short entry at next-day open.
  Exit Logic     : Price crosses back through the middle band (BB%B = 0.50),
                   OR initial stop (1.5× ATR below entry) hit.
  BB Parameters  : Period=20, StdDev=2.0
  ADX Filter     : Not required (BB itself provides range/trend context)
  RSI Filter     : Long only when RSI > 25 (not in capitulation);
                   Short only when RSI < 75 (not in extreme blow-off)

  [bold]Execution Rules[/bold]
  ─────────────────────────────────────────────────────────────────────────
  Entry          : Signal fires at day-i close → executes at day-(i+1) OPEN
                   (No same-bar execution — zero lookahead bias)
  Stop Loss      : 1.5× ATR below entry (Long) / above entry (Short)
  Target         : 2× risk-distance from entry (risk:reward = 1:2)
  Position Size  : Risk-based: risk_amount / (stop_distance × lot_size)
                   Index trades: 4% portfolio risk per trade (VIX-adjusted)
                   Stock trades: 2% portfolio risk per trade (fixed)
  MTM Settlement : Daily mark-to-market (NSE futures standard)
  Roll Cost      : 0.20% of position notional on last trading day of month
""")

cfg_tbl = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold",
                title="Per-Instrument Strategy Assignment")
cfg_tbl.add_column("Instrument",  min_width=13)
cfg_tbl.add_column("Asset Type",  min_width=9)
cfg_tbl.add_column("Lot Size",    justify="right", min_width=9)
cfg_tbl.add_column("Strategy",    min_width=18)
cfg_tbl.add_column("SPAN Margin", justify="right", min_width=12)
cfg_tbl.add_column("Risk/Trade",  justify="right", min_width=11)
cfg_tbl.add_column("Solo CAGR",   justify="right", min_width=10)
cfg_tbl.add_column("Solo Sharpe", justify="right", min_width=11)

lot_sizes = {"NIFTY":65,"BANKNIFTY":30,"MIDCPNIFTY":120,"FINNIFTY":60,"NIFTYIT":30,
             "RELIANCE":250,"HDFCBANK":550,"INFY":300,"TCS":150,"ICICIBANK":700,
             "AXISBANK":625,"SBIN":1500,"BAJFINANCE":125,"HCLTECH":700,
             "ITC":3200,"LT":300,"SUNPHARMA":700,"KOTAKBANK":400,"MARUTI":100,"WIPRO":1500}

for inst in (["NIFTY","BANKNIFTY","MIDCPNIFTY","FINNIFTY","NIFTYIT"] +
             ["RELIANCE","HDFCBANK","INFY","TCS","ICICIBANK","AXISBANK","SBIN",
              "BAJFINANCE","HCLTECH","ITC","LT","SUNPHARMA","KOTAKBANK","MARUTI","WIPRO"]):
    strat   = ASSIGN.get(inst, "bb_reversion")
    is_stk  = inst in STOCK_SET
    atype   = "STOCK" if is_stk else "INDEX"
    margin  = "15%" if is_stk else "10%"
    risk    = "2%" if is_stk else "4%"
    lot     = lot_sizes.get(inst, 1)
    gs      = D["discovery_grid"].get(inst, {}).get(strat, {})
    solo_c  = f"{gs.get('ann_ret',0):+.1f}%" if gs else "n/a"
    solo_sh = f"{gs.get('sharpe',0):.2f}"    if gs else "n/a"
    type_c  = "cyan" if atype == "INDEX" else "white"
    cfg_tbl.add_row(
        f"[bold]{inst}[/bold]",
        f"[{type_c}]{atype}[/{type_c}]",
        str(lot), strat, margin, risk, solo_c, solo_sh
    )
console.print(cfg_tbl)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — VERSION HISTORY
# ─────────────────────────────────────────────────────────────────────────────
console.print(Rule("[bold cyan]SECTION 6 — ALGORITHM VERSION HISTORY[/bold cyan]"))

VERSION_HISTORY = [
    ("v6", "Original algo (bugs present)",
     "—",     "—",     "—",     "—",      1_000_000,  "~1 Cr",   "Audit baseline"),
    ("v7", "Date UNION · Real drawdown · Rotation at next-open · Roll cost",
     "+26.6%","0.71",  "28.6%", "17.5 yr",1_000_000, "~1.33 Cr","4 structural fixes"),
    ("v8", "NIFTYIT added (0.31 corr vs BANKNIFTY) · SuperTrend · Momentum",
     "+28.9%","0.74",  "34.1%", "17.5 yr",1_000_000, "~1.74 Cr","5th instrument"),
    ("v9", "Trend-rider override · VIX>30 size reduced (data-proven losers)",
     "+28.4%","0.74",  "34.1%", "17.5 yr",1_000_000, "~1.68 Cr","Targeted fixes"),
    ("v10","Per-instrument strategy discovery on 5 indices",
     "+28.8%","0.75",  "32.9%", "17.5 yr",1_000_000, "~1.76 Cr","Discovery phase"),
    ("v11","20 instruments · BB reversion · MACD · Vol breakout\n"
           "       15 stock futures added · Per-instrument optimal",
     f"+{S['ann_ret']:.1f}%",
             f"{S['sharpe']:.2f}",
                      f"{S['max_dd']:.1f}%","18 yr",  1_000_000,
                                                        f"₹{FINAL_CAP/1e7:.1f} Cr",
                                                                   "★ CURRENT"),
]

vtbl = Table(box=box.DOUBLE_EDGE, show_header=True, header_style="bold magenta",
             title="P&L Statement  ·  Algorithm Version History  ·  NSE Futures India",
             min_width=155)
vtbl.add_column("Ver",     min_width=5)
vtbl.add_column("Key Changes",        min_width=54)
vtbl.add_column("CAGR",    justify="right", min_width=8)
vtbl.add_column("Sharpe",  justify="right", min_width=8)
vtbl.add_column("MaxDD",   justify="right", min_width=8)
vtbl.add_column("Period",  justify="right", min_width=8)
vtbl.add_column("Start",   justify="right", min_width=13)
vtbl.add_column("Final",   justify="right", min_width=14)
vtbl.add_column("Status",  min_width=14)

for ver, desc, cagr, sh, dd, yrs, start, final, status in VERSION_HISTORY:
    is_curr = (ver == "v11")
    c = "bold green" if is_curr else ("green" if ver >= "v8" else "white")
    vtbl.add_row(
        f"[{c}]{ver}[/{c}]",
        f"[{c}]{desc}[/{c}]",
        f"[{c}]{cagr}[/{c}]",
        sh, dd, yrs,
        f"₹{start:,.0f}",
        f"[{c}]{final}[/{c}]",
        f"[{c}]{status}[/{c}]"
    )
console.print(vtbl)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER PANEL
# ─────────────────────────────────────────────────────────────────────────────
console.print()
console.print(Panel(
    Text.assemble(
        ("ALGORITHM v11  ·  FINAL SUMMARY\n\n", "bold green"),
        ("Starting Capital : ", "bold"), (f"₹{START_CAP:>12,.0f}\n", "white"),
        ("Final Capital    : ", "bold"), (f"₹{FINAL_CAP:>12,.0f}    ", "bold green"),
        (f"({MULTIPLE:.0f}× return)\n", "bold green"),
        ("Net Profit       : ", "bold"), (f"₹{NET_PNL:>12,.0f}\n", "bold green"),
        ("Annualised CAGR  : ", "bold"), (f"{S['ann_ret']:>+11.1f}%\n", "bold green"),
        ("Sharpe Ratio     : ", "bold"), (f"{S['sharpe']:>12.2f}\n", "bold green"),
        ("Max Drawdown     : ", "bold"), (f"{S['max_dd']:>11.1f}%\n", "yellow"),
        ("Win Rate         : ", "bold"), (f"{S['win_rate']:>11.1f}%\n", "bold green"),
        ("Profit Factor    : ", "bold"), (f"{profit_factor:>12.2f}\n", "bold green"),
        ("Total Trades     : ", "bold"), (f"{S['num_trades']:>12,}\n", "white"),
        ("Backtest Period  : ", "bold"), (f"2008 – 2026  ({years} years)\n", "white"),
        ("Margin Calls     : ", "bold"), (f"{S['margin_calls']:>12}\n", "white"),
        ("\nDisclaimer: Past performance does not guarantee future results.\n"
         "Backtest results do not account for liquidity impact on large positions,\n"
         "lot size changes over time, or regulatory changes to F&O margins.\n"
         "This report is for research purposes only.", "dim"),
    ),
    title="[bold green]v11 RESULT[/bold green]",
    border_style="green", padding=(1, 3)
))

# ── Save plain-text version ──────────────────────────────────────────────────
out_txt = Path(f"reports/pnl_statement_v11_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
console2 = Console(file=open(out_txt, "w", encoding="utf-8"), width=160, no_color=True)

# Re-run everything into the file console (simplified)
console2.print(f"ALGORITHMIC TRADING STRATEGY — P&L STATEMENT — VERSION 11")
console2.print(f"Generated: {generated_at}   |   Source: {REPORT_FILE.name}\n")
console2.print(f"{'─'*120}")
console2.print(f"Starting Capital : ₹{START_CAP:>12,.0f}")
console2.print(f"Final Capital    : ₹{FINAL_CAP:>12,.0f}   ({MULTIPLE:.0f}× return)")
console2.print(f"Net Profit       : ₹{NET_PNL:>12,.0f}")
console2.print(f"CAGR             : {S['ann_ret']:>+11.1f}%")
console2.print(f"Sharpe Ratio     : {S['sharpe']:>12.2f}")
console2.print(f"Max Drawdown     : {S['max_dd']:>11.1f}%")
console2.print(f"Win Rate         : {S['win_rate']:>11.1f}%")
console2.print(f"Profit Factor    : {profit_factor:>12.2f}")
console2.print(f"Total Trades     : {S['num_trades']:>12,}")
console2.print(f"Margin Calls     : {S['margin_calls']:>12}")
console2.print(f"{'─'*120}\n")
console2.print("YEAR-BY-YEAR RETURNS")
console2.print(f"{'─'*120}")
console2.print(f"{'Year':>6}  {'Trades':>8}  {'Win%':>7}  {'P&L (₹)':>20}  {'Return%':>9}  {'Equity (₹)':>22}")
console2.print(f"{'─'*120}")
eq2 = START_CAP
for row in ANNUAL:
    eq2 += row["pnl"]
    console2.print(f"{row['year']:>6}  {row['trades']:>8,}  {row['win_rate']:>6.1f}%  "
                   f"{row['pnl']:>+20,.0f}  {row['ret_pct']:>+8.2f}%  {eq2:>22,.0f}")
console2.print(f"{'─'*120}")
console2.print(f"{'TOTAL':>6}  {S['num_trades']:>8,}  {S['win_rate']:>6.1f}%  "
               f"{NET_PNL:>+20,.0f}  {S['total_ret']:>+8.1f}%  {FINAL_CAP:>22,.0f}")
console2.print(f"\nReport saved to: {out_txt}")

console.print(f"\n[dim]Plain-text copy saved → {out_txt}[/dim]")
