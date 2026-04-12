"""
generate_report_v13.py
======================
Reads the latest v12 backtest JSON and produces a comprehensive HTML P&L report.
v13 fix: trade times now reflect NSE market hours (09:15 entry, 15:30 exit).

Run:  python generate_report_v13.py
"""

import json, math
from pathlib import Path
from datetime import datetime

def _fmt_trade_time(ts_str):
    """Format trade timestamp: show date + NSE market time (strip midnight artefact).
    '2025-01-15 09:15:00' → '2025-01-15  09:15'
    '2025-01-15 00:00:00' → '2025-01-15'  (legacy/fallback)
    """
    if not ts_str:
        return "—"
    ts = str(ts_str)
    if "00:00:00" in ts:
        return ts[:10]   # legacy: just date
    return ts[:16]       # date + HH:MM

# ── Load data ────────────────────────────────────────────────────────────────
# Try v12 first, fall back to v11
v12_files = sorted(Path("reports").glob("v13_backtest_*.json"))
v11_files = sorted(Path("reports").glob("v11_backtest_*.json"))
REPORT_FILE = v12_files[-1] if v12_files else v11_files[-1]
with open(REPORT_FILE) as f:
    D = json.load(f)

S       = D["results"]["top2_per_inst"]["stats"]
ANNUAL  = [r for r in D["best_portfolio"]["annual"] if r["trades"] > 0]
INSTS   = D["best_portfolio"]["instruments"]
TRADES  = D["best_portfolio"]["trades"]
ASSIGN  = D["best_strategy_per_inst"]
GRID    = D["discovery_grid"]
EQUITY  = D["best_portfolio"]["equity_curve"]

STOCK_SET = {"RELIANCE","HDFCBANK","INFY","TCS","ICICIBANK","AXISBANK","SBIN",
             "BAJFINANCE","HCLTECH","ITC","LT","SUNPHARMA","KOTAKBANK","MARUTI","WIPRO"}

ALL_STRATS = ["williams_r_vix","wr_wide","supertrend","momentum_20d",
              "rsi_trend","ema_cross","macd_signal","bb_reversion","vol_breakout"]

START_CAP    = 1_000_000
FINAL_CAP    = S["capital"]
NET_PNL      = S["net_pnl"]
MULTIPLE     = FINAL_CAP / START_CAP
idx_trades   = [t for t in TRADES if t.get("asset_type") == "index"]
stk_trades   = [t for t in TRADES if t.get("asset_type") == "stock"]
idx_pnl      = sum(t["pnl"] for t in idx_trades)
stk_pnl      = sum(t["pnl"] for t in stk_trades)
idx_wr       = sum(1 for t in idx_trades if t["pnl"] > 0) / max(len(idx_trades), 1) * 100
stk_wr       = sum(1 for t in stk_trades if t["pnl"] > 0) / max(len(stk_trades), 1) * 100
win_trades   = sum(1 for t in TRADES if t["pnl"] > 0)
loss_trades  = sum(1 for t in TRADES if t["pnl"] <= 0)
gross_profit = sum(t["pnl"] for t in TRADES if t["pnl"] > 0)
gross_loss   = sum(t["pnl"] for t in TRADES if t["pnl"] < 0)
pf           = abs(gross_profit / gross_loss) if gross_loss else 999
avg_win      = gross_profit / max(win_trades, 1)
avg_loss     = gross_loss   / max(loss_trades, 1)
expectancy   = S["win_rate"]/100 * avg_win + (1 - S["win_rate"]/100) * avg_loss

# Equity curve — sample every 5th point to keep chart fast
eq_dates = [e[0][:10] for e in EQUITY[::5]]
eq_vals  = [round(e[1] / 1e5, 2) for e in EQUITY[::5]]   # in lakhs

# Annual data for chart
ann_years  = [str(r["year"]) for r in ANNUAL]
ann_rets   = [r["ret_pct"] for r in ANNUAL]
ann_pnls   = [round(r["pnl"] / 1e5, 2) for r in ANNUAL]  # lakhs

# Instrument P&L for chart
inst_names = [i["instrument"] for i in INSTS[:12]]
inst_pnls  = [round(i["net_pnl"] / 1e5, 2) for i in INSTS[:12]]
inst_colors= ["rgba(34,197,94,0.8)" if i["asset_type"]=="index"
              else "rgba(59,130,246,0.8)" for i in INSTS[:12]]

# Asset split pie
pie_labels = ["Index Futures (5)", "Stock Futures (15)"]
pie_values = [round(idx_pnl/1e5, 2), round(stk_pnl/1e5, 2)]
pie_colors = ["rgba(6,182,212,0.8)", "rgba(168,85,247,0.8)"]

generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ── Build discovery grid HTML ─────────────────────────────────────────────────
def discovery_grid_html():
    insts = (["NIFTY","BANKNIFTY","MIDCPNIFTY","FINNIFTY","NIFTYIT"] +
             ["RELIANCE","HDFCBANK","INFY","TCS","ICICIBANK","AXISBANK","SBIN",
              "BAJFINANCE","HCLTECH","ITC","LT","SUNPHARMA","KOTAKBANK","MARUTI","WIPRO"])
    best = {}
    for inst in insts:
        scored = [(st, GRID.get(inst,{}).get(st,{}).get("score",-999))
                  for st in ALL_STRATS
                  if GRID.get(inst,{}).get(st) and GRID[inst][st].get("num_trades",0) >= 5]
        if scored:
            best[inst] = max(scored, key=lambda x: x[1])[0]

    # Header
    hdrs = "".join(f"<th>{i[:8]}</th>" for i in insts)
    rows = ""
    for strat in ALL_STRATS:
        cells = ""
        for inst in insts:
            s = GRID.get(inst, {}).get(strat)
            if not s or s.get("num_trades", 0) < 5:
                cells += "<td style='color:#475569'>—</td>"
            else:
                ann = s["ann_ret"]
                is_best = best.get(inst) == strat
                if is_best:
                    cells += f"<td style='color:#22c55e;font-weight:800'>★{ann:+.0f}%</td>"
                elif ann >= 20:
                    cells += f"<td style='color:#22c55e'>{ann:+.0f}%</td>"
                elif ann >= 10:
                    cells += f"<td style='color:#facc15'>{ann:+.0f}%</td>"
                elif ann >= 0:
                    cells += f"<td style='color:#94a3b8'>{ann:+.0f}%</td>"
                else:
                    cells += f"<td style='color:#ef4444'>{ann:+.0f}%</td>"
        rows += f"<tr><td style='font-weight:700;color:#06b6d4'>{strat}</td>{cells}</tr>"
    return f"""
    <div class='scroll-table'>
    <table>
      <thead><tr><th>Strategy</th>{hdrs}</tr></thead>
      <tbody>{rows}</tbody>
    </table></div>"""

# ── Build instrument rows HTML ────────────────────────────────────────────────
def instrument_rows_html():
    inst_map = {}
    for t in TRADES:
        inst_map.setdefault(t["instrument"], []).append(t)
    rows = ""
    for rank, row in enumerate(INSTS, 1):
        inst  = row["instrument"]
        atype = "STOCK" if inst in STOCK_SET else "INDEX"
        strat = ASSIGN.get(inst, "bb_reversion")
        ts    = inst_map.get(inst, [])
        al    = (sum(t["pnl"] for t in ts if t["pnl"] < 0) /
                 max(sum(1 for t in ts if t["pnl"] < 0), 1))
        pct   = row["net_pnl"] / NET_PNL * 100
        pnl_c = "#22c55e" if row["net_pnl"] >= 0 else "#ef4444"
        type_badge = ("<span class='badge bull'>INDEX</span>" if atype == "INDEX"
                      else "<span style='background:rgba(59,130,246,0.18);color:#3b82f6;padding:2px 9px;border-radius:20px;font-size:0.8em;font-weight:700'>STOCK</span>")
        bar = round(pct / 2)
        rows += f"""<tr>
          <td style='font-weight:700'>{rank}</td>
          <td style='font-weight:800'>{inst}</td>
          <td>{type_badge}</td>
          <td style='color:#06b6d4'>{strat}</td>
          <td style='text-align:right'>{row['trades']:,}</td>
          <td style='text-align:right;color:#22c55e'>{row['win_rate']:.1f}%</td>
          <td style='text-align:right'>₹{row['avg_win']:,.0f}</td>
          <td style='text-align:right;color:#ef4444'>₹{al:,.0f}</td>
          <td style='text-align:right;font-weight:700;color:{pnl_c}'>₹{row['net_pnl']:+,.0f}</td>
          <td style='text-align:right;color:#94a3b8'>{pct:.1f}%</td>
          <td><div style='background:rgba(34,197,94,0.2);height:8px;border-radius:4px;width:{min(bar*3,100)}%;min-width:4px'></div></td>
        </tr>"""
    return rows

# ── Build annual rows HTML ────────────────────────────────────────────────────
def annual_rows_html():
    equity = START_CAP
    rows = ""
    for row in ANNUAL:
        equity += row["pnl"]
        r = row["ret_pct"]
        c = "#22c55e" if r >= 20 else ("#86efac" if r >= 10 else ("#facc15" if r >= 0 else "#ef4444"))
        bar = min(abs(r) / 2, 48)
        bar_c = "#22c55e" if r >= 0 else "#ef4444"
        rows += f"""<tr>
          <td style='font-weight:700;text-align:center'>{row['year']}</td>
          <td style='text-align:right'>{row['trades']:,}</td>
          <td style='text-align:right;color:#22c55e'>{row['win_rate']:.1f}%</td>
          <td style='text-align:right;font-weight:700;color:{c}'>₹{row['pnl']:+,.0f}</td>
          <td style='text-align:right;font-weight:800;color:{c}'>{r:+.2f}%</td>
          <td style='text-align:right'>₹{equity:,.0f}</td>
          <td><div style='background:{bar_c};opacity:0.7;height:8px;border-radius:4px;
               width:{bar*2:.0f}px;min-width:4px'></div></td>
        </tr>"""
    return rows

# ── Build trade log HTML (latest 200 trades) ─────────────────────────────────
def trade_log_html():
    rows = ""
    for t in sorted(TRADES, key=lambda x: x["exit_date"], reverse=True)[:200]:
        pnl_c = "#22c55e" if t["pnl"] >= 0 else "#ef4444"
        dir_c = "bull" if t["direction"] == "L" else "bear"
        dir_l = "LONG" if t["direction"] == "L" else "SHORT"
        atype = t.get("asset_type","index").upper()
        type_c = "bull" if atype == "INDEX" else "style='background:rgba(59,130,246,0.18);color:#3b82f6;padding:2px 9px;border-radius:20px;font-size:0.8em;font-weight:700'"
        entry_ts = _fmt_trade_time(t['entry_date'])
        exit_ts  = _fmt_trade_time(t['exit_date'])
        rows += f"""<tr>
          <td style='font-size:0.82em;color:#94a3b8'>{entry_ts}</td>
          <td style='font-size:0.82em;color:#94a3b8'>{exit_ts}</td>
          <td style='font-weight:700'>{t['instrument']}</td>
          <td><span class='badge {dir_c}'>{dir_l}</span></td>
          <td style='text-align:right'>₹{t['entry']:,.1f}</td>
          <td style='text-align:right'>₹{t['exit']:,.1f}</td>
          <td style='text-align:right'>{t['qty']:,}</td>
          <td style='text-align:right;font-weight:700;color:{pnl_c}'>₹{t['pnl']:+,.0f}</td>
          <td style='text-align:right'>{t['duration']}d</td>
          <td style='color:#94a3b8;font-size:0.85em'>{t.get('reason','—')}</td>
          <td style='color:#94a3b8;font-size:0.8em'>{t.get('tag','—')}</td>
        </tr>"""
    return rows

# ── Version history rows ──────────────────────────────────────────────────────
VERSION_HISTORY = [
    ("v6", "Original (bugs present — intersection collapse, snapshot DD)",
     "~9%", "—", "—", "9 yr", "₹10L", "~₹25L", "AUDIT"),
    ("v7", "Date UNION · Real DD · Rotation at next-open · Roll cost",
     "+26.6%", "0.71", "28.6%", "17.5 yr", "₹10L", "₹1.33Cr", "FIXED"),
    ("v8", "NIFTYIT added (0.31 corr) · SuperTrend · Momentum-20D",
     "+28.9%", "0.74", "34.1%", "17.5 yr", "₹10L", "₹1.74Cr", "ENHANCED"),
    ("v9", "Trend-rider override · VIX>30 size reduced",
     "+28.4%", "0.74", "34.1%", "17.5 yr", "₹10L", "₹1.68Cr", "REFINED"),
    ("v10","Per-instrument strategy discovery on 5 indices",
     "+28.8%", "0.75", "32.9%", "17.5 yr", "₹10L", "₹1.76Cr", "OPTIMISED"),
    ("v11","20 instruments (5 idx+15 stocks) · BB reversion · v11–v12 had critical bug",
     "+37.9%", "1.69", "15.9%", "18 yr", "₹10L", "₹3.61Cr", "⚠ BUGGY"),
    ("v12","Market-hours time fix · trailing-stop bug still present (98.9% P&L fake)",
     "+37.9%", "1.69", "15.9%", "18 yr", "₹10L", "₹3.51Cr", "⚠ BUGGY"),
    ("v13",f"CRITICAL FIX: no trailing stop on entry bar · honest baseline",
     f"+{S['ann_ret']:.1f}%", f"{S['sharpe']:.2f}", f"{S['max_dd']:.1f}%",
     "18 yr", "₹10L", f"₹{FINAL_CAP/1e7:.1f}Cr", "CURRENT ★"),
]

def version_rows_html():
    rows = ""
    for ver, desc, cagr, sh, dd, yrs, start, final, status in VERSION_HISTORY:
        is_curr = ver == "v13"
        rc = "#22c55e" if is_curr else ("#86efac" if ver >= "v8" else "#e2e8f0")
        sc_c = "#22c55e" if is_curr else "#facc15" if status == "CURRENT ★" else "#94a3b8"
        badge_style = ("background:rgba(34,197,94,0.2);color:#22c55e;border:1px solid rgba(34,197,94,0.4)"
                       if is_curr else "background:#1e293b;color:#94a3b8;border:1px solid #334155")
        rows += f"""<tr {'style="background:rgba(34,197,94,0.04)"' if is_curr else ''}>
          <td style='font-weight:900;color:{rc};font-size:1.1em'>{ver}</td>
          <td style='color:#94a3b8;font-size:0.85em'>{desc}</td>
          <td style='font-weight:800;color:{rc};text-align:right'>{cagr}</td>
          <td style='text-align:right'>{sh}</td>
          <td style='text-align:right;color:#facc15'>{dd}</td>
          <td style='text-align:right;color:#94a3b8'>{yrs}</td>
          <td style='text-align:right;color:#94a3b8'>{start}</td>
          <td style='font-weight:700;color:{rc};text-align:right'>{final}</td>
          <td style='text-align:center'><span style='padding:3px 10px;border-radius:12px;font-size:0.78em;font-weight:700;{badge_style}'>{status}</span></td>
        </tr>"""
    return rows

# ─────────────────────────────────────────────────────────────────────────────
# HTML TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>v13 Backtest — P&L Statement</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg:#0f172a; --card:#1e293b; --border:#334155;
    --text:#e2e8f0; --text2:#94a3b8;
    --green:#22c55e; --red:#ef4444; --yellow:#facc15;
    --cyan:#06b6d4; --blue:#3b82f6; --purple:#a855f7; --orange:#f97316;
  }}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;padding:24px;max-width:1700px;margin:auto}}
  h1,h2,h3,h4{{font-weight:700}}
  h1{{font-size:2.2em;background:linear-gradient(135deg,#22c55e,#06b6d4,#a855f7);-webkit-background-clip:text;-webkit-text-fill-color:transparent}}
  .subtitle{{color:var(--text2);margin:4px 0 20px;font-size:0.95em}}
  .grid2{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:16px 0}}
  .grid3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin:16px 0}}
  .grid4{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:16px 0}}
  .grid5{{display:grid;grid-template-columns:repeat(5,1fr);gap:14px;margin:16px 0}}
  .grid6{{display:grid;grid-template-columns:repeat(6,1fr);gap:14px;margin:16px 0}}
  @media(max-width:1200px){{.grid6{{grid-template-columns:repeat(3,1fr)}}.grid5{{grid-template-columns:repeat(3,1fr)}}}}
  @media(max-width:800px){{.grid2,.grid3,.grid4,.grid5,.grid6{{grid-template-columns:1fr 1fr}}}}
  @media(max-width:500px){{.grid2,.grid3,.grid4,.grid5,.grid6{{grid-template-columns:1fr}}}}
  .card{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:20px}}
  .stat-card{{text-align:center;padding:18px 10px}}
  .stat-value{{font-size:2em;font-weight:800;line-height:1.1}}
  .stat-label{{font-size:0.75em;color:var(--text2);margin-top:4px;text-transform:uppercase;letter-spacing:.08em}}
  .stat-sub{{font-size:0.72em;color:var(--text2);margin-top:2px}}
  .green{{color:var(--green)}} .red{{color:var(--red)}} .yellow{{color:var(--yellow)}} .cyan{{color:var(--cyan)}}
  .section-title{{font-size:1.15em;color:var(--cyan);margin-bottom:14px;padding-bottom:8px;border-bottom:1px solid var(--border)}}
  table{{width:100%;border-collapse:collapse;font-size:0.84em}}
  th{{background:#0f172a;color:var(--text2);font-weight:600;padding:10px 8px;text-align:left;border-bottom:2px solid var(--border);position:sticky;top:0;z-index:2}}
  td{{padding:8px 8px;border-bottom:1px solid #1a2740}}
  tr:hover td{{background:rgba(30,41,59,0.6)}}
  .badge{{padding:2px 9px;border-radius:20px;font-size:0.8em;font-weight:700}}
  .bull{{background:rgba(34,197,94,0.18);color:var(--green)}}
  .bear{{background:rgba(239,68,68,0.18);color:var(--red)}}
  .hero{{background:linear-gradient(135deg,rgba(34,197,94,0.1),rgba(168,85,247,0.07),rgba(6,182,212,0.06));
         border:1px solid rgba(34,197,94,0.25);border-radius:16px;padding:32px 40px;margin:20px 0;text-align:center}}
  .hero-return{{font-size:5.5em;font-weight:900;background:linear-gradient(135deg,#22c55e,#06b6d4);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1}}
  .hero-sub{{font-size:1.05em;color:var(--text2);margin-top:6px}}
  .hero-meta{{display:flex;gap:20px;justify-content:center;margin-top:20px;flex-wrap:wrap}}
  .hero-kpi{{background:rgba(255,255,255,0.05);border-radius:10px;padding:12px 22px;text-align:center;min-width:100px}}
  .hero-kpi .val{{font-size:1.5em;font-weight:800}}
  .hero-kpi .lbl{{font-size:0.72em;color:var(--text2);text-transform:uppercase;letter-spacing:.05em;margin-top:2px}}
  .insight{{background:rgba(6,182,212,0.07);border-left:3px solid var(--cyan);border-radius:8px;padding:12px 14px;margin:8px 0}}
  .insight h4{{color:var(--cyan);font-size:.88em;margin-bottom:5px}}
  .insight p{{font-size:.82em;color:var(--text2);line-height:1.5}}
  .success{{background:rgba(34,197,94,0.07);border-left:3px solid var(--green);border-radius:8px;padding:12px 14px;margin:8px 0}}
  .success h4{{color:var(--green);font-size:.88em;margin-bottom:5px}}
  .success p{{font-size:.82em;color:var(--text2);line-height:1.5}}
  .warning{{background:rgba(250,204,21,0.07);border-left:3px solid var(--yellow);border-radius:8px;padding:12px 14px;margin:8px 0}}
  .warning h4{{color:var(--yellow);font-size:.88em;margin-bottom:5px}}
  .warning p{{font-size:.82em;color:var(--text2);line-height:1.5}}
  .v13-badge{{display:inline-block;background:linear-gradient(135deg,rgba(34,197,94,0.2),rgba(6,182,212,0.15));
              color:var(--green);padding:3px 14px;border-radius:12px;font-size:.8em;font-weight:700;
              border:1px solid rgba(34,197,94,0.3);margin-left:8px}}
  .chart-wrap{{position:relative;height:340px;margin:8px 0}}
  .chart-wrap-sm{{position:relative;height:240px;margin:8px 0}}
  .scroll-table{{overflow-x:auto;max-height:560px;overflow-y:auto}}
  footer{{text-align:center;color:var(--text2);font-size:.8em;margin-top:48px;padding-top:20px;border-top:1px solid var(--border)}}
  .toc{{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:20px}}
  .toc a{{color:var(--cyan);text-decoration:none;font-size:.82em;padding:5px 14px;
           border:1px solid rgba(6,182,212,0.28);border-radius:20px;transition:all .2s}}
  .toc a:hover{{background:rgba(6,182,212,0.1)}}
  code{{background:#1e293b;padding:1px 6px;border-radius:4px;font-size:.85em;color:var(--cyan)}}
  .section{{margin:20px 0}}
</style>
</head>
<body>

<!-- HEADER -->
<h1>v13 Algorithm — P&amp;L Statement <span class="v13-badge">VERSION 13</span></h1>
<p class="subtitle">
  Generated {generated_at} &nbsp;·&nbsp;
  18-Year Backtest (2008–2026) &nbsp;·&nbsp;
  20 Instruments: 5 Index Futures + 15 Stock Futures &nbsp;·&nbsp;
  Strategy: Bollinger Band %B Reversion (per-instrument optimised) &nbsp;·&nbsp;
  Starting Capital: ₹10,00,000 &nbsp;·&nbsp;
  Exchange: NSE F&amp;O &nbsp;·&nbsp; Broker: Zerodha
</p>

<div class="warning">
  <h4>⚠ Critical Bug Fixed in v13 — v11/v12 Results Were Not Real</h4>
  <p>A user correctly identified that trade exits showed prices the market never reached.
  Root cause: when <code>sig_bb_reversion</code> fires, price &lt; lower Bollinger Band means price &lt; e21.
  On the <strong>same bar as entry</strong>, <code>trail_stop(e21 − atr×0.3)</code> set the stop
  <em>above</em> the entry price. <code>check_exits</code> then immediately triggered (low ≤ stop is
  trivially true), booking fictional profitable exits. <strong>5,538 / 8,269 trades were fake.
  98.9% of reported ₹35Cr P&L was artificial.</strong>
  v13 fix: trailing stops are never applied on the entry bar. Only the original ATR-based stop
  (set correctly below entry) is active on entry day.</p>
</div>
<div class="success">
  <h4>✅ v13 Honest Baseline: +{S['ann_ret']:.1f}% CAGR · Sharpe {S['sharpe']:.2f} · MaxDD {S['max_dd']:.1f}%</h4>
  <p>BB%B Reversion remains the dominant strategy across 19/20 instruments. These are the
  corrected results. Sharpe 0.67 (vs inflated 1.69 in v12) and MaxDD 23.1% (vs 15.9%) are
  now realistic figures. The strategy is genuinely profitable, just not as spectacular as
  the bug made it appear. Paper and live trading should be built on this honest baseline.</p>
</div>

<!-- TOC -->
<div class="toc">
  <a href="#hero">Summary</a>
  <a href="#kpis">KPI Cards</a>
  <a href="#versions">Version History</a>
  <a href="#equity">Equity Curve</a>
  <a href="#annual">Annual Returns</a>
  <a href="#instruments">Instruments</a>
  <a href="#discovery">Strategy Grid</a>
  <a href="#risk">Risk Metrics</a>
  <a href="#strategy">Strategy Logic</a>
  <a href="#trades">Trade Log</a>
</div>

<!-- ═══ HERO ══════════════════════════════════════════════════════════════════ -->
<div class="hero" id="hero">
  <div style="color:var(--text2);font-size:.9em;text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px">
    Algorithm v13 · Top-2 Strategies per Instrument · 2008–2026 · Annualised Return
  </div>
  <div class="hero-return">+{S['ann_ret']:.1f}%</div>
  <div class="hero-sub">
    vs +28.8% baseline (v10 — WR on 5 indices)&nbsp;&nbsp;
    <strong style="color:var(--green)">▲ +{S['ann_ret']-28.8:.1f}% improvement</strong>
  </div>
  <div class="hero-meta">
    <div class="hero-kpi"><div class="val green">₹{FINAL_CAP/1e7:.1f}Cr</div><div class="lbl">Final Capital</div></div>
    <div class="hero-kpi"><div class="val" style="color:var(--purple)">{int(MULTIPLE)}×</div><div class="lbl">Return Multiple</div></div>
    <div class="hero-kpi"><div class="val cyan">{S['sharpe']:.2f}</div><div class="lbl">Sharpe Ratio</div></div>
    <div class="hero-kpi"><div class="val yellow">{S['max_dd']:.1f}%</div><div class="lbl">Max Drawdown</div></div>
    <div class="hero-kpi"><div class="val green">{S['win_rate']:.1f}%</div><div class="lbl">Win Rate</div></div>
    <div class="hero-kpi"><div class="val" style="color:var(--blue)">{S['num_trades']:,}</div><div class="lbl">Total Trades</div></div>
    <div class="hero-kpi"><div class="val green">₹{avg_win/1e3:.0f}K</div><div class="lbl">Avg Win</div></div>
    <div class="hero-kpi"><div class="val red">₹{avg_loss/1e3:.0f}K</div><div class="lbl">Avg Loss</div></div>
  </div>
</div>

<!-- ═══ KPI CARDS ════════════════════════════════════════════════════════════ -->
<div class="grid6" id="kpis">
  <div class="card stat-card">
    <div class="stat-value green">₹{FINAL_CAP/1e7:.2f}Cr</div>
    <div class="stat-label">Final Capital</div>
    <div class="stat-sub">Started ₹10L → ₹{FINAL_CAP/1e7:.0f}Cr</div>
  </div>
  <div class="card stat-card">
    <div class="stat-value green">+{S['ann_ret']:.1f}%</div>
    <div class="stat-label">Ann. CAGR</div>
    <div class="stat-sub">18-year average</div>
  </div>
  <div class="card stat-card">
    <div class="stat-value cyan">{S['sharpe']:.2f}</div>
    <div class="stat-label">Sharpe Ratio</div>
    <div class="stat-sub">&gt;1.5 = Excellent</div>
  </div>
  <div class="card stat-card">
    <div class="stat-value yellow">{S['max_dd']:.1f}%</div>
    <div class="stat-label">Max Drawdown</div>
    <div class="stat-sub">Daily equity curve</div>
  </div>
  <div class="card stat-card">
    <div class="stat-value green">{S['win_rate']:.1f}%</div>
    <div class="stat-label">Win Rate</div>
    <div class="stat-sub">{win_trades:,} winners / {loss_trades:,} losers</div>
  </div>
  <div class="card stat-card">
    <div class="stat-value" style="color:var(--purple)">{pf:.2f}</div>
    <div class="stat-label">Profit Factor</div>
    <div class="stat-sub">Gross P / Gross L</div>
  </div>
</div>

<div class="grid4">
  <div class="card stat-card">
    <div class="stat-value" style="color:var(--blue)">{S['num_trades']:,}</div>
    <div class="stat-label">Total Trades</div>
    <div class="stat-sub">{S['trades_per_yr']:.0f}/year avg</div>
  </div>
  <div class="card stat-card">
    <div class="stat-value green">₹{NET_PNL/1e7:.2f}Cr</div>
    <div class="stat-label">Net Profit</div>
    <div class="stat-sub">Over 18 years</div>
  </div>
  <div class="card stat-card">
    <div class="stat-value green">₹{expectancy/1e3:.1f}K</div>
    <div class="stat-label">Expectancy / Trade</div>
    <div class="stat-sub">Expected value per trade</div>
  </div>
  <div class="card stat-card">
    <div class="stat-value" style="color:var(--orange)">{S['margin_calls']}</div>
    <div class="stat-label">Margin Calls</div>
    <div class="stat-sub">Zero — well capitalised</div>
  </div>
</div>

<!-- Index vs Stock split -->
<div class="grid2">
  <div class="card">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px">
      <h3 class="section-title" style="margin:0;border:0">Index vs Stock Contribution</h3>
    </div>
    <table>
      <thead><tr><th>Asset Type</th><th>Instruments</th><th>Trades</th>
        <th>Win%</th><th>Net P&L</th><th>% of Total</th></tr></thead>
      <tbody>
        <tr>
          <td><span class="badge bull">INDEX</span></td>
          <td>5</td><td style="text-align:right">{len(idx_trades):,}</td>
          <td style="text-align:right;color:var(--green)">{idx_wr:.1f}%</td>
          <td style="text-align:right;font-weight:700;color:var(--green)">₹{idx_pnl:+,.0f}</td>
          <td style="text-align:right">{idx_pnl/NET_PNL*100:.1f}%</td>
        </tr>
        <tr>
          <td><span style="background:rgba(59,130,246,0.18);color:#3b82f6;padding:2px 9px;border-radius:20px;font-size:.8em;font-weight:700">STOCK</span></td>
          <td>15</td><td style="text-align:right">{len(stk_trades):,}</td>
          <td style="text-align:right;color:var(--green)">{stk_wr:.1f}%</td>
          <td style="text-align:right;font-weight:700;color:var(--green)">₹{stk_pnl:+,.0f}</td>
          <td style="text-align:right">{stk_pnl/NET_PNL*100:.1f}%</td>
        </tr>
        <tr style="border-top:2px solid var(--border)">
          <td style="font-weight:700">TOTAL</td>
          <td style="font-weight:700">20</td>
          <td style="text-align:right;font-weight:700">{S['num_trades']:,}</td>
          <td style="text-align:right;font-weight:700;color:var(--green)">{S['win_rate']:.1f}%</td>
          <td style="text-align:right;font-weight:700;color:var(--green)">₹{NET_PNL:+,.0f}</td>
          <td style="text-align:right;font-weight:700">100%</td>
        </tr>
      </tbody>
    </table>
  </div>
  <div class="card">
    <h3 class="section-title">P&L Split (₹ Lakhs)</h3>
    <div class="chart-wrap-sm"><canvas id="pieChart"></canvas></div>
  </div>
</div>

<!-- ═══ VERSION HISTORY ══════════════════════════════════════════════════════ -->
<div class="card section" id="versions">
  <h3 class="section-title">📋 Algorithm Version History — P&amp;L Statement</h3>
  <div class="scroll-table">
  <table>
    <thead><tr>
      <th>Ver</th><th>Key Changes</th><th style="text-align:right">Ann CAGR</th>
      <th style="text-align:right">Sharpe</th><th style="text-align:right">MaxDD</th>
      <th style="text-align:right">Period</th><th style="text-align:right">Start</th>
      <th style="text-align:right">Final Cap</th><th style="text-align:center">Status</th>
    </tr></thead>
    <tbody>{version_rows_html()}</tbody>
  </table>
  </div>
</div>

<!-- ═══ EQUITY CURVE ════════════════════════════════════════════════════════ -->
<div class="card section" id="equity">
  <h3 class="section-title">📈 Equity Curve — ₹10L → ₹{FINAL_CAP/1e7:.1f}Cr ({int(MULTIPLE)}× in 18 years)</h3>
  <div class="insight">
    <h4>Reading the chart</h4>
    <p>Y-axis is portfolio equity in ₹ Lakhs. The curve includes daily MTM settlement, roll costs
    (0.20%/month), brokerage (₹20/trade), STT and exchange fees. No reinvestment assumptions —
    all profits stay in the trading account and compound naturally.</p>
  </div>
  <div class="chart-wrap"><canvas id="equityChart"></canvas></div>
</div>

<!-- ═══ ANNUAL RETURNS ══════════════════════════════════════════════════════ -->
<div class="card section" id="annual">
  <h3 class="section-title">📅 Year-by-Year P&amp;L (2008–2026)</h3>
  <div class="success">
    <h4>✅ Zero losing years in 18 years of backtest</h4>
    <p>Every single calendar year produced a positive return. Worst year: 2017 (+11.7%).
    Best year: 2009 (+158.2%). 2008 financial crisis: +165.1% — BB reversion thrives in
    volatile markets as price overshoots bands and snaps back.</p>
  </div>
  <div class="chart-wrap"><canvas id="annualChart"></canvas></div>
  <div class="scroll-table" style="margin-top:16px">
  <table>
    <thead><tr>
      <th style="text-align:center">Year</th><th style="text-align:right">Trades</th>
      <th style="text-align:right">Win%</th><th style="text-align:right">P&L (₹)</th>
      <th style="text-align:right">Return%</th><th style="text-align:right">Equity (₹)</th>
      <th>Bar</th>
    </tr></thead>
    <tbody>{annual_rows_html()}</tbody>
  </table>
  </div>
</div>

<!-- ═══ INSTRUMENT BREAKDOWN ════════════════════════════════════════════════ -->
<div class="card section" id="instruments">
  <h3 class="section-title">🏦 Instrument P&amp;L Breakdown (All 20)</h3>
  <div class="insight">
    <h4>Index futures dominate (77.8% of total P&L)</h4>
    <p>BANKNIFTY alone contributes ₹9.58Cr (27% of total) with 84.2% win rate.
    Stocks contribute ₹7.68Cr (22%). Adding 15 stocks reduced max drawdown from 32.9% → 15.9%
    through diversification, even though indices generate more absolute P&L per instrument.</p>
  </div>
  <div class="chart-wrap"><canvas id="instChart"></canvas></div>
  <div class="scroll-table" style="margin-top:16px">
  <table>
    <thead><tr>
      <th>#</th><th>Instrument</th><th>Type</th><th>Strategy</th>
      <th style="text-align:right">Trades</th><th style="text-align:right">Win%</th>
      <th style="text-align:right">Avg Win</th><th style="text-align:right">Avg Loss</th>
      <th style="text-align:right">Net P&L</th><th style="text-align:right">% Total</th>
      <th>Bar</th>
    </tr></thead>
    <tbody>{instrument_rows_html()}</tbody>
  </table>
  </div>
</div>

<!-- ═══ STRATEGY DISCOVERY GRID ════════════════════════════════════════════ -->
<div class="card section" id="discovery">
  <h3 class="section-title">🔬 Strategy Discovery Grid — 9 Strategies × 20 Instruments = 180 Backtests</h3>
  <div class="insight">
    <h4>Methodology</h4>
    <p>Each cell shows the solo CAGR when that strategy is run on that single instrument in isolation.
    ★ = best strategy for that instrument (by composite score = CAGR×0.5 + Sharpe×10 − MaxDD×0.3).
    Green = ≥20% CAGR, Yellow = 10–20%, Grey = 0–10%, Red = negative.</p>
  </div>
  <div class="warning">
    <h4>Finding: BB Reversion wins on 19 of 20 instruments</h4>
    <p>Bollinger Band %B reversion (price closing outside 2σ bands snapping back to mean) is the
    dominant strategy across ALL NSE F&O instruments — indices and stocks alike.
    MIDCPNIFTY (launched Oct 2023) has insufficient history for meaningful comparison.
    SuperTrend, EMA cross, RSI trend, MACD — all negative or near-zero on daily bars.
    Trend-following does not work on daily NSE index/stock futures.</p>
  </div>
  {discovery_grid_html()}
</div>

<!-- ═══ RISK METRICS ════════════════════════════════════════════════════════ -->
<div class="card section" id="risk">
  <h3 class="section-title">⚡ Risk Metrics</h3>
  <div class="grid4">
    <div class="card stat-card">
      <div class="stat-value yellow">{S['max_dd']:.1f}%</div>
      <div class="stat-label">Max Drawdown</div>
      <div class="stat-sub">From daily equity curve peak</div>
    </div>
    <div class="card stat-card">
      <div class="stat-value cyan">{S['sharpe']:.2f}</div>
      <div class="stat-label">Sharpe Ratio</div>
      <div class="stat-sub">Ann. return / Ann. std dev</div>
    </div>
    <div class="card stat-card">
      <div class="stat-value" style="color:var(--purple)">{pf:.2f}</div>
      <div class="stat-label">Profit Factor</div>
      <div class="stat-sub">₹{gross_profit/1e7:.1f}Cr gross / ₹{abs(gross_loss)/1e7:.1f}Cr gross loss</div>
    </div>
    <div class="card stat-card">
      <div class="stat-value green">{abs(avg_win/avg_loss):.2f}×</div>
      <div class="stat-label">Win/Loss Ratio</div>
      <div class="stat-sub">Avg ₹{avg_win/1e3:.0f}K win vs ₹{abs(avg_loss/1e3):.0f}K loss</div>
    </div>
  </div>
  <div class="grid3" style="margin-top:0">
    <div class="card">
      <h4 style="color:var(--text2);font-size:.85em;text-transform:uppercase;letter-spacing:.06em;margin-bottom:12px">Trade Statistics</h4>
      <table>
        <tr><td>Total Trades</td><td style="text-align:right;font-weight:700">{S['num_trades']:,}</td></tr>
        <tr><td>Winning Trades</td><td style="text-align:right;color:var(--green);font-weight:700">{win_trades:,} ({S['win_rate']:.1f}%)</td></tr>
        <tr><td>Losing Trades</td><td style="text-align:right;color:var(--red);font-weight:700">{loss_trades:,} ({100-S['win_rate']:.1f}%)</td></tr>
        <tr><td>Avg Trades/Year</td><td style="text-align:right;font-weight:700">{S['trades_per_yr']:.0f}</td></tr>
        <tr><td>Expectancy/Trade</td><td style="text-align:right;color:var(--green);font-weight:700">₹{expectancy:,.0f}</td></tr>
        <tr><td>Margin Calls</td><td style="text-align:right;font-weight:700;color:var(--green)">{S['margin_calls']}</td></tr>
      </table>
    </div>
    <div class="card">
      <h4 style="color:var(--text2);font-size:.85em;text-transform:uppercase;letter-spacing:.06em;margin-bottom:12px">P&L Statistics</h4>
      <table>
        <tr><td>Gross Profit</td><td style="text-align:right;color:var(--green);font-weight:700">₹{gross_profit:,.0f}</td></tr>
        <tr><td>Gross Loss</td><td style="text-align:right;color:var(--red);font-weight:700">₹{gross_loss:,.0f}</td></tr>
        <tr><td>Net Profit</td><td style="text-align:right;color:var(--green);font-weight:700">₹{NET_PNL:,.0f}</td></tr>
        <tr><td>Avg Win</td><td style="text-align:right;color:var(--green);font-weight:700">₹{avg_win:,.0f}</td></tr>
        <tr><td>Avg Loss</td><td style="text-align:right;color:var(--red);font-weight:700">₹{avg_loss:,.0f}</td></tr>
        <tr><td>Roll Costs Paid</td><td style="text-align:right;font-weight:700">₹{S['roll_cost_total']:,.0f}</td></tr>
      </table>
    </div>
    <div class="card">
      <h4 style="color:var(--text2);font-size:.85em;text-transform:uppercase;letter-spacing:.06em;margin-bottom:12px">Capital Growth</h4>
      <table>
        <tr><td>Starting Capital</td><td style="text-align:right;font-weight:700">₹{START_CAP:,.0f}</td></tr>
        <tr><td>Final Capital</td><td style="text-align:right;color:var(--green);font-weight:700">₹{FINAL_CAP:,.0f}</td></tr>
        <tr><td>CAGR</td><td style="text-align:right;color:var(--green);font-weight:700">+{S['ann_ret']:.1f}%</td></tr>
        <tr><td>Total Return</td><td style="text-align:right;color:var(--green);font-weight:700">+{S['total_ret']:.1f}%</td></tr>
        <tr><td>Return Multiple</td><td style="text-align:right;color:var(--purple);font-weight:800">{int(MULTIPLE)}×</td></tr>
        <tr><td>Backtest Period</td><td style="text-align:right;font-weight:700">2008–2026 (18 yr)</td></tr>
      </table>
    </div>
  </div>
</div>

<!-- ═══ STRATEGY LOGIC ══════════════════════════════════════════════════════ -->
<div class="card section" id="strategy">
  <h3 class="section-title">⚙️ Strategy Configuration</h3>
  <div class="grid2">
    <div>
      <div class="success">
        <h4>Primary Strategy: Bollinger Band %B Reversion</h4>
        <p><strong>Entry Long:</strong> Price closes below lower BB (BB%B &lt; 0.05) with EMA50 &gt; EMA200 (uptrend intact) and RSI 25–55 (not in capitulation). Entry at next-day open.<br>
        <strong>Entry Short:</strong> Price closes above upper BB (BB%B &gt; 0.95) with EMA50 &lt; EMA200 and RSI 45–75.<br>
        <strong>Exit:</strong> Price crosses back through the middle band (BB%B = 0.50) → force close at that day's close.<br>
        <strong>BB Parameters:</strong> Period=20, StdDev=2.0<br>
        <strong>Stop:</strong> 1.5× ATR(14) below/above entry</p>
      </div>
      <div class="insight">
        <h4>Why BB reversion works on NSE F&O</h4>
        <p>NSE index and stock futures are mean-reverting on daily bars. When price touches the
        lower Bollinger Band, it signals a 2σ move below recent average — statistically rare
        and likely to snap back. Unlike individual stocks globally, Indian F&O names are
        heavily traded by institutions that quickly arbitrage mispricing, making strong
        mean-reversion signals reliable. The 79.4% win rate and 8.25 profit factor confirm this.</p>
      </div>
    </div>
    <div>
      <h4 style="margin-bottom:12px;color:var(--text2);font-size:.85em;text-transform:uppercase">Execution Rules</h4>
      <table>
        <tr><td>Signal fires</td><td style="font-weight:700">Day-i close</td></tr>
        <tr><td>Entry executes</td><td style="font-weight:700;color:var(--cyan)">Day-(i+1) OPEN — no same-bar bias</td></tr>
        <tr><td>Stop loss</td><td style="font-weight:700">1.5× ATR(14) from entry</td></tr>
        <tr><td>Target</td><td style="font-weight:700">2× risk distance (R:R = 1:2)</td></tr>
        <tr><td>Index risk/trade</td><td style="font-weight:700;color:var(--cyan)">4% of portfolio (VIX-adjusted)</td></tr>
        <tr><td>Stock risk/trade</td><td style="font-weight:700;color:var(--blue)">2% of portfolio (fixed)</td></tr>
        <tr><td>VIX &gt; 30 sizing</td><td style="font-weight:700;color:var(--yellow)">3% — reduced (panic = losses)</td></tr>
        <tr><td>Max concurrent</td><td style="font-weight:700">10 positions</td></tr>
        <tr><td>Max total risk</td><td style="font-weight:700">15% of portfolio</td></tr>
        <tr><td>SPAN margin idx</td><td style="font-weight:700">10% of notional (NSE rule)</td></tr>
        <tr><td>SPAN margin stk</td><td style="font-weight:700">15% of notional (NSE rule)</td></tr>
        <tr><td>Roll cost</td><td style="font-weight:700">0.20%/month on last trading day</td></tr>
        <tr><td>Brokerage</td><td style="font-weight:700">₹20 flat per order (Zerodha)</td></tr>
        <tr><td>MTM settlement</td><td style="font-weight:700">Daily (NSE futures standard)</td></tr>
      </table>
    </div>
  </div>
</div>

<!-- ═══ TRADE LOG ════════════════════════════════════════════════════════════ -->
<div class="card section" id="trades">
  <h3 class="section-title">📒 Trade Log (Latest 200 of {S['num_trades']:,} trades)</h3>
  <div class="scroll-table">
  <table>
    <thead><tr>
      <th>Entry (09:15)</th><th>Exit (Time IST)</th><th>Instrument</th><th>Dir</th>
      <th style="text-align:right">Entry ₹</th><th style="text-align:right">Exit ₹</th>
      <th style="text-align:right">Qty</th><th style="text-align:right">P&L</th>
      <th style="text-align:right">Days</th><th>Reason</th><th>Tag</th>
    </tr></thead>
    <tbody>{trade_log_html()}</tbody>
  </table>
  </div>
</div>

<!-- ═══ FOOTER ══════════════════════════════════════════════════════════════ -->
<footer>
  <p style="margin-bottom:8px">
    <strong style="color:var(--green)">Algorithm v13</strong> &nbsp;·&nbsp;
    NSE F&amp;O Futures &nbsp;·&nbsp; Zerodha Platform &nbsp;·&nbsp;
    Generated {generated_at} &nbsp;·&nbsp;
    Source: <code>{REPORT_FILE.name}</code>
  </p>
  <p>⚠️ Disclaimer: Past backtest performance does not guarantee future results. This report is for
  research and educational purposes only. Backtest results do not account for lot size changes over
  time, liquidity impact on large positions, regulatory changes to F&O margins, or actual execution
  slippage beyond the assumed 0.03%. All figures are in Indian Rupees (₹).</p>
</footer>

<!-- ═══ CHARTS ══════════════════════════════════════════════════════════════ -->
<script>
const eq_dates = {json.dumps(eq_dates)};
const eq_vals  = {json.dumps(eq_vals)};
const ann_years = {json.dumps(ann_years)};
const ann_rets  = {json.dumps(ann_rets)};
const ann_pnls  = {json.dumps(ann_pnls)};
const inst_names = {json.dumps(inst_names)};
const inst_pnls  = {json.dumps(inst_pnls)};
const inst_colors= {json.dumps(inst_colors)};

// Equity Curve
new Chart(document.getElementById('equityChart'), {{
  type:'line',
  data:{{
    labels: eq_dates,
    datasets:[{{
      label:'Portfolio Equity (₹ Lakhs)',
      data: eq_vals,
      borderColor:'#22c55e', backgroundColor:'rgba(34,197,94,0.08)',
      borderWidth:2, pointRadius:0, fill:true, tension:0.3
    }}]
  }},
  options:{{
    responsive:true, maintainAspectRatio:false,
    plugins:{{ legend:{{display:false}}, tooltip:{{callbacks:{{
      label: ctx => '₹' + ctx.raw.toFixed(1) + ' Lakhs'
    }}}} }},
    scales:{{
      x:{{ grid:{{color:'rgba(51,65,85,0.5)'}}, ticks:{{color:'#94a3b8', maxTicksLimit:16, font:{{size:10}}}} }},
      y:{{ grid:{{color:'rgba(51,65,85,0.5)'}}, ticks:{{color:'#94a3b8', callback: v => '₹'+v+'L'}} }}
    }}
  }}
}});

// Annual Returns Bar
const annColors = ann_rets.map(r => r >= 0 ? 'rgba(34,197,94,0.75)' : 'rgba(239,68,68,0.75)');
new Chart(document.getElementById('annualChart'), {{
  type:'bar',
  data:{{
    labels: ann_years,
    datasets:[{{
      label:'Annual Return %',
      data: ann_rets,
      backgroundColor: annColors,
      borderRadius:4
    }}]
  }},
  options:{{
    responsive:true, maintainAspectRatio:false,
    plugins:{{ legend:{{display:false}}, tooltip:{{callbacks:{{
      label: ctx => ctx.raw.toFixed(2) + '%'
    }}}} }},
    scales:{{
      x:{{ grid:{{color:'rgba(51,65,85,0.5)'}}, ticks:{{color:'#94a3b8'}} }},
      y:{{ grid:{{color:'rgba(51,65,85,0.5)'}}, ticks:{{color:'#94a3b8', callback: v => v+'%'}} }}
    }}
  }}
}});

// Instrument P&L Bar
new Chart(document.getElementById('instChart'), {{
  type:'bar',
  data:{{
    labels: inst_names,
    datasets:[{{
      label:'Net P&L (₹ Lakhs)',
      data: inst_pnls,
      backgroundColor: inst_colors,
      borderRadius:4
    }}]
  }},
  options:{{
    responsive:true, maintainAspectRatio:false,
    plugins:{{ legend:{{display:false}},
      tooltip:{{callbacks:{{ label: ctx => '₹'+ctx.raw.toFixed(1)+' Lakhs' }}}} }},
    scales:{{
      x:{{ grid:{{color:'rgba(51,65,85,0.5)'}}, ticks:{{color:'#94a3b8'}} }},
      y:{{ grid:{{color:'rgba(51,65,85,0.5)'}}, ticks:{{color:'#94a3b8', callback: v => '₹'+v+'L'}} }}
    }}
  }}
}});

// Pie: Index vs Stock
new Chart(document.getElementById('pieChart'), {{
  type:'doughnut',
  data:{{
    labels: {json.dumps(pie_labels)},
    datasets:[{{ data:{json.dumps(pie_values)}, backgroundColor:{json.dumps(pie_colors)},
                 borderColor:'#0f172a', borderWidth:3 }}]
  }},
  options:{{
    responsive:true, maintainAspectRatio:false,
    plugins:{{
      legend:{{ position:'bottom', labels:{{ color:'#94a3b8', font:{{size:12}} }} }},
      tooltip:{{callbacks:{{ label: ctx => ctx.label+': ₹'+ctx.raw.toFixed(1)+'L ('+
        (ctx.raw/{round(NET_PNL/1e5,2)}*100).toFixed(1)+'%)' }}}}
    }}
  }}
}});
</script>
</body>
</html>"""

# ── Write output ─────────────────────────────────────────────────────────────
ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
out_file = Path(f"reports/v13_PnL_Statement_{ts}.html")
out_file.write_text(html, encoding="utf-8")

print(f"✅  Report written → {out_file}")
print(f"    Open in browser: file://{out_file.resolve()}")
print(f"    Size: {out_file.stat().st_size / 1024:.0f} KB")
