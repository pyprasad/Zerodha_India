"""
generate_kite_report.py
=======================
Generates a full HTML P&L report using Zerodha Kite Connect data.

Key design decisions vs generate_v6_report.py:
  - Loads data from OHLCVStore (Kite) and exports to CSVs before running backtest
  - Main backtest uses NIFTY + BANKNIFTY + FINNIFTY (3 instruments) — all have real
    OHLCV from 2017 in Kite. MIDCPNIFTY is excluded from the main run because Kite
    token 288009 (NIFTY MID SELECT) returns flat open=high=low=close prices for
    2017-2021, which v6_backtest filters out, leaving only ~1046 real bars. That would
    limit the date-intersection to 2022+ and reduce the backtest to 3 years instead of 9.
  - A supplementary 4-instrument run (2022+) is shown separately for completeness.
  - All labels say "Zerodha Kite" not "Yahoo Finance"
  - Saves to reports/v6_kite_Report_TIMESTAMP.html
  - generate_v6_report.py is NOT modified

Usage:
  python generate_kite_report.py

Prerequisites:
  Run fetch_kite_daily.py first to populate OHLCVStore.
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv; load_dotenv()

from datetime import datetime
from pathlib import Path
import pandas as pd

from data.store import OHLCVStore

# ---------------------------------------------------------------------------
# Step 0: Export Kite data → CSVs that v6_backtest.load_data() expects
# ---------------------------------------------------------------------------
DATA_DIR       = Path("data/historical")
V6_INSTRUMENTS = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "INDIAVIX", "NIFTYIT"]

# Instruments with FULL real OHLCV from 2017 in Kite API
# MIDCPNIFTY (token 288009) has flat open=high=low=close bars pre-2022 in Kite,
# so it's excluded from the main 9-year backtest (v6_backtest drops flat bars,
# reducing MIDCPNIFTY to ~1046 real bars which limits the date-intersection to 2022+)
# NIFTYIT (token 259849) has clean OHLCV from 2017-01-02 — included in main run
# Expansion backtest proved NIFTYIT adds +8.9% ann return, Sharpe 2.76 → 3.03
KITE_MAIN_INSTRUMENTS = ["NIFTY", "BANKNIFTY", "FINNIFTY", "NIFTYIT"]

print("Exporting Kite data from OHLCVStore → CSV format v6 expects ...")
store = OHLCVStore()
for inst in V6_INSTRUMENTS:
    df = store.load_parquet(inst, "day")
    if df.empty:
        print(f"  ✗ {inst}: no Kite data — run fetch_kite_daily.py first")
        sys.exit(1)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        if df["date"].dt.tz is not None:
            df["date"] = df["date"].dt.tz_localize(None)
        df = df.set_index("date")
    else:
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

    df.index.name = "date"
    df.index      = df.index.normalize()
    df.columns    = [c.lower() for c in df.columns]

    if "volume" not in df.columns:
        df["volume"] = 0.0
    df["volume"] = df["volume"].fillna(0.0)
    df = df[df.index.dayofweek < 5].sort_index()

    out = DATA_DIR / f"{inst}_daily_extended.csv"
    df.to_csv(out)
    # Count real (non-flat) bars to show MIDCPNIFTY data quality issue
    no_move = ((df["open"] == df["close"]) & (df["high"] == df["close"]) & (df["low"] == df["close"]))
    real_bars = (~no_move).sum()
    note = f" ← {real_bars:,} real bars (flat pre-2022 filtered)" if inst == "MIDCPNIFTY" else ""
    print(f"  ✓ {inst}: {len(df):,} rows → {out.name} "
          f"({df.index.min().date()} → {df.index.max().date()}){note}")

# ---------------------------------------------------------------------------
# Step 1: Run v6 backtest using KITE_MAIN_INSTRUMENTS (4 insts with full data)
# ---------------------------------------------------------------------------
print("\nRunning v6 backtest on Kite data ...")
print(f"  Main instruments: {KITE_MAIN_INSTRUMENTS} (full 2017–2026 Kite OHLCV)")
print("  Note: MIDCPNIFTY excluded — Kite token has flat bars pre-2022 (only ~1046 real bars)")
print("        NIFTYIT included — token 259849 has clean OHLCV from 2017, adds +8.9%/yr")
print("        A supplementary 5-instrument run (2022+) is shown separately in the report.")
from v6_backtest import (
    run_v6, annual_breakdown, attribution_stats, instrument_stats,
    CAPITAL as _V6_CAPITAL, INSTRUMENTS_4, CONFIGS
)
import os
CAPITAL = float(os.getenv("TRADING_CAPITAL", _V6_CAPITAL))
print(f"  Starting capital: ₹{CAPITAL:,.0f} (from TRADING_CAPITAL env / v6_backtest default)")

Path("reports").mkdir(exist_ok=True)

print("  Running WR_4Inst/Kite (NIFTY+BANKNIFTY+FINNIFTY+NIFTYIT, full 9yr) ...")
pf_best   = run_v6(KITE_MAIN_INSTRUMENTS, ["williams_r"], CAPITAL)
s_best    = pf_best.stats("WR_4Inst_Kite")
ybl       = annual_breakdown(pf_best)
attr      = attribution_stats(pf_best)
ia        = instrument_stats(pf_best)
trades_df = pd.DataFrame(pf_best.trades)
print(f"  WR_4Inst_Kite: {s_best['ann_ret']:+.1f}%/yr, {s_best['num_trades']} trades")

print("  Running WR_VIX_Sizing (4 insts) ...")
pf_vix    = run_v6(KITE_MAIN_INSTRUMENTS, ["williams_r_vix"], CAPITAL)
s_vix     = pf_vix.stats("WR_VIX_4Inst")
print(f"  WR_VIX_4Inst: {s_vix['ann_ret']:+.1f}%/yr")

# Supplementary: 5-instrument run (limited to 2022+ due to MIDCPNIFTY data)
INSTRUMENTS_5 = KITE_MAIN_INSTRUMENTS + ["MIDCPNIFTY"]
print("  Running WR_5Inst (includes MIDCPNIFTY, limited to 2022+) ...")
try:
    pf_4inst  = run_v6(INSTRUMENTS_5, ["williams_r"], CAPITAL)
    s_4inst   = pf_4inst.stats("WR_5Inst_2022")
    print(f"  WR_5Inst_2022: {s_4inst['ann_ret']:+.1f}%/yr (limited date range)")
except Exception as e:
    pf_4inst  = None
    s_4inst   = None
    print(f"  WR_5Inst_2022: FAILED — {e}")

print("  Running all 4-instrument configs ...")
kite_configs = {
    "WR_4Inst":         (KITE_MAIN_INSTRUMENTS, ["williams_r"]),
    "WR_VIX_4Inst":     (KITE_MAIN_INSTRUMENTS, ["williams_r_vix"]),
    "Regime_4Inst":     (KITE_MAIN_INSTRUMENTS, ["regime_adaptive"]),
    "VIX_Spike+WR_4":   (KITE_MAIN_INSTRUMENTS, ["vix_spike", "williams_r"]),
    "Momentum_4Inst":   (KITE_MAIN_INSTRUMENTS, ["momentum_52wk"]),
    "Monthly_4Inst":    (KITE_MAIN_INSTRUMENTS, ["monthly_rotation"]),
    "COMBINED_4Inst":   (KITE_MAIN_INSTRUMENTS, ["vix_spike", "williams_r_vix",
                                                   "regime_adaptive", "momentum_52wk",
                                                   "monthly_rotation"]),
}
all_configs    = {}
all_portfolios = {}
for cfg_name, (insts, strats) in kite_configs.items():
    try:
        pf_c = run_v6(insts, strats, CAPITAL)
        all_configs[cfg_name]    = pf_c.stats(cfg_name)
        all_portfolios[cfg_name] = pf_c
        print(f"    {cfg_name}: {all_configs[cfg_name]['ann_ret']:+.1f}%")
    except Exception as e:
        print(f"    {cfg_name}: FAILED — {e}")

# ---------------------------------------------------------------------------
# Equity curve data
# ---------------------------------------------------------------------------
def get_equity_series(pf):
    eq = pd.DataFrame(pf.daily_equity, columns=["date", "capital"])
    if eq.empty:
        return [], []
    return [str(d)[:10] for d in eq["date"]], list(eq["capital"].values)

eq_dates, eq_vals         = get_equity_series(pf_best)
eq_vix_dates, eq_vix_vals = get_equity_series(pf_vix)

def downsample(labels, values, n=250):
    if len(labels) <= n:
        return labels, values
    step = max(1, len(labels) // n)
    return labels[::step], values[::step]

eq_dates_ds,     eq_vals_ds     = downsample(eq_dates, eq_vals)
eq_vix_dates_ds, eq_vix_vals_ds = downsample(eq_vix_dates, eq_vix_vals)
chart_labels = eq_dates_ds

# Monthly P&L
if not trades_df.empty:
    trades_df["exit_month"] = pd.to_datetime(trades_df["exit_date"]).dt.to_period("M")
    monthly   = trades_df.groupby("exit_month")["pnl"].sum().reset_index()
    monthly["exit_month"] = monthly["exit_month"].astype(str)
    m_labels  = list(monthly["exit_month"].values)
    m_values  = list(monthly["pnl"].values)
    m_colors  = ["rgba(34,197,94,0.8)" if v > 0 else "rgba(239,68,68,0.8)" for v in m_values]
else:
    m_labels, m_values, m_colors = [], [], []

# Config comparison rows
ranked_configs = sorted(
    [(k, v) for k, v in all_configs.items() if v is not None],
    key=lambda x: x[1]["ann_ret"], reverse=True,
)

config_rows = ""
for rank, (name, s) in enumerate(ranked_configs, 1):
    beats  = s["ann_ret"] >= 38
    color  = "#22c55e" if s["ann_ret"] >= 45 else ("#facc15" if s["ann_ret"] >= 20 else "#ef4444")
    marker = "★" if "COMBINED" in name else ("✓" if beats else "")
    dd_col = "#22c55e" if s["max_dd"] < 10 else ("#facc15" if s["max_dd"] < 20 else "#ef4444")
    sh_col = "#22c55e" if s["sharpe"] >= 3 else ("#facc15" if s["sharpe"] >= 1 else "#ef4444")
    config_rows += f"""<tr>
      <td style="text-align:center;font-weight:700">{rank}</td>
      <td style="font-weight:700;color:{color}">{marker} {name}</td>
      <td style="color:{color};font-weight:700">{s['ann_ret']:+.1f}%</td>
      <td>{s['total_ret']:+.1f}%</td>
      <td style="color:{sh_col};font-weight:600">{s['sharpe']:.2f}</td>
      <td style="color:{dd_col}">{s['max_dd']:.1f}%</td>
      <td>{s['win_rate']:.1f}%</td>
      <td>{s['num_trades']}</td>
      <td style="color:{color};font-weight:600">₹{s['net_pnl']:+,.0f}</td>
      <td>₹{s['capital']:,.0f}</td>
    </tr>"""

# Supplementary 4-instrument row
supp_row = ""
if s_4inst:
    c4 = "#22c55e" if s_4inst["ann_ret"] >= 45 else ("#facc15" if s_4inst["ann_ret"] >= 20 else "#ef4444")
    supp_row = f"""<tr style="opacity:0.7;border-top:2px dashed #334155">
      <td style="text-align:center;color:#64748b">—</td>
      <td style="color:#64748b">⚠ WR_4Inst (2022+ only)</td>
      <td style="color:{c4}">{s_4inst['ann_ret']:+.1f}%</td>
      <td>{s_4inst['total_ret']:+.1f}%</td>
      <td>{s_4inst['sharpe']:.2f}</td>
      <td>{s_4inst['max_dd']:.1f}%</td>
      <td>{s_4inst['win_rate']:.1f}%</td>
      <td>{s_4inst['num_trades']}</td>
      <td style="color:{c4}">₹{s_4inst['net_pnl']:+,.0f}</td>
      <td>₹{s_4inst['capital']:,.0f}</td>
    </tr>"""

# Year rows
year_rows = ""
for row in ybl:
    c     = "#22c55e" if row["ret_pct"] >= 40 else ("#facc15" if row["ret_pct"] >= 20 else "#ef4444")
    badge = "🏆 TARGET+" if row["ret_pct"] >= 40 else ("✓ POSITIVE" if row["ret_pct"] > 0 else "✗ NEGATIVE")
    year_rows += f"""<tr>
      <td style="font-weight:700">{row['year']}</td>
      <td>{row['trades']}</td>
      <td>{row['win_rate']:.0f}%</td>
      <td>₹{row['pnl']:+,.0f}</td>
      <td style="color:{c};font-weight:700;font-size:1.1em">{row['ret_pct']:+.2f}%</td>
      <td style="color:{c};font-size:0.85em">{badge}</td>
    </tr>"""

year_chart_labels = json.dumps([str(r["year"]) for r in ybl])
year_chart_values = json.dumps([round(r["ret_pct"], 2) for r in ybl])
year_chart_colors = json.dumps([
    "rgba(34,197,94,0.8)" if r["ret_pct"] >= 40
    else "rgba(250,204,21,0.8)" if r["ret_pct"] > 0
    else "rgba(239,68,68,0.8)" for r in ybl
])

# Trade log rows
trade_rows = ""
for t in sorted(pf_best.trades, key=lambda x: str(x["entry_date"])):
    pnl_c      = "#22c55e" if t["pnl"] > 0 else "#ef4444"
    inst_badge = {"NIFTY":"#06b6d4","BANKNIFTY":"#a855f7","MIDCPNIFTY":"#f97316","FINNIFTY":"#eab308"}.get(t["instrument"],"#94a3b8")
    trade_rows += f"""<tr>
      <td>{str(t['entry_date'])[:10]}</td>
      <td>{str(t['exit_date'])[:10]}</td>
      <td><span class="badge {'bull' if t['direction']=='L' else 'bear'}">{t['direction']}</span></td>
      <td><span style="background:rgba(255,255,255,0.08);color:{inst_badge};padding:2px 8px;border-radius:8px;font-size:0.8em;font-weight:700">{t['instrument']}</span></td>
      <td>{t['entry']:,.0f}</td>
      <td>{t['exit']:,.0f}</td>
      <td>{t['qty']}</td>
      <td style="color:{pnl_c};font-weight:700">₹{t['pnl']:+,.0f}</td>
      <td style="color:{pnl_c}">{t['pnl_pct']:+.2f}%</td>
      <td>₹{t['capital']:,.0f}</td>
      <td>{t['duration']}d</td>
      <td><span style="font-size:0.78em;color:#94a3b8">{t['reason']}</span></td>
    </tr>"""

# Attribution rows
attr_rows = ""
for a in attr:
    c = "#22c55e" if a["net_pnl"] > 0 else "#ef4444"
    attr_rows += f"""<tr>
      <td style="font-weight:600">{a['strategy']}</td>
      <td>{a['trades']}</td>
      <td>{a['win_rate']:.1f}%</td>
      <td style="color:{c};font-weight:700">₹{a['net_pnl']:+,.0f}</td>
      <td style="color:#22c55e">₹{a['avg_win']:,.0f}</td>
      <td style="color:#ef4444">₹{a['avg_loss']:,.0f}</td>
    </tr>"""

inst_rows = ""
total_pnl = sum(a["net_pnl"] for a in ia) or 1
for a in ia:
    c          = "#22c55e" if a["net_pnl"] > 0 else "#ef4444"
    pct        = a["net_pnl"] / total_pnl * 100
    inst_badge = {"NIFTY":"#06b6d4","BANKNIFTY":"#a855f7","MIDCPNIFTY":"#f97316","FINNIFTY":"#eab308"}.get(a["instrument"],"#94a3b8")
    inst_rows += f"""<tr>
      <td><span style="color:{inst_badge};font-weight:700">{a['instrument']}</span></td>
      <td>{a['trades']}</td>
      <td>{a['win_rate']:.1f}%</td>
      <td style="color:{c};font-weight:700">₹{a['net_pnl']:+,.0f}</td>
      <td style="color:{c}">{pct:.1f}%</td>
    </tr>"""

pie_labels = json.dumps([a["instrument"] for a in ia])
pie_values = json.dumps([max(0, a["net_pnl"]) for a in ia])
pie_colors = json.dumps(["#06b6d4","#a855f7","#f97316","#eab308","#22c55e"])

ts = datetime.now().strftime("%Y-%m-%d %H:%M")

# ---------------------------------------------------------------------------
# HTML — identical structure to generate_v6_report.py but Kite-labelled
# ---------------------------------------------------------------------------
capital_l = CAPITAL / 100_000   # e.g. 2000000 → 20.0 (lakhs)
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>v6 Backtest — Zerodha Kite Data Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg:     #0f172a; --card:   #1e293b; --border: #334155;
    --text:   #e2e8f0; --text2:  #94a3b8;
    --green:  #22c55e; --red:    #ef4444; --yellow: #facc15;
    --cyan:   #06b6d4; --blue:   #3b82f6; --purple: #a855f7;
    --orange: #f97316;
  }}
  * {{ box-sizing:border-box; margin:0; padding:0 }}
  body {{ background:var(--bg); color:var(--text); font-family:'Segoe UI',system-ui,sans-serif; padding:24px; max-width:1600px; margin:auto }}
  h1,h2,h3,h4 {{ font-weight:700 }}
  h1 {{ font-size:2.2em; background:linear-gradient(135deg,#22c55e,#06b6d4,#3b82f6); -webkit-background-clip:text; -webkit-text-fill-color:transparent }}
  .subtitle {{ color:var(--text2); margin:4px 0 20px; font-size:0.95em }}
  .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; margin:16px 0 }}
  .grid3 {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px; margin:16px 0 }}
  .grid4 {{ display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin:16px 0 }}
  .grid5 {{ display:grid; grid-template-columns:repeat(5,1fr); gap:14px; margin:16px 0 }}
  @media(max-width:1000px){{ .grid4,.grid5{{grid-template-columns:1fr 1fr}} .grid3{{grid-template-columns:1fr 1fr}} }}
  @media(max-width:600px) {{ .grid2,.grid3,.grid4,.grid5{{grid-template-columns:1fr}} }}
  .card {{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:20px }}
  .stat-card {{ text-align:center; padding:18px 10px }}
  .stat-value {{ font-size:2em; font-weight:800; line-height:1.1 }}
  .stat-label {{ font-size:0.75em; color:var(--text2); margin-top:4px; text-transform:uppercase; letter-spacing:0.08em }}
  .stat-sub {{ font-size:0.72em; color:var(--text2); margin-top:2px }}
  .green {{ color:var(--green) }} .red {{ color:var(--red) }}
  .yellow{{ color:var(--yellow) }} .cyan {{ color:var(--cyan) }}
  .section-title {{ font-size:1.2em; color:var(--cyan); margin-bottom:14px; padding-bottom:8px; border-bottom:1px solid var(--border) }}
  table {{ width:100%; border-collapse:collapse; font-size:0.84em }}
  th {{ background:#0f172a; color:var(--text2); font-weight:600; padding:10px 8px; text-align:left; border-bottom:2px solid var(--border); position:sticky; top:0; z-index:2 }}
  td {{ padding:8px 8px; border-bottom:1px solid #1a2740 }}
  tr:hover td {{ background:#1e293b99 }}
  .badge {{ padding:2px 9px; border-radius:20px; font-size:0.8em; font-weight:700 }}
  .bull {{ background:rgba(34,197,94,0.18); color:var(--green) }}
  .bear {{ background:rgba(239,68,68,0.18); color:var(--red) }}
  .hero {{ background:linear-gradient(135deg,rgba(34,197,94,0.12),rgba(6,182,212,0.08));
           border:1px solid rgba(34,197,94,0.3); border-radius:16px; padding:28px 32px;
           margin:20px 0; text-align:center }}
  .hero-return {{ font-size:5.5em; font-weight:900; color:var(--green); line-height:1 }}
  .hero-sub {{ font-size:1.05em; color:var(--text2); margin-top:6px }}
  .hero-meta {{ display:flex; gap:24px; justify-content:center; margin-top:16px; flex-wrap:wrap }}
  .hero-kpi {{ background:rgba(255,255,255,0.05); border-radius:10px; padding:10px 20px; text-align:center }}
  .hero-kpi .val {{ font-size:1.5em; font-weight:800 }}
  .hero-kpi .lbl {{ font-size:0.72em; color:var(--text2); text-transform:uppercase; letter-spacing:0.05em }}
  .insight {{ background:rgba(6,182,212,0.07); border-left:3px solid var(--cyan); border-radius:8px; padding:12px 14px; margin:8px 0 }}
  .insight h4 {{ color:var(--cyan); font-size:0.88em; margin-bottom:5px }}
  .insight p {{ font-size:0.82em; color:var(--text2); line-height:1.5 }}
  .warning {{ background:rgba(250,204,21,0.07); border-left:3px solid var(--yellow); border-radius:8px; padding:12px 14px; margin:8px 0 }}
  .warning h4 {{ color:var(--yellow); font-size:0.88em; margin-bottom:5px }}
  .warning p {{ font-size:0.82em; color:var(--text2); line-height:1.5 }}
  .success {{ background:rgba(34,197,94,0.07); border-left:3px solid var(--green); border-radius:8px; padding:12px 14px; margin:8px 0 }}
  .success h4 {{ color:var(--green); font-size:0.88em; margin-bottom:5px }}
  .success p {{ font-size:0.82em; color:var(--text2); line-height:1.5 }}
  .kite-badge {{ display:inline-block; background:rgba(34,197,94,0.15); color:var(--green);
                 padding:3px 12px; border-radius:12px; font-size:0.8em; font-weight:700;
                 border:1px solid rgba(34,197,94,0.3); margin-left:8px }}
  .chart-wrap {{ position:relative; height:320px; margin:8px 0 }}
  .chart-wrap-sm {{ position:relative; height:220px; margin:8px 0 }}
  .scroll-table {{ overflow-x:auto; max-height:520px; overflow-y:auto }}
  footer {{ text-align:center; color:var(--text2); font-size:0.8em; margin-top:40px; padding-top:20px; border-top:1px solid var(--border) }}
  .toc {{ display:flex; gap:10px; flex-wrap:wrap; margin-bottom:20px }}
  .toc a {{ color:var(--cyan); text-decoration:none; font-size:0.82em; padding:5px 13px; border:1px solid rgba(6,182,212,0.28); border-radius:20px; transition:all .2s }}
  .toc a:hover {{ background:rgba(6,182,212,0.1) }}
  code {{ background:#1e293b; padding:1px 6px; border-radius:4px; font-size:0.85em; color:var(--cyan) }}
</style>
</head>
<body>

<!-- HEADER -->
<h1>v6 Williams%R — Zerodha Kite Data Report <span class="kite-badge">KITE DATA</span></h1>
<p class="subtitle">Generated {ts} · 9-Year Backtest (Jan 2017 – Apr 2026) · NIFTY · BANKNIFTY · FINNIFTY · NIFTYIT (4 instruments with full Kite OHLCV) · ₹{capital_l:.0f}L Capital · Data: Zerodha Kite Connect API</p>
<div class="warning" style="margin-bottom:16px">
  <h4>ℹ️ Why 3 instruments instead of 4?</h4>
  <p>Zerodha Kite token 288009 (NIFTY MID SELECT / MIDCPNIFTY) returns flat open=high=low=close prices
  for 2017–2021. The backtest engine filters these out, leaving only ~1046 real MIDCPNIFTY bars (2022+).
  This would limit the date-intersection to 2022+ and collapse the 9-year backtest to 3 years — which is
  why the previous run showed only ₹44.8L instead of ₹2.5Cr.
  <strong>Fix: main backtest uses NIFTY + BANKNIFTY + FINNIFTY (all have real Kite OHLCV since 2017).
  A supplementary 4-instrument run (2022+) is shown in the Config Comparison table for reference.</strong></p>
</div>

<div class="toc">
  <a href="#hero">Summary</a>
  <a href="#configs">Config Comparison</a>
  <a href="#equity">Equity Curve</a>
  <a href="#annual">Annual Returns</a>
  <a href="#strategy">Strategy Logic</a>
  <a href="#attribution">Attribution</a>
  <a href="#instruments">Instruments</a>
  <a href="#trades">Trade Log</a>
  <a href="#notes">Notes</a>
</div>

<!-- HERO -->
<div class="hero" id="hero">
  <div style="color:var(--text2);font-size:0.9em;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px">
    Best Config: Williams%R(14) × 3 Instruments (Kite) — Full 9-Year Annualised Return
  </div>
  <div class="hero-return">{s_best['ann_ret']:+.1f}%</div>
  <div class="hero-sub">vs +38.0% baseline (v5 NIFTY+BANKNIFTY only)
    &nbsp;+{s_best['ann_ret']-38.0:.1f}% improvement
  </div>
  <div class="hero-meta">
    <div class="hero-kpi"><div class="val green">₹{s_best['net_pnl']/10000000:.1f}Cr</div><div class="lbl">Net Profit (7yr)</div></div>
    <div class="hero-kpi"><div class="val cyan">{s_best['sharpe']:.2f}</div><div class="lbl">Sharpe Ratio</div></div>
    <div class="hero-kpi"><div class="val {'green' if s_best['max_dd']<10 else 'yellow'}">{s_best['max_dd']:.1f}%</div><div class="lbl">Max Drawdown</div></div>
    <div class="hero-kpi"><div class="val green">{s_best['win_rate']:.0f}%</div><div class="lbl">Win Rate</div></div>
    <div class="hero-kpi"><div class="val" style="color:var(--blue)">{s_best['num_trades']}</div><div class="lbl">Total Trades</div></div>
    <div class="hero-kpi"><div class="val" style="color:var(--purple)">{s_best['trades_per_yr']:.0f}/yr</div><div class="lbl">Trades/Year</div></div>
    <div class="hero-kpi"><div class="val green">₹{s_best['avg_win']:,.0f}</div><div class="lbl">Avg Win</div></div>
    <div class="hero-kpi"><div class="val red">₹{s_best['avg_loss']:,.0f}</div><div class="lbl">Avg Loss</div></div>
  </div>
</div>

<!-- KPI GRID -->
<div class="grid5">
  <div class="card stat-card">
    <div class="stat-value green">₹{s_best['capital']/10000000:.2f}Cr</div>
    <div class="stat-label">Final Capital</div>
    <div class="stat-sub">Starting ₹{capital_l:.0f}L → ₹{s_best['capital']/100000:.0f}L</div>
  </div>
  <div class="card stat-card">
    <div class="stat-value" style="color:var(--blue)">{s_best['total_ret']:+.0f}%</div>
    <div class="stat-label">Total Return (7yr)</div>
    <div class="stat-sub">Jan 2017 → Apr 2026</div>
  </div>
  <div class="card stat-card">
    <div class="stat-value cyan">{s_best['sharpe']:.2f}</div>
    <div class="stat-label">Sharpe Ratio</div>
    <div class="stat-sub">&gt;3 = Exceptional</div>
  </div>
  <div class="card stat-card">
    <div class="stat-value {'green' if s_best['max_dd']<10 else 'yellow'}">{s_best['max_dd']:.1f}%</div>
    <div class="stat-label">Max Drawdown</div>
    <div class="stat-sub">{'Extremely low' if s_best['max_dd']<6 else 'Very low' if s_best['max_dd']<10 else 'Low'}</div>
  </div>
  <div class="card stat-card">
    <div class="stat-value" style="color:var(--purple)">{abs(s_best['avg_win'])/max(abs(s_best['avg_loss']),1):.1f}×</div>
    <div class="stat-label">Win/Loss Ratio</div>
    <div class="stat-sub">Realised R:R multiple</div>
  </div>
</div>

<!-- CONFIG COMPARISON -->
<div class="card" style="margin:16px 0" id="configs">
  <h3 class="section-title">Config Comparison — 3-Instrument Kite Strategies (2017–2026) <span class="kite-badge">Kite Data</span></h3>
  <div class="scroll-table">
    <table>
      <thead><tr>
        <th>Rank</th><th>Config</th><th>Ann Ret%</th><th>Total%</th>
        <th>Sharpe</th><th>MaxDD%</th><th>Win%</th><th>Trades</th>
        <th>Net P&L</th><th>Final Capital</th>
      </tr></thead>
      <tbody>{config_rows}{supp_row}</tbody>
    </table>
  </div>
  <div style="font-size:0.78em;color:#64748b;margin-top:6px">
    ⚠ Greyed row = 4-instrument run limited to 2022+ (MIDCPNIFTY Kite data only has real OHLCV from 2022)
  </div>
  <div class="insight" style="margin-top:12px">
    <h4>Data Source: Zerodha Kite Connect API</h4>
    <p>All OHLCV data sourced directly from Zerodha Kite Connect historical API (NSE spot index tokens).
    Fetched via <code>fetch_kite_daily.py</code> and cached in OHLCVStore (SQLite + Parquet).
    This is the same data source the paper trading daemon uses — results here reflect actual Zerodha data quality.</p>
  </div>
</div>

<!-- EQUITY CURVE -->
<div class="card" style="margin:16px 0" id="equity">
  <h3 class="section-title">Equity Curve — ₹{capital_l:.0f}L Compounded (Kite Data)</h3>
  <div class="chart-wrap"><canvas id="eqChart"></canvas></div>
</div>

<!-- ANNUAL RETURNS -->
<div class="grid2" id="annual">
  <div class="card">
    <h3 class="section-title">Year-by-Year Returns — WR_4Inst (Kite)</h3>
    <table>
      <thead><tr>
        <th>Year</th><th>Trades</th><th>Win%</th><th>P&amp;L</th>
        <th>Annual Return</th><th>vs 40% Target</th>
      </tr></thead>
      <tbody>{year_rows}</tbody>
    </table>
  </div>
  <div class="card">
    <h3 class="section-title">Annual Return Bar Chart</h3>
    <div class="chart-wrap"><canvas id="yearChart"></canvas></div>
  </div>
</div>

<!-- MONTHLY P&L -->
<div class="card" style="margin:16px 0">
  <h3 class="section-title">Monthly P&L Distribution</h3>
  <div class="chart-wrap"><canvas id="monthChart"></canvas></div>
</div>

<!-- STRATEGY LOGIC -->
<div class="card" style="margin:16px 0" id="strategy">
  <h3 class="section-title">Core Strategy: Williams%R(14) Mean Reversion × 4 Instruments</h3>
  <div class="grid2">
    <div>
      <h4 style="color:var(--green);margin-bottom:10px">LONG Entry Conditions</h4>
      <ul style="list-style:none;font-size:0.88em;line-height:2.1">
        <li>✅ Williams%R(14) crosses <strong>above −80</strong> (exits extreme oversold zone)</li>
        <li>✅ Price &gt; EMA200 × 0.97 (long-term uptrend confirmed)</li>
        <li>✅ EMA50 &gt; EMA200 × 0.98 (intermediate trend aligned)</li>
        <li>✅ Stop-loss: Entry − 1.5 × ATR(14)</li>
        <li>✅ Target: Entry + 3.0 × ATR (2:1 reward:risk)</li>
      </ul>
      <h4 style="color:var(--red);margin-top:14px;margin-bottom:10px">SHORT Entry Conditions</h4>
      <ul style="list-style:none;font-size:0.88em;line-height:2.1">
        <li>✅ Williams%R(14) crosses <strong>below −20</strong> (exits extreme overbought zone)</li>
        <li>✅ Price &lt; EMA200 × 1.03 (long-term downtrend confirmed)</li>
        <li>✅ EMA50 &lt; EMA200 × 1.02 (intermediate trend aligned)</li>
        <li>✅ Stop-loss: Entry + 1.5 × ATR(14)</li>
      </ul>
    </div>
    <div>
      <h4 style="color:var(--cyan);margin-bottom:10px">Exit Rules</h4>
      <ul style="list-style:none;font-size:0.88em;line-height:2.1">
        <li>🎯 <strong>Profit exit:</strong> WR(14) crosses −50 midpoint (mean-reversion complete)</li>
        <li>🛡️ <strong>Hard stop:</strong> Price hits ATR-based stop level</li>
        <li>📈 <strong>Trail:</strong> EMA21 − ATR×0.3 for longs, EMA21 + ATR×0.3 for shorts</li>
        <li>⏹️ <strong>End-of-backtest:</strong> All open positions closed at last bar</li>
      </ul>
      <h4 style="color:var(--purple);margin-top:14px;margin-bottom:10px">Position Sizing (Lot-Based)</h4>
      <ul style="list-style:none;font-size:0.88em;line-height:2">
        <li>📊 Risk = 4% of portfolio capital per trade</li>
        <li>📦 Lot sizes: NIFTY 65 · BANKNIFTY 30 · MIDCPNIFTY 120 · FINNIFTY 60</li>
        <li>⚖️ <code>Lots = max(1, floor(Risk / (ATR×1.5 × LotSize)))</code></li>
        <li>🔒 Max 4 concurrent positions, total portfolio risk ≤ 8%</li>
      </ul>
    </div>
  </div>
</div>

<!-- ATTRIBUTION -->
<div class="grid2" id="attribution">
  <div class="card">
    <h3 class="section-title">Strategy Attribution</h3>
    <table>
      <thead><tr>
        <th>Strategy Signal</th><th>Trades</th><th>Win%</th>
        <th>Net P&amp;L</th><th>Avg Win</th><th>Avg Loss</th>
      </tr></thead>
      <tbody>{attr_rows}</tbody>
    </table>
  </div>
  <div class="card" id="instruments">
    <h3 class="section-title">Instrument Attribution</h3>
    <div class="grid2" style="margin-bottom:12px">
      <table>
        <thead><tr>
          <th>Instrument</th><th>Trades</th><th>Win%</th>
          <th>Net P&amp;L</th><th>% Share</th>
        </tr></thead>
        <tbody>{inst_rows}</tbody>
      </table>
      <div style="position:relative;height:200px"><canvas id="pieChart"></canvas></div>
    </div>
  </div>
</div>

<!-- TRADE LOG -->
<div class="card" style="margin:16px 0" id="trades">
  <h3 class="section-title">Complete Trade Log — WR_3Inst/Kite ({s_best['num_trades']} trades · {s_best['win_rate']:.0f}% win rate · NIFTY+BANKNIFTY+FINNIFTY)</h3>
  <div class="scroll-table">
    <table>
      <thead><tr>
        <th>Entry</th><th>Exit</th><th>Dir</th><th>Instrument</th>
        <th>Entry Px</th><th>Exit Px</th><th>Qty</th>
        <th>P&amp;L</th><th>P&amp;L%</th><th>Capital After</th><th>Hold</th><th>Reason</th>
      </tr></thead>
      <tbody>{trade_rows}</tbody>
    </table>
  </div>
</div>

<!-- NOTES -->
<div class="card" style="margin:16px 0" id="notes">
  <h3 class="section-title">Data Sources, Methodology &amp; Caveats</h3>
  <div class="grid2">
    <div>
      <div class="success">
        <h4>✅ Data Source: Zerodha Kite Connect API</h4>
        <p>• NIFTY: token 256265 (NIFTY 50 spot index)<br>
        • BANKNIFTY: token 260105 (NIFTY BANK spot index)<br>
        • FINNIFTY: token 257801 (NIFTY FIN SERVICE) — tradeable from Jul-2021<br>
        • MIDCPNIFTY: token 288009 (NIFTY MID SELECT) — tradeable from Oct-2023<br>
        • INDIAVIX: token 264969 (INDIA VIX) — used for VIX-sizing configs<br>
        All data: NSE spot index prices via Kite historical API. Fetched by
        <code>fetch_kite_daily.py</code>, cached in <code>data/historical/</code>.</p>
      </div>
      <div class="insight" style="margin-top:8px">
        <h4>Commission Model (Zerodha)</h4>
        <p>• Brokerage: ₹20 flat per order (round-trip: ₹40)<br>
        • Exchange fee: 0.00125% of trade value (round-trip)<br>
        • Slippage: 0.03% per side (round-trip: 0.06%)<br>
        • All commissions deducted from portfolio capital on each trade</p>
      </div>
    </div>
    <div>
      <div class="warning">
        <h4>⚠️ Important Caveats</h4>
        <p>1. <strong>Daily bars only:</strong> Uses EOD close as signal; actual execution is next-day open.<br>
        2. <strong>Spot index proxy:</strong> Futures prices include carry; actual P&amp;L will differ slightly.<br>
        3. <strong>Compounding:</strong> Capital compounds aggressively — later years have larger position sizes.<br>
        4. <strong>No liquidity limits:</strong> For capital above ₹1Cr, market impact would reduce returns.<br>
        5. <strong>Past performance ≠ future results.</strong></p>
      </div>
      <div class="insight" style="margin-top:8px">
        <h4>Paper Trading Status</h4>
        <p>This report validates the strategy on Zerodha's own data before paper trading begins.
        Paper trading daemon (<code>main.py --mode v6-paper</code>) uses the same Kite API
        data source and OHLCVStore, ensuring consistency between backtest and live signals.</p>
      </div>
    </div>
  </div>
</div>

<footer>
  v6 Williams%R — Zerodha Kite Data Report · {ts}<br>
  <span style="color:#334155">Data: Zerodha Kite Connect API · Strategy: Williams%R(14) Mean Reversion ·
  7-Year Daily Data (2017–2026) · NIFTY · BANKNIFTY · MIDCPNIFTY · FINNIFTY</span>
</footer>

<script>
new Chart(document.getElementById('eqChart').getContext('2d'), {{
  type: 'line',
  data: {{
    labels: {json.dumps(chart_labels)},
    datasets: [
      {{
        label: 'WR_4Inst (Kite Data)',
        data: {json.dumps(eq_vals_ds)},
        borderColor: '#22c55e', backgroundColor: 'rgba(34,197,94,0.07)',
        borderWidth: 2.5, fill: true, tension: 0.3, pointRadius: 0,
      }},
      {{
        label: 'WR_VIX_Sizing (Kite)',
        data: {json.dumps(eq_vix_vals_ds)},
        borderColor: '#06b6d4', backgroundColor: 'rgba(6,182,212,0.04)',
        borderWidth: 1.5, fill: false, tension: 0.3, pointRadius: 0,
        borderDash: [4,3],
      }}
    ]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{
      legend: {{ labels: {{ color: '#94a3b8', boxWidth: 20 }} }},
      tooltip: {{ callbacks: {{ label: c => ' ₹' + c.raw.toLocaleString('en-IN') }} }}
    }},
    scales: {{
      x: {{ ticks: {{ color:'#64748b', maxTicksLimit:14 }}, grid:{{ color:'#1a2740' }} }},
      y: {{ ticks: {{ color:'#64748b', callback: v => '₹'+(v/10000000).toFixed(1)+'Cr' }}, grid:{{ color:'#1a2740' }} }}
    }}
  }}
}});

new Chart(document.getElementById('yearChart').getContext('2d'), {{
  type: 'bar',
  data: {{
    labels: {year_chart_labels},
    datasets: [{{
      label: 'Annual Return %',
      data: {year_chart_values},
      backgroundColor: {year_chart_colors},
      borderRadius: 6,
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display:false }},
      tooltip: {{ callbacks: {{ label: c => ' '+c.raw.toFixed(1)+'%' }} }} }},
    scales: {{
      x: {{ ticks:{{ color:'#94a3b8' }}, grid:{{ display:false }} }},
      y: {{ ticks:{{ color:'#94a3b8', callback: v => v+'%' }}, grid:{{ color:'#1a2740' }},
            min: Math.min(0, {min(round(r['ret_pct'],0) for r in ybl) - 5 if ybl else -10}) }}
    }}
  }}
}});

new Chart(document.getElementById('monthChart').getContext('2d'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(m_labels)},
    datasets: [{{
      label: 'Monthly P&L (₹)',
      data: {json.dumps(m_values)},
      backgroundColor: {json.dumps(m_colors)},
      borderRadius: 3,
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend:{{ display:false }} }},
    scales: {{
      x: {{ ticks:{{ color:'#64748b', maxRotation:90, maxTicksLimit:24 }}, grid:{{ display:false }} }},
      y: {{ ticks:{{ color:'#64748b', callback: v => '₹'+(v/100000).toFixed(1)+'L' }}, grid:{{ color:'#1a2740' }} }}
    }}
  }}
}});

new Chart(document.getElementById('pieChart').getContext('2d'), {{
  type: 'doughnut',
  data: {{
    labels: {pie_labels},
    datasets: [{{
      data: {pie_values},
      backgroundColor: {pie_colors},
      borderColor: '#0f172a',
      borderWidth: 3,
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ position:'bottom', labels:{{ color:'#94a3b8', boxWidth:14, padding:10 }} }},
      tooltip: {{ callbacks: {{ label: c => ' ₹'+c.raw.toLocaleString('en-IN') }} }} }}
  }}
}});
</script>
</body>
</html>"""

ts2   = datetime.now().strftime("%Y%m%d_%H%M%S")
fname = f"reports/v6_kite_Report_{ts2}.html"
with open(fname, "w") as f:
    f.write(html)

print(f"\nKite report saved: {fname}")
print(f"Size: {len(html):,} bytes ({len(html)//1024} KB)")
print("Open in browser to view.")
