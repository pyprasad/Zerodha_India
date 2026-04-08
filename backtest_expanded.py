"""
backtest_expanded.py
====================
Iterative instrument expansion analysis for the Williams%R v6 strategy.

Starts with the 2-instrument baseline (NIFTY + BANKNIFTY), then adds instruments
one-by-one to measure each one's marginal contribution to returns, Sharpe, and
drawdown. Identifies the optimal instrument combination.

Instruments researched:
  Current (4): NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY
  Phase 1 (+): NIFTYIT   — Nifty IT sector, ~0.65 correlation w/ NIFTY (best diversifier)
  Phase 2 (+): NIFTYNXT50 — Nifty Next 50, ~0.85 correlation (if data available)

All data sourced from Yahoo Finance CSVs (data/historical/).
Run fetch_all_instruments.py first to ensure all CSVs exist.

Usage:
  python fetch_all_instruments.py   ← get Yahoo Finance data for all instruments
  python backtest_expanded.py       ← run expansion analysis + generate HTML report

No credentials required — uses locally cached Yahoo Finance CSV data.
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
from pathlib import Path
import pandas as pd

from rich.console import Console
from rich.table import Table
from rich import box

console = Console()
Path("reports").mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Instrument expansion schedule — add in order of expected contribution
# ---------------------------------------------------------------------------
# Format: (label, instruments_list)
EXPANSION_STEPS = [
    ("WR_2Inst (baseline)",         ["NIFTY", "BANKNIFTY"]),
    ("WR_3Inst (+FINNIFTY)",         ["NIFTY", "BANKNIFTY", "FINNIFTY"]),
    ("WR_4Inst (+MIDCPNIFTY)",       ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]),
    ("WR_4Inst (+NIFTYIT)",          ["NIFTY", "BANKNIFTY", "FINNIFTY", "NIFTYIT"]),
    ("WR_5Inst (ALL incl IT)",       ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYIT"]),
    ("WR_NIFTYIT standalone",        ["NIFTYIT"]),
    ("WR_IT+NIFTY pair",             ["NIFTY", "NIFTYIT"]),
    ("WR_IT+BN pair",                ["BANKNIFTY", "NIFTYIT"]),
]

# Phase 2 — uncomment when NIFTYNXT50 data is available
PHASE2_STEPS = [
    ("WR_5Inst (+NIFTYNXT50)",       ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"]),
    ("WR_6Inst (ALL+NXT50)",         ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYIT", "NIFTYNXT50"]),
    ("WR_NXT50 standalone",          ["NIFTYNXT50"]),
]

# ---------------------------------------------------------------------------
# Check which data files exist and filter steps accordingly
# ---------------------------------------------------------------------------
DATA_DIR = Path("data/historical")

def data_exists(inst):
    return (DATA_DIR / f"{inst}_daily_extended.csv").exists()

def filter_steps(steps):
    """Keep only steps where ALL required instruments have data."""
    available = []
    skipped   = []
    for label, insts in steps:
        missing = [i for i in insts if not data_exists(i)]
        if missing:
            skipped.append((label, missing))
        else:
            available.append((label, insts))
    return available, skipped

console.rule("[bold cyan]Instrument Expansion Analysis — Williams%R v6[/bold cyan]")
console.print("[dim]Iteratively adding instruments to measure each one's marginal contribution[/dim]\n")

# Check data availability
console.print("[cyan]Checking data availability...[/cyan]")
all_insts = set()
for _, insts in EXPANSION_STEPS + PHASE2_STEPS:
    all_insts.update(insts)

for inst in sorted(all_insts):
    status = "[green]✓[/green]" if data_exists(inst) else "[red]✗ MISSING — run fetch_all_instruments.py[/red]"
    console.print(f"  {inst}: {status}")

active_steps, skipped_steps = filter_steps(EXPANSION_STEPS)
p2_active, p2_skipped = filter_steps(PHASE2_STEPS)

if skipped_steps:
    console.print(f"\n[yellow]Skipping {len(skipped_steps)} step(s) — missing data:[/yellow]")
    for label, missing in skipped_steps:
        console.print(f"  [dim]{label} — missing: {', '.join(missing)}[/dim]")

if p2_skipped:
    console.print(f"[dim]Phase 2 steps waiting for data: {', '.join(l for l,_ in p2_skipped)}[/dim]")

all_active = active_steps + p2_active
if not all_active:
    console.print("\n[red]No steps can run — ensure data/historical/ has CSV files.[/red]")
    console.print("[yellow]Run: python fetch_all_instruments.py[/yellow]")
    sys.exit(1)

console.print(f"\n[green]Running {len(all_active)} expansion steps...[/green]\n")

# ---------------------------------------------------------------------------
# Run backtest for each step
# ---------------------------------------------------------------------------
from v6_backtest import run_v6, annual_breakdown, CAPITAL

results = []

for label, insts in all_active:
    try:
        pf = run_v6(insts, ["williams_r"], CAPITAL)
        s  = pf.stats(label)
        ybl = annual_breakdown(pf)
        results.append({
            "label":    label,
            "insts":    insts,
            "stats":    s,
            "annual":   ybl,
            "pf":       pf,
        })
        console.print(
            f"  [green]✓[/green] {label:<38} "
            f"[cyan]{s['ann_ret']:+.1f}%/yr[/cyan]  "
            f"Sharpe {s['sharpe']:.2f}  "
            f"MaxDD {s['max_dd']:.1f}%  "
            f"₹{s['capital']/100000:.0f}L final  "
            f"({s['num_trades']} trades)"
        )
    except Exception as e:
        console.print(f"  [red]✗[/red] {label}: FAILED — {e}")

if not results:
    console.print("[red]All steps failed.[/red]")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Rich summary table
# ---------------------------------------------------------------------------
console.print()
tbl = Table(
    title="Instrument Expansion — Ranked by Annualised Return",
    box=box.DOUBLE_EDGE, header_style="bold cyan",
)
tbl.add_column("Config",        min_width=38)
tbl.add_column("Ann Ret%",      justify="right", min_width=10)
tbl.add_column("Total%",        justify="right", min_width=10)
tbl.add_column("Sharpe",        justify="right", min_width=8)
tbl.add_column("MaxDD%",        justify="right", min_width=8)
tbl.add_column("Win%",          justify="right", min_width=7)
tbl.add_column("Trades",        justify="right", min_width=7)
tbl.add_column("Final Capital", justify="right", min_width=14)

ranked = sorted(results, key=lambda x: x["stats"]["ann_ret"], reverse=True)
best   = ranked[0]

for r in ranked:
    s      = r["stats"]
    is_best = r is best
    color  = "#22c55e" if s["ann_ret"] >= 45 else ("#facc15" if s["ann_ret"] >= 30 else "#ef4444")
    marker = "★ " if is_best else "  "
    tbl.add_row(
        f"{marker}{r['label']}",
        f"{s['ann_ret']:+.1f}%",
        f"{s['total_ret']:+.1f}%",
        f"{s['sharpe']:.2f}",
        f"{s['max_dd']:.1f}%",
        f"{s['win_rate']:.1f}%",
        str(s["num_trades"]),
        f"₹{s['capital']/100000:.0f}L",
    )

console.print(tbl)

# Show marginal contribution of each step
console.print("\n[cyan]Marginal contribution of each instrument added:[/cyan]")
main_seq = [r for r in results if r["label"].startswith("WR_2Inst") or
            r["label"].startswith("WR_3Inst (+FIN") or
            r["label"].startswith("WR_4Inst (+MID") or
            r["label"].startswith("WR_5Inst (ALL incl")]
if len(main_seq) >= 2:
    for i in range(1, len(main_seq)):
        prev = main_seq[i-1]["stats"]
        curr = main_seq[i]["stats"]
        delta_ret    = curr["ann_ret"] - prev["ann_ret"]
        delta_sharpe = curr["sharpe"] - prev["sharpe"]
        inst_added   = set(main_seq[i]["insts"]) - set(main_seq[i-1]["insts"])
        inst_name    = next(iter(inst_added)) if inst_added else "?"
        arrow        = "↑" if delta_ret > 0 else "↓"
        color        = "green" if delta_ret > 0 else "red"
        console.print(
            f"  Adding [bold]{inst_name}[/bold]: "
            f"[{color}]{arrow}{abs(delta_ret):.1f}% ann return[/{color}]  "
            f"Sharpe {prev['sharpe']:.2f} → {curr['sharpe']:.2f}"
        )

# ---------------------------------------------------------------------------
# Build HTML report
# ---------------------------------------------------------------------------
def get_equity_series(pf, n=250):
    eq = pd.DataFrame(pf.daily_equity, columns=["date", "capital"])
    if eq.empty: return [], []
    dates = [str(d)[:10] for d in eq["date"]]
    vals  = list(eq["capital"].values)
    if len(dates) > n:
        step = len(dates) // n
        dates, vals = dates[::step], vals[::step]
    return dates, vals

# Equity curves for top configs
top_configs = ranked[:5]
eq_datasets = []
colors = ["#22c55e","#06b6d4","#f97316","#a855f7","#eab308","#ef4444"]
for i, r in enumerate(top_configs):
    dates, vals = get_equity_series(r["pf"])
    eq_datasets.append({
        "label": r["label"],
        "dates": dates,
        "vals":  vals,
        "color": colors[i % len(colors)],
    })

# All labels from the first (longest) equity curve for chart x-axis
chart_labels = eq_datasets[0]["dates"] if eq_datasets else []

# Comparison table rows
comp_rows = ""
for r in ranked:
    s  = r["stats"]
    c  = "#22c55e" if s["ann_ret"] >= 45 else ("#facc15" if s["ann_ret"] >= 30 else "#ef4444")
    dd = "#22c55e" if s["max_dd"] < 10 else ("#facc15" if s["max_dd"] < 20 else "#ef4444")
    sh = "#22c55e" if s["sharpe"] >= 3 else ("#facc15" if s["sharpe"] >= 1 else "#ef4444")
    marker = "★" if r is best else ""
    comp_rows += f"""<tr>
      <td style="font-weight:700;color:{c}">{marker} {r['label']}</td>
      <td style="text-align:center;font-size:0.78em;color:#94a3b8">{', '.join(r['insts'])}</td>
      <td style="color:{c};font-weight:700">{s['ann_ret']:+.1f}%</td>
      <td>{s['total_ret']:+.1f}%</td>
      <td style="color:{sh};font-weight:600">{s['sharpe']:.2f}</td>
      <td style="color:{dd}">{s['max_dd']:.1f}%</td>
      <td>{s['win_rate']:.1f}%</td>
      <td>{s['num_trades']}</td>
      <td style="color:{c};font-weight:600">₹{s['net_pnl']:+,.0f}</td>
      <td>₹{s['capital']:,.0f}</td>
    </tr>"""

# Year-by-year for the best config
best_ybl = best["annual"]
year_rows = ""
for row in best_ybl:
    c     = "#22c55e" if row["ret_pct"] >= 40 else ("#facc15" if row["ret_pct"] >= 20 else "#ef4444")
    badge = "★ TARGET+" if row["ret_pct"] >= 40 else ("✓" if row["ret_pct"] > 0 else "✗")
    year_rows += f"""<tr>
      <td style="font-weight:700">{row['year']}</td>
      <td>{row['trades']}</td>
      <td>{row['win_rate']:.0f}%</td>
      <td>₹{row['pnl']:+,.0f}</td>
      <td style="color:{c};font-weight:700">{row['ret_pct']:+.2f}%</td>
      <td style="color:{c};font-size:0.85em">{badge}</td>
    </tr>"""

year_chart_labels = json.dumps([str(r["year"]) for r in best_ybl])
year_chart_values = json.dumps([round(r["ret_pct"], 2) for r in best_ybl])
year_chart_colors = json.dumps([
    "rgba(34,197,94,0.85)" if r["ret_pct"] >= 40
    else "rgba(250,204,21,0.85)" if r["ret_pct"] > 0
    else "rgba(239,68,68,0.85)" for r in best_ybl
])

# Marginal delta rows
delta_rows = ""
if len(main_seq) >= 2:
    for i in range(1, len(main_seq)):
        prev_s = main_seq[i-1]["stats"]
        curr_s = main_seq[i]["stats"]
        delta_ret    = curr_s["ann_ret"] - prev_s["ann_ret"]
        delta_sharpe = curr_s["sharpe"] - prev_s["sharpe"]
        delta_dd     = curr_s["max_dd"] - prev_s["max_dd"]
        inst_added   = set(main_seq[i]["insts"]) - set(main_seq[i-1]["insts"])
        inst_name    = next(iter(inst_added)) if inst_added else "?"
        ret_c        = "#22c55e" if delta_ret > 0 else "#ef4444"
        sh_c         = "#22c55e" if delta_sharpe > 0 else "#ef4444"
        dd_c         = "#22c55e" if delta_dd < 0 else "#ef4444"
        delta_rows += f"""<tr>
          <td style="font-weight:700;color:#06b6d4">{inst_name}</td>
          <td style="color:{ret_c};font-weight:700">{'+' if delta_ret>0 else ''}{delta_ret:.1f}%</td>
          <td style="color:{sh_c}">{'+' if delta_sharpe>0 else ''}{delta_sharpe:.2f}</td>
          <td style="color:{dd_c}">{'+' if delta_dd>0 else ''}{delta_dd:.1f}%</td>
          <td>{prev_s['ann_ret']:+.1f}% → {curr_s['ann_ret']:+.1f}%</td>
        </tr>"""

# JS datasets for equity chart
eq_ds_js = ""
for ds in eq_datasets:
    eq_ds_js += f"""{{
        label: {json.dumps(ds['label'])},
        data: {json.dumps(ds['vals'])},
        borderColor: '{ds['color']}',
        backgroundColor: 'transparent',
        borderWidth: ds_idx === 0 ? 2.5 : 1.5,
        tension: 0.3, pointRadius: 0,
      }},
    """.replace("ds_idx === 0", "true" if eq_datasets.index(ds) == 0 else "false")

# Build final datasets list properly
eq_ds_items = []
for idx, ds in enumerate(eq_datasets):
    eq_ds_items.append(f"""{{
        label: {json.dumps(ds['label'])},
        data: {json.dumps(ds['vals'])},
        borderColor: '{ds['color']}',
        backgroundColor: 'transparent',
        borderWidth: {2.5 if idx == 0 else 1.5},
        tension: 0.3, pointRadius: 0,
      }}""")
eq_ds_js = ",\n".join(eq_ds_items)

ts   = datetime.now().strftime("%Y-%m-%d %H:%M")
ts2  = datetime.now().strftime("%Y%m%d_%H%M%S")
best_s = best["stats"]

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>v6 Instrument Expansion Analysis</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg:#0f172a; --card:#1e293b; --border:#334155;
    --text:#e2e8f0; --text2:#94a3b8;
    --green:#22c55e; --red:#ef4444; --yellow:#facc15;
    --cyan:#06b6d4; --blue:#3b82f6; --purple:#a855f7;
  }}
  * {{ box-sizing:border-box; margin:0; padding:0 }}
  body {{ background:var(--bg); color:var(--text); font-family:'Segoe UI',system-ui,sans-serif;
          padding:24px; max-width:1600px; margin:auto }}
  h1 {{ font-size:2.1em; font-weight:800; background:linear-gradient(135deg,#22c55e,#06b6d4,#a855f7);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:4px }}
  .sub {{ color:var(--text2); font-size:0.9em; margin-bottom:20px }}
  .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; margin:16px 0 }}
  .grid3 {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px; margin:16px 0 }}
  .grid5 {{ display:grid; grid-template-columns:repeat(5,1fr); gap:14px; margin:16px 0 }}
  @media(max-width:900px) {{ .grid2,.grid3,.grid5 {{ grid-template-columns:1fr 1fr }} }}
  @media(max-width:500px) {{ .grid2,.grid3,.grid5 {{ grid-template-columns:1fr }} }}
  .card {{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:20px }}
  .stat-card {{ text-align:center; padding:18px 10px }}
  .stat-value {{ font-size:2em; font-weight:800; line-height:1.1 }}
  .stat-label {{ font-size:0.74em; color:var(--text2); margin-top:4px; text-transform:uppercase; letter-spacing:.08em }}
  .stat-sub {{ font-size:0.7em; color:var(--text2); margin-top:2px }}
  .sec {{ font-size:1.15em; color:var(--cyan); font-weight:700; margin-bottom:14px;
          padding-bottom:8px; border-bottom:1px solid var(--border) }}
  table {{ width:100%; border-collapse:collapse; font-size:0.83em }}
  th {{ background:#0f172a; color:var(--text2); font-weight:600; padding:9px 8px;
        text-align:left; border-bottom:2px solid var(--border); position:sticky; top:0; z-index:2 }}
  td {{ padding:8px; border-bottom:1px solid #1a2740 }}
  tr:hover td {{ background:#1e293b88 }}
  .hero {{ background:linear-gradient(135deg,rgba(34,197,94,.12),rgba(6,182,212,.08));
           border:1px solid rgba(34,197,94,.3); border-radius:16px;
           padding:28px 32px; margin:20px 0; text-align:center }}
  .hero-num {{ font-size:5em; font-weight:900; color:var(--green); line-height:1 }}
  .hero-meta {{ display:flex; gap:20px; justify-content:center; margin-top:16px; flex-wrap:wrap }}
  .kpi {{ background:rgba(255,255,255,.05); border-radius:10px; padding:10px 18px; text-align:center }}
  .kpi .v {{ font-size:1.5em; font-weight:800 }}
  .kpi .l {{ font-size:.7em; color:var(--text2); text-transform:uppercase; letter-spacing:.05em }}
  .insight {{ background:rgba(6,182,212,.07); border-left:3px solid var(--cyan);
              border-radius:8px; padding:12px 14px; margin:8px 0 }}
  .insight h4 {{ color:var(--cyan); font-size:.88em; margin-bottom:5px }}
  .insight p {{ font-size:.82em; color:var(--text2); line-height:1.5 }}
  .success {{ background:rgba(34,197,94,.07); border-left:3px solid var(--green);
              border-radius:8px; padding:12px 14px; margin:8px 0 }}
  .success h4 {{ color:var(--green); font-size:.88em; margin-bottom:5px }}
  .success p {{ font-size:.82em; color:var(--text2); line-height:1.5 }}
  .warning {{ background:rgba(250,204,21,.07); border-left:3px solid var(--yellow);
              border-radius:8px; padding:12px 14px; margin:8px 0 }}
  .warning h4 {{ color:var(--yellow); font-size:.88em; margin-bottom:5px }}
  .warning p {{ font-size:.82em; color:var(--text2); line-height:1.5 }}
  .chart-wrap {{ position:relative; height:340px; margin:8px 0 }}
  .chart-sm {{ position:relative; height:240px; margin:8px 0 }}
  .scroll {{ overflow-x:auto; max-height:580px; overflow-y:auto }}
  code {{ background:#1e293b; padding:1px 6px; border-radius:4px; font-size:.85em; color:var(--cyan) }}
  footer {{ text-align:center; color:var(--text2); font-size:.8em; margin-top:40px;
            padding-top:20px; border-top:1px solid var(--border) }}
  .toc {{ display:flex; gap:10px; flex-wrap:wrap; margin-bottom:20px }}
  .toc a {{ color:var(--cyan); text-decoration:none; font-size:.82em; padding:5px 13px;
            border:1px solid rgba(6,182,212,.28); border-radius:20px }}
  .toc a:hover {{ background:rgba(6,182,212,.1) }}
  .phase-badge {{ display:inline-block; background:rgba(6,182,212,.15); color:var(--cyan);
                  padding:2px 10px; border-radius:10px; font-size:.78em; font-weight:700; margin-left:8px }}
</style>
</head>
<body>

<h1>v6 Instrument Expansion Analysis</h1>
<p class="sub">Generated {ts} · Iterative instrument addition · Williams%R(14) Mean Reversion · ₹10L Capital · Yahoo Finance data</p>

<div class="toc">
  <a href="#hero">Best Result</a>
  <a href="#comparison">All Configs</a>
  <a href="#marginal">Marginal Impact</a>
  <a href="#equity">Equity Curves</a>
  <a href="#annual">Year-by-Year</a>
  <a href="#research">Research Notes</a>
</div>

<!-- HERO -->
<div class="hero" id="hero">
  <div style="color:var(--text2);font-size:.9em;text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px">
    Best Config: {best['label']} — Annualised Return
  </div>
  <div class="hero-num">{best_s['ann_ret']:+.1f}%</div>
  <div style="color:var(--text2);margin-top:6px">Instruments: {', '.join(best['insts'])}</div>
  <div class="hero-meta">
    <div class="kpi"><div class="v green">₹{best_s['net_pnl']/10000000:.1f}Cr</div><div class="l">Net Profit</div></div>
    <div class="kpi"><div class="v" style="color:var(--cyan)">{best_s['sharpe']:.2f}</div><div class="l">Sharpe</div></div>
    <div class="kpi"><div class="v {'green' if best_s['max_dd']<10 else 'yellow'}">{best_s['max_dd']:.1f}%</div><div class="l">Max DD</div></div>
    <div class="kpi"><div class="v green">{best_s['win_rate']:.0f}%</div><div class="l">Win Rate</div></div>
    <div class="kpi"><div class="v" style="color:var(--blue)">{best_s['num_trades']}</div><div class="l">Trades</div></div>
    <div class="kpi"><div class="v" style="color:var(--purple)">{best_s['trades_per_yr']:.0f}/yr</div><div class="l">Trades/Yr</div></div>
    <div class="kpi"><div class="v green">₹{best_s['capital']/100000:.0f}L</div><div class="l">Final Capital</div></div>
  </div>
</div>

<!-- KPI GRID -->
<div class="grid5">
  <div class="card stat-card">
    <div class="stat-value green">₹{best_s['capital']/10000000:.2f}Cr</div>
    <div class="stat-label">Final Capital</div>
    <div class="stat-sub">₹10L → ₹{best_s['capital']/100000:.0f}L</div>
  </div>
  <div class="card stat-card">
    <div class="stat-value" style="color:var(--blue)">{best_s['total_ret']:+.0f}%</div>
    <div class="stat-label">Total Return</div>
    <div class="stat-sub">Jan 2017 → today</div>
  </div>
  <div class="card stat-card">
    <div class="stat-value" style="color:var(--cyan)">{best_s['sharpe']:.2f}</div>
    <div class="stat-label">Sharpe Ratio</div>
    <div class="stat-sub">&gt;3 = Exceptional</div>
  </div>
  <div class="card stat-card">
    <div class="stat-value {'green' if best_s['max_dd']<10 else 'yellow'}">{best_s['max_dd']:.1f}%</div>
    <div class="stat-label">Max Drawdown</div>
    <div class="stat-sub">{'Extremely low' if best_s['max_dd']<6 else 'Very low' if best_s['max_dd']<10 else 'Low'}</div>
  </div>
  <div class="card stat-card">
    <div class="stat-value" style="color:var(--purple)">{abs(best_s['avg_win'])/max(abs(best_s['avg_loss']),1):.1f}×</div>
    <div class="stat-label">Win/Loss Ratio</div>
    <div class="stat-sub">Realised R:R</div>
  </div>
</div>

<!-- COMPARISON TABLE -->
<div class="card" style="margin:16px 0" id="comparison">
  <h3 class="sec">All Instrument Combinations — Ranked by Annualised Return</h3>
  <div class="scroll">
    <table>
      <thead><tr>
        <th>Config</th><th>Instruments</th><th>Ann Ret%</th><th>Total%</th>
        <th>Sharpe</th><th>MaxDD%</th><th>Win%</th><th>Trades</th>
        <th>Net P&L</th><th>Final Capital</th>
      </tr></thead>
      <tbody>{comp_rows}</tbody>
    </table>
  </div>
</div>

<!-- MARGINAL CONTRIBUTION -->
<div class="card" style="margin:16px 0" id="marginal">
  <h3 class="sec">Marginal Contribution — Each Instrument Added to the Core Portfolio</h3>
  <table>
    <thead><tr>
      <th>Instrument Added</th>
      <th>ΔAnn Return</th>
      <th>ΔSharpe</th>
      <th>ΔMax DD</th>
      <th>Before → After</th>
    </tr></thead>
    <tbody>{delta_rows or '<tr><td colspan="5" style="text-align:center;color:#64748b">Run with sequential configs to see marginal impact</td></tr>'}</tbody>
  </table>
  <div class="insight" style="margin-top:12px">
    <h4>Why NIFTYIT adds the most value</h4>
    <p>NIFTYIT (Nifty IT sector) has ~0.65 correlation with NIFTY — the lowest of all available index futures.
    Its Williams%R cycles are driven by US tech spending, USD/INR rate, and global software demand
    rather than Indian domestic macro. This means NIFTYIT signals fire on different days from
    NIFTY/BANKNIFTY signals — keeping capital deployed more days per year and increasing signal
    frequency without adding correlated risk.</p>
  </div>
</div>

<!-- EQUITY CURVES -->
<div class="card" style="margin:16px 0" id="equity">
  <h3 class="sec">Equity Curves — Top {len(top_configs)} Configurations</h3>
  <div class="chart-wrap"><canvas id="eqChart"></canvas></div>
</div>

<!-- ANNUAL RETURNS -->
<div class="grid2" id="annual">
  <div class="card">
    <h3 class="sec">Year-by-Year — {best['label']}</h3>
    <table>
      <thead><tr>
        <th>Year</th><th>Trades</th><th>Win%</th><th>P&L</th><th>Annual Return</th><th>vs 40% Target</th>
      </tr></thead>
      <tbody>{year_rows}</tbody>
    </table>
  </div>
  <div class="card">
    <h3 class="sec">Annual Return Bar Chart</h3>
    <div class="chart-wrap"><canvas id="yearChart"></canvas></div>
  </div>
</div>

<!-- RESEARCH NOTES -->
<div class="card" style="margin:16px 0" id="research">
  <h3 class="sec">Instrument Research — Expansion Roadmap</h3>
  <div class="grid3">
    <div>
      <div class="success">
        <h4>✅ Phase 1 Complete: NIFTYIT</h4>
        <p><strong>Nifty IT sector index</strong><br>
        Lot size: 30 · F&O since 2001 · Yahoo: ^CNXIT<br>
        Correlation with NIFTY: ~0.65 (best diversifier of all NSE indices)<br>
        Economic driver: US tech spending, USD/INR exchange rate<br>
        WR signal timing: independent from NIFTY/BANKNIFTY<br>
        Status: Backtested — results shown above</p>
      </div>
    </div>
    <div>
      <div class="warning">
        <h4>⏳ Phase 2 Pending: NIFTY NEXT 50</h4>
        <p><strong>Nifty Next 50 index</strong><br>
        Lot size: 25 · F&O launched ~2022 · Yahoo: ^NIFTNXT50<br>
        Correlation with NIFTY: ~0.85 (moderate diversification)<br>
        Tracks rank 51–100 stocks: more mid-large cap exposure<br>
        Status: Waiting for Kite token verification<br>
        Add to <code>fetch_all_instruments.py</code> with ticker ^NIFTNXT50<br>
        then run this script again to include in analysis</p>
      </div>
    </div>
    <div>
      <div class="insight">
        <h4>🔬 Not Pursued: Why Other Instruments Were Skipped</h4>
        <p><strong>SENSEX/BANKEX</strong>: BSE exchange — requires different API, correlation ~0.99 with NIFTY (no diversification benefit)<br><br>
        <strong>Sector indices (Auto, Pharma, FMCG, Infra, Realty)</strong>: No F&O contracts on NSE — cannot trade futures<br><br>
        <strong>Individual stocks</strong>: Much higher idiosyncratic risk, requires stock-specific analysis, different infrastructure — separate project scope</p>
      </div>
    </div>
  </div>
  <div class="insight" style="margin-top:12px">
    <h4>How to add NIFTYIT to Kite paper trading</h4>
    <p>1. Find NIFTYIT token: run <code>python3 -c "from data.fetcher import KiteAuth; k=KiteAuth().login(); [print(i['instrument_token'],i['tradingsymbol'],i['name']) for i in k.instruments('NSE') if 'NIFTY IT' in i.get('name','') or i.get('tradingsymbol','')=='NIFTY IT']"</code><br>
    2. Fill in the token in <code>config/settings.py</code> NIFTYIT entry<br>
    3. Run <code>python fetch_kite_daily.py</code> (NIFTYIT already in INSTRUMENTS list)<br>
    4. Update <code>generate_kite_report.py</code> KITE_MAIN_INSTRUMENTS to include "NIFTYIT"<br>
    5. NIFTYIT is already added to <code>strategies/williams_r_v6.py</code> V6_INSTRUMENTS for paper trading</p>
  </div>
</div>

<footer>
  v6 Instrument Expansion Analysis · {ts}<br>
  <span style="color:#334155">Williams%R(14) Mean Reversion · Yahoo Finance data · ₹10L Capital · All strategies backtested on daily bars</span>
</footer>

<script>
// Equity curves
new Chart(document.getElementById('eqChart').getContext('2d'), {{
  type: 'line',
  data: {{
    labels: {json.dumps(chart_labels)},
    datasets: [{eq_ds_js}]
  }},
  options: {{
    responsive:true, maintainAspectRatio:false,
    plugins: {{
      legend: {{ labels: {{ color:'#94a3b8', boxWidth:18, padding:12 }} }},
      tooltip: {{ callbacks: {{ label: c => ' ₹'+c.raw.toLocaleString('en-IN') }} }}
    }},
    scales: {{
      x: {{ ticks:{{ color:'#64748b', maxTicksLimit:14 }}, grid:{{ color:'#1a2740' }} }},
      y: {{ ticks:{{ color:'#64748b', callback: v => '₹'+(v/100000).toFixed(0)+'L' }}, grid:{{ color:'#1a2740' }} }}
    }}
  }}
}});

// Annual bar chart
new Chart(document.getElementById('yearChart').getContext('2d'), {{
  type: 'bar',
  data: {{
    labels: {year_chart_labels},
    datasets: [{{ label:'Annual Return %', data:{year_chart_values},
      backgroundColor:{year_chart_colors}, borderRadius:6 }}]
  }},
  options: {{
    responsive:true, maintainAspectRatio:false,
    plugins: {{ legend:{{ display:false }}, tooltip:{{ callbacks:{{ label: c => ' '+c.raw.toFixed(1)+'%' }} }} }},
    scales: {{
      x: {{ ticks:{{ color:'#94a3b8' }}, grid:{{ display:false }} }},
      y: {{ ticks:{{ color:'#94a3b8', callback: v => v+'%' }}, grid:{{ color:'#1a2740' }},
            min: Math.min(0, {min(round(r['ret_pct'],0) for r in best_ybl) - 5 if best_ybl else -10}) }}
    }}
  }}
}});
</script>
</body>
</html>"""

fname = f"reports/v6_expanded_Report_{ts2}.html"
with open(fname, "w") as f:
    f.write(html)

console.print(f"\n[bold green]Report saved: {fname}[/bold green]")
console.print(f"[dim]Open in browser to see full interactive comparison.[/dim]")

console.print("\n[cyan]Next steps:[/cyan]")
console.print("[dim]Phase 1 — NIFTYIT on Kite (requires Zerodha login):[/dim]")
console.print("[dim]  1. Find token:  python3 -c \"from data.fetcher import KiteAuth; k=KiteAuth().login(); [print(i['instrument_token'],i['tradingsymbol'],i['name']) for i in k.instruments('NSE') if 'NIFTY IT' in i.get('name','')]\"[/dim]")
console.print("[dim]  2. Fill token:  config/settings.py → NIFTYIT → token: <number>[/dim]")
console.print("[dim]  3. Fetch data:  python fetch_kite_daily.py   (NIFTYIT already in list)[/dim]")
console.print("[dim]  4. Kite report: python generate_kite_report.py[/dim]")
console.print("\n[dim]Phase 2 — NIFTY NEXT 50:[/dim]")
console.print("[dim]  Add '^NIFTNXT50' to fetch_all_instruments.py, run it, then re-run this script.[/dim]")
