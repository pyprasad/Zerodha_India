"""
HTML P&L Report Generator.
Reads backtest results from run_backtest.py and outputs a standalone HTML file.
Run AFTER run_backtest.py has been executed.
"""

import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings; warnings.filterwarnings("ignore")

from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

# ─── Re-run backtests inline to collect results ────────────────────────────────
# We import the classes and data functions from run_backtest.py
exec(open("run_backtest.py").read().replace(
    'if __name__ == "__main__":\n    main()',
    '# skipping main'
))

def build_html(all_results: list, data: dict) -> str:

    ts = datetime.now().strftime("%d %B %Y, %H:%M IST")

    # ── Helper to colour-code a number
    def clr(val, fmt="+,.2f"):
        c = "#27ae60" if val >= 0 else "#e74c3c"
        return f'<span style="color:{c};font-weight:600">₹{val:{fmt}}</span>'

    def clrp(val):
        c = "#27ae60" if val >= 0 else "#e74c3c"
        return f'<span style="color:{c};font-weight:600">{val:+.2f}%</span>'

    # ── Summary table rows
    summary_rows = ""
    for r in sorted(all_results, key=lambda x: x["returns_pct"], reverse=True):
        ret  = r["returns_pct"]
        c    = "#27ae60" if ret >= 0 else "#e74c3c"
        badge = f'<span style="background:{c};color:#fff;padding:2px 8px;border-radius:3px">{ret:+.2f}%</span>'
        pf   = f"{r['profit_factor']:.3f}" if r["profit_factor"] else "N/A"
        sh   = f"{r['sharpe']:.3f}" if r["sharpe"] else "N/A"
        summary_rows += f"""
        <tr>
          <td><strong>{r['strategy']}</strong></td>
          <td>{r['instrument']}</td>
          <td>₹{r['capital']:,.0f}</td>
          <td>₹{r['final_value']:,.2f}</td>
          <td>{clr(r['net_pnl'])}</td>
          <td>{badge}</td>
          <td>{sh}</td>
          <td>{r['max_drawdown']:.2f}%</td>
          <td>{r['total_trades']}</td>
          <td>{r['win_rate']:.1f}% ({r['won']}W/{r['lost']}L)</td>
          <td>{pf}</td>
        </tr>"""

    # ── Annual returns table
    years = sorted({yr for r in all_results for yr in r["annual"].keys()})
    ann_header = "".join(f"<th>{y}</th>" for y in years)
    ann_rows = ""
    for r in all_results:
        cells = ""
        for yr in years:
            v = r["annual"].get(yr)
            if v is not None:
                p = round(v * 100, 1)
                c = "#27ae60" if p >= 0 else "#e74c3c"
                cells += f'<td style="color:{c};font-weight:600">{p:+.1f}%</td>'
            else:
                cells += "<td>—</td>"
        ann_rows += f"<tr><td><strong>{r['strategy']}</strong></td><td>{r['instrument']}</td>{cells}</tr>"

    # ── Monthly P&L tables
    monthly_sections = ""
    for r in all_results:
        trades = r["trade_log"]
        if not trades:
            continue
        df_t = pd.DataFrame(trades)
        df_t["exit_date"] = pd.to_datetime(df_t["exit_date"])
        df_t["month"] = df_t["exit_date"].dt.to_period("M")
        monthly = df_t.groupby("month")["pnlcomm"].sum()
        rows = ""
        for month, pnl in monthly.items():
            c = "#27ae60" if pnl >= 0 else "#e74c3c"
            bar_w = int(abs(pnl) / monthly.abs().max() * 120) if monthly.abs().max() > 0 else 0
            rows += f"""<tr>
              <td>{month}</td>
              <td style="color:{c};font-weight:600">₹{pnl:+,.2f}</td>
              <td><div style="width:{bar_w}px;height:14px;background:{c};border-radius:2px;display:inline-block"></div></td>
            </tr>"""
        cumulative = monthly.cumsum().iloc[-1]
        cc = "#27ae60" if cumulative >= 0 else "#e74c3c"
        monthly_sections += f"""
        <div class="monthly-block">
          <h4>{r['strategy']} / {r['instrument']}</h4>
          <table class="trade-table">
            <thead><tr><th>Month</th><th>Net P&L (₹)</th><th>Bar</th></tr></thead>
            <tbody>{rows}</tbody>
            <tfoot>
              <tr><td><strong>Total</strong></td>
              <td style="color:{cc};font-weight:700">₹{cumulative:+,.2f}</td><td></td></tr>
            </tfoot>
          </table>
        </div>"""

    # ── Trade log tables (top 4 strategies)
    trade_sections = ""
    for r in sorted(all_results, key=lambda x: x["returns_pct"], reverse=True)[:6]:
        trades = r["trade_log"]
        if not trades:
            continue
        rows = ""
        for i, t in enumerate(trades, 1):
            pnl = t["pnlcomm"]
            c   = "#27ae60" if pnl >= 0 else "#e74c3c"
            wl  = f'<span style="color:{c};font-weight:700">{"W" if t["won"] else "L"}</span>'
            rows += f"""<tr>
              <td>{i}</td>
              <td>{t['date']}</td>
              <td>{t['exit_date']}</td>
              <td>₹{t['entry']:,.2f}</td>
              <td>₹{t['exit']:,.2f}</td>
              <td>₹{t['pnl']:+,.2f}</td>
              <td style="color:{c};font-weight:600">₹{pnl:+,.2f}</td>
              <td>{wl}</td>
            </tr>"""
        trade_sections += f"""
        <div class="trade-block">
          <h4>{r['strategy']} / {r['instrument']}
            <span class="badge {'green' if r['returns_pct']>=0 else 'red'}">{r['returns_pct']:+.2f}%</span>
          </h4>
          <table class="trade-table">
            <thead><tr>
              <th>#</th><th>Entry Date</th><th>Exit Date</th>
              <th>Entry ₹</th><th>Exit ₹</th>
              <th>Gross P&L</th><th>Net P&L</th><th>W/L</th>
            </tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </div>"""

    # ── Best strategy cards
    best_cards = ""
    for inst in ["NIFTY", "BANKNIFTY"]:
        inst_r = [r for r in all_results if r["instrument"] == inst]
        if not inst_r:
            continue
        best = max(inst_r, key=lambda x: x["returns_pct"])
        c    = "#27ae60" if best["returns_pct"] >= 0 else "#e74c3c"
        best_cards += f"""
        <div class="best-card">
          <div class="best-title">{inst}</div>
          <div class="best-strategy">{best['strategy']}</div>
          <div class="best-metrics">
            <div class="metric"><span>Return</span><span style="color:{c}">{best['returns_pct']:+.2f}%</span></div>
            <div class="metric"><span>Net P&L</span><span style="color:{c}">₹{best['net_pnl']:+,.2f}</span></div>
            <div class="metric"><span>Sharpe</span><span>{best['sharpe']}</span></div>
            <div class="metric"><span>Max Drawdown</span><span>{best['max_drawdown']:.2f}%</span></div>
            <div class="metric"><span>Win Rate</span><span>{best['win_rate']:.1f}%</span></div>
            <div class="metric"><span>Profit Factor</span><span>{best['profit_factor']}</span></div>
            <div class="metric"><span>Total Trades</span><span>{best['total_trades']}</span></div>
          </div>
        </div>"""

    data_period = ""
    for name, df in data.items():
        data_period += f"<li>{name}: {len(df)} daily candles ({df.index[0].date()} → {df.index[-1].date()})</li>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NIFTY / BANKNIFTY — Strategy Backtest Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #0d1117; color: #c9d1d9; font-size: 14px; }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}
  h1 {{ font-size: 28px; color: #58a6ff; margin-bottom: 6px; }}
  h2 {{ font-size: 20px; color: #58a6ff; margin: 32px 0 12px; border-bottom: 1px solid #21262d; padding-bottom: 8px; }}
  h3 {{ font-size: 16px; color: #8b949e; margin: 20px 0 8px; }}
  h4 {{ font-size: 14px; color: #c9d1d9; margin: 14px 0 6px; }}
  .meta {{ color: #8b949e; font-size: 13px; margin-bottom: 24px; }}
  .meta ul {{ list-style: none; padding-left: 0; }}
  .meta li {{ margin: 3px 0; }}
  table {{ width: 100%; border-collapse: collapse; margin-bottom: 16px; }}
  th {{ background: #161b22; color: #8b949e; font-weight: 600; text-align: left;
        padding: 8px 10px; font-size: 12px; text-transform: uppercase; letter-spacing: .5px; border-bottom: 1px solid #30363d; }}
  td {{ padding: 7px 10px; border-bottom: 1px solid #21262d; font-size: 13px; }}
  tr:hover td {{ background: #161b22; }}
  tfoot td {{ background: #161b22; font-weight: 700; }}
  .summary-table th, .summary-table td {{ text-align: right; }}
  .summary-table th:nth-child(1), .summary-table th:nth-child(2),
  .summary-table td:nth-child(1), .summary-table td:nth-child(2) {{ text-align: left; }}
  .trade-table th, .trade-table td {{ font-size: 12px; }}
  .badge {{ padding: 2px 8px; border-radius: 3px; font-size: 12px; font-weight: 700; color: #fff; margin-left: 8px; }}
  .badge.green {{ background: #27ae60; }}
  .badge.red   {{ background: #e74c3c; }}
  .best-grid   {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px; }}
  .best-card   {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; }}
  .best-title  {{ font-size: 13px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }}
  .best-strategy {{ font-size: 22px; color: #58a6ff; font-weight: 700; margin: 6px 0 14px; }}
  .best-metrics {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
  .metric      {{ display: flex; justify-content: space-between; font-size: 13px;
                  padding: 5px 0; border-bottom: 1px solid #21262d; }}
  .metric span:first-child {{ color: #8b949e; }}
  .metric span:last-child  {{ font-weight: 600; }}
  .monthly-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  .monthly-block, .trade-block {{ background: #161b22; border: 1px solid #30363d;
                                   border-radius: 6px; padding: 16px; margin-bottom: 16px; }}
  .alert-box  {{ background: #1f2937; border: 1px solid #f59e0b; border-radius: 6px;
                 padding: 12px 16px; margin-bottom: 20px; color: #f59e0b; font-size: 13px; }}
  @media (max-width: 900px) {{ .best-grid, .monthly-grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<div class="container">

  <h1>NIFTY / BANKNIFTY — Strategy Backtest Report</h1>
  <p class="meta">Generated: {ts}</p>

  <div class="alert-box">
    ℹ️ <strong>Note:</strong> This backtest uses <strong>daily OHLCV data</strong> from Yahoo Finance
    (free, no credentials required). Strategies are adapted to daily bars. Intraday signals
    (especially ORB) will show stronger results with 5-minute data once Zerodha API credentials are
    connected. The Mean Reversion and EMA Crossover strategies are well-suited to daily timeframes.
  </div>

  <div class="meta">
    <strong>Configuration</strong>
    <ul>
      <li>Starting Capital: ₹{CAPITAL:,.0f}</li>
      <li>Brokerage: ₹{BROKERAGE_PER_ORDER}/order (Zerodha flat rate)</li>
      <li>Slippage: {SLIPPAGE_PCT*100:.3f}%</li>
      <li>Exchange Fees: {EXCHANGE_FEE_PCT*100:.4f}%</li>
      {data_period}
    </ul>
  </div>

  <h2>Best Strategy Per Instrument</h2>
  <div class="best-grid">{best_cards}</div>

  <h2>Full Performance Summary</h2>
  <table class="summary-table">
    <thead>
      <tr>
        <th>Strategy</th><th>Instrument</th><th>Capital</th><th>Final Value</th>
        <th>Net P&L</th><th>Return %</th><th>Sharpe</th><th>Max DD</th>
        <th>Trades</th><th>Win Rate</th><th>Profit Factor</th>
      </tr>
    </thead>
    <tbody>{summary_rows}</tbody>
  </table>

  <h2>Annual Returns</h2>
  <table>
    <thead><tr><th>Strategy</th><th>Instrument</th>{ann_header}</tr></thead>
    <tbody>{ann_rows}</tbody>
  </table>

  <h2>Monthly P&L Breakdown</h2>
  <div class="monthly-grid">{monthly_sections}</div>

  <h2>Trade-by-Trade Log</h2>
  {trade_sections}

  <div style="margin-top:40px;color:#8b949e;font-size:12px;text-align:center;border-top:1px solid #21262d;padding-top:16px">
    Mayu Solutions — NIFTY/BANKNIFTY Autonomous Trading System &nbsp;|&nbsp;
    Data: Yahoo Finance (^NSEI, ^NSEBANK) &nbsp;|&nbsp;
    Backtest Engine: Backtrader 1.9.78 &nbsp;|&nbsp;
    {ts}
  </div>

</div>
</body>
</html>"""
    return html


def main():
    console.print("[bold cyan]Fetching data and running backtests for HTML report...[/bold cyan]")

    data = {
        "NIFTY":     fetch_yfinance("^NSEI",    "NIFTY 50"),
        "BANKNIFTY": fetch_yfinance("^NSEBANK", "NIFTY BANK"),
    }

    strategies = [
        ("ORB",            ORBStrategy,          "Opening Range Breakout"),
        ("EMA Crossover",  EMACrossoverStrategy,  "EMA 9/21/55 + Trend Filter"),
        ("SuperTrend",     SuperTrendStrategy,    "SuperTrend ATR(10,3)"),
        ("Mean Reversion", MeanReversionStrategy, "Bollinger(20,2) + RSI(14)"),
    ]

    all_results = []
    for sname, scls, desc in strategies:
        for inst, df in data.items():
            console.print(f"  {desc} / {inst} ...", end=" ")
            try:
                result = run_bt(scls, df.copy(), inst, sname)
                all_results.append(result)
                ret = result["returns_pct"]
                c   = "green" if ret >= 0 else "red"
                console.print(f"[{c}]{ret:+.2f}%[/{c}]  win={result['win_rate']}%  trades={result['total_trades']}")
            except Exception as e:
                console.print(f"[red]FAILED: {e}[/red]")

    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    html = build_html(all_results, data)
    out  = Path("reports") / f"PnL_Report_{ts}.html"
    out.write_text(html, encoding="utf-8")

    console.print(f"\n[bold green]✓ HTML report saved: {out}[/bold green]")
    console.print(f"  Open in browser: file://{out.resolve()}")

    # Also run the console backtest
    console.print("\n[bold]Running full console report...[/bold]")
    import subprocess
    subprocess.run(["python3", "run_backtest.py"], check=False)


if __name__ == "__main__":
    main()
