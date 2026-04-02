# NIFTY/BANKNIFTY Autonomous Trading System — v6

Algorithmic trading system for Indian index futures on Zerodha Kite Connect.
Strategy: **Williams%R(14) Mean Reversion** across NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY.

---

## Backtest Results (v6 — Corrected, All Bugs Fixed)

**Best Strategy: Williams%R(14) × 4 Instruments**

| Metric | Result | Notes |
|---|---|---|
| Annualised Return | **+50.0% / year** | 9-year backtest Jan 2017 → Mar 2026 |
| Total Return | +2,561% | ₹10L → ₹2.66Cr |
| Net Profit | ₹2.56Cr on ₹10L capital | |
| Sharpe Ratio | **2.74** | Daily equity-curve Sharpe (correct method) |
| Max Drawdown | **7.3%** | Survived COVID crash without a long trade |
| Win Rate | 72.4% | 377 trades across 9 years |
| Trades / year | ~42 | Daily timeframe, not intraday |

**Year-by-year:**

| Year | Return | Notes |
|---|---|---|
| 2018 | +93% | First full year, smaller lots |
| 2019 | +32% | |
| 2020 | +40% | COVID crash — EMA200 filter blocked all longs during crash |
| 2021 | +55% | FINNIFTY added (F&O launched Jul 2021) |
| 2022 | +31% | Bear market — strategy switched to shorts |
| 2023 | +55% | MIDCPNIFTY added (F&O launched Oct 2023) |
| 2024 | +58% | |
| 2025 | +40% | |

**v5 baseline (2 instruments):** +38% / year — v6 adds +12 percentage points via FINNIFTY + MIDCPNIFTY.

### Backtest integrity — bugs found and fixed

| Bug | Fix |
|---|---|
| FINNIFTY / MIDCPNIFTY traded before F&O launch | Enforced live dates: FINNIFTY from 2021-07-28, MIDCPNIFTY from 2023-10-03 |
| Runaway lot compounding (300+ lots in late years) | Capped at 20 lots (NIFTY/BANKNIFTY) and 15 lots (FINNIFTY/MIDCPNIFTY) |
| `min(1, lots)` override bypassed risk budget | Removed — returns 0 lots if budget insufficient |
| Same-bar entry (signal at close, entry at close) | Pending-entry system: signal fires at close, executes at **next day's open** |
| VIX `pct_change` masked by `ffill` on holiday gaps | Computed on raw series before reindex — missing days yield NaN, no false signals |
| STT (Securities Transaction Tax) not modelled | Added 0.01% on sell-side of every futures trade |
| Sharpe computed trade-by-trade (inflated) | Switched to daily equity-curve returns, annualised |

---

## Project Structure

```
zerodha_india/
│
├── v6_backtest.py           ★ Core strategy — backtest engine + all 6 strategies
├── fetch_all_instruments.py ★ Downloads 7-year daily OHLCV for all 6 instruments
├── generate_v6_report.py    ★ Produces full HTML P&L report
│
├── main.py                  Unified CLI — paper / live / fetch / test
│
├── config/
│   ├── credentials.py       Loads .env credentials (never hardcoded)
│   └── settings.py          Strategy params, VIX thresholds, risk limits
│
├── data/
│   ├── fetcher.py           Kite Connect historical + live quotes + PCR/VIX
│   ├── store.py             SQLite / Parquet OHLCV cache
│   └── historical/          CSV + Parquet files (git-ignored, created by fetch script)
│
├── execution/
│   ├── broker.py            PaperBroker (simulation) + LiveBroker (Kite Connect orders)
│   ├── risk.py              Position sizing, daily loss halt, VIX scaling
│   └── tracker.py           Trade logger, live MTM P&L
│
├── strategies/              Backtrader strategy modules (used by main.py)
├── scheduler/runner.py      Market-hours scheduler 08:55–15:30 IST
├── alerts/notifier.py       Console (rich) + optional Telegram alerts
├── tests/                   Unit tests — no credentials needed
│
├── .env.example             Credential template — safe to commit
├── requirements.txt
└── reports/                 Generated HTML reports (git-ignored)
```

---

## Quick Start

### Step 1 — Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2 — Download historical data (no credentials needed)

```bash
python fetch_all_instruments.py
```

Downloads 7-year daily OHLCV from Yahoo Finance for all 6 instruments:
NIFTY, BANKNIFTY, MIDCPNIFTY, FINNIFTY, NIFTYIT, INDIAVIX → saved to `data/historical/`.

Expected output: `6/6 instruments OK` in ~60 seconds.

### Step 3 — Run the backtest

```bash
python v6_backtest.py
```

Runs 7 strategy configurations across 4 instruments, prints ranked comparison table,
saves full results to `reports/v6_backtest_TIMESTAMP.json`.

### Step 4 — Generate the HTML P&L report

```bash
python generate_v6_report.py
```

Produces `reports/v6_Report_TIMESTAMP.html` — opens automatically in browser.
Includes: equity curve, year-by-year returns, monthly P&L bars, strategy and instrument attribution, full trade log.

---

## Zerodha Paper / Live Trading Setup

### Prerequisites

1. **Zerodha account** with F&O trading enabled
2. **Kite Connect API subscription** — ₹2,000/month from [kite.trade](https://kite.trade)
   - After subscribing, create an app at [developers.kite.trade](https://developers.kite.trade)
   - Note your **API Key** and **API Secret**
3. **TOTP authenticator** set up on your Zerodha account (Google Authenticator / Authy)
   - You need the **TOTP secret** (the seed shown during setup), not the 6-digit code
   - This allows the system to auto-login daily without manual intervention

### Step 5 — Configure credentials

```bash
cp .env.example .env
```

Edit `.env` and fill in:

```env
KITE_API_KEY=your_api_key          # From developers.kite.trade
KITE_API_SECRET=your_api_secret    # From developers.kite.trade
ZERODHA_USER_ID=AB1234             # Your Zerodha client ID
ZERODHA_PASSWORD=your_password
ZERODHA_TOTP_SECRET=BASE32SECRET   # Seed from TOTP setup (not the 6-digit code)

TRADING_CAPITAL=1000000            # ₹10L starting capital
PER_TRADE_RISK_PCT=0.04            # 4% risk per trade (matches backtest)
MAX_DAILY_LOSS_PCT=0.02            # Halt trading if daily P&L drops 2%
TRADING_MODE=paper                 # paper or live
```

> `.env` is in `.gitignore` and will never be committed.

### Step 6 — Validate credentials

```bash
python main.py --mode test
```

Runs unit tests without placing any orders. Confirms credentials load correctly.

### Step 7 — Run paper trading (demo mode) ✅ Start here

```bash
python main.py --mode paper
```

**What this does:**
- Logs into Zerodha at 08:55 IST using your credentials
- Fetches live NIFTY / BANKNIFTY quotes via Kite Connect
- Computes Williams%R(14) signal on end-of-day data each evening
- **Prints exactly what orders it would place — but places nothing**
- Generates a daily P&L report in `reports/`

Run this for **at least 2–4 weeks** before going live. Verify:
- [ ] Signals match what you'd expect from the backtest
- [ ] Position sizes are correct (lot-based, 4% risk)
- [ ] Stop-loss levels make sense
- [ ] Daily report generates cleanly

### Step 8 — Run the v6 strategy in paper mode

The live execution engine (`main.py`) uses the Backtrader-based strategies in `strategies/`.
To run the exact v6 Williams%R logic in paper mode, the simplest approach is:

```bash
# Run v6 signal check on today's data (no orders, just signals)
python - <<'EOF'
from v6_backtest import load_data, precompute, signal_williams_r
import pandas as pd

for inst in ["NIFTY", "BANKNIFTY", "FINNIFTY"]:
    df  = load_data(inst)
    ind = precompute(df)
    i   = len(df) - 1          # today's bar (last bar in CSV)
    sig = signal_williams_r(i, ind)
    if sig:
        direction, stop_dist, tag = sig
        price = ind["close"].iloc[i]
        stop  = price - (1 if direction=="L" else -1) * stop_dist
        print(f"{inst}: {direction} signal | Entry ~{price:,.0f} | Stop {stop:,.0f} | Tag: {tag}")
    else:
        print(f"{inst}: No signal today")
EOF
```

This reads your locally cached daily data and tells you whether today's close triggered a Williams%R signal.

### Step 9 — Live trading ⚠️ Real money

```bash
python main.py --mode live
# You will be prompted to type: YES I UNDERSTAND
```

**Before going live, confirm:**
- [ ] Paper trading ran cleanly for 2–4 weeks
- [ ] Capital in trading account ≥ `TRADING_CAPITAL` in `.env`
- [ ] F&O margin available: NIFTY 1 lot (~₹1.2L SPAN margin), BANKNIFTY 1 lot (~₹90K)
- [ ] Daily loss cap set in `.env` (`MAX_DAILY_LOSS_PCT=0.02`)
- [ ] Reviewed the risk section below

---

## Strategy Logic (Williams%R Mean Reversion)

### Entry signals

**LONG** — triggered when all conditions met at market close:
1. Williams%R(14) crosses **above −80** (exits extreme oversold zone)
2. Close price > EMA(200) × 0.97 (long-term uptrend confirmed)
3. EMA(50) > EMA(200) × 0.98 (intermediate trend aligned)
4. Stop-loss set at: `entry − ATR(14) × 1.5`

**SHORT** — triggered when all conditions met at market close:
1. Williams%R(14) crosses **below −20** (exits extreme overbought zone)
2. Close price < EMA(200) × 1.03 (long-term downtrend confirmed)
3. EMA(50) < EMA(200) × 1.02 (intermediate trend aligned)
4. Stop-loss set at: `entry + ATR(14) × 1.5`

### Exit rules
- **Primary:** Williams%R crosses back to −50 midpoint (mean reversion complete) → exit at next open
- **Trail stop:** Moves with EMA(21) ± ATR × 0.3 each day
- **Hard stop:** Hit intraday on any bar

### Why EMA200 matters (COVID example)
During the March 2020 crash, NIFTY fell 38% (12,000 → 7,610). Williams%R hit −99 (maximum oversold) — normally a strong buy signal. The EMA200 filter blocked every single long entry from **Feb 25 → Aug 25 2020** (6 months). Zero long losses during the crash. The strategy resumed longs once NIFTY recovered above EMA200 in August and captured the full rally.

### Instrument schedule

| Instrument | F&O Launch | Lot Size | Yahoo Proxy | Backtest from |
|---|---|---|---|---|
| NIFTY | 2000 | 65 | `^NSEI` | 2017 |
| BANKNIFTY | 2000 | 30 | `^NSEBANK` | 2017 |
| FINNIFTY | Jul 2021 | 60 | `NIFTY_FIN_SERVICE.NS` | Jul 2021 |
| MIDCPNIFTY | Oct 2023 | 120 | `^NSMIDCP` | Oct 2023 |

---

## Risk Management

| Parameter | Value | Where set |
|---|---|---|
| Risk per trade | 4% of portfolio | `RISK_PER_TRADE` in `v6_backtest.py` / `.env` |
| Max concurrent positions | 4 (one per instrument) | `MAX_CONCURRENT` |
| Total portfolio risk cap | 8% | `MAX_TOTAL_RISK` |
| Max lots per instrument | 20 (NIFTY/BN), 15 (FIN/MID) | `MAX_LOTS_PER_INST` |
| Daily loss halt | 2% of capital | `MAX_DAILY_LOSS_PCT` in `.env` |
| Stop-loss | 1.5 × ATR(14) | Signal logic |
| Min stop distance | 0.4% of price | `MIN_STOP_PCT` |

**Position sizing formula:**
```
lots = floor( (Capital × 4%) / (ATR × 1.5 × lot_size) )
qty  = lots × lot_size
```
Example at ₹10L capital, NIFTY ATR=220, lot=65:
```
lots = floor( 40,000 / (330 × 65) ) = floor(1.86) = 1 lot
qty  = 65 units
```

**Margin required per lot (approximate, check Zerodha for current SPAN):**

| Instrument | SPAN margin | Exposure margin | Total approx |
|---|---|---|---|
| NIFTY (1 lot = 65 units) | ₹95,000 | ₹55,000 | ~₹1.5L |
| BANKNIFTY (1 lot = 30 units) | ₹55,000 | ₹45,000 | ~₹1.0L |
| FINNIFTY (1 lot = 60 units) | ₹45,000 | ₹30,000 | ~₹75K |
| MIDCPNIFTY (1 lot = 120 units) | ₹65,000 | ₹40,000 | ~₹1.05L |

Minimum capital to trade all 4 instruments simultaneously: **~₹5L** (margin + buffer).
Recommended starting capital for 4% risk sizing to work correctly: **₹10L**.

---

## Commission Model (as modelled in backtest)

| Cost | Rate | Applied |
|---|---|---|
| Brokerage | ₹20 flat | Per order (both legs) |
| NSE exchange fee | 0.00125% | Per side on notional |
| Slippage | 0.03% | Per side on notional |
| **STT** | **0.01%** | **Sell side only (mandatory, futures)** |

All costs are deducted from portfolio capital on every trade in the backtest.

---

## Daily Scheduler (paper / live modes)

| Time (IST) | Action |
|---|---|
| 08:55 | Auto-login to Zerodha, refresh access token |
| 09:00 | Fetch India VIX + PCR, configure strategy parameters |
| 09:15 | Market open — start live feed |
| 15:00 | Begin closing positions if stop/target not hit |
| 15:25 | Cancel all pending orders |
| 15:30 | Market close — evaluate signals on today's close |
| 15:31 | If WR signal fires: queue order for tomorrow's open |
| 15:35 | Generate daily P&L report |

> The v6 strategy uses **daily bars** — signals are evaluated at 15:30 close and executed at next morning's open (~09:16). This is not an intraday strategy.

---

## Alerts

Optional Telegram alerts for every signal and order:

```env
TELEGRAM_BOT_TOKEN=123456:ABC-your-token
TELEGRAM_CHAT_ID=your_chat_id
```

To get your chat ID:
1. Message `@userinfobot` on Telegram
2. It replies with your chat ID

---

## Running Tests

```bash
python main.py --mode test
# or directly:
python -m pytest tests/ -v
```

Tests in `tests/test_strategies.py` and `tests/test_risk.py` run without credentials.

---

## Frequently Asked Questions

**Q: Do I need a Zerodha account to run the backtest?**
No. `v6_backtest.py` and `fetch_all_instruments.py` use Yahoo Finance only. No credentials needed.

**Q: What capital do I need to start paper trading?**
None — paper trading simulates orders without using real money. For live trading, minimum ₹5L is recommended (to afford at least 1 lot margin per instrument plus buffer).

**Q: The backtest shows 2018 returning +93% — is that realistic?**
The signal logic is real, but the first year uses maximum lot sizes relative to ₹10L capital. In practice, margin requirements limit you to 2–3 lots in year 1. The 2019–2025 range of 30–58%/year is more representative.

**Q: Why did 2020 (COVID) barely affect results?**
The EMA200 trend filter blocked all LONG entries from Feb 25 → Aug 25 2020 (6 months). NIFTY was 23% below its 200-day average at the crash bottom. The strategy sat out the crash, then caught the recovery rally. See the `COVID analysis` in the backtest output for full details.

**Q: Can I add more instruments (e.g., NIFTYIT, SENSEX)?**
Yes. Add an entry to `INSTRUMENTS_4` in `v6_backtest.py`, add the instrument to `INSTRUMENT_LIVE_DATE` and `MAX_LOTS_PER_INST`, then re-run the backtest to validate. NIFTYIT futures were listed in 2001 and are already in `fetch_all_instruments.py`.

**Q: Is the FINNIFTY / MIDCPNIFTY data accurate?**
The backtest uses **spot index proxies** from Yahoo Finance (`NIFTY_FIN_SERVICE.NS` and `^NSMIDCP`). These correlate >0.97 with futures prices for daily signal generation. Actual futures prices include carry cost (~0.5–1% per month), which slightly reduces mean-reversion bounce size on the long side.

---

## Security

- `.env` is in `.gitignore` — credentials are **never committed**
- TOTP secret in `.env` enables fully headless daily re-login via `pyotp`
- Access token is memory-only — regenerated each session, never written to disk
- No hardcoded credentials anywhere in source code
- `.claude/` is in `.gitignore` — AI session files never committed

---

## Requirements

- Python 3.9+
- Zerodha Kite Connect API subscription (for paper/live modes only — not needed for backtest)
- See `requirements.txt` for all Python package dependencies

Key packages: `kiteconnect`, `yfinance`, `pandas`, `numpy`, `rich`, `pyotp`, `python-dotenv`
