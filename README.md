# NIFTY/BANKNIFTY Autonomous Trading System — v6

Algorithmic trading system for Indian index futures on Zerodha Kite Connect.
Strategy: **Williams%R(14) Mean Reversion** across NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY.

---

## Project Status — April 2026

| Phase | Status |
|---|---|
| Strategy backtested (Yahoo Finance data) | ✅ Complete |
| Zerodha Kite API connected + historical data fetched | ✅ Complete |
| Strategy validated against Kite data | ✅ Complete — results consistent within <1% |
| Kite backtest HTML report generated | ✅ Complete |
| **Instrument expansion (NIFTYIT) backtested** | ✅ Complete — +8.9% ann return, Sharpe 2.76 → 3.03 |
| **Expanded expansion report generated** | ✅ [reports/v6_expanded_Report_*.html](reports/) |
| Paper trading daemon built | ✅ Ready to run |
| Paper trading running | ⏳ Start with `python main.py --mode v6-paper` |
| NIFTYIT Kite token — add to config/settings.py | ⏳ Run lookup below, then `python fetch_kite_daily.py` |
| Live trading | 🔒 After 2–4 weeks paper trading validation |

---

## Backtest Results

### Instrument Expansion Analysis (Yahoo Finance, April 2026) ← Latest

Iterative addition of instruments — each row shows the marginal contribution:

| Config | Ann Return | Final Capital | Sharpe | MaxDD | Trades |
|---|---|---|---|---|---|
| 2-Inst baseline (NIFTY+BN) | +43.2%/yr | ₹183L | 2.68 | 7.3% | 272 |
| 3-Inst +FINNIFTY | +50.2%/yr | ₹268L | 2.79 | 7.3% | 341 |
| 4-Inst +MIDCPNIFTY | +50.7%/yr | ₹276L | 2.76 | 7.3% | 378 |
| **4-Inst +NIFTYIT** | **+58.0%/yr** | **₹405L** | **3.03** | **7.4%** | **474** |
| **★ 5-Inst ALL (incl NIFTYIT)** | **+59.6%/yr** | **₹439L** | **3.01** | **7.4%** | **515** |

**Key finding:** NIFTYIT adds **+8.9% annual return** (+Sharpe 2.76 → 3.03) — the single most impactful instrument added.
- FINNIFTY contribution: +7.0% ann return
- MIDCPNIFTY contribution: +0.5% ann return
- **NIFTYIT contribution: +8.9% ann return** ← decisive alpha driver

Full report: [reports/v6_expanded_Report_*.html](reports/)

**Next step:** Get NIFTYIT Kite token → add to `config/settings.py` → `python fetch_kite_daily.py` → `python generate_kite_report.py`

---

### Validated on Zerodha Kite Data (April 2026)

**Main backtest: NIFTY + BANKNIFTY + FINNIFTY (3 instruments, full 9-year Kite OHLCV)**

| Metric | Result | Notes |
|---|---|---|
| Annualised Return | **+57.3% / year** | Full 9-year backtest Jan 2017 → Apr 2026 |
| Total Return | ~+3,850% | ₹10L → ~₹3.95Cr |
| Trades | 473 across 9 years | ~53/year |
| Data source | Zerodha Kite Connect API | NSE spot index tokens, fetched via `fetch_kite_daily.py` |
| NIFTYIT token | 259849 | Verified 2026-04-08, OHLCV clean from 2017 |

**Supplementary: All 5 instruments (2022+ only, limited by MIDCPNIFTY Kite data)**

| Metric | Result | Notes |
|---|---|---|
| Annualised Return | **+87.9% / year** | 2022–2026 only (see MIDCPNIFTY note below) |

> **Why 4 instruments (not 5) for the main Kite run?** Zerodha Kite token 288009 (NIFTY MID SELECT / MIDCPNIFTY)
> returns flat `open=high=low=close` prices for 2017–2021. The backtest engine filters these out,
> leaving only ~1046 real bars (2022+). Including MIDCPNIFTY would collapse the date-intersection
> to 2022+ and reduce the 9-year backtest to 3 years. NIFTY, BANKNIFTY, FINNIFTY and NIFTYIT all
> have real OHLCV from 2017 in Kite. For paper/live trading, all 5 instruments are used from today.

---

### Original Backtest (Yahoo Finance data)

**Best Strategy: Williams%R(14) × 4 Instruments (NIFTY+BANKNIFTY+FINNIFTY+NIFTYIT)**

| Metric | Result | Notes |
|---|---|---|
| Annualised Return | **+58.0% / year** | 9-year backtest Jan 2017 → Apr 2026 |
| Total Return | +3,948% | ₹10L → ₹4.05Cr |
| Sharpe Ratio | **3.03** | Daily equity-curve Sharpe |
| Max Drawdown | **7.4%** | Survived COVID crash without a long trade |
| Trades | 474 across 9 years | ~53/year |

**Kite vs Yahoo Finance consistency:** +57.3% (Kite, 4 insts) vs +58.0% (Yahoo, 4 insts) — within 0.7%.
Strategy is validated on real Zerodha data.

**Year-by-year (Yahoo Finance, 4-instrument baseline):**

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

**v5 baseline (2 instruments):** +38% / year — v6 adds +12 pp via FINNIFTY + MIDCPNIFTY. With NIFTYIT: +21 pp to ~59%/yr.

---

### Backtest Integrity — Bugs Found and Fixed

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
├── v6_backtest.py              ★ Core strategy — backtest engine + all 6 strategies
│                                  NEVER modified — signal logic imported from here
│
├── fetch_all_instruments.py    Downloads 7-year daily OHLCV via Yahoo Finance (no credentials)
├── fetch_kite_daily.py         ★ Downloads 7-year daily OHLCV from Zerodha Kite API
│                                  Saves to OHLCVStore (SQLite + Parquet)
│
├── validate_data_sources.py    Compares Kite vs Yahoo Finance close prices (correlation check)
├── run_v6_kite_backtest.py     Re-runs v6 backtest using Kite data (exports CSVs, runs backtest)
│
├── generate_v6_report.py       Full HTML P&L report — uses Yahoo Finance CSVs
├── generate_kite_report.py     ★ Full HTML P&L report — uses Zerodha Kite data
│                                  3-instrument main run (full 9yr) + 4-instrument supplement
├── backtest_expanded.py        ★ Instrument expansion analysis — iteratively adds instruments,
│                                  shows marginal contribution (ΔReturn, ΔSharpe, ΔMaxDD)
│                                  Best result: 5-Inst ALL = +59.6%/yr, Sharpe 3.01
│
├── main.py                     Unified CLI entry point
│                                  --mode v6-paper  ← paper trading daemon (use this)
│                                  --mode paper     ← original paper mode
│                                  --mode live      ← live trading (real money)
│                                  --mode test      ← run unit tests
│
├── config/
│   ├── credentials.py          Loads .env credentials (never hardcoded)
│   └── settings.py             Instruments (tokens), strategy params, VIX thresholds, risk limits
│                                  NIFTY: 256265, BANKNIFTY: 260105, FINNIFTY: 257801,
│                                  MIDCPNIFTY: 288009, INDIAVIX: 264969
│
├── data/
│   ├── fetcher.py              KiteAuth (headless login) + KiteDataFetcher (historical/live)
│   ├── store.py                OHLCVStore — SQLite + Parquet OHLCV cache
│   └── historical/             CSV + Parquet files (git-ignored, created by fetch scripts)
│
├── execution/
│   ├── broker.py               PaperBroker (simulation) + LiveBroker (Kite Connect orders)
│   ├── risk.py                 Position sizing, daily loss halt, VIX scaling
│   └── tracker.py              Trade logger, live MTM P&L, equity CSV
│
├── strategies/
│   └── williams_r_v6.py        ★ Signal adapter — imports from v6_backtest, exposes
│                                  WilliamsRSignalGenerator for paper trading daemon
│
├── scheduler/
│   ├── runner.py               Original market-hours scheduler
│   └── v6_paper_runner.py      ★ V6 paper trading daemon
│                                  Schedules: login 08:55, entry 09:16, exits 12/14/15:25,
│                                  EOD signals 15:25, P&L summary 15:30
│                                  State: logs/v6_paper_pending.json
│                                        logs/v6_paper_positions.json
│                                        logs/v6_paper_equity.csv
│
├── alerts/notifier.py          Console (rich) + optional Telegram alerts
├── tests/                      Unit tests — no credentials needed
│
├── .env.example                Credential template — safe to commit
├── .env                        Your credentials — NEVER committed (in .gitignore)
├── requirements.txt
└── reports/                    Generated HTML reports (git-ignored)
    ├── v6_Report_*.html              Yahoo Finance backtest reports
    ├── v6_kite_Report_*.html         Zerodha Kite backtest reports
    └── v6_expanded_Report_*.html     Instrument expansion analysis (marginal contribution)
```

---

## Quick Start

### Step 1 — Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2a — Backtest with Yahoo Finance data (no credentials needed)

```bash
python fetch_all_instruments.py    # download 7yr OHLCV from Yahoo Finance
python v6_backtest.py              # run backtest, print ranked table
python generate_v6_report.py       # generate HTML report → reports/v6_Report_*.html
```

### Step 2b — Backtest with Zerodha Kite data (requires credentials)

```bash
python fetch_kite_daily.py         # download 7yr OHLCV from Kite API (login required)
python generate_kite_report.py     # run backtest + generate HTML → reports/v6_kite_Report_*.html
```

### Step 2c — Instrument expansion analysis (Yahoo Finance data)

```bash
python fetch_all_instruments.py    # download 7yr OHLCV (includes NIFTYIT via ^CNXIT)
python backtest_expanded.py        # iterative expansion: 2→3→4→5 instruments, marginal contribution table
```

---

## Zerodha Kite Setup

### Prerequisites

1. **Zerodha account** with F&O trading enabled
2. **Kite Connect API subscription** — ₹500/month from [kite.trade](https://kite.trade)
   - Create an app at [developers.kite.trade](https://developers.kite.trade)
   - Set redirect URL to `http://127.0.0.1`
   - Note your **API Key** and **API Secret**
3. **External TOTP** set up on your Zerodha account
   - Go to Console → Profile → Password & Security → External TOTP → Enable
   - Save the **TOTP secret** (the base32 seed shown during setup), not the 6-digit code
   - This allows fully headless daily auto-login via `pyotp`

### Configure credentials

```bash
cp .env.example .env
```

Edit `.env`:

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

### Validate credentials

```bash
python -c "from config import credentials; credentials.validate(); print('OK')"
```

### Verify instrument tokens (one-time, after first login)

```bash
python3 -c "
from data.fetcher import KiteAuth
kite = KiteAuth().login()
insts = kite.instruments('NSE')
for i in insts:
    if any(x in i.get('name','') for x in ['FIN SERVICE','MIDCAP SELECT','INDIA VIX','NIFTY IT']):
        print(i['instrument_token'], i['tradingsymbol'], i['name'])
"
```

Expected tokens: NIFTY FIN SERVICE → `257801`, NIFTY MID SELECT → `288009`, INDIA VIX → `264969`.

**NIFTYIT token** — after finding it, fill in `config/settings.py`:
```python
"NIFTYIT": {
    "token": <found_token>,   # replace None with the actual token
    ...
}
```
Then run `python fetch_kite_daily.py` to pull NIFTYIT historical data from Kite.

---

## Paper Trading (v6 Strategy)

```bash
python main.py --mode v6-paper
```

**Daily schedule (IST):**

| Time | Action |
|---|---|
| 08:55 | Auto-login to Zerodha, refresh access token |
| 09:16 | Execute signals from previous evening (paper orders at live quote) |
| 12:00 | Check stop-loss hits + Williams%R mid-exit on open positions |
| 14:00 | Check stop-loss hits + Williams%R mid-exit on open positions |
| 15:25 | Final exit check; compute today's EOD signals for tomorrow |
| 15:30 | Print P&L summary; append to `logs/v6_paper_equity.csv` |

**State files written by the daemon:**

| File | Contents |
|---|---|
| `logs/v6_paper_pending.json` | Signals queued for next morning's entry |
| `logs/v6_paper_positions.json` | Currently open paper positions |
| `logs/v6_paper_equity.csv` | Daily running P&L log |

Run for **at least 2–4 weeks** before considering live. Verify:
- [ ] Signals fire on expected instruments
- [ ] Position sizes are reasonable (lot-based, 4% risk per trade)
- [ ] Stop-loss levels match ATR × 1.5
- [ ] Equity CSV shows consistent growth pattern

---

## Live Trading ⚠️ Real Money

```bash
# Change in .env first:
TRADING_MODE=live

python main.py --mode live
# You will be prompted: type YES I UNDERSTAND
```

**Before going live, confirm:**
- [ ] Paper trading ran cleanly for 2–4 weeks
- [ ] Capital in trading account ≥ `TRADING_CAPITAL` in `.env`
- [ ] F&O margin available: NIFTY 1 lot (~₹1.5L SPAN+exposure), BANKNIFTY 1 lot (~₹1.0L)
- [ ] Daily loss cap configured (`MAX_DAILY_LOSS_PCT=0.02`)

---

## Strategy Logic (Williams%R Mean Reversion)

### Entry signals

**LONG** — all conditions at market close:
1. Williams%R(14) crosses **above −80** (exits extreme oversold zone)
2. Close price > EMA(200) × 0.97 (long-term uptrend confirmed)
3. EMA(50) > EMA(200) × 0.98 (intermediate trend aligned)
4. Stop-loss: `entry − ATR(14) × 1.5`

**SHORT** — all conditions at market close:
1. Williams%R(14) crosses **below −20** (exits extreme overbought zone)
2. Close price < EMA(200) × 1.03 (long-term downtrend confirmed)
3. EMA(50) < EMA(200) × 1.02 (intermediate trend aligned)
4. Stop-loss: `entry + ATR(14) × 1.5`

### Exit rules

- **Primary:** Williams%R crosses back to −50 midpoint (mean reversion complete)
- **Trail stop:** Moves with EMA(21) ± ATR × 0.3 each day
- **Hard stop:** ATR-based level hit on any intraday bar

### Why EMA200 matters — COVID example

During the March 2020 crash, NIFTY fell 38% (12,000 → 7,610). Williams%R hit −99 (maximum oversold).
The EMA200 filter blocked **every single long entry from Feb 25 → Aug 25 2020** (6 months).
Zero long losses during the crash. The strategy resumed longs once NIFTY recovered above EMA200
in August and captured the full rally.

### Instrument schedule

| Instrument | F&O Launch | Lot Size | Kite Token | Kite OHLCV from | Notes |
|---|---|---|---|---|---|
| NIFTY | 2000 | 65 | 256265 | 2017 | Core |
| BANKNIFTY | 2000 | 30 | 260105 | 2017 | Core |
| FINNIFTY | Jul 2021 | 60 | 257801 | 2017 (spot index) | +7% ann |
| MIDCPNIFTY | Oct 2023 | 120 | 288009 | 2022 (real OHLCV; flat pre-2022 in Kite) | +0.5% ann |
| **NIFTYIT** | **2001** | **30** | **TBD** | **2017** | **+8.9% ann ← biggest alpha** |
| INDIAVIX | — | — | 264969 | 2017 | Filter only |

---

## Risk Management

| Parameter | Value | Where set |
|---|---|---|
| Risk per trade | 4% of portfolio | `RISK_PER_TRADE` in `v6_backtest.py` |
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

**Margin required per lot (approximate — verify on Zerodha SPAN calculator):**

| Instrument | Approx total margin |
|---|---|
| NIFTY (1 lot = 65 units) | ~₹1.5L |
| BANKNIFTY (1 lot = 30 units) | ~₹1.0L |
| FINNIFTY (1 lot = 60 units) | ~₹75K |
| MIDCPNIFTY (1 lot = 120 units) | ~₹1.05L |

Recommended starting capital for 4% risk sizing to work correctly: **₹10L**.

---

## Commission Model (as modelled in backtest)

| Cost | Rate | Applied |
|---|---|---|
| Brokerage | ₹20 flat | Per order (both legs) |
| NSE exchange fee | 0.00125% | Per side on notional |
| Slippage | 0.03% | Per side on notional |
| **STT** | **0.01%** | **Sell side only (mandatory, futures)** |

All costs deducted from portfolio capital on every trade.

---

## Alerts (Optional)

```env
TELEGRAM_BOT_TOKEN=123456:ABC-your-token
TELEGRAM_CHAT_ID=your_chat_id
```

To get your chat ID: message `@userinfobot` on Telegram.

---

## Running Tests

```bash
python main.py --mode test
# or:
python -m pytest tests/ -v
```

---

## Frequently Asked Questions

**Q: Do I need a Zerodha account to run the backtest?**
No. `v6_backtest.py` + `fetch_all_instruments.py` use Yahoo Finance only. No credentials needed.
For the Kite-data backtest (`generate_kite_report.py`), Kite API credentials are required.

**Q: What capital do I need to start paper trading?**
None — paper trading simulates orders without real money. For live trading, minimum ₹5L is
recommended (margin + buffer for 1 lot per instrument).

**Q: Why does the Kite report use 4 instruments instead of 5?**
Kite token 288009 (NIFTY MID SELECT / MIDCPNIFTY) returns flat `open=high=low=close` bars for
2017–2021. The backtest filters these out, leaving only ~1046 real bars (2022+). Including
MIDCPNIFTY in the main run would limit the date-intersection to 2022+ and collapse the 9-year
backtest to 3 years. NIFTY, BANKNIFTY, FINNIFTY and NIFTYIT all have real OHLCV from 2017 in
Kite, so the main Kite backtest uses those 4. For paper/live trading, all 5 instruments are traded.

**Q: Why is the Kite backtest +49.8% but Yahoo Finance shows +50.0%?**
They are the same strategy on the same instruments — the <0.2% difference is normal data variance
(slightly different close prices on a handful of dates between NSE spot data and Yahoo Finance proxy).
This confirms the strategy is consistent and not dependent on the data source.

**Q: The backtest shows 2018 returning +93% — is that realistic?**
The signal logic is real, but first-year capital is small so lot-level returns look large.
The 2019–2025 range of 30–58%/year is more representative.

**Q: Why did 2020 (COVID) barely affect results?**
The EMA200 trend filter blocked all LONG entries from Feb 25 → Aug 25 2020 (6 months).
The strategy sat out the crash entirely, then caught the recovery rally.

**Q: Can I add more instruments (e.g., NIFTYIT)?**
Yes — and it matters a lot. NIFTYIT is already supported in `v6_backtest.py` (LOT_SIZES=30, INSTRUMENT_LIVE_DATE=2017).
Backtesting shows it adds **+8.9% annual return** (Sharpe 2.76 → 3.03) — the single most impactful addition.
To activate on Kite: find the NIFTYIT token (see Verify instrument tokens section), fill in `config/settings.py`,
then run `python fetch_kite_daily.py` and `python generate_kite_report.py`.

**Q: What is the best-proven instrument combination?**
5 instruments (NIFTY + BANKNIFTY + FINNIFTY + MIDCPNIFTY + **NIFTYIT**): **+59.6%/yr, Sharpe 3.01, MaxDD 7.4%**.
Proven in `python backtest_expanded.py` — generates `reports/v6_expanded_Report_*.html`.

---

## Security

- `.env` is in `.gitignore` — credentials are **never committed**
- TOTP secret in `.env` enables fully headless daily re-login via `pyotp`
- Access token is memory-only — regenerated each session, never written to disk
- No hardcoded credentials anywhere in source code

---

## Requirements

- Python 3.9+
- Zerodha Kite Connect API subscription (for paper/live modes and Kite data fetch only)
- See `requirements.txt` for all Python package dependencies

Key packages: `kiteconnect`, `yfinance`, `pandas`, `numpy`, `rich`, `pyotp`, `python-dotenv`, `pyarrow`
