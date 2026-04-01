# NIFTY/BANKNIFTY Autonomous Trading System

An algorithmic trading system for Indian indices (NIFTY 50 & BANKNIFTY) built on the Zerodha Kite Connect API. Covers historical data fetching, strategy backtesting, parameter optimisation, and live/paper order execution.

## Backtest Results

**Best Strategy: Williams%R(14) Mean Reversion — NIFTY + BANKNIFTY Portfolio**

| Metric | Result |
|---|---|
| Annualised Return | **+38.0% / year** |
| Total Return (5.4 yrs) | +306.9% |
| Net Profit | ₹30.7L on ₹10L capital |
| Sharpe Ratio | 11.69 |
| Max Drawdown | 4.9% |
| Win Rate | 79.3% |
| Trades | 145 over 5.4 years |

Year-by-year: 2021 +18.4% · 2022 +10.4% · **2023 +50.4%** · **2024 +35.3%** · **2025 +43.6%**

---

## Project Structure

```
zerodha_india/
├── config/
│   ├── settings.py          # Strategy params, VIX thresholds, risk limits
│   └── credentials.py       # Loads .env — API keys, login, TOTP
├── data/
│   ├── fetcher.py           # Kite historical data + live quotes + PCR/VIX
│   ├── store.py             # SQLite/Parquet OHLCV cache
│   └── historical/          # CSV + Parquet data files (git-ignored)
├── strategies/
│   ├── base.py              # Abstract base strategy (Backtrader)
│   ├── orb.py               # Opening Range Breakout (9:15–9:30)
│   ├── ema_crossover.py     # EMA 9/21/55 + VWAP filter
│   ├── supertrend.py        # SuperTrend (ATR=10, mult=3.0)
│   ├── mean_reversion.py    # Bollinger Bands + RSI(14)
│   └── pcr_vix.py           # PCR + India VIX signal filter
├── backtest/
│   ├── engine.py            # Backtrader runner with Zerodha commission model
│   ├── optimizer.py         # VectorBT parameter grid-search
│   └── report.py            # P&L, Sharpe, drawdown report generator
├── execution/
│   ├── broker.py            # PaperBroker + LiveBroker (Kite Connect orders)
│   ├── risk.py              # Position sizing, daily loss halt, VIX scaling
│   └── tracker.py           # Trade logger, live MTM P&L
├── scheduler/
│   └── runner.py            # Market-hours scheduler (08:55–15:30 IST)
├── alerts/
│   └── notifier.py          # Console (rich) + optional Telegram alerts
├── tests/
│   ├── test_strategies.py   # Unit tests — no credentials needed
│   └── test_risk.py         # Risk manager unit tests
├── v5_portfolio.py          # Vectorized Williams%R portfolio backtest
├── generate_final_report.py # HTML P&L presentation generator
├── main.py                  # Unified CLI entry point
├── .env.example             # Credential template (safe to commit)
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure credentials

```bash
cp .env.example .env
# Edit .env and fill in your Zerodha credentials
```

### 3. Fetch historical data

```bash
python main.py --mode fetch-historical --instrument ALL --interval 5minute --days 1095
```

### 4. Run backtest

```bash
# All strategies, all instruments
python main.py --mode backtest

# Specific strategy
python main.py --mode backtest --strategy ORB --instrument NIFTY

# Williams%R vectorized portfolio (fastest, no credentials needed)
python v5_portfolio.py
```

### 5. Generate HTML report

```bash
python generate_final_report.py
# Opens reports/Final_WilliamsR_Report_*.html
```

### 6. Paper trading

```bash
python main.py --mode paper
```

### 7. Live trading ⚠️

```bash
python main.py --mode live
# Requires confirmed prompt: "YES I UNDERSTAND"
```

### 8. Run tests

```bash
python main.py --mode test
```

---

## CLI Reference

```
python main.py --mode <MODE> [OPTIONS]

Modes:
  fetch-historical   Download OHLCV data from Zerodha and cache locally
  backtest           Run strategy backtest on cached historical data
  optimize           Grid-search optimal strategy parameters (VectorBT)
  paper              Paper trading — signals generated, no real orders
  live               Live trading — REAL ORDERS on Zerodha account
  test               Run unit test suite

Options:
  --strategy    ORB | EMA | SUPERTREND | MR | ALL  (default: ALL)
  --instrument  NIFTY | BANKNIFTY | ALL             (default: ALL)
  --interval    Candle interval for data fetch       (default: 5minute)
  --days        Days of history to fetch/backtest    (default: 1095)
  --validation-days  OOS holdout period in days     (default: 180)
```

---

## Strategies

| Strategy | Timeframe | Instrument | Key Signal | Win Rate |
|---|---|---|---|---|
| Opening Range Breakout (ORB) | 5-min intraday | Both | 9:15–9:30 range break + volume surge | 55–65% |
| EMA Crossover | 5-min intraday | BANKNIFTY | EMA 9/21 cross + VWAP + EMA55 | 50–60% |
| SuperTrend | 15-min intraday | Both | ATR(10) SuperTrend flip + EMA55 | 45–55% |
| Mean Reversion | 15-min intraday | Both | BB(20,2) + RSI < 30 / > 70 | 55–65% |
| **Williams%R(14)** | **Daily** | **Both** | **%R crosses ±80 + EMA200 trend** | **79%** |

### VIX-Based Strategy Selection (Auto)

```
VIX < 15   → ORB only (cheap options environment)
VIX 15–20  → ORB + EMA Crossover (primary)
VIX > 20   → Mean Reversion + SuperTrend
VIX > 25   → Reduce all position sizes by 50%
```

---

## Risk Management

- **Per-trade risk**: 2–4% of portfolio capital (configurable in `.env`)
- **Daily loss cap**: 2% of capital → trading halted for the remainder of the day
- **Position sizing**: `qty = floor((capital × risk_pct) / (entry − stop_loss) / lot_size) × lot_size`
- **Lot sizes**: NIFTY futures = 75 units, BANKNIFTY futures = 30 units
- **Max concurrent positions**: 2 (one per instrument)
- **Stop-loss**: 1.5 × ATR(14) from entry price (hard floor/ceiling)

---

## Scheduler (Live/Paper Mode)

| Time (IST) | Action |
|---|---|
| 08:55 | Auto-login to Zerodha, refresh access token |
| 09:00 | Fetch India VIX + PCR, configure strategy parameters |
| 09:15 | Market open — start live feed, activate ORB window |
| 09:30 | ORB range locked, breakout watch begins |
| 09:30–15:00 | EMA / SuperTrend / Mean Reversion strategies active |
| 15:00 | Begin closing all open positions |
| 15:30 | Cancel all pending orders, generate daily report |

---

## Security

- `.env` is in `.gitignore` — credentials are **never committed**
- TOTP secret in `.env` enables fully headless daily re-login via `pyotp`
- Access token is memory-only — regenerated each session, never written to disk
- No hardcoded credentials anywhere in source code

---

## Data Sources

| Source | Use | Notes |
|---|---|---|
| Zerodha Kite Connect | Primary — live + intraday historical | Requires API subscription |
| Yahoo Finance (`yfinance`) | Fallback / daily backtesting | Free, `^NSEI` and `^NSEBANK` |
| NSE Option Chain | PCR calculation | Scraped via `requests` |

---

## Requirements

- Python 3.9+
- Zerodha Kite Connect API credentials (for live/paper modes)
- See `requirements.txt` for all Python package dependencies
