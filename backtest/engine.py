"""
Backtrader-based backtesting engine.
Runs any strategy against historical OHLCV data with realistic cost simulation.
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Type

import backtrader as bt
import pandas as pd

from config.settings import (
    INSTRUMENTS,
    BROKERAGE_PER_ORDER,
    SLIPPAGE_PCT,
    EXCHANGE_FEES_PCT,
    TRADING_CAPITAL,
)
from data.store import OHLCVStore
from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class ZerodhaCommission(bt.CommInfoBase):
    """
    Zerodha brokerage model:
    - ₹20 flat per order (regardless of quantity)
    - STT + exchange charges: ~0.0125%
    """
    params = (
        ("commission", BROKERAGE_PER_ORDER),  # flat ₹20
        ("exchange_fee_pct", EXCHANGE_FEES_PCT),
        ("stocklike", False),
        ("commtype", bt.CommInfoBase.COMM_FIXED),
    )

    def getcommission(self, size, price):
        return self.p.commission + abs(size * price * self.p.exchange_fee_pct)


def load_bt_data(df: pd.DataFrame) -> bt.feeds.PandasData:
    """Convert a stored OHLCV DataFrame to a Backtrader PandasData feed."""
    df = df.copy()
    df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    df.index.name = "datetime"
    return bt.feeds.PandasData(
        dataname=df,
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume",
        openinterest=-1,
    )


def run_backtest(
    strategy_cls: Type[BaseStrategy],
    instrument: str,
    interval: str,
    from_date: datetime,
    to_date: datetime,
    strategy_params: dict = None,
    capital: float = TRADING_CAPITAL,
    store: OHLCVStore = None,
) -> dict:
    """
    Run a full backtest for a strategy on an instrument.

    Args:
        strategy_cls: Strategy class (e.g., ORBStrategy)
        instrument: "NIFTY" or "BANKNIFTY"
        interval: e.g., "5minute", "15minute"
        from_date / to_date: Backtest date range
        strategy_params: Override strategy parameters
        capital: Starting capital in INR
        store: OHLCVStore instance (uses default if None)

    Returns:
        dict with keys: final_value, returns_pct, sharpe, max_drawdown, trade_log
    """
    if store is None:
        store = OHLCVStore()

    df = store.load_parquet(instrument, interval)
    if df.empty:
        df = store.load(instrument, interval, from_date, to_date)

    # Filter to backtest window
    df = df[(df["date"] >= pd.Timestamp(from_date)) & (df["date"] <= pd.Timestamp(to_date))]
    if df.empty:
        raise ValueError(
            f"No data for {instrument} ({interval}) in range {from_date} → {to_date}. "
            "Run --mode fetch-historical first."
        )

    logger.info(
        f"Backtesting {strategy_cls.__name__} on {instrument} ({interval}) "
        f"with {len(df)} candles | capital=₹{capital:,.0f}"
    )

    cerebro = bt.Cerebro()
    cerebro.adddata(load_bt_data(df))

    # Add strategy with params
    params = {
        "instrument": instrument,
        "lot_size": INSTRUMENTS[instrument]["lot_size"],
        "capital": capital,
    }
    if strategy_params:
        params.update(strategy_params)
    cerebro.addstrategy(strategy_cls, **params)

    # Broker settings
    cerebro.broker.setcash(capital)
    cerebro.broker.addcommissioninfo(ZerodhaCommission())
    cerebro.broker.set_slippage_perc(SLIPPAGE_PCT)

    # Analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.065)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")

    results = cerebro.run()
    strat = results[0]

    final_value = cerebro.broker.getvalue()
    returns_pct = ((final_value - capital) / capital) * 100

    sharpe_data = strat.analyzers.sharpe.get_analysis()
    sharpe = sharpe_data.get("sharperatio", None)

    drawdown_data = strat.analyzers.drawdown.get_analysis()
    max_dd = drawdown_data.get("max", {}).get("drawdown", 0)

    trade_data = strat.analyzers.trades.get_analysis()
    total_trades = trade_data.get("total", {}).get("closed", 0)
    won = trade_data.get("won", {}).get("total", 0)
    lost = trade_data.get("lost", {}).get("total", 0)
    win_rate = (won / total_trades * 100) if total_trades > 0 else 0

    result = {
        "strategy": strategy_cls.__name__,
        "instrument": instrument,
        "interval": interval,
        "from_date": str(from_date.date()),
        "to_date": str(to_date.date()),
        "capital": capital,
        "final_value": round(final_value, 2),
        "returns_pct": round(returns_pct, 2),
        "sharpe": round(sharpe, 3) if sharpe else None,
        "max_drawdown_pct": round(max_dd, 2),
        "total_trades": total_trades,
        "won": won,
        "lost": lost,
        "win_rate_pct": round(win_rate, 1),
        "trade_log": strat.trade_log,
    }

    logger.info(
        f"Backtest complete: returns={returns_pct:.2f}% sharpe={sharpe:.3f if sharpe else 'N/A'} "
        f"max_dd={max_dd:.2f}% win_rate={win_rate:.1f}% ({won}W/{lost}L)"
    )
    return result
