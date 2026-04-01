"""
SQLite-based OHLCV data store using SQLAlchemy.
Caches historical data locally to avoid redundant API calls during backtesting.
"""
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, Column, String, Float, DateTime, Integer, Index
from sqlalchemy.orm import DeclarativeBase, Session

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "historical" / "ohlcv.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


class Base(DeclarativeBase):
    pass


class OHLCVRecord(Base):
    __tablename__ = "ohlcv"

    id = Column(Integer, primary_key=True, autoincrement=True)
    instrument = Column(String(32), nullable=False)
    interval = Column(String(16), nullable=False)
    date = Column(DateTime, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False, default=0)

    __table_args__ = (
        Index("ix_ohlcv_lookup", "instrument", "interval", "date", unique=True),
    )


class OHLCVStore:
    """Read/write OHLCV data to local SQLite database."""

    def __init__(self, db_path: Path = DB_PATH):
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        logger.info(f"OHLCVStore initialized at {db_path}")

    def save(self, instrument: str, interval: str, df: pd.DataFrame) -> int:
        """
        Upsert OHLCV records for an instrument+interval.
        Returns number of new records inserted.
        """
        if df.empty:
            return 0

        records = []
        for _, row in df.iterrows():
            records.append(OHLCVRecord(
                instrument=instrument,
                interval=interval,
                date=pd.Timestamp(row["date"]).to_pydatetime(),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row.get("volume", 0)),
            ))

        inserted = 0
        with Session(self.engine) as session:
            for rec in records:
                # Use INSERT OR IGNORE pattern via merge (upsert)
                existing = session.query(OHLCVRecord).filter_by(
                    instrument=rec.instrument,
                    interval=rec.interval,
                    date=rec.date,
                ).first()
                if existing is None:
                    session.add(rec)
                    inserted += 1
            session.commit()

        logger.info(f"Saved {inserted} new records for {instrument} ({interval})")
        return inserted

    def load(
        self,
        instrument: str,
        interval: str,
        from_date: datetime = None,
        to_date: datetime = None,
    ) -> pd.DataFrame:
        """Load OHLCV data from SQLite into a DataFrame."""
        with Session(self.engine) as session:
            query = session.query(OHLCVRecord).filter_by(
                instrument=instrument,
                interval=interval,
            )
            if from_date:
                query = query.filter(OHLCVRecord.date >= from_date)
            if to_date:
                query = query.filter(OHLCVRecord.date <= to_date)
            query = query.order_by(OHLCVRecord.date)
            rows = query.all()

        if not rows:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame([{
            "date": r.date,
            "open": r.open,
            "high": r.high,
            "low": r.low,
            "close": r.close,
            "volume": r.volume,
        } for r in rows])
        df["date"] = pd.to_datetime(df["date"])
        return df

    def get_date_range(self, instrument: str, interval: str) -> tuple:
        """Returns (min_date, max_date) of stored data for an instrument+interval."""
        with Session(self.engine) as session:
            result = session.query(
                OHLCVRecord.date
            ).filter_by(instrument=instrument, interval=interval).order_by(OHLCVRecord.date)
            dates = [r.date for r in result]
        if not dates:
            return None, None
        return dates[0], dates[-1]

    def has_data(self, instrument: str, interval: str) -> bool:
        """True if any data exists for this instrument+interval."""
        with Session(self.engine) as session:
            count = session.query(OHLCVRecord).filter_by(
                instrument=instrument, interval=interval
            ).count()
        return count > 0

    def save_parquet(self, instrument: str, interval: str) -> Path:
        """Export stored data to parquet for fast backtest loading."""
        df = self.load(instrument, interval)
        path = DB_PATH.parent / f"{instrument}_{interval}.parquet"
        df.to_parquet(path, index=False)
        logger.info(f"Exported {instrument} ({interval}) to {path}")
        return path

    def load_parquet(self, instrument: str, interval: str) -> pd.DataFrame:
        """Load from parquet if available, else fall back to SQLite."""
        path = DB_PATH.parent / f"{instrument}_{interval}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df["date"] = pd.to_datetime(df["date"])
            return df
        return self.load(instrument, interval)
