"""
Timeseries Store
Append-only storage for candles and ticks
"""
import sqlite3
import logging
from typing import List, Optional
from datetime import datetime
import pandas as pd
from pathlib import Path

from connectors.market_data_adapter import Candle, Tick

logger = logging.getLogger(__name__)


class TimeseriesStore:
    """Store for time series data (candles and ticks)"""
    
    def __init__(self, db_path: str = "data/trading.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Candles table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS candles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                interval TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                UNIQUE(symbol, timestamp, interval)
            )
        """)
        
        # Ticks table (for high-frequency data if needed)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ticks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                price REAL NOT NULL,
                volume REAL,
                bid REAL,
                ask REAL
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_candles_symbol_time ON candles(symbol, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ticks_symbol_time ON ticks(symbol, timestamp)")
        
        conn.commit()
        conn.close()
    
    def store_candle(self, candle: Candle):
        """Store a single candle"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO candles 
                (symbol, timestamp, interval, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                candle.symbol,
                candle.timestamp,
                candle.interval,
                candle.open,
                candle.high,
                candle.low,
                candle.close,
                candle.volume
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"Error storing candle: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def store_candles(self, candles: List[Candle]):
        """Store multiple candles"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            for candle in candles:
                cursor.execute("""
                    INSERT OR REPLACE INTO candles 
                    (symbol, timestamp, interval, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    candle.symbol,
                    candle.timestamp,
                    candle.interval,
                    candle.open,
                    candle.high,
                    candle.low,
                    candle.close,
                    candle.volume
                ))
            conn.commit()
        except Exception as e:
            logger.error(f"Error storing candles: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def get_candles(
        self,
        symbol: str,
        interval: str = "15m",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Candle]:
        """Retrieve candles from store"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT symbol, timestamp, interval, open, high, low, close, volume
            FROM candles
            WHERE symbol = ? AND interval = ?
        """
        params = [symbol, interval]
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        query += " ORDER BY timestamp ASC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        try:
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            candles = []
            for _, row in df.iterrows():
                candles.append(Candle(
                    symbol=row['symbol'],
                    timestamp=pd.to_datetime(row['timestamp']).to_pydatetime(),
                    interval=row['interval'],
                    open=row['open'],
                    high=row['high'],
                    low=row['low'],
                    close=row['close'],
                    volume=row['volume']
                ))
            return candles
        except Exception as e:
            logger.error(f"Error retrieving candles: {e}")
            conn.close()
            return []
    
    def get_latest_candle(self, symbol: str, interval: str = "15m") -> Optional[Candle]:
        """Get the most recent candle"""
        candles = self.get_candles(symbol, interval, limit=1)
        return candles[-1] if candles else None
    
    def store_tick(self, tick: Tick):
        """Store a single tick"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO ticks (symbol, timestamp, price, volume, bid, ask)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                tick.symbol,
                tick.timestamp,
                tick.price,
                tick.volume,
                tick.bid,
                tick.ask
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"Error storing tick: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def get_candles_as_dataframe(
        self,
        symbol: str,
        interval: str = "15m",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Get candles as pandas DataFrame for analysis"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM candles
            WHERE symbol = ? AND interval = ?
        """
        params = [symbol, interval]
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        query += " ORDER BY timestamp ASC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        try:
            df = pd.read_sql_query(query, conn, params=params, parse_dates=['timestamp'])
            df.set_index('timestamp', inplace=True)
            conn.close()
            return df
        except Exception as e:
            logger.error(f"Error retrieving candles as DataFrame: {e}")
            conn.close()
            return pd.DataFrame()

