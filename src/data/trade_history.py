"""
Trade History
Store trade events, P&L history, and grid state snapshots
"""
import sqlite3
import json
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class TradeHistory:
    """Store trade events and P&L history"""
    
    def __init__(self, db_path: str = "data/trading.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT UNIQUE NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                price REAL NOT NULL,
                timestamp DATETIME NOT NULL,
                order_id TEXT,
                fill_id TEXT,
                pnl REAL,
                commission REAL
            )
        """)
        
        # P&L snapshots
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pnl_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                realized_pnl REAL NOT NULL,
                unrealized_pnl REAL NOT NULL,
                total_pnl REAL NOT NULL,
                daily_pnl REAL NOT NULL,
                drawdown REAL NOT NULL,
                peak_balance REAL NOT NULL
            )
        """)
        
        # Grid state snapshots
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS grid_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                symbol TEXT NOT NULL,
                grid_mid REAL NOT NULL,
                spacing REAL NOT NULL,
                atr REAL NOT NULL,
                levels_json TEXT NOT NULL,
                open_orders_count INTEGER NOT NULL,
                net_position INTEGER NOT NULL
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pnl_timestamp ON pnl_snapshots(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_grid_timestamp ON grid_snapshots(timestamp)")
        
        conn.commit()
        conn.close()
    
    def record_trade(
        self,
        trade_id: str,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        timestamp: datetime,
        order_id: Optional[str] = None,
        fill_id: Optional[str] = None,
        pnl: Optional[float] = None,
        commission: Optional[float] = None
    ):
        """Record a trade"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO trades
                (trade_id, symbol, side, quantity, price, timestamp, order_id, fill_id, pnl, commission)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_id, symbol, side, quantity, price, timestamp,
                order_id, fill_id, pnl, commission
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"Error recording trade: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def record_pnl_snapshot(
        self,
        realized_pnl: float,
        unrealized_pnl: float,
        total_pnl: float,
        daily_pnl: float,
        drawdown: float,
        peak_balance: float,
        timestamp: Optional[datetime] = None
    ):
        """Record a P&L snapshot"""
        if timestamp is None:
            timestamp = datetime.now()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO pnl_snapshots
                (timestamp, realized_pnl, unrealized_pnl, total_pnl, daily_pnl, drawdown, peak_balance)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (timestamp, realized_pnl, unrealized_pnl, total_pnl, daily_pnl, drawdown, peak_balance))
            conn.commit()
        except Exception as e:
            logger.error(f"Error recording P&L snapshot: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def record_grid_snapshot(
        self,
        symbol: str,
        grid_mid: float,
        spacing: float,
        atr: float,
        levels: List[Dict[str, Any]],
        open_orders_count: int,
        net_position: int,
        timestamp: Optional[datetime] = None
    ):
        """Record a grid state snapshot"""
        if timestamp is None:
            timestamp = datetime.now()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO grid_snapshots
                (timestamp, symbol, grid_mid, spacing, atr, levels_json, open_orders_count, net_position)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp, symbol, grid_mid, spacing, atr,
                json.dumps(levels), open_orders_count, net_position
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"Error recording grid snapshot: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def get_daily_pnl(self, date: Optional[datetime] = None) -> float:
        """Get daily P&L for a specific date"""
        if date is None:
            date = datetime.now()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT daily_pnl FROM pnl_snapshots
                WHERE DATE(timestamp) = DATE(?)
                ORDER BY timestamp DESC
                LIMIT 1
            """, (date,))
            result = cursor.fetchone()
            return result[0] if result else 0.0
        except Exception as e:
            logger.error(f"Error getting daily P&L: {e}")
            return 0.0
        finally:
            conn.close()
    
    def get_trades(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get trades from history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM trades WHERE 1=1"
        params = []
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        query += " ORDER BY timestamp DESC"
        
        try:
            cursor.execute(query, params)
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            logger.error(f"Error getting trades: {e}")
            return []
        finally:
            conn.close()

