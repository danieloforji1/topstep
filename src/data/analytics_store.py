"""
Analytics Store
Comprehensive data storage for AI/ML analysis and strategy optimization
"""
import sqlite3
import json
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class AnalyticsStore:
    """Store comprehensive analytics data for AI/ML analysis"""
    
    def __init__(self, db_path: str = "data/trading.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize analytics database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Enhanced trades table with market context
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades_analytics (
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
                commission REAL,
                -- Market context at time of trade
                atr REAL,
                volatility REAL,
                correlation REAL,
                current_price REAL,
                grid_mid REAL,
                grid_spacing REAL,
                level_index INTEGER,
                -- Position context
                position_before INTEGER,
                position_after INTEGER,
                avg_price_before REAL,
                avg_price_after REAL,
                -- Risk context
                daily_pnl REAL,
                drawdown REAL,
                exposure REAL,
                -- Market conditions
                volume REAL,
                price_change REAL,
                price_change_pct REAL
            )
        """)
        
        # Position history snapshots
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS position_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                symbol TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                avg_price REAL NOT NULL,
                current_price REAL,
                realized_pnl REAL NOT NULL,
                unrealized_pnl REAL NOT NULL,
                total_pnl REAL NOT NULL,
                exposure REAL NOT NULL
            )
        """)
        
        # Order events (all order lifecycle events)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS order_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                order_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                price REAL,
                event_type TEXT NOT NULL,  -- 'placed', 'filled', 'cancelled', 'modified', 'rejected'
                status_before TEXT,
                status_after TEXT,
                fill_price REAL,
                fill_quantity INTEGER,
                reason TEXT,  -- Why order was placed/cancelled
                market_context_json TEXT  -- JSON with market conditions
            )
        """)
        
        # Strategy decisions (why we made certain decisions)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategy_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                decision_type TEXT NOT NULL,  -- 'grid_rebuild', 'hedge_placed', 'order_placed', 'risk_check'
                symbol TEXT,
                decision_data_json TEXT NOT NULL,  -- JSON with decision details
                market_conditions_json TEXT,  -- Market state at decision time
                outcome_json TEXT,  -- Result of decision (filled, cancelled, etc.)
                reasoning TEXT  -- Why this decision was made
            )
        """)
        
        # Market conditions snapshots
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_conditions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                symbol TEXT NOT NULL,
                price REAL NOT NULL,
                atr REAL,
                volatility REAL,
                volume REAL,
                -- Price movement
                price_change REAL,
                price_change_pct REAL,
                high REAL,
                low REAL,
                -- Grid context
                grid_mid REAL,
                grid_spacing REAL,
                levels_count INTEGER,
                open_orders_count INTEGER,
                -- Correlation with hedge
                correlation REAL,
                hedge_price REAL
            )
        """)
        
        # Performance metrics (detailed breakdowns)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                -- P&L breakdown
                realized_pnl REAL NOT NULL,
                unrealized_pnl REAL NOT NULL,
                total_pnl REAL NOT NULL,
                daily_pnl REAL NOT NULL,
                -- Risk metrics
                drawdown REAL NOT NULL,
                peak_balance REAL NOT NULL,
                current_balance REAL NOT NULL,
                exposure REAL NOT NULL,
                max_exposure REAL NOT NULL,
                -- Trade metrics
                total_trades INTEGER NOT NULL,
                winning_trades INTEGER NOT NULL,
                losing_trades INTEGER NOT NULL,
                avg_win REAL,
                avg_loss REAL,
                win_rate REAL,
                profit_factor REAL,
                -- Grid metrics
                grid_fills INTEGER NOT NULL,
                grid_rebuilds INTEGER NOT NULL,
                hedge_activations INTEGER NOT NULL,
                -- Market metrics
                avg_atr REAL,
                avg_volatility REAL,
                avg_correlation REAL
            )
        """)
        
        # Trade pairs (entry/exit pairs for analysis)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_pairs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_trade_id TEXT NOT NULL,
                exit_trade_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                entry_time DATETIME NOT NULL,
                exit_time DATETIME NOT NULL,
                quantity INTEGER NOT NULL,
                pnl REAL NOT NULL,
                pnl_pct REAL NOT NULL,
                duration_seconds INTEGER NOT NULL,
                -- Context
                entry_atr REAL,
                exit_atr REAL,
                entry_volatility REAL,
                exit_volatility REAL,
                grid_level_entry INTEGER,
                grid_level_exit INTEGER
            )
        """)
        
        # Create indexes for fast queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_analytics_timestamp ON trades_analytics(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_analytics_symbol ON trades_analytics(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_position_history_timestamp ON position_history(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_order_events_timestamp ON order_events(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_order_events_order_id ON order_events(order_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_strategy_decisions_timestamp ON strategy_decisions(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_conditions_timestamp ON market_conditions(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_performance_metrics_timestamp ON performance_metrics(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trade_pairs_symbol ON trade_pairs(symbol)")
        
        conn.commit()
        conn.close()
        logger.info("Analytics database initialized")
    
    def record_trade_with_context(
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
        commission: Optional[float] = None,
        # Market context
        atr: Optional[float] = None,
        volatility: Optional[float] = None,
        correlation: Optional[float] = None,
        current_price: Optional[float] = None,
        grid_mid: Optional[float] = None,
        grid_spacing: Optional[float] = None,
        level_index: Optional[int] = None,
        # Position context
        position_before: Optional[int] = None,
        position_after: Optional[int] = None,
        avg_price_before: Optional[float] = None,
        avg_price_after: Optional[float] = None,
        # Risk context
        daily_pnl: Optional[float] = None,
        drawdown: Optional[float] = None,
        exposure: Optional[float] = None,
        # Market conditions
        volume: Optional[float] = None,
        price_change: Optional[float] = None,
        price_change_pct: Optional[float] = None
    ):
        """Record trade with full market and strategy context"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO trades_analytics
                (trade_id, symbol, side, quantity, price, timestamp, order_id, fill_id, pnl, commission,
                 atr, volatility, correlation, current_price, grid_mid, grid_spacing, level_index,
                 position_before, position_after, avg_price_before, avg_price_after,
                 daily_pnl, drawdown, exposure, volume, price_change, price_change_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_id, symbol, side, quantity, price, timestamp, order_id, fill_id, pnl, commission,
                atr, volatility, correlation, current_price, grid_mid, grid_spacing, level_index,
                position_before, position_after, avg_price_before, avg_price_after,
                daily_pnl, drawdown, exposure, volume, price_change, price_change_pct
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"Error recording trade with context: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def record_position_snapshot(
        self,
        symbol: str,
        quantity: int,
        avg_price: float,
        current_price: Optional[float],
        realized_pnl: float,
        unrealized_pnl: float,
        total_pnl: float,
        exposure: float,
        timestamp: Optional[datetime] = None
    ):
        """Record position snapshot"""
        if timestamp is None:
            timestamp = datetime.now()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO position_history
                (timestamp, symbol, quantity, avg_price, current_price, realized_pnl, unrealized_pnl, total_pnl, exposure)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (timestamp, symbol, quantity, avg_price, current_price, realized_pnl, unrealized_pnl, total_pnl, exposure))
            conn.commit()
        except Exception as e:
            logger.error(f"Error recording position snapshot: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def record_order_event(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: int,
        event_type: str,  # 'placed', 'filled', 'cancelled', 'modified', 'rejected'
        timestamp: Optional[datetime] = None,
        price: Optional[float] = None,
        status_before: Optional[str] = None,
        status_after: Optional[str] = None,
        fill_price: Optional[float] = None,
        fill_quantity: Optional[int] = None,
        reason: Optional[str] = None,
        market_context: Optional[Dict] = None
    ):
        """Record order lifecycle event"""
        if timestamp is None:
            timestamp = datetime.now()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            market_context_json = json.dumps(market_context) if market_context else None
            
            cursor.execute("""
                INSERT INTO order_events
                (timestamp, order_id, symbol, side, quantity, price, event_type,
                 status_before, status_after, fill_price, fill_quantity, reason, market_context_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp, order_id, symbol, side, quantity, price, event_type,
                status_before, status_after, fill_price, fill_quantity, reason, market_context_json
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"Error recording order event: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def record_strategy_decision(
        self,
        decision_type: str,  # 'grid_rebuild', 'hedge_placed', 'order_placed', 'risk_check'
        decision_data: Dict,
        symbol: Optional[str] = None,
        market_conditions: Optional[Dict] = None,
        outcome: Optional[Dict] = None,
        reasoning: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ):
        """Record strategy decision with context"""
        if timestamp is None:
            timestamp = datetime.now()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO strategy_decisions
                (timestamp, decision_type, symbol, decision_data_json, market_conditions_json, outcome_json, reasoning)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp,
                decision_type,
                symbol,
                json.dumps(decision_data),
                json.dumps(market_conditions) if market_conditions else None,
                json.dumps(outcome) if outcome else None,
                reasoning
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"Error recording strategy decision: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def record_market_conditions(
        self,
        symbol: str,
        price: float,
        timestamp: Optional[datetime] = None,
        atr: Optional[float] = None,
        volatility: Optional[float] = None,
        volume: Optional[float] = None,
        price_change: Optional[float] = None,
        price_change_pct: Optional[float] = None,
        high: Optional[float] = None,
        low: Optional[float] = None,
        grid_mid: Optional[float] = None,
        grid_spacing: Optional[float] = None,
        levels_count: Optional[int] = None,
        open_orders_count: Optional[int] = None,
        correlation: Optional[float] = None,
        hedge_price: Optional[float] = None
    ):
        """Record market conditions snapshot"""
        if timestamp is None:
            timestamp = datetime.now()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO market_conditions
                (timestamp, symbol, price, atr, volatility, volume, price_change, price_change_pct,
                 high, low, grid_mid, grid_spacing, levels_count, open_orders_count, correlation, hedge_price)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp, symbol, price, atr, volatility, volume, price_change, price_change_pct,
                high, low, grid_mid, grid_spacing, levels_count, open_orders_count, correlation, hedge_price
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"Error recording market conditions: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def record_performance_metrics(
        self,
        realized_pnl: float,
        unrealized_pnl: float,
        total_pnl: float,
        daily_pnl: float,
        drawdown: float,
        peak_balance: float,
        current_balance: float,
        exposure: float,
        max_exposure: float,
        total_trades: int,
        winning_trades: int,
        losing_trades: int,
        avg_win: float,
        avg_loss: float,
        win_rate: float,
        profit_factor: float,
        grid_fills: int,
        grid_rebuilds: int,
        hedge_activations: int,
        avg_atr: float,
        avg_volatility: float,
        avg_correlation: float,
        timestamp: Optional[datetime] = None
    ):
        """Record comprehensive performance metrics"""
        if timestamp is None:
            timestamp = datetime.now()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO performance_metrics
                (timestamp, realized_pnl, unrealized_pnl, total_pnl, daily_pnl,
                 drawdown, peak_balance, current_balance, exposure, max_exposure,
                 total_trades, winning_trades, losing_trades, avg_win, avg_loss, win_rate, profit_factor,
                 grid_fills, grid_rebuilds, hedge_activations,
                 avg_atr, avg_volatility, avg_correlation)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp, realized_pnl, unrealized_pnl, total_pnl, daily_pnl,
                drawdown, peak_balance, current_balance, exposure, max_exposure,
                total_trades, winning_trades, losing_trades, avg_win, avg_loss, win_rate, profit_factor,
                grid_fills, grid_rebuilds, hedge_activations,
                avg_atr, avg_volatility, avg_correlation
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"Error recording performance metrics: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def record_trade_pair(
        self,
        entry_trade_id: str,
        exit_trade_id: str,
        symbol: str,
        entry_price: float,
        exit_price: float,
        entry_time: datetime,
        exit_time: datetime,
        quantity: int,
        pnl: float,
        pnl_pct: float,
        duration_seconds: int,
        entry_atr: Optional[float] = None,
        exit_atr: Optional[float] = None,
        entry_volatility: Optional[float] = None,
        exit_volatility: Optional[float] = None,
        grid_level_entry: Optional[int] = None,
        grid_level_exit: Optional[int] = None
    ):
        """Record trade pair (entry/exit) for analysis"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO trade_pairs
                (entry_trade_id, exit_trade_id, symbol, entry_price, exit_price, entry_time, exit_time,
                 quantity, pnl, pnl_pct, duration_seconds,
                 entry_atr, exit_atr, entry_volatility, exit_volatility, grid_level_entry, grid_level_exit)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry_trade_id, exit_trade_id, symbol, entry_price, exit_price, entry_time, exit_time,
                quantity, pnl, pnl_pct, duration_seconds,
                entry_atr, exit_atr, entry_volatility, exit_volatility, grid_level_entry, grid_level_exit
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"Error recording trade pair: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def get_trades_for_analysis(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get trades with full context for analysis"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM trades_analytics WHERE 1=1"
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
        
        query += " ORDER BY timestamp ASC"
        
        try:
            cursor.execute(query, params)
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            logger.error(f"Error getting trades for analysis: {e}")
            return []
        finally:
            conn.close()
    
    def get_performance_summary(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get performance summary for analysis"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                AVG(CASE WHEN pnl > 0 THEN pnl END) as avg_win,
                AVG(CASE WHEN pnl < 0 THEN pnl END) as avg_loss,
                SUM(pnl) as total_pnl,
                AVG(atr) as avg_atr,
                AVG(volatility) as avg_volatility
            FROM trades_analytics
            WHERE 1=1
        """
        params = []
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        try:
            cursor.execute(query, params)
            row = cursor.fetchone()
            if row:
                return {
                    "total_trades": row[0] or 0,
                    "winning_trades": row[1] or 0,
                    "losing_trades": row[2] or 0,
                    "avg_win": row[3] or 0.0,
                    "avg_loss": row[4] or 0.0,
                    "total_pnl": row[5] or 0.0,
                    "avg_atr": row[6] or 0.0,
                    "avg_volatility": row[7] or 0.0
                }
            return {}
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {}
        finally:
            conn.close()

