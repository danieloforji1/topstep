"""
State Persistence
Persists grid state, orders, and positions to database for continuity
"""
import sqlite3
import logging
import json
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class StatePersistence:
    """Persists strategy state to database"""
    
    def __init__(self, db_path: str = "data/trading.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables for state persistence"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Grid state table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS grid_state (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                grid_mid REAL,
                spacing REAL,
                levels_json TEXT,
                filled_levels_json TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol)
            )
        """)
        
        # Grid orders table (tracks orders for each grid level)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS grid_orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                price REAL NOT NULL,
                side TEXT NOT NULL,
                size INTEGER NOT NULL,
                level_index INTEGER NOT NULL,
                order_id TEXT,
                filled BOOLEAN DEFAULT 0,
                fill_price REAL,
                fill_quantity INTEGER DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, price, side)
            )
        """)
        
        # Strategy session table (tracks active sessions)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategy_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                account_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                initial_balance REAL,
                notes TEXT
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_grid_orders_symbol ON grid_orders(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_grid_orders_filled ON grid_orders(filled)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_active ON strategy_sessions(is_active)")
        
        conn.commit()
        conn.close()
    
    def save_grid_state(
        self,
        symbol: str,
        grid_mid: Optional[float],
        spacing: Optional[float],
        levels: List[Dict],
        filled_levels: List[float]
    ):
        """Save grid state"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            levels_json = json.dumps(levels)
            filled_levels_json = json.dumps(filled_levels)
            
            cursor.execute("""
                INSERT OR REPLACE INTO grid_state 
                (symbol, grid_mid, spacing, levels_json, filled_levels_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                symbol,
                grid_mid,
                spacing,
                levels_json,
                filled_levels_json,
                datetime.now().isoformat()
            ))
            conn.commit()
            logger.debug(f"Saved grid state for {symbol}")
        except Exception as e:
            logger.error(f"Error saving grid state: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def load_grid_state(self, symbol: str) -> Optional[Dict]:
        """Load grid state"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT grid_mid, spacing, levels_json, filled_levels_json
                FROM grid_state
                WHERE symbol = ?
            """, (symbol,))
            
            row = cursor.fetchone()
            if row:
                grid_mid, spacing, levels_json, filled_levels_json = row
                return {
                    "grid_mid": grid_mid,
                    "spacing": spacing,
                    "levels": json.loads(levels_json) if levels_json else [],
                    "filled_levels": json.loads(filled_levels_json) if filled_levels_json else []
                }
            return None
        except Exception as e:
            logger.error(f"Error loading grid state: {e}")
            return None
        finally:
            conn.close()
    
    def save_grid_order(
        self,
        symbol: str,
        price: float,
        side: str,
        size: int,
        level_index: int,
        order_id: Optional[str] = None,
        filled: bool = False,
        fill_price: Optional[float] = None,
        fill_quantity: int = 0
    ):
        """Save or update a grid order"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO grid_orders
                (symbol, price, side, size, level_index, order_id, filled, fill_price, fill_quantity, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol,
                price,
                side,
                size,
                level_index,
                order_id,
                1 if filled else 0,
                fill_price,
                fill_quantity,
                datetime.now().isoformat()
            ))
            conn.commit()
            logger.debug(f"Saved grid order: {symbol} {side} {size} @ {price}")
        except Exception as e:
            logger.error(f"Error saving grid order: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def load_grid_orders(self, symbol: str, only_open: bool = False) -> List[Dict]:
        """Load grid orders"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            if only_open:
                cursor.execute("""
                    SELECT price, side, size, level_index, order_id, filled, fill_price, fill_quantity
                    FROM grid_orders
                    WHERE symbol = ? AND filled = 0
                """, (symbol,))
            else:
                cursor.execute("""
                    SELECT price, side, size, level_index, order_id, filled, fill_price, fill_quantity
                    FROM grid_orders
                    WHERE symbol = ?
                """, (symbol,))
            
            rows = cursor.fetchall()
            orders = []
            for row in rows:
                price, side, size, level_index, order_id, filled, fill_price, fill_quantity = row
                orders.append({
                    "price": price,
                    "side": side,
                    "size": size,
                    "level_index": level_index,
                    "order_id": order_id,
                    "filled": bool(filled),
                    "fill_price": fill_price,
                    "fill_quantity": fill_quantity
                })
            return orders
        except Exception as e:
            logger.error(f"Error loading grid orders: {e}")
            return []
        finally:
            conn.close()
    
    def mark_order_filled(self, symbol: str, order_id: str, fill_price: float, fill_quantity: int):
        """Mark an order as filled"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE grid_orders
                SET filled = 1, fill_price = ?, fill_quantity = ?, updated_at = ?
                WHERE symbol = ? AND order_id = ?
            """, (
                fill_price,
                fill_quantity,
                datetime.now().isoformat(),
                symbol,
                order_id
            ))
            conn.commit()
            logger.debug(f"Marked order {order_id} as filled")
        except Exception as e:
            logger.error(f"Error marking order as filled: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def delete_grid_orders(self, symbol: str):
        """Delete all grid orders for a symbol (cleanup)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("DELETE FROM grid_orders WHERE symbol = ?", (symbol,))
            conn.commit()
            logger.debug(f"Deleted all grid orders for {symbol}")
        except Exception as e:
            logger.error(f"Error deleting grid orders: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def create_session(
        self,
        account_id: int,
        symbol: str,
        initial_balance: float,
        notes: Optional[str] = None
    ) -> int:
        """Create a new strategy session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Mark previous sessions as inactive
            cursor.execute("""
                UPDATE strategy_sessions
                SET is_active = 0, last_updated = ?
                WHERE account_id = ? AND symbol = ? AND is_active = 1
            """, (datetime.now().isoformat(), account_id, symbol))
            
            # Create new session
            cursor.execute("""
                INSERT INTO strategy_sessions
                (account_id, symbol, initial_balance, notes, started_at, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                account_id,
                symbol,
                initial_balance,
                notes,
                datetime.now().isoformat(),
                datetime.now().isoformat()
            ))
            
            session_id = cursor.lastrowid
            conn.commit()
            logger.info(f"Created strategy session {session_id} for {symbol}")
            return session_id
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            conn.rollback()
            return 0
        finally:
            conn.close()
    
    def update_session(self, account_id: int, symbol: str):
        """Update session last_updated timestamp"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE strategy_sessions
                SET last_updated = ?
                WHERE account_id = ? AND symbol = ? AND is_active = 1
            """, (datetime.now().isoformat(), account_id, symbol))
            conn.commit()
        except Exception as e:
            logger.error(f"Error updating session: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def get_active_session(self, account_id: int, symbol: str) -> Optional[Dict]:
        """Get active session for account/symbol"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT id, started_at, last_updated, initial_balance, notes
                FROM strategy_sessions
                WHERE account_id = ? AND symbol = ? AND is_active = 1
                ORDER BY started_at DESC
                LIMIT 1
            """, (account_id, symbol))
            
            row = cursor.fetchone()
            if row:
                return {
                    "id": row[0],
                    "started_at": row[1],
                    "last_updated": row[2],
                    "initial_balance": row[3],
                    "notes": row[4]
                }
            return None
        except Exception as e:
            logger.error(f"Error getting active session: {e}")
            return None
        finally:
            conn.close()
    
    def clear_state(self, symbol: str):
        """Clear all state for a symbol (use with caution)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("DELETE FROM grid_state WHERE symbol = ?", (symbol,))
            cursor.execute("DELETE FROM grid_orders WHERE symbol = ?", (symbol,))
            conn.commit()
            logger.info(f"Cleared all state for {symbol}")
        except Exception as e:
            logger.error(f"Error clearing state: {e}")
            conn.rollback()
        finally:
            conn.close()

