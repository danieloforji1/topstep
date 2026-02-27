"""
Position Manager
Tracks net positions, realized/unrealized P&L, and enforces exposure caps
"""
import logging
import json
from typing import Dict, Optional, List, Any
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


class Position:
    """Represents a position"""
    def __init__(self, symbol: str, quantity: int, avg_price: float):
        self.symbol = symbol
        self.quantity = quantity  # Positive for long, negative for short
        self.avg_price = avg_price
        self.realized_pnl = 0.0
        self.trades = []  # List of (price, quantity, side) tuples
    
    def add_trade(self, price: float, quantity: int, side: str):
        """Add a trade to the position"""
        self.trades.append((price, quantity, side, datetime.now()))
        
        # Update average price and quantity
        if side.upper() == "BUY":
            if self.quantity >= 0:  # Adding to long or opening long
                total_cost = (self.quantity * self.avg_price) + (quantity * price)
                self.quantity += quantity
                if self.quantity > 0:
                    self.avg_price = total_cost / self.quantity
            else:  # Closing short
                # Calculate realized P&L
                pnl = (self.avg_price - price) * min(abs(self.quantity), quantity)
                self.realized_pnl += pnl
                self.quantity += quantity
                if self.quantity < 0:
                    # Still short, update avg price
                    total_cost = (abs(self.quantity) * self.avg_price) - (quantity * price)
                    self.avg_price = total_cost / abs(self.quantity) if self.quantity != 0 else price
        else:  # SELL
            if self.quantity <= 0:  # Adding to short or opening short
                total_cost = (abs(self.quantity) * self.avg_price) + (quantity * price)
                self.quantity -= quantity
                if self.quantity < 0:
                    self.avg_price = total_cost / abs(self.quantity)
            else:  # Closing long
                # Calculate realized P&L
                pnl = (price - self.avg_price) * min(self.quantity, quantity)
                self.realized_pnl += pnl
                self.quantity -= quantity
                if self.quantity > 0:
                    # Still long, update avg price
                    total_cost = (self.quantity * self.avg_price) - (quantity * price)
                    self.avg_price = total_cost / self.quantity if self.quantity != 0 else price
    
    def get_unrealized_pnl(self, current_price: float, tick_value: float = 5.0, tick_size: float = 0.25) -> float:
        """Calculate unrealized P&L (properly converts price_diff to ticks first)"""
        if self.quantity == 0:
            return 0.0
        
        price_diff = current_price - self.avg_price
        if self.quantity < 0:  # Short position
            price_diff = -price_diff
        
        # Convert price movement to ticks, then calculate P&L
        ticks = price_diff / tick_size if tick_size > 0 else 0.0
        return ticks * tick_value * abs(self.quantity)
    
    def get_total_pnl(self, current_price: float, tick_value: float = 5.0, tick_size: float = 0.25) -> float:
        """Get total P&L (realized + unrealized)"""
        return self.realized_pnl + self.get_unrealized_pnl(current_price, tick_value, tick_size)


class PositionManager:
    """Manages positions across multiple instruments"""
    
    def __init__(
        self,
        max_net_notional: float = 1200.0,
        tick_values: Optional[Dict[str, float]] = None,
        tick_sizes: Optional[Dict[str, float]] = None,
        contract_multipliers: Optional[Dict[str, float]] = None
    ):
        self.max_net_notional = max_net_notional
        self.tick_values = tick_values or {"MES": 5.0, "MNQ": 2.0}  # Default tick values
        # Contract multipliers: GC = 100 oz, MGC = 10 oz
        self.contract_multipliers = contract_multipliers or {"GC": 100.0, "MGC": 10.0, "MES": 5.0, "MNQ": 2.0}
        # Tick sizes for proper P&L calculation
        self.tick_sizes = tick_sizes or {"GC": 0.10, "MGC": 0.10, "MES": 0.25, "MNQ": 0.25}
        self.positions: Dict[str, Position] = {}
        self.current_prices: Dict[str, float] = {}
        
        # API-sourced P&L values (source of truth from TopstepX)
        self.api_realized_pnl: Dict[str, float] = {}  # symbol -> realized P&L from API
        self.api_unrealized_pnl: Dict[str, float] = {}  # symbol -> unrealized P&L from API
        self.api_total_pnl: Dict[str, float] = {}  # symbol -> total P&L from API
        self.last_api_sync: Optional[datetime] = None
    
    def update_price(self, symbol: str, price: float):
        """Update current price for a symbol"""
        self.current_prices[symbol] = price
    
    def on_fill(self, symbol: str, side: str, quantity: int, price: float):
        """
        Handle a fill.
        
        NOTE: This is for immediate tracking only. Positions are periodically
        reconciled with API (every 30s) which is the source of truth.
        """
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol, 0, price)
        
        # Only add trade if we haven't reconciled recently
        # If last sync was recent (< 5 seconds), skip internal tracking
        # to avoid double-counting with API reconciliation
        if self.last_api_sync and (datetime.now() - self.last_api_sync).total_seconds() < 5:
            logger.debug(f"Skipping internal fill tracking for {symbol} - recent API sync, will use API positions")
            self.update_price(symbol, price)
            return
        
        self.positions[symbol].add_trade(price, quantity, side)
        self.update_price(symbol, price)
        
        logger.info(f"Fill: {side} {quantity} {symbol} @ {price}")
    
    def get_position(self, symbol: str) -> Position:
        """Get position for a symbol"""
        return self.positions.get(symbol, Position(symbol, 0, 0.0))
    
    def get_net_position(self, symbol: str) -> int:
        """Get net position quantity"""
        pos = self.get_position(symbol)
        return pos.quantity
    
    def reconcile_with_api_positions(self, api_positions: List[Dict[str, Any]]):
        """
        Reconcile internal positions with TopstepX API positions.
        Uses API values as source of truth for P&L.
        
        CRITICAL: This REPLACES internal positions with API positions.
        If a symbol is not in api_positions, it means position is 0 (closed).
        
        Args:
            api_positions: List of position dicts from TopstepX API
        """
        # Track which symbols we see in API response
        symbols_seen = set()
        
        if not api_positions:
            # API says no positions - clear all internal positions
            logger.debug("API reports no positions - clearing all internal positions")
            self.positions.clear()
            return
        
        # Log the full API response structure ONCE (first time only) to see what fields are available
        if api_positions and not hasattr(self, '_api_structure_logged'):
            logger.info(f"TopstepX API Position Data Structure (first position):")
            logger.info(f"  Full response: {json.dumps(api_positions[0], indent=2, default=str)}")
            logger.info(f"  Available fields: {list(api_positions[0].keys())}")
            self._api_structure_logged = True
        
        # Map API positions by contract/symbol
        for api_pos in api_positions:
            contract_id = api_pos.get('contractId', '')
            # Extract symbol from contract ID (e.g., "CON.F.US.MES.H25" -> "MES")
            symbol = None
            if contract_id:
                parts = contract_id.split('.')
                if len(parts) >= 4:
                    symbol = parts[3]  # Usually the symbol is in position 3
            
            if not symbol:
                logger.debug(f"Could not extract symbol from contract_id: {contract_id}")
                continue
            
            # Get position data from API
            size = api_pos.get('size', 0)
            avg_price = api_pos.get('averagePrice', 0.0)
            position_type = api_pos.get('type', 0)  # 1=Long, 2=Short
            
            # Changed to DEBUG to reduce log noise - only log if position changed
            if symbol not in self.positions or self.positions[symbol].quantity != (size if position_type == 1 else -size if position_type == 2 else 0):
                logger.debug(f"Position from API - Symbol: {symbol}, Size: {size}, AvgPrice: {avg_price}, Type: {position_type}")
            
            # Convert to our format (positive=long, negative=short)
            quantity = size if position_type == 1 else -size if position_type == 2 else 0
            symbols_seen.add(symbol)
            
            # Update or create position
            # CRITICAL: API is the source of truth - REPLACE, don't add
            if symbol not in self.positions:
                self.positions[symbol] = Position(symbol, quantity, avg_price)
            else:
                # Update position from API (source of truth)
                # Clear trade history to prevent accumulation
                self.positions[symbol].quantity = quantity
                self.positions[symbol].avg_price = avg_price
                self.positions[symbol].trades = []  # Clear trade history - API is source of truth
                self.positions[symbol].realized_pnl = 0.0  # Reset - API P&L will be used
            
            # Store API P&L values if available (these are the source of truth)
            # Note: TopstepX API may provide P&L in different fields
            # Check common field names (trying both camelCase and snake_case)
            api_unrealized = (
                api_pos.get('unrealizedPnl') or
                api_pos.get('unrealizedProfitLoss') or
                api_pos.get('openProfitLoss') or
                api_pos.get('unrealized_pnl') or
                None
            )
            api_realized = (
                api_pos.get('realizedPnl') or
                api_pos.get('closedProfitLoss') or
                api_pos.get('realized_profit_loss') or
                api_pos.get('realized_pnl') or
                None
            )
            api_pnl = (
                api_pos.get('profitAndLoss') or
                api_pos.get('pnl') or
                api_pos.get('realizedPnl') or
                api_pos.get('unrealizedPnl') or
                api_pos.get('profit_and_loss') or  # Try snake_case
                api_pos.get('realized_pnl') or
                api_pos.get('unrealized_pnl') or
                None
            )
            
            # Changed to DEBUG - only log P&L fields on first reconciliation or if P&L becomes available
            if api_pnl is not None:
                # If API provides total P&L, we'll use it
                if symbol not in self.api_total_pnl or self.api_total_pnl[symbol] != float(api_pnl):
                    logger.debug(f"P&L fields in API response for {symbol}: Using API P&L ${api_pnl:.2f}")
                self.api_total_pnl[symbol] = float(api_pnl)
            else:
                # Only log warning once per symbol per session (not every reconciliation)
                if not hasattr(self, '_pnl_warning_logged'):
                    self._pnl_warning_logged = set()
                if symbol not in self._pnl_warning_logged:
                    logger.debug(f"P&L fields in API response for {symbol}: No P&L field found, will calculate internally")
                    self._pnl_warning_logged.add(symbol)
            
            if api_unrealized is not None:
                self.api_unrealized_pnl[symbol] = float(api_unrealized)
            else:
                self.api_unrealized_pnl.pop(symbol, None)
            
            if api_realized is not None:
                self.api_realized_pnl[symbol] = float(api_realized)
            else:
                self.api_realized_pnl.pop(symbol, None)
        
        # CRITICAL: Clear positions for symbols NOT in API response (they were closed)
        # This prevents stale positions from accumulating
        symbols_to_remove = set(self.positions.keys()) - symbols_seen
        for symbol in symbols_to_remove:
            logger.debug(f"Position {symbol} not in API response (closed) - removing from internal tracking")
            del self.positions[symbol]
            # Also clear API P&L tracking
            self.api_total_pnl.pop(symbol, None)
            self.api_realized_pnl.pop(symbol, None)
            self.api_unrealized_pnl.pop(symbol, None)
        
        self.last_api_sync = datetime.now()
    
    def get_realized_pnl(self, symbol: Optional[str] = None) -> float:
        """
        Get realized P&L.
        Uses API values if available (source of truth), otherwise calculates from internal tracking.
        """
        if symbol:
            # Prefer API value if available
            if symbol in self.api_realized_pnl:
                return self.api_realized_pnl[symbol]
            # Fallback to internal calculation
            return self.positions.get(symbol, Position(symbol, 0, 0.0)).realized_pnl
        
        # Total across all symbols
        total = 0.0
        for sym in set(list(self.api_realized_pnl.keys()) + list(self.positions.keys())):
            if sym in self.api_realized_pnl:
                total += self.api_realized_pnl[sym]
            elif sym in self.positions:
                total += self.positions[sym].realized_pnl
        return total
    
    def get_unrealized_pnl(self, symbol: Optional[str] = None) -> float:
        """
        Get unrealized P&L.
        Uses API values if available (source of truth), otherwise calculates from current prices.
        """
        if symbol:
            # Prefer API value if available
            if symbol in self.api_unrealized_pnl:
                return self.api_unrealized_pnl[symbol]
            # Fallback to internal calculation
            pos = self.positions.get(symbol)
            if not pos:
                return 0.0
            tick_value = self.tick_values.get(symbol, 5.0)
            tick_size = self.tick_sizes.get(symbol, 0.25)
            price = self.current_prices.get(symbol, pos.avg_price)
            return pos.get_unrealized_pnl(price, tick_value, tick_size)
        
        # Total across all symbols
        total = 0.0
        for sym in set(list(self.api_unrealized_pnl.keys()) + list(self.positions.keys())):
            if sym in self.api_unrealized_pnl:
                total += self.api_unrealized_pnl[sym]
            elif sym in self.positions:
                pos = self.positions[sym]
                tick_value = self.tick_values.get(sym, 5.0)
                tick_size = self.tick_sizes.get(sym, 0.25)
                price = self.current_prices.get(sym, pos.avg_price)
                total += pos.get_unrealized_pnl(price, tick_value, tick_size)
        return total
    
    def get_total_pnl(self) -> float:
        """
        Get total P&L (realized + unrealized).
        Uses API values if available (source of truth), otherwise calculates.
        """
        # If we have API total P&L for all positions, use that
        if self.api_total_pnl:
            total = sum(self.api_total_pnl.values())
            if total != 0 or not self.positions:  # If API has values or no internal positions
                return total
        
        # Fallback to calculated values
        return self.get_realized_pnl() + self.get_unrealized_pnl()
    
    def net_exposure_dollars(self) -> float:
        """Calculate net exposure in dollars (gross notional value)"""
        total = 0.0
        for symbol, pos in self.positions.items():
            price = self.current_prices.get(symbol, pos.avg_price)
            # Proper notional calculation: quantity * price * contract_multiplier
            # GC multiplier = 100 oz, MGC multiplier = 10 oz
            contract_multiplier = self.contract_multipliers.get(symbol, 100.0)
            notional = abs(pos.quantity) * price * contract_multiplier
            total += notional
        
        return total
    
    def is_exposure_capped(self) -> bool:
        """Check if exposure cap is reached"""
        exposure = self.net_exposure_dollars()
        return exposure >= self.max_net_notional
    
    def flatten_all(self):
        """Reset all positions (for emergency flatten)"""
        logger.warning("Flattening all positions")
        self.positions.clear()
    
    def get_all_positions(self) -> Dict[str, Position]:
        """Get all positions"""
        return self.positions.copy()

