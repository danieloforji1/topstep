"""
Grid Manager
Maintains grid levels, places limit orders, tracks fills, reconciles book
"""
import logging
from typing import List, Dict, Optional, Tuple, Any
from collections import namedtuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

Level = namedtuple('Level', ['price', 'side', 'size', 'level_index'])


@dataclass
class GridOrder:
    """Represents a grid order"""
    level: Level
    order_id: Optional[str] = None
    filled: bool = False
    fill_price: Optional[float] = None
    fill_quantity: int = 0


class GridManager:
    """Manages grid levels and orders"""
    
    def __init__(
        self,
        symbol: str,
        levels_each_side: int = 5,
        tick_size: float = 0.25,
        state_persistence: Optional[Any] = None
    ):
        self.symbol = symbol
        self.levels_each_side = levels_each_side
        self.tick_size = tick_size
        self.state_persistence = state_persistence  # StatePersistence instance
        
        self.grid_mid: Optional[float] = None
        self.spacing: Optional[float] = None
        self.levels: List[Level] = []
        self.orders: Dict[float, GridOrder] = {}  # price -> GridOrder
        self.filled_levels: set = set()
    
    def round_to_tick(self, price: float) -> float:
        """Round price to nearest tick"""
        return round(price / self.tick_size) * self.tick_size
    
    def generate_grid(
        self,
        mid_price: float,
        spacing: float,
        base_lot: int
    ) -> List[Level]:
        """
        Generate grid levels around midpoint
        
        Args:
            mid_price: Center price for grid
            spacing: Distance between levels (in price points)
            base_lot: Base lot size for orders
        """
        # Don't clear existing orders when regenerating grid - preserve them
        # Only clear if grid_mid is None (first time)
        if self.grid_mid is None:
            self.orders = {}
            self.filled_levels = set()
        
        self.grid_mid = mid_price
        self.spacing = spacing
        
        levels = []
        
        # Generate levels on each side
        for i in range(1, self.levels_each_side + 1):
            # Buy levels (below mid)
            buy_price = self.round_to_tick(mid_price - spacing * i)
            levels.append(Level(
                price=buy_price,
                side='BUY',
                size=base_lot,
                level_index=-i
            ))
            
            # Sell levels (above mid)
            sell_price = self.round_to_tick(mid_price + spacing * i)
            levels.append(Level(
                price=sell_price,
                side='SELL',
                size=base_lot,
                level_index=i
            ))
        
        # Sort by price
        levels.sort(key=lambda l: l.price)
        
        self.levels = levels
        logger.info(f"Generated grid: {len(levels)} levels around ${mid_price:.2f}, spacing=${spacing:.2f}")
        
        return levels
    
    def rebuild_grid_if_needed(
        self,
        current_price: float,
        spacing: float,
        base_lot: int,
        threshold: float = 0.5
    ) -> bool:
        """
        Rebuild grid if price has moved significantly
        
        Args:
            current_price: Current market price
            spacing: Current grid spacing
            threshold: Fraction of spacing to trigger rebuild (0.5 = rebuild if price moved 50% of spacing)
        
        Returns:
            True if grid was rebuilt
        """
        if self.grid_mid is None:
            self.generate_grid(current_price, spacing, base_lot)
            return True
        
        # Check if price has moved beyond threshold
        price_move = abs(current_price - self.grid_mid)
        move_threshold = spacing * threshold
        
        if price_move > move_threshold:
            logger.info(f"Price moved {price_move:.2f} from grid mid, rebuilding grid")
            self.generate_grid(current_price, spacing, base_lot)
            return True
        
        return False
    
    def get_levels_to_place(self) -> List[Level]:
        """Get levels that need orders placed"""
        levels_to_place = []
        
        for level in self.levels:
            # Skip if already filled
            if level.price in self.filled_levels:
                continue
            
            # Skip if order already exists
            if level.price in self.orders and self.orders[level.price].order_id:
                continue
            
            levels_to_place.append(level)
        
        return levels_to_place
    
    def register_order(self, level: Level, order_id: str):
        """Register an order for a level"""
        if level.price not in self.orders:
            self.orders[level.price] = GridOrder(level=level)
        
        self.orders[level.price].order_id = order_id
        logger.debug(f"Registered order {order_id} for level {level.price:.2f} ({level.side})")
    
    def on_fill(self, order_id: str, price: float, quantity: int):
        """Handle a fill"""
        # Find the order
        for grid_order in self.orders.values():
            if grid_order.order_id == order_id:
                grid_order.filled = True
                grid_order.fill_price = price
                grid_order.fill_quantity = quantity
                self.filled_levels.add(price)
                logger.info(f"Grid fill: {grid_order.level.side} {quantity} @ {price:.2f}")
                return grid_order.level
        
        logger.warning(f"Fill for unknown order: {order_id}")
        return None
    
    def cancel_all_orders(self) -> List[str]:
        """Get list of all order IDs to cancel"""
        order_ids = []
        for grid_order in self.orders.values():
            if grid_order.order_id and not grid_order.filled:
                order_ids.append(grid_order.order_id)
        return order_ids
    
    def get_open_orders_count(self) -> int:
        """Get count of open orders"""
        count = 0
        for grid_order in self.orders.values():
            if grid_order.order_id and not grid_order.filled:
                count += 1
        return count
    
    def get_open_orders(self) -> List[Dict[str, Any]]:
        """Get list of open orders with details"""
        open_orders = []
        
        # Build a set of prices that have registered orders
        registered_orders = {}
        for price, grid_order in self.orders.items():
            if grid_order.order_id and not grid_order.filled:
                registered_orders[price] = grid_order
        
        # Get all levels that should have orders (not filled)
        for level in self.levels:
            if level.price in self.filled_levels:
                continue  # Skip filled levels
            
            # If we have a registered order for this level, use it
            if level.price in registered_orders:
                grid_order = registered_orders[level.price]
                open_orders.append({
                    "order_id": grid_order.order_id,
                    "symbol": self.symbol,
                    "side": level.side,
                    "quantity": level.size,
                    "price": level.price,
                    "level_index": level.level_index
                })
            elif level.price in self.orders:
                # Order exists but might not have order_id yet (being placed)
                grid_order = self.orders[level.price]
                open_orders.append({
                    "order_id": grid_order.order_id or f"pending_{level.price}",
                    "symbol": self.symbol,
                    "side": level.side,
                    "quantity": level.size,
                    "price": level.price,
                    "level_index": level.level_index
                })
            else:
                # Level exists but no order registered yet - still show it
                open_orders.append({
                    "order_id": f"pending_{level.price}",
                    "symbol": self.symbol,
                    "side": level.side,
                    "quantity": level.size,
                    "price": level.price,
                    "level_index": level.level_index
                })
        
        return open_orders
    
    def get_filled_levels(self) -> List[GridOrder]:
        """Get all filled levels"""
        return [order for order in self.orders.values() if order.filled]
    
    def get_grid_state(self) -> Dict:
        """Get current grid state for snapshot"""
        return {
            "grid_mid": self.grid_mid,
            "spacing": self.spacing,
            "levels_count": len(self.levels),
            "open_orders": self.get_open_orders_count(),
            "filled_levels": len(self.filled_levels),
            "levels": [
                {
                    "price": level.price,
                    "side": level.side,
                    "size": level.size,
                    "level_index": level.level_index
                }
                for level in self.levels
            ]
        }
    
    def save_state(self):
        """Save current grid state to database"""
        if not self.state_persistence:
            return
        
        try:
            levels_data = [
                {
                    "price": level.price,
                    "side": level.side,
                    "size": level.size,
                    "level_index": level.level_index
                }
                for level in self.levels
            ]
            
            self.state_persistence.save_grid_state(
                symbol=self.symbol,
                grid_mid=self.grid_mid,
                spacing=self.spacing,
                levels=levels_data,
                filled_levels=list(self.filled_levels)
            )
            
            # Save all orders
            for price, grid_order in self.orders.items():
                self.state_persistence.save_grid_order(
                    symbol=self.symbol,
                    price=price,
                    side=grid_order.level.side,
                    size=grid_order.level.size,
                    level_index=grid_order.level.level_index,
                    order_id=grid_order.order_id,
                    filled=grid_order.filled,
                    fill_price=grid_order.fill_price,
                    fill_quantity=grid_order.fill_quantity
                )
            
            logger.debug(f"Saved grid state for {self.symbol}")
        except Exception as e:
            logger.error(f"Error saving grid state: {e}")
    
    def load_state(self) -> bool:
        """Load grid state from database"""
        if not self.state_persistence:
            return False
        
        try:
            state = self.state_persistence.load_grid_state(self.symbol)
            if not state:
                return False
            
            # Restore grid parameters
            self.grid_mid = state.get("grid_mid")
            self.spacing = state.get("spacing")
            
            # Restore levels
            levels_data = state.get("levels", [])
            self.levels = [
                Level(
                    price=level["price"],
                    side=level["side"],
                    size=level["size"],
                    level_index=level["level_index"]
                )
                for level in levels_data
            ]
            
            # Restore filled levels
            self.filled_levels = set(state.get("filled_levels", []))
            
            # Restore orders
            orders_data = self.state_persistence.load_grid_orders(self.symbol)
            self.orders = {}
            for order_data in orders_data:
                price = order_data["price"]
                level = Level(
                    price=price,
                    side=order_data["side"],
                    size=order_data["size"],
                    level_index=order_data["level_index"]
                )
                grid_order = GridOrder(
                    level=level,
                    order_id=order_data["order_id"],
                    filled=order_data["filled"],
                    fill_price=order_data["fill_price"],
                    fill_quantity=order_data["fill_quantity"]
                )
                self.orders[price] = grid_order
                
                # If order was filled, add to filled_levels
                if grid_order.filled:
                    self.filled_levels.add(price)
            
            logger.info(f"Loaded grid state for {self.symbol}: {len(self.levels)} levels, {len(self.orders)} orders")
            return True
        except Exception as e:
            logger.error(f"Error loading grid state: {e}")
            return False
    
    def reconcile_with_api_orders(self, api_orders: List[Dict]) -> Dict[str, Any]:
        """
        Reconcile internal state with API orders
        
        Returns:
            Dict with 'matched', 'missing', 'extra' orders
        """
        if not api_orders:
            return {"matched": [], "missing": [], "extra": []}
        
        # Build map of API orders by price
        api_orders_by_price = {}
        for order in api_orders:
            price = order.get("limitPrice") or order.get("price")
            if price:
                api_orders_by_price[price] = order
        
        matched = []
        missing = []
        extra = []
        
        # Check our orders against API
        for price, grid_order in self.orders.items():
            if grid_order.order_id and not grid_order.filled:
                if price in api_orders_by_price:
                    matched.append({
                        "price": price,
                        "order_id": grid_order.order_id,
                        "api_order": api_orders_by_price[price]
                    })
                else:
                    # Our order not in API - might have been filled or cancelled
                    missing.append({
                        "price": price,
                        "order_id": grid_order.order_id
                    })
        
        # Check for API orders we don't know about
        for price, api_order in api_orders_by_price.items():
            if price not in self.orders:
                extra.append({
                    "price": price,
                    "api_order": api_order
                })
        
        return {
            "matched": matched,
            "missing": missing,
            "extra": extra
        }

