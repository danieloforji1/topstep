"""
Fill Handler
Reconciles fills, updates strategy state, triggers post-fill actions
"""
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


class FillHandler:
    """Handles order fills and updates strategy state"""
    
    def __init__(
        self,
        position_manager,
        grid_manager,
        order_client
    ):
        self.position_manager = position_manager
        self.grid_manager = grid_manager
        self.order_client = order_client
    
    def on_fill(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        fill_id: Optional[str] = None
    ):
        """
        Handle an order fill
        
        Args:
            order_id: Order ID
            symbol: Instrument symbol
            side: Buy or Sell
            quantity: Filled quantity
            price: Fill price
            fill_id: Optional fill ID
        """
        logger.info(f"Fill received: {order_id} - {side} {quantity} {symbol} @ {price:.2f}")
        
        # Update position manager
        self.position_manager.on_fill(symbol, side, quantity, price)
        
        # Update grid manager if this is a grid order
        if symbol == self.grid_manager.symbol:
            level = self.grid_manager.on_fill(order_id, price, quantity)
            if level:
                logger.debug(f"Grid level filled: {level.side} @ {level.price:.2f}")
        
        # Record trade in history (would be done by main loop)
        return {
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "fill_id": fill_id,
            "timestamp": datetime.now()
        }
    
    def process_fills_from_api(self, fills: List[Dict[str, Any]]):
        """Process fills from API response"""
        processed = []
        for fill in fills:
            try:
                result = self.on_fill(
                    order_id=fill.get("orderId") or fill.get("id"),
                    symbol=fill.get("symbol") or fill.get("contractId"),
                    side="BUY" if fill.get("side") == 0 or fill.get("side", "").upper() == "BUY" else "SELL",
                    quantity=fill.get("quantity") or fill.get("size"),
                    price=fill.get("price") or fill.get("fillPrice"),
                    fill_id=fill.get("fillId") or fill.get("id")
                )
                processed.append(result)
            except Exception as e:
                logger.error(f"Error processing fill: {e}")
        
        return processed

