"""
Order Client
Atomic order submit/cancel/modify with retry logic and idempotency
"""
import logging
import time
import uuid
from typing import Optional, Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


class OrderClient:
    """Handles order execution with retry and idempotency"""
    
    def __init__(self, api_client, dry_run: bool = True):
        self.api_client = api_client
        self.dry_run = dry_run
        self.pending_orders: Dict[str, Dict[str, Any]] = {}  # order_id -> order info
        self.idempotency_keys: Dict[str, str] = {}  # order_key -> order_id
    
    def place_limit_order(
        self,
        contract_id: str,
        side: str,
        quantity: int,
        price: float,
        idempotency_key: Optional[str] = None
    ) -> Optional[str]:
        """
        Place a limit order with idempotency
        
        Returns:
            Order ID if successful, None otherwise
        """
        if idempotency_key is None:
            idempotency_key = str(uuid.uuid4())
        
        # Check if we already have this order
        if idempotency_key in self.idempotency_keys:
            existing_order_id = self.idempotency_keys[idempotency_key]
            logger.debug(f"Order with key {idempotency_key} already exists: {existing_order_id}")
            return existing_order_id
        
        # Place order with retry
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = self.api_client.place_order(
                    contract_id=contract_id,
                    side=side,
                    quantity=quantity,
                    order_type="Limit",
                    price=price
                )
                
                if result and (result.get("success", True) or "orderId" in result):
                    order_id = result.get("orderId") or result.get("id")
                    if order_id:
                        self.pending_orders[order_id] = {
                            "contract_id": contract_id,
                            "side": side,
                            "quantity": quantity,
                            "price": price,
                            "timestamp": datetime.now(),
                            "idempotency_key": idempotency_key
                        }
                        self.idempotency_keys[idempotency_key] = order_id
                        logger.info(f"Placed order: {order_id} - {side} {quantity} @ {price:.2f}")
                        return order_id
                
                logger.warning(f"Order placement returned no order ID: {result}")
                
            except Exception as e:
                logger.error(f"Error placing order (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    return None
        
        return None

    def place_market_order(
        self,
        contract_id: str,
        side: str,
        quantity: int,
        idempotency_key: Optional[str] = None
    ) -> Optional[str]:
        """
        Place a market order with idempotency.
        """
        if idempotency_key is None:
            idempotency_key = str(uuid.uuid4())

        if idempotency_key in self.idempotency_keys:
            existing_order_id = self.idempotency_keys[idempotency_key]
            logger.debug(f"Order with key {idempotency_key} already exists: {existing_order_id}")
            return existing_order_id

        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = self.api_client.place_order(
                    contract_id=contract_id,
                    side=side,
                    quantity=quantity,
                    order_type="Market",
                    price=None
                )

                if result and (result.get("success", True) or "orderId" in result):
                    order_id = result.get("orderId") or result.get("id")
                    if order_id:
                        self.pending_orders[order_id] = {
                            "contract_id": contract_id,
                            "side": side,
                            "quantity": quantity,
                            "price": None,
                            "timestamp": datetime.now(),
                            "idempotency_key": idempotency_key
                        }
                        self.idempotency_keys[idempotency_key] = order_id
                        logger.info(f"Placed market order: {order_id} - {side} {quantity}")
                        return order_id

                logger.warning(f"Market order placement returned no order ID: {result}")
            except Exception as e:
                logger.error(f"Error placing market order (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return None

        return None
    
    def place_stop_order(
        self,
        contract_id: str,
        side: str,
        quantity: int,
        stop_price: float,
        idempotency_key: Optional[str] = None
    ) -> Optional[str]:
        """
        Place a stop order with idempotency.
        
        Args:
            contract_id: Contract ID
            side: "BUY" or "SELL"
            quantity: Number of contracts
            stop_price: Stop price that triggers the order
            idempotency_key: Optional key to prevent duplicate orders
        """
        if idempotency_key is None:
            idempotency_key = str(uuid.uuid4())
        
        if idempotency_key in self.idempotency_keys:
            existing_order_id = self.idempotency_keys[idempotency_key]
            logger.debug(f"Order with key {idempotency_key} already exists: {existing_order_id}")
            return existing_order_id
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = self.api_client.place_order(
                    contract_id=contract_id,
                    side=side,
                    quantity=quantity,
                    order_type="Stop",
                    price=None,
                    stop_price=stop_price
                )
                
                if result and (result.get("success", True) or "orderId" in result):
                    order_id = result.get("orderId") or result.get("id")
                    if order_id:
                        self.pending_orders[order_id] = {
                            "contract_id": contract_id,
                            "side": side,
                            "quantity": quantity,
                            "stop_price": stop_price,
                            "timestamp": datetime.now(),
                            "idempotency_key": idempotency_key
                        }
                        self.idempotency_keys[idempotency_key] = order_id
                        logger.info(f"Placed stop order: {order_id} - {side} {quantity} @ stop {stop_price:.2f}")
                        return order_id
                
                logger.warning(f"Stop order placement returned no order ID: {result}")
            except Exception as e:
                logger.error(f"Error placing stop order (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return None
        
        return None
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                success = self.api_client.cancel_order(order_id)
                if success:
                    if order_id in self.pending_orders:
                        del self.pending_orders[order_id]
                    logger.info(f"Canceled order: {order_id}")
                    return True
            except Exception as e:
                logger.error(f"Error canceling order (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return False
        
        return False
    
    def cancel_all_orders(self, order_ids: List[str]) -> int:
        """
        Cancel multiple orders with rate limiting to avoid 429 errors.
        
        Adds a small delay between cancellations and handles rate limits gracefully.
        """
        if not order_ids:
            return 0
        
        canceled = 0
        failed = 0
        rate_limited = False
        
        logger.info(f"Canceling {len(order_ids)} orders with rate limiting...")
        
        for i, order_id in enumerate(order_ids):
            # Add delay between cancellations to respect rate limits
            # Start with 0.2s delay, increase if we hit rate limits
            if i > 0:  # Don't delay before first order
                delay = 0.3 if rate_limited else 0.2
                time.sleep(delay)
            
            try:
                success = self.cancel_order(order_id)
                if success:
                    canceled += 1
                    rate_limited = False  # Reset flag on success
                else:
                    failed += 1
                    # If cancellation failed, it might be due to rate limiting
                    # Check if we should increase delay
                    if i < len(order_ids) - 1:  # Not the last order
                        time.sleep(0.5)  # Extra delay after failure
                        rate_limited = True
            except Exception as e:
                logger.error(f"Exception canceling order {order_id}: {e}")
                failed += 1
                # If we get an exception, likely rate limited - add extra delay
                if i < len(order_ids) - 1:
                    time.sleep(1.0)  # Longer delay after exception
                    rate_limited = True
        
        logger.info(f"Canceled {canceled}/{len(order_ids)} orders ({failed} failed)")
        return canceled
    
    def modify_order(
        self,
        order_id: str,
        quantity: Optional[int] = None,
        price: Optional[float] = None
    ) -> bool:
        """Modify an existing order"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                success = self.api_client.modify_order(
                    order_id=order_id,
                    quantity=quantity,
                    price=price
                )
                if success:
                    if order_id in self.pending_orders:
                        if quantity is not None:
                            self.pending_orders[order_id]["quantity"] = quantity
                        if price is not None:
                            self.pending_orders[order_id]["price"] = price
                    logger.info(f"Modified order: {order_id}")
                    return True
            except Exception as e:
                logger.error(f"Error modifying order (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return False
        
        return False
    
    def get_pending_orders(self) -> Dict[str, Dict[str, Any]]:
        """Get all pending orders"""
        return self.pending_orders.copy()
    
    def reconcile_orders(self, api_orders: List[Dict[str, Any]]):
        """Reconcile pending orders with API state"""
        api_order_ids = {order.get("orderId") or order.get("id") for order in api_orders if order}
        
        # Remove orders that no longer exist in API
        to_remove = []
        for order_id in self.pending_orders:
            if order_id not in api_order_ids:
                to_remove.append(order_id)
        
        for order_id in to_remove:
            logger.debug(f"Removing reconciled order: {order_id}")
            del self.pending_orders[order_id]

