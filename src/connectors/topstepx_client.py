"""
TopstepX API Client
Handles authentication, REST API calls, and WebSocket connections
"""
import os
import time
import logging
import requests
from typing import Optional, Dict, List, Any, Callable
from datetime import datetime
import json
import threading

logger = logging.getLogger(__name__)

# SignalR is optional - we'll use REST polling as fallback
try:
    from signalrcore.hub_connection_builder import HubConnectionBuilder
    SIGNALR_AVAILABLE = True
except ImportError:
    SIGNALR_AVAILABLE = False
    logger.warning("SignalR not available, will use REST polling for real-time updates")


class TopstepXClient:
    """Client for TopstepX Gateway API"""
    
    def __init__(
        self,
        username: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: str = "https://api.topstepx.com",
        user_hub_url: str = "https://rtc.topstepx.com/hubs/user",
        market_hub_url: str = "https://rtc.topstepx.com/hubs/market",
        dry_run: bool = True
    ):
        self.username = username or os.getenv("TOPSTEPX_USERNAME")
        self.api_key = api_key or os.getenv("TOPSTEPX_API_KEY")
        self.base_url = base_url.rstrip('/')
        self.user_hub_url = user_hub_url
        self.market_hub_url = market_hub_url
        self.dry_run = dry_run
        
        self.session_token: Optional[str] = None
        self.token_expiry: Optional[datetime] = None
        self.account_id: Optional[int] = None
        
        self.user_hub_connection: Optional[Any] = None
        self.market_hub_connection: Optional[Any] = None
        self.realtime_callbacks: Dict[str, Callable] = {}
        
        # Session validation caching
        self._last_validation: Optional[datetime] = None
        self._validation_cache_seconds: int = 300  # Only validate every 5 minutes
        self._auth_retry_count: int = 0
        self._last_auth_attempt: Optional[datetime] = None
        
        self._session = requests.Session()
        self._session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
    def authenticate(self) -> bool:
        """Authenticate with TopstepX API and obtain session token"""
        if not self.username or not self.api_key:
            logger.error("Missing TOPSTEPX_USERNAME or TOPSTEPX_API_KEY")
            return False
        
        # Exponential backoff for rate limiting
        if self._last_auth_attempt:
            time_since_last = (datetime.now() - self._last_auth_attempt).total_seconds()
            backoff_seconds = min(2 ** self._auth_retry_count, 60)  # Max 60 seconds
            if time_since_last < backoff_seconds:
                logger.warning(f"Rate limited - waiting {backoff_seconds - time_since_last:.1f}s before retry")
                return False
        
        url = f"{self.base_url}/api/Auth/loginKey"
        payload = {
            "userName": self.username,
            "apiKey": self.api_key
        }
        
        try:
            self._last_auth_attempt = datetime.now()
            response = self._session.post(url, json=payload, timeout=10)
            
            # Handle rate limiting
            if response.status_code == 429:
                self._auth_retry_count += 1
                logger.warning(f"Rate limited (429). Retry count: {self._auth_retry_count}")
                return False
            
            response.raise_for_status()
            data = response.json()
            
            if data.get("success") and data.get("errorCode") == 0:
                self.session_token = data.get("token")
                # Tokens typically last 24 hours, but set expiry to 23 hours to be safe
                from datetime import timedelta
                self.token_expiry = datetime.now() + timedelta(hours=23)
                self._session.headers.update({
                    'Authorization': f'Bearer {self.session_token}'
                })
                self._auth_retry_count = 0  # Reset on success
                logger.info("Successfully authenticated with TopstepX")
                return True
            else:
                logger.error(f"Authentication failed: {data.get('errorMessage')}")
                self._auth_retry_count += 1
                return False
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            self._auth_retry_count += 1
            return False
    
    def validate_session(self) -> bool:
        """Validate and refresh session token if needed (with caching)"""
        if not self.session_token:
            return self.authenticate()
        
        # Check if token is close to expiry (within 1 hour)
        if self.token_expiry:
            time_until_expiry = (self.token_expiry - datetime.now()).total_seconds()
            if time_until_expiry > 3600:  # More than 1 hour left
                # Token is still valid, skip validation
                return True
        
        # Check if we validated recently (cache validation for 5 minutes)
        if self._last_validation:
            time_since_validation = (datetime.now() - self._last_validation).total_seconds()
            if time_since_validation < self._validation_cache_seconds:
                # Recently validated, skip
                return True
        
        # Only validate if token is close to expiry or cache expired
        url = f"{self.base_url}/api/Auth/validate"
        
        try:
            response = self._session.post(url, timeout=10)
            
            # Handle rate limiting gracefully
            if response.status_code == 429:
                logger.warning("Rate limited on validation - using cached session")
                self._last_validation = datetime.now()  # Update cache to avoid retrying immediately
                return True  # Assume session is still valid
            
            response.raise_for_status()
            data = response.json()
            
            if data.get("success"):
                if "newToken" in data:
                    self.session_token = data["newToken"]
                    self._session.headers.update({
                        'Authorization': f'Bearer {self.session_token}'
                    })
                    # Reset expiry when we get a new token
                    from datetime import timedelta
                    self.token_expiry = datetime.now() + timedelta(hours=23)
                self._last_validation = datetime.now()
                return True
            return False
        except Exception as e:
            logger.warning(f"Session validation failed: {e}, will re-authenticate if needed")
            # Don't immediately re-authenticate on validation failure
            # Only re-authenticate if token is actually expired
            self._last_validation = datetime.now()  # Update cache to avoid retrying immediately
            return True  # Assume session is still valid for now
    
    def _ensure_authenticated(self):
        """Ensure we have a valid session token (only validates when needed)"""
        # If no token, authenticate
        if not self.session_token:
            if not self.authenticate():
                raise Exception("Failed to authenticate with TopstepX")
            return
        
        # Check if token is expired
        if self.token_expiry and datetime.now() >= self.token_expiry:
            logger.info("Token expired, re-authenticating")
            if not self.authenticate():
                raise Exception("Failed to authenticate with TopstepX")
            return
        
        # Only validate session if token is close to expiry (within 1 hour)
        # This reduces API calls significantly
        if self.token_expiry:
            time_until_expiry = (self.token_expiry - datetime.now()).total_seconds()
            if time_until_expiry <= 3600:  # Less than 1 hour left
                # Token is close to expiry, validate it
                if not self.validate_session():
                    # Validation failed, try to re-authenticate
                    logger.warning("Session validation failed, re-authenticating")
                    if not self.authenticate():
                        raise Exception("Failed to authenticate with TopstepX")
    
    def get_accounts(self, only_active: bool = True) -> List[Dict[str, Any]]:
        """Get list of trading accounts"""
        self._ensure_authenticated()
        url = f"{self.base_url}/api/Account/search"
        
        payload = {
            "onlyActiveAccounts": only_active
        }
        
        try:
            response = self._session.post(url, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data.get("success") and "accounts" in data:
                return data["accounts"]
            return []
        except Exception as e:
            logger.error(f"Error fetching accounts: {e}")
            return []
    
    def set_account(self, account_id: int):
        """Set the active trading account"""
        self.account_id = account_id
        logger.info(f"Set active account: {account_id}")
    
    def search_contracts(self, symbol: Optional[str] = None, live: bool = False) -> List[Dict[str, Any]]:
        """Search for available contracts"""
        self._ensure_authenticated()
        url = f"{self.base_url}/api/Contract/search"
        
        if not symbol:
            logger.warning("No symbol provided for contract search")
            return []
        
        payload = {
            "searchText": symbol,
            "live": live
        }
        
        try:
            response = self._session.post(url, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data.get("success") and "contracts" in data:
                return data["contracts"]
            return []
        except Exception as e:
            logger.error(f"Error searching contracts: {e}")
            return []
    
    def get_contract_by_id(self, contract_id: str) -> Optional[Dict[str, Any]]:
        """Get contract details by ID"""
        self._ensure_authenticated()
        url = f"{self.base_url}/api/MarketData/Contracts/{contract_id}"
        
        try:
            response = self._session.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching contract: {e}")
            return None
    
    def get_bars(
        self,
        contract_id: str,
        interval: str = "15m",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
        live: bool = False
    ) -> List[Dict[str, Any]]:
        """Retrieve historical bars/candles"""
        self._ensure_authenticated()
        url = f"{self.base_url}/api/History/retrieveBars"
        
        # Convert interval string to unit and unitNumber
        # interval format: "15m" = 15 minutes, "1h" = 1 hour, etc.
        unit_map = {
            "s": 1,  # Second
            "m": 2,  # Minute
            "h": 3,  # Hour
            "d": 4,  # Day
            "w": 5,  # Week
            "M": 6   # Month
        }
        
        unit = 2  # Default to minutes
        unit_number = 15  # Default to 15
        
        if interval:
            # Parse interval like "15m", "1h", etc.
            import re
            match = re.match(r'(\d+)([smhdwM])', interval)
            if match:
                unit_number = int(match.group(1))
                unit_char = match.group(2)
                unit = unit_map.get(unit_char, 2)
        
        # Default to last 24 hours if no times specified
        if not end_time:
            end_time = datetime.now()
        if not start_time:
            from datetime import timedelta
            start_time = end_time - timedelta(days=1)
        
        payload = {
            "contractId": contract_id,
            "live": live,
            "startTime": start_time.isoformat() + "Z" if start_time.tzinfo is None else start_time.isoformat(),
            "endTime": end_time.isoformat() + "Z" if end_time.tzinfo is None else end_time.isoformat(),
            "unit": unit,
            "unitNumber": unit_number,
            "limit": limit,
            "includePartialBar": False
        }
        
        # Calculate dynamic timeout based on limit
        # Base timeout: 10 seconds for 100 bars
        # Scale up: ~0.01 seconds per bar, minimum 10s, maximum 60s
        base_timeout = 10
        timeout = max(base_timeout, min(60, base_timeout + (limit * 0.01)))
        
        # For very large requests (2000+ bars), use longer timeout and retry logic
        max_retries = 3 if limit > 1000 else 1
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                response = self._session.post(url, json=payload, timeout=timeout)
                response.raise_for_status()
                data = response.json()
                if data.get("success") and "bars" in data:
                    return data["bars"]
                return []
            except (requests.exceptions.Timeout, requests.Timeout) as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Timeout fetching bars (attempt {attempt + 1}/{max_retries}, "
                        f"limit={limit}, timeout={timeout}s). Retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
                    # Increase timeout for retry
                    timeout = min(120, timeout * 1.5)
                    retry_delay *= 2
                else:
                    logger.error(f"Error fetching bars: Timeout after {max_retries} attempts (limit={limit}, timeout={timeout}s)")
                    return []
            except Exception as e:
                # For non-timeout errors, don't retry (might be auth, network, etc.)
                if attempt == 0:  # Only log on first attempt
                    logger.error(f"Error fetching bars: {e}")
                return []
        
        return []
    
    def place_order(
        self,
        contract_id: str,
        side: str,  # "Buy" or "Sell"
        quantity: int,
        order_type: str = "Limit",  # Limit, Market, Stop, etc.
        price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Place an order"""
        if self.dry_run:
            logger.info(f"[DRY RUN] Would place order: {side} {quantity} {contract_id} @ {price}")
            return {
                "orderId": f"dry_run_{int(time.time())}",
                "status": "Open",
                "success": True
            }
        
        self._ensure_authenticated()
        if not self.account_id:
            raise Exception("No account selected")
        
        url = f"{self.base_url}/api/Order/place"
        
        order_side = 0 if side.lower() == "buy" else 1  # Bid=0, Ask=1
        order_type_map = {
            "Limit": 1,
            "Market": 2,
            "Stop": 4,
            "StopLimit": 3,
            "TrailingStop": 5,
            "JoinBid": 6,
            "JoinAsk": 7
        }
        order_type_code = order_type_map.get(order_type, 1)
        
        payload = {
            "accountId": self.account_id,
            "contractId": contract_id,
            "type": order_type_code,
            "side": order_side,
            "size": quantity,
            "limitPrice": price,
            "stopPrice": stop_price,
            "trailPrice": None,
            "customTag": None
        }
        
        try:
            response = self._session.post(url, json=payload, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order with rate limiting and 429 handling"""
        if self.dry_run:
            logger.info(f"[DRY RUN] Would cancel order: {order_id}")
            return True
        
        self._ensure_authenticated()
        if not self.account_id:
            raise Exception("No account selected")
        
        url = f"{self.base_url}/api/Order/cancel"
        
        payload = {
            "accountId": self.account_id,
            "orderId": int(order_id) if isinstance(order_id, str) and order_id.isdigit() else order_id
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self._session.post(url, json=payload, timeout=10)
                
                # Handle 429 rate limit errors specifically
                if response.status_code == 429:
                    # Extract retry-after header if available
                    retry_after = response.headers.get('Retry-After')
                    if retry_after:
                        wait_time = int(retry_after)
                    else:
                        # Exponential backoff: 1s, 2s, 4s
                        wait_time = 2 ** attempt
                    
                    if attempt < max_retries - 1:
                        logger.warning(f"Rate limited (429) when canceling order {order_id}, waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Rate limited (429) when canceling order {order_id} after {max_retries} attempts")
                        return False
                
                response.raise_for_status()
                data = response.json()
                return data.get("success", False)
                
            except requests.exceptions.HTTPError as e:
                if e.response and e.response.status_code == 429:
                    # Already handled above, but catch here too
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        logger.warning(f"Rate limited (429) when canceling order {order_id}, waiting {wait_time}s")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Rate limited (429) when canceling order {order_id} after {max_retries} attempts")
                        return False
                else:
                    logger.error(f"HTTP error canceling order {order_id}: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                    else:
                        return False
            except Exception as e:
                logger.error(f"Error canceling order {order_id} (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return False
        
        return False
    
    def modify_order(
        self,
        order_id: str,
        quantity: Optional[int] = None,
        price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> bool:
        """Modify an existing order"""
        if self.dry_run:
            logger.info(f"[DRY RUN] Would modify order: {order_id}")
            return True
        
        self._ensure_authenticated()
        if not self.account_id:
            raise Exception("No account selected")
        
        url = f"{self.base_url}/api/Order/modify"
        
        payload = {
            "accountId": self.account_id,
            "orderId": int(order_id) if isinstance(order_id, str) and order_id.isdigit() else order_id
        }
        
        if quantity is not None:
            payload["size"] = quantity
        if price is not None:
            payload["limitPrice"] = price
        if stop_price is not None:
            payload["stopPrice"] = stop_price
        
        try:
            response = self._session.post(url, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get("success", False)
        except Exception as e:
            logger.error(f"Error modifying order: {e}")
            return False
    
    def get_orders(
        self,
        account_id: Optional[int] = None,
        start_timestamp: Optional[datetime] = None,
        end_timestamp: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get all orders for an account"""
        self._ensure_authenticated()
        account = account_id or self.account_id
        if not account:
            return []
        
        url = f"{self.base_url}/api/Order/search"
        
        # startTimestamp is required - default to 24 hours ago if not provided
        if not start_timestamp:
            from datetime import timedelta
            start_timestamp = datetime.now() - timedelta(days=1)
        
        payload = {
            "accountId": account,
            "startTimestamp": start_timestamp.isoformat() + "Z" if start_timestamp.tzinfo is None else start_timestamp.isoformat()
        }
        
        if end_timestamp:
            payload["endTimestamp"] = end_timestamp.isoformat() + "Z" if end_timestamp.tzinfo is None else end_timestamp.isoformat()
        
        try:
            response = self._session.post(url, json=payload, timeout=10)
            
            # Handle rate limiting (429 errors)
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 5))
                logger.warning(f"Rate limited (429) on get_orders. Waiting {retry_after} seconds...")
                import time
                time.sleep(retry_after)
                # Retry once after waiting
                response = self._session.post(url, json=payload, timeout=10)
            
            response.raise_for_status()
            data = response.json()
            if data.get("success") and "orders" in data:
                return data["orders"]
            return []
        except Exception as e:
            # Don't log 429 errors as ERROR (they're expected during rate limits)
            if "429" in str(e) or "Too Many Requests" in str(e):
                logger.warning(f"Rate limited on get_orders: {e}")
            else:
                logger.error(f"Error fetching orders: {e}")
            return []
    
    def get_open_orders(self, account_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get open orders for an account"""
        self._ensure_authenticated()
        account = account_id or self.account_id
        if not account:
            return []
        
        url = f"{self.base_url}/api/Order/searchOpen"
        
        payload = {
            "accountId": account
        }
        
        try:
            response = self._session.post(url, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data.get("success") and "orders" in data:
                return data["orders"]
            return []
        except Exception as e:
            logger.error(f"Error fetching open orders: {e}")
            return []
    
    def get_positions(self, account_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get positions for an account"""
        self._ensure_authenticated()
        account = account_id or self.account_id
        if not account:
            return []
        
        url = f"{self.base_url}/api/Position/searchOpen"
        
        payload = {
            "accountId": account
        }
        
        try:
            response = self._session.post(url, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data.get("success") and "positions" in data:
                return data["positions"]
            return []
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return []
    
    def close_position(self, contract_id: str, account_id: Optional[int] = None) -> bool:
        """Close a position by contract ID"""
        if self.dry_run:
            logger.info(f"[DRY RUN] Would close position: {contract_id}")
            return True
        
        self._ensure_authenticated()
        account = account_id or self.account_id
        if not account:
            raise Exception("No account selected")
        
        url = f"{self.base_url}/api/Position/closeContract"
        
        payload = {
            "accountId": account,
            "contractId": contract_id
        }
        
        try:
            response = self._session.post(url, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get("success", False)
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False
    
    def get_account_balance(self, account_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Get account balance and P&L"""
        self._ensure_authenticated()
        account = account_id or self.account_id
        if not account:
            return None
        
        # This would typically be part of the Account API
        # For now, we'll use positions to calculate P&L
        positions = self.get_positions(account)
        # In a real implementation, you'd get balance from Account endpoint
        return {
            "accountId": account,
            "positions": positions
        }
    
    def connect_realtime(self, account_id: Optional[int] = None, contract_ids: Optional[List[str]] = None):
        """Connect to real-time WebSocket hubs using SignalR
        
        Args:
            account_id: Account ID for user hub subscriptions
            contract_ids: List of contract IDs to subscribe to for market hub quotes
        """
        if not self.session_token:
            self.authenticate()
        
        if not SIGNALR_AVAILABLE:
            logger.info("SignalR not available, using REST polling for real-time updates")
            return False
        
        account = account_id or self.account_id
        if not account:
            logger.warning("No account ID provided for SignalR connection")
            return False
        
        try:
            # Convert https:// to wss:// for WebSocket connections
            # signalrcore's websocket-client requires wss:// not https://
            user_hub_ws_url = self.user_hub_url.replace("https://", "wss://")
            user_hub_url = f"{user_hub_ws_url}?access_token={self.session_token}"
            
            # Create access token factory function
            def get_token():
                return self.session_token
            
            self.user_hub_connection = HubConnectionBuilder()\
                .with_url(user_hub_url, options={
                    "access_token_factory": get_token,
                    "skip_negotiation": True
                })\
                .with_automatic_reconnect({
                    "type": "raw",
                    "keep_alive_interval": 10,
                    "reconnect_interval": 5,
                    "max_attempts": 5
                })\
                .build()
            
            # Register connection event handlers
            self.user_hub_connection.on_open(lambda: logger.info("SignalR User Hub: Connection opened and ready"))
            self.user_hub_connection.on_close(lambda: logger.warning("SignalR User Hub: Connection closed"))
            self.user_hub_connection.on_error(lambda data: logger.error(f"SignalR User Hub error: {data}"))
            
            # Register event handlers for user hub events
            self.user_hub_connection.on("GatewayUserAccount", self._on_account_update)
            self.user_hub_connection.on("GatewayUserOrder", self._on_order_update)
            self.user_hub_connection.on("GatewayUserPosition", self._on_position_update)
            self.user_hub_connection.on("GatewayUserTrade", self._on_trade_update)
            
            # Start connection
            self.user_hub_connection.start()
            
            # Wait a bit for connection to establish
            import time
            time.sleep(2)
            
            # Subscribe to updates using send() method (signalrcore uses send, not invoke)
            # send() takes method name and list of parameters
            try:
                self.user_hub_connection.send("SubscribeAccounts", [])
                self.user_hub_connection.send("SubscribeOrders", [account])
                self.user_hub_connection.send("SubscribePositions", [account])
                self.user_hub_connection.send("SubscribeTrades", [account])
                logger.info("Subscribed to User Hub events (accounts, orders, positions, trades)")
            except Exception as send_err:
                logger.error(f"Error subscribing to User Hub events: {send_err}")
            
            logger.info("Connected to SignalR User Hub")
            
            # Market Hub connection (optional, for market data)
            if contract_ids:
                market_hub_ws_url = self.market_hub_url.replace("https://", "wss://")
                market_hub_url = f"{market_hub_ws_url}?access_token={self.session_token}"
                
                self.market_hub_connection = HubConnectionBuilder()\
                    .with_url(market_hub_url, options={
                        "access_token_factory": get_token,
                        "skip_negotiation": True
                    })\
                    .with_automatic_reconnect({
                        "type": "raw",
                        "keep_alive_interval": 10,
                        "reconnect_interval": 5,
                        "max_attempts": 5
                    })\
                    .build()
                
                # Register connection event handlers
                self.market_hub_connection.on_open(lambda: logger.info("SignalR Market Hub: Connection opened and ready"))
                self.market_hub_connection.on_close(lambda: logger.warning("SignalR Market Hub: Connection closed"))
                self.market_hub_connection.on_error(lambda data: logger.error(f"SignalR Market Hub error: {data}"))
                
                # Register event handlers for market hub events
                # Market Hub events receive contractId as first parameter, then data
                self.market_hub_connection.on("GatewayQuote", self._on_market_quote_update)
                self.market_hub_connection.on("GatewayTrade", self._on_market_trade_update)
                self.market_hub_connection.on("GatewayDepth", self._on_depth_update)
                
                self.market_hub_connection.start()
                
                # Wait for market hub connection
                time.sleep(2)
                
                # Subscribe to contract quotes and trades for each contract
                try:
                    for contract_id in contract_ids:
                        self.market_hub_connection.send("SubscribeContractQuotes", [contract_id])
                        self.market_hub_connection.send("SubscribeContractTrades", [contract_id])
                        logger.info(f"Subscribed to market data for contract: {contract_id}")
                except Exception as sub_err:
                    logger.warning(f"Error subscribing to market hub contracts: {sub_err}")
                
                logger.info("Connected to SignalR Market Hub")
            
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to SignalR: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            logger.info("Falling back to REST polling")
            return False
    
    def _on_account_update(self, data: Dict[str, Any]):
        """Handle account update from SignalR"""
        if "on_account_update" in self.realtime_callbacks:
            self.realtime_callbacks["on_account_update"](data)
        logger.debug(f"Account update: {data}")
    
    def _on_order_update(self, data: Dict[str, Any]):
        """Handle order update from SignalR"""
        if "on_order_update" in self.realtime_callbacks:
            self.realtime_callbacks["on_order_update"](data)
        logger.debug(f"Order update: {data}")
    
    def _on_position_update(self, data: Dict[str, Any]):
        """Handle position update from SignalR"""
        if "on_position_update" in self.realtime_callbacks:
            self.realtime_callbacks["on_position_update"](data)
        logger.debug(f"Position update: {data}")
    
    def _on_trade_update(self, data: Dict[str, Any]):
        """Handle trade update from SignalR"""
        if "on_trade_update" in self.realtime_callbacks:
            self.realtime_callbacks["on_trade_update"](data)
        logger.debug(f"Trade update: {data}")
    
    def _on_quote_update(self, data: Dict[str, Any]):
        """Handle market quote update from SignalR (User Hub - not used for market data)"""
        logger.debug(f"Quote update (User Hub): {data}")
    
    def _on_market_quote_update(self, *args):
        """Handle market quote update from SignalR Market Hub
        
        Market Hub GatewayQuote may receive:
        - (contract_id, data) - two arguments
        - (data,) - one argument (data only)
        - data might be a list or dict
        """
        # Handle both cases: (contract_id, data) or (data,)
        if len(args) == 2:
            contract_id, data = args
        elif len(args) == 1:
            data = args[0]
            contract_id = None
        else:
            logger.warning(f"Unexpected GatewayQuote arguments: {args}")
            return
        
        # Convert list to dict if needed (signalrcore sometimes passes lists)
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict):
                data = data[0]  # Take first element if it's a dict
            elif len(data) == 2:
                # Might be [contract_id, data_dict]
                contract_id, data = data
            else:
                logger.debug(f"Quote data is list but not in expected format: {data}")
                return
        
        # Ensure data is a dict before processing
        if not isinstance(data, dict):
            logger.debug(f"Quote data is not a dict: {type(data)}, value: {data}")
            return
        
        logger.debug(f"Market quote update for contract {contract_id}: {data}")
        
        # Ensure callback can match updates by contract ID.
        if contract_id and isinstance(data, dict) and "contractId" not in data:
            data["contractId"] = contract_id
        
        # Pass data to callback
        if "on_quote_update" in self.realtime_callbacks:
            self.realtime_callbacks["on_quote_update"](data)
    
    def _on_market_trade_update(self, *args):
        """Handle market trade update from SignalR
        
        Market Hub GatewayTrade may receive:
        - (contract_id, data) - two arguments
        - (data,) - one argument (data only)
        - data might be a list or dict
        """
        # Handle both cases: (contract_id, data) or (data,)
        if len(args) == 2:
            contract_id, data = args
        elif len(args) == 1:
            data = args[0]
            contract_id = None
        else:
            logger.warning(f"Unexpected GatewayTrade arguments: {args}")
            return
        
        # Convert list to dict if needed (signalrcore sometimes passes lists)
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict):
                data = data[0]  # Take first element if it's a dict
            elif len(data) == 2:
                # Might be [contract_id, data_dict]
                contract_id, data = data
            else:
                logger.debug(f"Trade data is list but not in expected format: {data}")
                return
        
        # Ensure data is a dict before processing
        if not isinstance(data, dict):
            logger.debug(f"Trade data is not a dict: {type(data)}, value: {data}")
            return
        
        logger.debug(f"Market trade update for contract {contract_id}: {data}")
        
        # Ensure callback can match updates by contract ID.
        if contract_id and isinstance(data, dict) and "contractId" not in data:
            data["contractId"] = contract_id
        
        # Pass data to callback
        if "on_market_trade_update" in self.realtime_callbacks:
            self.realtime_callbacks["on_market_trade_update"](data)
    
    def _on_depth_update(self, data: Dict[str, Any]):
        """Handle market depth (DOM) update from SignalR"""
        logger.debug(f"Market depth update: {data}")
    
    def register_realtime_callback(self, event: str, callback: Callable):
        """Register a callback for real-time events"""
        self.realtime_callbacks[event] = callback
    
    def disconnect(self):
        """Disconnect from API and close connections"""
        try:
            if self.user_hub_connection:
                self.user_hub_connection.stop()
                self.user_hub_connection = None
            if self.market_hub_connection:
                self.market_hub_connection.stop()
                self.market_hub_connection = None
            logger.info("Disconnected from TopstepX SignalR hubs")
        except Exception as e:
            logger.error(f"Error disconnecting SignalR: {e}")

