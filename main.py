"""
Main Trading Loop
Grid Strategy with Cross-Asset Hedging for TopstepX
"""
import os
import sys
import time
import logging
import yaml
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from connectors.topstepx_client import TopstepXClient
from connectors.market_data_adapter import MarketDataAdapter
from data.timeseries_store import TimeseriesStore
from data.trade_history import TradeHistory
from data.state_persistence import StatePersistence
from data.analytics_store import AnalyticsStore
from indicators.technical import calculate_atr, calculate_volatility, calculate_correlation
from strategy.grid_manager import GridManager
from strategy.position_manager import PositionManager
from strategy.risk_manager import RiskManager
from strategy.sizer import Sizer
from strategy.hedge_manager import HedgeManager
from execution.order_client import OrderClient
from execution.fill_handler import FillHandler
from observability.metrics import MetricsExporter
from api.ops import app, set_strategy_instance
import uvicorn
from threading import Thread

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class GridStrategy:
    """Main grid strategy implementation"""
    
    def __init__(self, config_path: str = "config.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        # Check environment variable first (overrides config), then config, default to True (safe)
        env_dry_run = os.getenv('DRY_RUN', '').lower()
        if env_dry_run:
            self.dry_run = env_dry_run == 'true'
        else:
            self.dry_run = self.config.get('dry_run', True)
        
        # API Client
        self.api_client = TopstepXClient(
            username=os.getenv('TOPSTEPX_USERNAME'),
            api_key=os.getenv('TOPSTEPX_API_KEY'),
            base_url=self.config.get('api_base_url', 'https://api.topstepx.com'),
            user_hub_url=self.config.get('user_hub_url', 'https://rtc.topstepx.com/hubs/user'),
            market_hub_url=self.config.get('market_hub_url', 'https://rtc.topstepx.com/hubs/market'),
            dry_run=self.dry_run
        )
        
        # Data stores
        self.timeseries_store = TimeseriesStore()
        self.trade_history = TradeHistory()
        self.state_persistence = StatePersistence()
        self.analytics_store = AnalyticsStore()
        
        # Strategy components
        self.primary_symbol = self.config.get('primary_instrument', 'MES')
        self.hedge_symbol = self.config.get('hedge_instrument', 'MNQ')
        
        self.grid_manager = GridManager(
            symbol=self.primary_symbol,
            levels_each_side=self.config.get('grid_levels_each_side', 5),
            tick_size=0.25,  # MES tick size
            state_persistence=self.state_persistence
        )
        
        self.position_manager = PositionManager(
            max_net_notional=self.config.get('max_net_notional', 1200.0),
            tick_values={self.primary_symbol: 5.0, self.hedge_symbol: 2.0}
        )
        
        self.risk_manager = RiskManager(
            max_daily_loss=self.config.get('max_daily_loss', 900.0),
            trailing_drawdown_limit=self.config.get('trailing_drawdown_limit', 1800.0),
            max_net_notional=self.config.get('max_net_notional', 1200.0)
        )
        
        # Get max position size from config (for 50k challenge: 5 contracts)
        max_position_size = self.config.get('max_position_size', 10)
        max_lot_per_order = min(max_position_size, 5)  # Never exceed 5 per order for challenge
        
        self.sizer = Sizer(
            R_per_trade=self.config.get('sizer_R_per_trade', 150.0),
            tick_value=5.0,  # MES tick value
            min_lot=1,
            max_lot=max_lot_per_order  # Respect challenge limit
        )
        
        # Store max position size for total position checking
        self.max_position_size = max_position_size
        
        self.hedge_manager = HedgeManager(
            primary_symbol=self.primary_symbol,
            hedge_symbol=self.hedge_symbol,
            min_hedge_ratio=self.config.get('hedge_ratio_min', 0.5),
            max_hedge_ratio=self.config.get('hedge_ratio_max', 1.25),
            correlation_threshold=self.config.get('correlation_threshold', 0.6),
            max_hedge_contracts_multiplier=self.config.get('max_hedge_contracts_multiplier', 1.5)
        )
        
        # Execution
        self.order_client = OrderClient(self.api_client, dry_run=self.dry_run)
        self.fill_handler = FillHandler(
            self.position_manager,
            self.grid_manager,
            self.order_client
        )
        
        # State
        self.running = False
        self.paused = False
        self.last_recalc_time = datetime.now()
        self.recalc_interval = timedelta(seconds=self.config.get('recalc_interval_seconds', 60))
        self.last_state_save = datetime.now()
        self.state_save_interval = timedelta(seconds=30)  # Save state every 30 seconds
        
        # Contract IDs (will be fetched on startup)
        self.primary_contract_id: Optional[str] = None
        self.hedge_contract_id: Optional[str] = None
        
        # Current prices
        self.current_price: Optional[float] = None
        self.hedge_price: Optional[float] = None
        
        logger.info(f"Grid Strategy initialized (dry_run={self.dry_run})")
        if not self.dry_run:
            logger.warning("=" * 60)
            logger.warning("LIVE TRADING MODE ENABLED - REAL ORDERS WILL BE PLACED!")
            logger.warning("=" * 60)
    
    def initialize(self) -> bool:
        """Initialize connection and fetch contract info"""
        logger.info("Initializing strategy...")
        
        # Authenticate
        if not self.api_client.authenticate():
            logger.error("Failed to authenticate with TopstepX")
            return False
        
        # Get accounts
        accounts = self.api_client.get_accounts()
        if not accounts:
            logger.error("No accounts found")
            return False
        
        # Log all available accounts for debugging
        logger.info(f"Found {len(accounts)} account(s):")
        for acc in accounts:
            acc_id = acc.get('id') or acc.get('accountId', 'Unknown')
            acc_name = acc.get('name', 'Unknown')
            is_sim = acc.get('simulated', False)
            balance = acc.get('balance', 0.0)
            logger.info(f"  - {acc_name} (ID: {acc_id}, Simulated: {is_sim}, Balance: ${balance:.2f})")
        
        # Select account based on configuration
        account = None
        prefer_practice = self.config.get('prefer_practice_account', True)
        specified_account_id = self.config.get('account_id')
        
        if specified_account_id:
            # Use specified account ID
            account = next((acc for acc in accounts if (acc.get('id') == specified_account_id or 
                                                        acc.get('accountId') == specified_account_id)), None)
            if account:
                logger.info(f"Using specified account ID: {specified_account_id}")
            else:
                logger.warning(f"Specified account ID {specified_account_id} not found, falling back to auto-selection")
        
        if not account:
            if prefer_practice:
                # Prefer practice accounts - look for accounts with "PRAC" or "Practice" in name
                # Prioritize accounts that START with "PRAC"
                practice_accounts_start = []  # Accounts starting with PRAC
                practice_accounts_contain = []  # Accounts containing PRAC
                practice_accounts_sim = []  # Accounts with simulated flag
                
                for acc in accounts:
                    name = acc.get('name', '').upper()
                    is_simulated = acc.get('simulated', False)
                    
                    if name.startswith('PRAC'):
                        practice_accounts_start.append(acc)
                    elif 'PRAC' in name or 'PRACTICE' in name:
                        practice_accounts_contain.append(acc)
                    elif is_simulated:
                        practice_accounts_sim.append(acc)
                
                # Priority: start with PRAC > contains PRAC > simulated flag
                if practice_accounts_start:
                    account = practice_accounts_start[0]
                    logger.info(f"Selected practice account (starts with PRAC): {account.get('name', 'Unknown')} (ID: {account.get('id')})")
                elif practice_accounts_contain:
                    account = practice_accounts_contain[0]
                    logger.info(f"Selected practice account (contains PRAC): {account.get('name', 'Unknown')} (ID: {account.get('id')})")
                elif practice_accounts_sim:
                    account = practice_accounts_sim[0]
                    logger.info(f"Selected practice account (simulated flag): {account.get('name', 'Unknown')} (ID: {account.get('id')})")
                else:
                    # Fallback to any account if no practice accounts found
                    account = accounts[0]
                    logger.warning(f"No practice accounts found (looking for 'PRAC' in name or simulated=true), using first available account: {account.get('name', 'Unknown')}")
            else:
                # Use first account (could be live)
                account = accounts[0]
                logger.info(f"Using first account: {account.get('name', 'Unknown')} (ID: {account.get('id')})")
        
        if account:
            account_id = account.get('id') or account.get('accountId')
            if account_id:
                self.api_client.set_account(account_id)
                account_name = account.get('name', 'Unknown')
                is_simulated = account.get('simulated', False)
                logger.info(f"Using account: {account_name} (ID: {account_id}, Simulated: {is_simulated})")
                
                # Get initial balance from account
                initial_balance = account.get('balance', 0.0)
                if initial_balance > 0:
                    self._initial_balance = initial_balance
                    self.risk_manager.update_balance(initial_balance)
                    logger.info(f"Initial account balance: ${initial_balance:.2f}")
                else:
                    # Fallback to config or default
                    self._initial_balance = self.config.get('capital_equiv', 150000.0)
                    self.risk_manager.update_balance(self._initial_balance)
                    logger.warning(f"Balance not in account data, using config default: ${self._initial_balance:.2f}")
            else:
                logger.error("Selected account has no ID")
                return False
        else:
            logger.error("Could not select an account")
            return False
        
        # Search for contracts
        primary_contracts = self.api_client.search_contracts(self.primary_symbol)
        hedge_contracts = self.api_client.search_contracts(self.hedge_symbol)
        
        if primary_contracts:
            self.primary_contract_id = primary_contracts[0].get('contractId') or primary_contracts[0].get('id')
            logger.info(f"Primary contract: {self.primary_symbol} -> {self.primary_contract_id}")
        
        if hedge_contracts:
            self.hedge_contract_id = hedge_contracts[0].get('contractId') or hedge_contracts[0].get('id')
            logger.info(f"Hedge contract: {self.hedge_symbol} -> {self.hedge_contract_id}")
        
        if not self.primary_contract_id:
            logger.error(f"Could not find contract for {self.primary_symbol}")
            return False
        
        # Load historical data
        self._load_historical_data()
        
        # Restore state from database and reconcile with API
        self._restore_and_reconcile_state()
        
        # Create/update strategy session
        if self._initial_balance:
            self.state_persistence.create_session(
                account_id=self.api_client.account_id,
                symbol=self.primary_symbol,
                initial_balance=self._initial_balance,
                notes=f"Grid strategy session - {self.primary_symbol}"
            )
        
        # Connect to real-time data and register callbacks
        self._setup_realtime_callbacks()
        
        # Try to connect to SignalR for real-time updates
        # Pass contract IDs for market hub subscriptions
        contract_ids = []
        if self.primary_contract_id:
            contract_ids.append(self.primary_contract_id)
        if self.hedge_contract_id:
            contract_ids.append(self.hedge_contract_id)
        
        self.api_client.connect_realtime(
            account_id=self.api_client.account_id,
            contract_ids=contract_ids if contract_ids else None
        )
        
        logger.info("Strategy initialized successfully")
        return True
    
    def _load_historical_data(self):
        """Load historical candles for indicators"""
        logger.info("Loading historical data...")
        
        # Get bars from API
        bars = self.api_client.get_bars(
            contract_id=self.primary_contract_id,
            interval="15m",
            limit=max(self.config.get('atr_window', 14), self.config.get('volatility_window', 100))
        )
        
        if bars:
            adapter = MarketDataAdapter()
            candles = adapter.normalize_bars(bars, self.primary_symbol, "15m")
            self.timeseries_store.store_candles(candles)
            logger.info(f"Loaded {len(candles)} historical candles")
            
            # Initialize current_price from most recent candle
            if candles and not self.current_price:
                self.current_price = candles[-1].close
                self.position_manager.update_price(self.primary_symbol, self.current_price)
                logger.info(f"Initialized {self.primary_symbol} price from historical data: {self.current_price:.2f}")
        
        # Load hedge data
        if self.hedge_contract_id:
            hedge_bars = self.api_client.get_bars(
                contract_id=self.hedge_contract_id,
                interval="15m",
                limit=self.config.get('volatility_window', 100)
            )
            if hedge_bars:
                adapter = MarketDataAdapter()
                hedge_candles = adapter.normalize_bars(hedge_bars, self.hedge_symbol, "15m")
                self.timeseries_store.store_candles(hedge_candles)
                logger.info(f"Loaded {len(hedge_candles)} hedge candles")
                
                # Initialize hedge_price from most recent candle
                if hedge_candles and not self.hedge_price:
                    self.hedge_price = hedge_candles[-1].close
                    self.position_manager.update_price(self.hedge_symbol, self.hedge_price)
                    logger.info(f"Initialized {self.hedge_symbol} price from historical data: {self.hedge_price:.2f}")
    
    def _restore_and_reconcile_state(self):
        """Restore state from database and reconcile with API"""
        logger.info("Restoring state from database...")
        
        # CRITICAL: Reconcile positions FIRST (before loading grid state)
        # This ensures we start with correct positions from API
        if not self.dry_run and self.api_client.account_id:
            try:
                import time
                time.sleep(1)  # Small delay to avoid rate limits on startup
                api_positions = self.api_client.get_positions(self.api_client.account_id)
                if api_positions:
                    logger.info(f"Reconciling positions on startup: {len(api_positions)} positions from API")
                    self.position_manager.reconcile_with_api_positions(api_positions)
                    # Log current positions after reconciliation
                    for symbol in ['MES', 'MNQ']:
                        pos = self.position_manager.get_net_position(symbol)
                        if pos != 0:
                            logger.info(f"  {symbol}: {pos} contracts (from API)")
            except Exception as e:
                logger.warning(f"Could not reconcile positions on startup: {e}")
        
        # Try to load grid state
        state_loaded = self.grid_manager.load_state()
        
        if state_loaded:
            logger.info("Loaded grid state from database")
            
            # Reconcile with API orders if not in dry_run
            if not self.dry_run and self.api_client.account_id:
                try:
                    import time
                    time.sleep(1)  # Small delay between API calls to avoid rate limits
                    api_orders = self.api_client.get_open_orders(self.api_client.account_id)
                    reconciliation = self.grid_manager.reconcile_with_api_orders(api_orders)
                    
                    logger.info(f"Reconciliation: {len(reconciliation['matched'])} matched, "
                              f"{len(reconciliation['missing'])} missing, "
                              f"{len(reconciliation['extra'])} extra orders")
                    
                    # Handle missing orders (might have been filled or cancelled)
                    # CRITICAL: Fetch recent orders ONCE, not in a loop (prevents 429 errors)
                    if reconciliation['missing']:
                        try:
                            import time
                            time.sleep(1)  # Delay before fetching orders to avoid rate limits
                            # Fetch recent orders once for all missing orders
                            recent_orders = self.api_client.get_orders(
                                self.api_client.account_id,
                                start_timestamp=datetime.now() - timedelta(hours=1)
                            )
                            # Create a lookup dict for fast searching
                            recent_orders_dict = {
                                str(o.get('id')): o 
                                for o in recent_orders 
                                if o.get('id')
                            }
                        except Exception as e:
                            logger.warning(f"Could not fetch recent orders for reconciliation: {e}")
                            recent_orders_dict = {}
                        
                        # Now process each missing order using the pre-fetched data
                        for missing in reconciliation['missing']:
                            order_id = missing['order_id']
                            price = missing['price']
                            
                            # Check if it was filled using pre-fetched data
                            order_data = recent_orders_dict.get(str(order_id))
                            if order_data and order_data.get('status') == 2:  # Status 2 = Filled
                                # Order was filled - update state
                                fill_price = order_data.get('filledPrice') or order_data.get('limitPrice') or price
                                fill_quantity = order_data.get('fillVolume') or order_data.get('size', 0)
                                self.grid_manager.on_fill(str(order_id), fill_price, fill_quantity)
                                logger.info(f"Reconciled: Order {order_id} was filled at {fill_price}")
                            else:
                                # Order was cancelled or doesn't exist - remove from state
                                if price in self.grid_manager.orders:
                                    self.grid_manager.orders[price].order_id = None
                                logger.debug(f"Reconciled: Order {order_id} not found in API (likely cancelled)")
                    
                    # Handle extra orders (orders in API we don't know about)
                    # These might be from a previous session or manual orders
                    for extra in reconciliation['extra']:
                        logger.warning(f"Found extra order in API: {extra['api_order']}")
                    
                except Exception as e:
                    logger.error(f"Error reconciling with API: {e}")
            
            # Save state after reconciliation
            self.grid_manager.save_state()
        else:
            logger.info("No previous state found - starting fresh")
    
    def _setup_realtime_callbacks(self):
        """Setup callbacks for real-time market data updates"""
        # Get symbol IDs for matching quotes
        # TopstepX uses format like "F.US.MES" or "F.US.MNQ"
        primary_symbol_id = f"F.US.{self.primary_symbol}"
        hedge_symbol_id = f"F.US.{self.hedge_symbol}"
        
        def on_quote_update(data):
            """Handle real-time quote updates"""
            try:
                # Handle case where data might be a list
                if isinstance(data, list):
                    if len(data) > 0 and isinstance(data[0], dict):
                        data = data[0]
                    else:
                        logger.debug(f"Quote data is list but not in expected format: {data}")
                        return
                
                # Ensure data is a dict
                if not isinstance(data, dict):
                    logger.debug(f"Quote data is not a dict: {type(data)}")
                    return
                
                symbol_id = data.get('symbol', '')
                last_price = data.get('lastPrice')
                
                if last_price is None:
                    return
                
                # Update current price for primary instrument
                if symbol_id == primary_symbol_id:
                    self.current_price = float(last_price)
                    self.position_manager.update_price(self.primary_symbol, self.current_price)
                    logger.debug(f"Updated {self.primary_symbol} price: {self.current_price:.2f}")
                
                # Update current price for hedge instrument
                elif symbol_id == hedge_symbol_id:
                    self.hedge_price = float(last_price)
                    self.position_manager.update_price(self.hedge_symbol, self.hedge_price)
                    logger.debug(f"Updated {self.hedge_symbol} price: {self.hedge_price:.2f}")
                
                # Also try to match by contract ID if symbol doesn't match
                # Some quotes might use different symbol formats
                if not self.current_price and self.primary_contract_id:
                    # Try to infer from contract ID or use lastPrice if it's the only quote
                    if not self.current_price:
                        self.current_price = float(last_price)
                        self.position_manager.update_price(self.primary_symbol, self.current_price)
                        
            except Exception as e:
                logger.error(f"Error processing quote update: {e}")
        
        def on_market_trade_update(data):
            """Handle real-time trade updates (can also update price)"""
            try:
                # Handle case where data might be a list
                if isinstance(data, list):
                    if len(data) > 0 and isinstance(data[0], dict):
                        data = data[0]
                    else:
                        logger.debug(f"Trade data is list but not in expected format: {data}")
                        return
                
                # Ensure data is a dict
                if not isinstance(data, dict):
                    logger.debug(f"Trade data is not a dict: {type(data)}")
                    return
                
                symbol_id = data.get('symbolId', '')
                price = data.get('price')
                
                if price is None:
                    return
                
                # Update price from trade data
                if symbol_id == primary_symbol_id or not self.current_price:
                    self.current_price = float(price)
                    self.position_manager.update_price(self.primary_symbol, self.current_price)
                elif symbol_id == hedge_symbol_id:
                    self.hedge_price = float(price)
                    self.position_manager.update_price(self.hedge_symbol, self.hedge_price)
                    
            except Exception as e:
                logger.error(f"Error processing trade update: {e}")
        
        # Register callbacks
        self.api_client.register_realtime_callback("on_quote_update", on_quote_update)
        self.api_client.register_realtime_callback("on_market_trade_update", on_market_trade_update)
        
        logger.info("Registered real-time market data callbacks")
    
    def _recalculate_indicators(self):
        """Recalculate ATR, volatility, and correlation"""
        # Get candles
        primary_candles = self.timeseries_store.get_candles(
            self.primary_symbol,
            interval="15m",
            limit=self.config.get('atr_window', 14) + 10
        )
        
        if len(primary_candles) < self.config.get('atr_window', 14):
            logger.warning("Insufficient data for ATR calculation")
            return
        
        # Calculate ATR
        atr = calculate_atr(primary_candles, self.config.get('atr_window', 14))
        if atr:
            logger.debug(f"ATR: {atr:.2f}")
            MetricsExporter.update_atr(self.primary_symbol, atr)
        
        # Calculate volatility
        vol_window = self.config.get('volatility_window', 100)
        vol_candles = self.timeseries_store.get_candles(
            self.primary_symbol,
            interval="15m",
            limit=vol_window + 10
        )
        primary_vol = calculate_volatility(vol_candles, vol_window)
        
        # Calculate hedge volatility and correlation
        hedge_vol = None
        correlation = None
        
        if self.hedge_contract_id:
            hedge_candles = self.timeseries_store.get_candles(
                self.hedge_symbol,
                interval="15m",
                limit=vol_window + 10
            )
            if len(hedge_candles) >= vol_window:
                hedge_vol = calculate_volatility(hedge_candles, vol_window)
                correlation = calculate_correlation(vol_candles, hedge_candles, vol_window)
        
        # Update managers
        if atr:
            spacing = atr * self.config.get('atr_multiplier_for_spacing', 0.45)
            base_lot = self.sizer.compute_base_lot(atr)
            self.sizer.update_volatility(atr)
            
            # Store for analytics
            self._last_atr = atr
            if primary_vol:
                self._last_volatility = primary_vol
            if correlation is not None:
                self._last_correlation = correlation
            
            # Update grid if needed
            grid_rebuilt = False
            if self.current_price:
                grid_rebuilt = self.grid_manager.rebuild_grid_if_needed(
                    self.current_price,
                    spacing,
                    base_lot
                )
                
                # Record grid rebuild decision
                if grid_rebuilt:
                    self.analytics_store.record_strategy_decision(
                        decision_type="grid_rebuild",
                        symbol=self.primary_symbol,
                        decision_data={
                            "grid_mid": self.grid_manager.grid_mid,
                            "spacing": spacing,
                            "atr": atr,
                            "base_lot": base_lot,
                            "levels_count": len(self.grid_manager.levels)
                        },
                        market_conditions={
                            "atr": atr,
                            "volatility": primary_vol,
                            "current_price": self.current_price,
                            "correlation": correlation
                        },
                        reasoning=f"Price moved significantly, rebuilding grid with spacing={spacing:.2f}"
                    )
        
        if primary_vol and hedge_vol:
            self.hedge_manager.update_volatilities(primary_vol, hedge_vol)
        
        if correlation is not None:
            self.hedge_manager.update_correlation(correlation)
            MetricsExporter.update_correlation(correlation)
        
        if self.hedge_manager.hedge_ratio:
            MetricsExporter.update_hedge_ratio(self.hedge_manager.hedge_ratio)
    
    def _place_grid_orders(self):
        """Place orders for grid levels"""
        levels = self.grid_manager.get_levels_to_place()
        
        # Check current total position size (for 50k challenge: max 5 contracts)
        if hasattr(self, 'max_position_size') and self.max_position_size:
            primary_pos = abs(self.position_manager.get_net_position(self.primary_symbol))
            hedge_pos = abs(self.position_manager.get_net_position(self.hedge_symbol))
            total_pos = primary_pos + hedge_pos
            
            if total_pos >= self.max_position_size:
                logger.warning(
                    f"⚠️ Total position size ({total_pos}) at or exceeds limit ({self.max_position_size}). "
                    f"Skipping new grid orders. Primary: {primary_pos}, Hedge: {hedge_pos}"
                )
                return  # Don't place new orders if at limit
        
        for level in levels:
            if not self.primary_contract_id:
                continue
            
            # Additional check: ensure this order won't exceed limit
            if hasattr(self, 'max_position_size') and self.max_position_size:
                primary_pos = abs(self.position_manager.get_net_position(self.primary_symbol))
                hedge_pos = abs(self.position_manager.get_net_position(self.hedge_symbol))
                projected_total = primary_pos + hedge_pos + level.size
                
                if projected_total > self.max_position_size:
                    logger.warning(
                        f"⚠️ Order would exceed position limit: {projected_total} > {self.max_position_size}. "
                        f"Skipping order for level {level.level_index}"
                    )
                    continue
            
            order_id = self.order_client.place_limit_order(
                contract_id=self.primary_contract_id,
                side=level.side,
                quantity=level.size,
                price=level.price
            )
            
            if order_id:
                self.grid_manager.register_order(level, order_id)
                MetricsExporter.record_order(self.primary_symbol, level.side, "placed")
                
                # Record order event for analytics
                self.analytics_store.record_order_event(
                    order_id=str(order_id),
                    symbol=self.primary_symbol,
                    side=level.side,
                    quantity=level.size,
                    event_type="placed",
                    price=level.price,
                    reason=f"Grid level {level.level_index}",
                    market_context={
                        "atr": getattr(self, '_last_atr', None),
                        "volatility": getattr(self, '_last_volatility', None),
                        "current_price": self.current_price,
                        "grid_mid": self.grid_manager.grid_mid,
                        "grid_spacing": self.grid_manager.spacing,
                        "level_index": level.level_index
                    }
                )
                
                # Record strategy decision
                self.analytics_store.record_strategy_decision(
                    decision_type="order_placed",
                    symbol=self.primary_symbol,
                    decision_data={
                        "order_id": str(order_id),
                        "side": level.side,
                        "quantity": level.size,
                        "price": level.price,
                        "level_index": level.level_index
                    },
                    market_conditions={
                        "atr": getattr(self, '_last_atr', None),
                        "volatility": getattr(self, '_last_volatility', None),
                        "current_price": self.current_price,
                        "grid_mid": self.grid_manager.grid_mid,
                        "grid_spacing": self.grid_manager.spacing
                    },
                    reasoning=f"Placing grid order at level {level.level_index} ({level.side})"
                )
                
                # Save state after placing order
                self.grid_manager.save_state()
    
    def _check_and_place_hedge(self):
        """Check if hedge should be activated and place if needed"""
        if not self.hedge_contract_id or not self.current_price:
            return
        
        # Cooldown: Don't check hedge more than once every 10 seconds
        # This prevents rapid-fire hedge orders
        if not hasattr(self, '_last_hedge_check'):
            self._last_hedge_check = datetime.now()
        
        time_since_last_check = (datetime.now() - self._last_hedge_check).total_seconds()
        if time_since_last_check < 10:
            return  # Still in cooldown
        
        self._last_hedge_check = datetime.now()
        
        net_exposure = self.position_manager.net_exposure_dollars()
        primary_position = self.position_manager.get_net_position(self.primary_symbol)
        
        if primary_position == 0:
            return
        
        # Check if hedge should be activated
        spacing = self.grid_manager.spacing or 0
        grid_mid = self.grid_manager.grid_mid or self.current_price
        
        should_hedge = self.hedge_manager.should_activate_hedge(
            net_exposure,
            spacing,
            self.current_price,
            grid_mid,
            self.config.get('hedge_activation_multiplier', 1.5)
        )
        
        if should_hedge:
            # CRITICAL: Reconcile positions FIRST to get accurate current positions
            # This prevents placing hedge orders based on stale position data
            if not self.dry_run and self.api_client.account_id:
                try:
                    api_positions = self.api_client.get_positions(self.api_client.account_id)
                    if api_positions:
                        self.position_manager.reconcile_with_api_positions(api_positions)
                        logger.debug(f"Reconciled positions before hedge check: {len(api_positions)} positions")
                except Exception as e:
                    logger.debug(f"Could not reconcile positions before hedge check: {e}")
            
            # Check existing hedge position (after reconciliation)
            existing_hedge_position = self.position_manager.get_net_position(self.hedge_symbol)
            logger.info(f"Current hedge position check: {self.hedge_symbol} = {existing_hedge_position} contracts")
            
            # Check for existing open hedge orders to avoid duplicate orders
            api_orders = []
            try:
                api_orders = self.api_client.get_open_orders(self.api_client.account_id)
            except Exception as e:
                logger.debug(f"Could not check open orders: {e}")
            
            # Count open hedge orders
            open_hedge_orders = [
                o for o in api_orders 
                if o.get('contractId') == self.hedge_contract_id or 
                   str(o.get('contractId', '')).endswith(self.hedge_symbol)
            ]
            
            if open_hedge_orders:
                logger.debug(f"Found {len(open_hedge_orders)} existing hedge orders, skipping new hedge placement")
                return  # Don't place duplicate hedge orders
            
            # Get tick values for proper risk-based sizing
            primary_tick_value = self.position_manager.tick_values.get(self.primary_symbol, 5.0)
            hedge_tick_value = self.position_manager.tick_values.get(self.hedge_symbol, 2.0)
            
            # Get current volatilities and correlation for risk-based hedge calculation
            # These are already stored in hedge_manager, but pass explicitly for clarity
            primary_vol = self.hedge_manager.primary_volatility
            hedge_vol = self.hedge_manager.hedge_volatility
            correlation = self.hedge_manager.current_correlation
            
            # Compute target hedge size based on RISK exposure (accounts for volatility differences)
            # This properly hedges drawdowns by matching risk, not just dollar exposure
            target_hedge_size = self.hedge_manager.compute_hedge_size(
                primary_position=primary_position,
                primary_price=self.current_price,
                hedge_price=self.hedge_price,
                primary_tick_value=primary_tick_value,
                hedge_tick_value=hedge_tick_value,
                primary_volatility=primary_vol,
                hedge_volatility=hedge_vol,
                correlation=correlation
            )
            
            # Calculate how much hedge we need to ADD (account for existing position)
            # target_hedge_size is the total desired position (opposite of primary)
            # existing_hedge_position is what we currently have
            # hedge_needed = target_hedge_size - existing_hedge_position
            
            # CRITICAL CHECK: If existing hedge is already at or above target, don't add more
            # Both should be negative (short) if primary is long, or both positive if primary is short
            if (primary_position > 0 and existing_hedge_position <= target_hedge_size) or \
               (primary_position < 0 and existing_hedge_position >= target_hedge_size):
                logger.info(
                    f"Hedge already adequate or over-hedged: primary={primary_position}, "
                    f"existing_hedge={existing_hedge_position}, target_hedge={target_hedge_size}. "
                    f"Skipping hedge order."
                )
                return  # Don't place hedge order if already hedged enough
            
            hedge_needed = target_hedge_size - existing_hedge_position
            
            # CRITICAL: Cap hedge_needed to prevent oversized adjustments
            # Never adjust by more than the primary position size
            max_adjustment = abs(primary_position) * self.hedge_manager.max_hedge_contracts_multiplier
            if abs(hedge_needed) > max_adjustment:
                logger.error(
                    f"🚨 CRITICAL: Hedge adjustment ({hedge_needed}) exceeds max adjustment ({max_adjustment}). "
                    f"CAPPING adjustment to {max_adjustment} contracts."
                )
                hedge_needed = max_adjustment if hedge_needed > 0 else -max_adjustment
            
            # Additional safety: Never adjust by more than primary position itself
            if abs(hedge_needed) > abs(primary_position):
                logger.error(
                    f"🚨 CRITICAL: Hedge adjustment ({hedge_needed}) exceeds primary position ({primary_position}). "
                    f"CAPPING to primary position size."
                )
                hedge_needed = abs(primary_position) if hedge_needed > 0 else -abs(primary_position)
            
            # Final safety check: Ensure total hedge position (after adjustment) never exceeds primary
            # This prevents accumulation of oversized hedges
            projected_total_hedge = abs(existing_hedge_position + hedge_needed)
            max_allowed_hedge = abs(primary_position) * self.hedge_manager.max_hedge_contracts_multiplier
            
            if projected_total_hedge > max_allowed_hedge:
                # Calculate maximum allowed adjustment
                max_allowed_adjustment = max_allowed_hedge - abs(existing_hedge_position)
                if max_allowed_adjustment < 0:
                    max_allowed_adjustment = 0  # Already over-hedged, don't add more
                
                logger.error(
                    f"🚨 CRITICAL: Projected total hedge ({projected_total_hedge}) exceeds max ({max_allowed_hedge}). "
                    f"Limiting adjustment to {max_allowed_adjustment} contracts."
                )
                # Cap the adjustment to keep total hedge within limits
                hedge_needed = max_allowed_adjustment if hedge_needed > 0 else -max_allowed_adjustment
            
            # Only place order if we need to adjust the hedge
            hedge_size = 0  # Default: no hedge order needed
            if abs(hedge_needed) > 0:
                # Limit the order size to what's actually needed
                # Never place an order larger than the difference
                hedge_size = hedge_needed
                
                logger.info(
                    f"Hedge adjustment: primary={primary_position}, "
                    f"existing_hedge={existing_hedge_position}, "
                    f"target_hedge={target_hedge_size}, "
                    f"hedge_needed={hedge_size} (capped), "
                    f"projected_total={abs(existing_hedge_position + hedge_size)}"
                )
            else:
                logger.debug(
                    f"Hedge already adequate: primary={primary_position}, "
                    f"existing_hedge={existing_hedge_position}, "
                    f"target_hedge={target_hedge_size}"
                )
            
            if hedge_size != 0:
                # CRITICAL: Check total position size limit (50k challenge: 5 contracts max)
                if hasattr(self, 'max_position_size') and self.max_position_size:
                    primary_pos = abs(self.position_manager.get_net_position(self.primary_symbol))
                    existing_hedge_pos = abs(self.position_manager.get_net_position(self.hedge_symbol))
                    projected_hedge_pos = abs(existing_hedge_position + hedge_size)
                    projected_total = primary_pos + projected_hedge_pos
                    
                    if projected_total > self.max_position_size:
                        logger.error(
                            f"🚨 CRITICAL: Hedge order would exceed position limit: "
                            f"{projected_total} > {self.max_position_size}. "
                            f"Primary: {primary_pos}, Projected Hedge: {projected_hedge_pos}. "
                            f"Skipping hedge order."
                        )
                        return  # Don't place hedge if it would exceed limit
                
                # Place hedge order (opposite direction)
                # hedge_size is negative when we want to go short (SELL)
                # hedge_size is positive when we want to go long (BUY)
                hedge_side = "SELL" if hedge_size < 0 else "BUY"
                order_id = self.order_client.place_limit_order(
                    contract_id=self.hedge_contract_id,
                    side=hedge_side,
                    quantity=abs(hedge_size),
                    price=self.hedge_price or self.current_price  # Use market price for hedge
                )
                
                if order_id:
                    logger.info(f"Placed hedge order: {hedge_side} {abs(hedge_size)} {self.hedge_symbol}")
                    MetricsExporter.record_order(self.hedge_symbol, hedge_side, "placed")
                    
                    # Record hedge decision for analytics
                    self.analytics_store.record_strategy_decision(
                        decision_type="hedge_placed",
                        symbol=self.hedge_symbol,
                        decision_data={
                            "order_id": str(order_id),
                            "side": hedge_side,
                            "quantity": abs(hedge_size),
                            "price": self.hedge_price or self.current_price,
                            "primary_position": primary_position,
                            "net_exposure": net_exposure,
                            "hedge_ratio": self.hedge_manager.hedge_ratio
                        },
                        market_conditions={
                            "atr": getattr(self, '_last_atr', None),
                            "volatility": getattr(self, '_last_volatility', None),
                            "correlation": getattr(self, '_last_correlation', None),
                            "current_price": self.current_price,
                            "hedge_price": self.hedge_price,
                            "grid_mid": grid_mid,
                            "spacing": spacing
                        },
                        reasoning=f"Hedge activated: net_exposure={net_exposure:.2f}, primary_position={primary_position}"
                    )
    
    def _process_fills(self):
        """Process fills from API"""
        if not self.api_client.account_id:
            return
        
        # Get open orders first (more efficient for checking fills)
        # Also get recent orders to catch any fills that might have happened
        open_orders = self.api_client.get_open_orders(self.api_client.account_id)
        
        # Get recent orders from last hour to check for fills
        from datetime import timedelta
        recent_orders = self.api_client.get_orders(
            self.api_client.account_id,
            start_timestamp=datetime.now() - timedelta(hours=1)
        )
        
        # Combine and deduplicate
        all_orders_dict = {order.get('id'): order for order in open_orders + recent_orders}
        all_orders = list(all_orders_dict.values())
        
        for order in all_orders:
            order_id = order.get('id') or order.get('orderId')
            status = order.get('status')  # 1=Open, 2=Filled, 3=Cancelled, etc.
            
            # Check if filled (status 2 = Filled)
            if status == 2:
                # Process fill
                contract_id = order.get('contractId')
                side = "BUY" if order.get('side') == 0 else "SELL"  # 0=Bid (buy), 1=Ask (sell)
                quantity = order.get('size') or order.get('fillVolume')
                price = order.get('filledPrice') or order.get('limitPrice')
                
                if order_id and contract_id and quantity and price:
                    # Map contract_id to symbol (simplified - you might want a mapping)
                    symbol = self.primary_symbol if contract_id == self.primary_contract_id else (
                        self.hedge_symbol if contract_id == self.hedge_contract_id else contract_id
                    )
                    
                    # Get position context before fill
                    position_before = self.position_manager.get_net_position(symbol)
                    pos = self.position_manager.get_position(symbol)
                    avg_price_before = pos.avg_price if pos.quantity != 0 else None
                    
                    # Process fill
                    self.fill_handler.on_fill(
                        order_id=str(order_id),
                        symbol=symbol,
                        side=side,
                        quantity=quantity,
                        price=price
                    )
                    MetricsExporter.record_trade(symbol, side)
                    
                    # Get position context after fill
                    position_after = self.position_manager.get_net_position(symbol)
                    pos_after = self.position_manager.get_position(symbol)
                    avg_price_after = pos_after.avg_price if pos_after.quantity != 0 else None
                    
                    # Get market context for analytics
                    atr = None
                    volatility = None
                    correlation = None
                    if hasattr(self, '_last_atr'):
                        atr = self._last_atr
                    if hasattr(self, '_last_volatility'):
                        volatility = self._last_volatility
                    if hasattr(self, '_last_correlation'):
                        correlation = self._last_correlation
                    
                    # Calculate P&L for this trade
                    trade_pnl = None
                    if symbol == self.primary_symbol:
                        # Calculate realized P&L change
                        realized_before = self.position_manager.get_realized_pnl(symbol)
                        # Fill handler updates position, so get new realized
                        realized_after = self.position_manager.get_realized_pnl(symbol)
                        trade_pnl = realized_after - realized_before
                    
                    # Record trade with full context
                    self.analytics_store.record_trade_with_context(
                        trade_id=f"{order_id}_{int(time.time())}",
                        symbol=symbol,
                        side=side,
                        quantity=quantity,
                        price=price,
                        timestamp=datetime.now(),
                        order_id=str(order_id),
                        fill_id=str(order_id),
                        pnl=trade_pnl,
                        atr=atr,
                        volatility=volatility,
                        correlation=correlation,
                        current_price=price,
                        grid_mid=self.grid_manager.grid_mid,
                        grid_spacing=self.grid_manager.spacing,
                        level_index=None,  # Will be set if it's a grid order
                        position_before=position_before,
                        position_after=position_after,
                        avg_price_before=avg_price_before,
                        avg_price_after=avg_price_after,
                        daily_pnl=self.risk_manager.get_daily_pnl(),
                        drawdown=self.risk_manager.get_trailing_drawdown(),
                        exposure=self.position_manager.net_exposure_dollars()
                    )
                    
                    # Record order event
                    self.analytics_store.record_order_event(
                        order_id=str(order_id),
                        symbol=symbol,
                        side=side,
                        quantity=quantity,
                        event_type="filled",
                        price=price,
                        fill_price=price,
                        fill_quantity=quantity,
                        market_context={
                            "atr": atr,
                            "volatility": volatility,
                            "current_price": price,
                            "grid_mid": self.grid_manager.grid_mid
                        }
                    )
                    
                    # Record position snapshot
                    pos_snapshot = self.position_manager.get_position(symbol)
                    if pos_snapshot:
                        self.analytics_store.record_position_snapshot(
                            symbol=symbol,
                            quantity=pos_snapshot.quantity,
                            avg_price=pos_snapshot.avg_price,
                            current_price=price,
                            realized_pnl=pos_snapshot.realized_pnl,
                            unrealized_pnl=self.position_manager.get_unrealized_pnl(symbol),
                            total_pnl=pos_snapshot.realized_pnl + self.position_manager.get_unrealized_pnl(symbol),
                            exposure=self.position_manager.net_exposure_dollars()
                        )
                    
                    # Save state after fill
                    if symbol == self.primary_symbol:
                        self.grid_manager.save_state()
    
    def _update_risk_checks(self) -> bool:
        """Perform risk checks, return True if should stop"""
        # Reconcile positions with API periodically (every 30 seconds) to get accurate P&L
        if not hasattr(self, '_last_position_sync') or \
           (datetime.now() - self._last_position_sync).total_seconds() > 30:
            if not self.dry_run and self.api_client.account_id:
                try:
                    api_positions = self.api_client.get_positions(self.api_client.account_id)
                    if api_positions:
                        self.position_manager.reconcile_with_api_positions(api_positions)
                        logger.debug(f"Reconciled {len(api_positions)} positions with API")
                    self._last_position_sync = datetime.now()
                except Exception as e:
                    logger.debug(f"Error reconciling positions: {e}")
        
        total_pnl = self.position_manager.get_total_pnl()
        exposure = self.position_manager.net_exposure_dollars()
        
        # Update risk manager balance from account API FIRST
        # This ensures we have the latest balance before calculating P&L
        # Poll account balance every risk check (but cache to avoid too many API calls)
        if not hasattr(self, '_last_balance_check') or \
           (datetime.now() - self._last_balance_check).total_seconds() > 5:  # Check every 5 seconds
            accounts = self.api_client.get_accounts()
            if accounts and len(accounts) > 0:
                # Find the account matching our selected account ID
                selected_account = next(
                    (acc for acc in accounts if (acc.get('id') == self.api_client.account_id or 
                                                  acc.get('accountId') == self.api_client.account_id)),
                    None
                )
                
                if selected_account:
                    current_balance = selected_account.get('balance')
                    if current_balance is not None:
                        self.risk_manager.update_balance(current_balance)
                        self._last_balance_check = datetime.now()
                    else:
                        # Fallback: use starting balance + P&L if API doesn't return balance
                        if not hasattr(self, '_initial_balance'):
                            self._initial_balance = self.config.get('capital_equiv', 150000.0)
                        self.risk_manager.update_balance(self._initial_balance + total_pnl)
                else:
                    # Selected account not found in list, use fallback
                    logger.warning(f"Selected account {self.api_client.account_id} not found in accounts list")
                    if not hasattr(self, '_initial_balance'):
                        self._initial_balance = self.config.get('capital_equiv', 150000.0)
                    self.risk_manager.update_balance(self._initial_balance + total_pnl)
            else:
                # Fallback if accounts not available
                if not hasattr(self, '_initial_balance'):
                    self._initial_balance = self.config.get('capital_equiv', 150000.0)
                    self.risk_manager.update_balance(self._initial_balance + total_pnl)
        
        # Now calculate P&L metrics after balance is updated
        daily_pnl = self.risk_manager.get_daily_pnl()
        drawdown = self.risk_manager.get_trailing_drawdown()
        
        # Check limits
        should_stop, reason = self.risk_manager.check_all_limits(
            daily_pnl,
            drawdown,
            exposure
        )
        
        if should_stop:
            logger.critical(f"Risk limit triggered: {reason}")
            self.emergency_flatten(reason)
            return True
        
        # Update metrics
        MetricsExporter.update_pnl(
            self.position_manager.get_realized_pnl(),
            self.position_manager.get_unrealized_pnl(),
            total_pnl,
            daily_pnl
        )
        MetricsExporter.update_drawdown(drawdown)
        MetricsExporter.update_exposure(exposure)
        
        return False
    
    def _check_session_end(self) -> bool:
        """
        Check if we're near session end and should flatten positions
        
        Futures market close: 3:45 PM CT (4:45 PM ET) for ES/NQ
        Returns True if we should flatten
        """
        try:
            import pytz
            
            # Get current time in CT (Chicago timezone)
            ct_tz = pytz.timezone('America/Chicago')
            now_ct = datetime.now(ct_tz)
            
            # Market close is 3:45 PM CT
            market_close = now_ct.replace(hour=15, minute=45, second=0, microsecond=0)
            
            # Safety margin from config
            safety_margin = timedelta(minutes=self.config.get('session_close_safety_margin_minutes', 5))
            flatten_time = market_close - safety_margin
            
            # Check if we're past the flatten time
            if now_ct >= flatten_time:
                # Only flatten if we have positions
                positions = self.position_manager.get_all_positions()
                has_positions = any(pos.quantity != 0 for pos in positions.values())
                
                if has_positions:
                    logger.warning(f"Session end approaching: {now_ct.strftime('%H:%M:%S CT')} >= {flatten_time.strftime('%H:%M:%S CT')}")
                    return True
            
            # Also check if it's after market close (shouldn't happen, but safety check)
            if now_ct.hour >= 16:  # After 4 PM CT
                positions = self.position_manager.get_all_positions()
                has_positions = any(pos.quantity != 0 for pos in positions.values())
                if has_positions:
                    logger.warning("After market close - should flatten")
                    return True
                    
        except ImportError:
            # pytz not available, use simple time check (ET timezone)
            now = datetime.now()
            # Market close is approximately 4:45 PM ET (3:45 PM CT)
            # Using 4:40 PM ET as safety margin (5 minutes before)
            if now.hour >= 16 and now.minute >= 40:
                positions = self.position_manager.get_all_positions()
                has_positions = any(pos.quantity != 0 for pos in positions.values())
                if has_positions:
                    logger.warning(f"Session end check (simplified): {now.strftime('%H:%M:%S')}")
                    return True
        except Exception as e:
            logger.error(f"Error checking session end: {e}")
        
        return False
    
    def _poll_price_update(self):
        """
        Fallback: Poll API for latest price if SignalR not available
        This is less efficient but ensures we have price data
        """
        try:
            # Try to get latest quote from API
            # Note: TopstepX API might have a quote endpoint, but for now we'll use
            # the last price from recent bars or positions
            if not self.current_price and self.primary_contract_id:
                # Get most recent bar
                bars = self.api_client.get_bars(
                    contract_id=self.primary_contract_id,
                    interval="1m",
                    limit=1
                )
                if bars and len(bars) > 0:
                    last_bar = bars[0]
                    close_price = last_bar.get('c') or last_bar.get('close')
                    if close_price:
                        self.current_price = float(close_price)
                        self.position_manager.update_price(self.primary_symbol, self.current_price)
                        logger.debug(f"Polled {self.primary_symbol} price: {self.current_price:.2f}")
        except Exception as e:
            logger.debug(f"Error polling price update: {e}")
    
    def emergency_flatten(self, reason: str = "Emergency"):
        """Emergency flatten all positions"""
        logger.critical(f"EMERGENCY FLATTEN: {reason}")
        
        # Cancel all orders
        order_ids = self.grid_manager.cancel_all_orders()
        self.order_client.cancel_all_orders(order_ids)
        
        # Close all positions
        positions = self.api_client.get_positions()
        for position in positions:
            contract_id = position.get('contractId')
            if contract_id:
                self.api_client.close_position(contract_id)
        
        self.position_manager.flatten_all()
        self.running = False
    
    def set_paused(self, paused: bool):
        """Pause or resume strategy"""
        self.paused = paused
        logger.info(f"Strategy {'paused' if paused else 'resumed'}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status"""
        # Get current balance from risk manager
        current_balance = self.risk_manager.current_balance
        if current_balance == 0 and hasattr(self, '_initial_balance'):
            current_balance = self._initial_balance
        
        # Get real orders from TopstepX API
        api_orders = []
        if self.api_client.account_id and not self.dry_run:
            try:
                api_orders = self.api_client.get_open_orders(self.api_client.account_id)
            except Exception as e:
                logger.debug(f"Error fetching API orders: {e}")
        
        # Convert API orders to display format
        open_orders_list = []
        for order in api_orders:
            contract_id = order.get('contractId', '')
            # Map contract_id to symbol
            symbol = self.primary_symbol if contract_id == self.primary_contract_id else (
                self.hedge_symbol if contract_id == self.hedge_contract_id else contract_id
            )
            
            side = "BUY" if order.get('side') == 0 else "SELL"  # 0=Bid (buy), 1=Ask (sell)
            price = order.get('limitPrice') or order.get('stopPrice') or 0.0
            
            open_orders_list.append({
                "order_id": str(order.get('id') or order.get('orderId', '')),
                "symbol": symbol,
                "side": side,
                "quantity": order.get('size', 0),
                "price": price,
                "level_index": None,  # API doesn't provide this
                "status": order.get('status', 0),  # 1=Open, 2=Filled, etc.
                "type": order.get('type', 0)  # Order type
            })
        
        # In dry_run mode or if no API orders, use grid manager orders (internal tracking)
        if not open_orders_list or self.dry_run:
            grid_orders = self.grid_manager.get_open_orders()
            # Merge with API orders, preferring API orders if they exist
            if api_orders:
                # Use API orders as primary source
                pass  # Already added above
            else:
                # Use grid manager orders
                open_orders_list = grid_orders
        
        return {
            "status": "running" if self.running else "stopped",
            "timestamp": datetime.now().isoformat(),
            "trading_enabled": not self.paused and self.running,
            "dry_run": self.dry_run,
            "account_balance": current_balance,
            "daily_pnl": self.risk_manager.get_daily_pnl(),
            "total_pnl": self.position_manager.get_total_pnl(),
            "net_exposure": self.position_manager.net_exposure_dollars(),
            "open_orders": len(open_orders_list),
            "open_orders_list": open_orders_list,
            "drawdown": self.risk_manager.get_trailing_drawdown(),
            "current_price": self.current_price,
            "primary_symbol": self.primary_symbol,
            "positions": {
                sym: pos.quantity
                for sym, pos in self.position_manager.get_all_positions().items()
            }
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics"""
        return self.get_status()
    
    def run(self):
        """Main trading loop"""
        if not self.initialize():
            logger.error("Failed to initialize strategy")
            return
        
        self.running = True
        logger.info("Starting trading loop...")
        
        try:
            while self.running:
                if self.paused:
                    time.sleep(1)
                    continue
                
                # Check if time to recalculate
                if datetime.now() - self.last_recalc_time >= self.recalc_interval:
                    self._recalculate_indicators()
                    self.last_recalc_time = datetime.now()
                
                # Process fills
                self._process_fills()
                
                # Place grid orders
                self._place_grid_orders()
                
                # Check and place hedge
                self._check_and_place_hedge()
                
                # Risk checks
                if self._update_risk_checks():
                    break
                
                # Check session end and flatten if needed
                if self._check_session_end():
                    logger.info("Session end detected - flattening positions")
                    self.emergency_flatten("Session end")
                    break
                
                # Update metrics
                MetricsExporter.update_orders(self.primary_symbol, self.grid_manager.get_open_orders_count())
                
                # Periodic state save and analytics
                if datetime.now() - self.last_state_save >= self.state_save_interval:
                    self.grid_manager.save_state()
                    if self.api_client.account_id:
                        self.state_persistence.update_session(
                            self.api_client.account_id,
                            self.primary_symbol
                        )
                    
                    # Record market conditions snapshot
                    if self.current_price:
                        self.analytics_store.record_market_conditions(
                            symbol=self.primary_symbol,
                            price=self.current_price,
                            atr=getattr(self, '_last_atr', None),
                            volatility=getattr(self, '_last_volatility', None),
                            grid_mid=self.grid_manager.grid_mid,
                            grid_spacing=self.grid_manager.spacing,
                            levels_count=len(self.grid_manager.levels),
                            open_orders_count=self.grid_manager.get_open_orders_count(),
                            correlation=getattr(self, '_last_correlation', None),
                            hedge_price=self.hedge_price
                        )
                    
                    # Record position snapshots
                    for symbol, pos in self.position_manager.get_all_positions().items():
                        if pos.quantity != 0:
                            self.analytics_store.record_position_snapshot(
                                symbol=symbol,
                                quantity=pos.quantity,
                                avg_price=pos.avg_price,
                                current_price=self.current_price if symbol == self.primary_symbol else self.hedge_price,
                                realized_pnl=pos.realized_pnl,
                                unrealized_pnl=self.position_manager.get_unrealized_pnl(symbol),
                                total_pnl=pos.realized_pnl + self.position_manager.get_unrealized_pnl(symbol),
                                exposure=self.position_manager.net_exposure_dollars()
                            )
                    
                    self.last_state_save = datetime.now()
                
                # Fallback: Poll for price updates if SignalR not available
                if not self.current_price:
                    self._poll_price_update()
                
                # Sleep
                time.sleep(1)
        
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        except Exception as e:
            logger.error(f"Error in trading loop: {e}", exc_info=True)
        finally:
            logger.info("Shutting down...")
            # Save final state before shutdown
            self.grid_manager.save_state()
            self.emergency_flatten("Shutdown")
            self.api_client.disconnect()


def main():
    """Main entry point"""
    # Start API server in background
    def run_api():
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    
    api_thread = Thread(target=run_api, daemon=True)
    api_thread.start()
    
    # Create and run strategy
    strategy = GridStrategy()
    set_strategy_instance(strategy)
    
    try:
        strategy.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

