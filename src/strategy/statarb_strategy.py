"""
Statistical Arbitrage Strategy for GC/MGC Gold Futures - Production Implementation
Cross-Asset Statistical Arbitrage: MGC ↔ GC (Micro Gold ↔ Full Gold)
"""
import os
import sys
import time
import logging
import yaml
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from threading import Lock

# Get absolute paths
current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(os.path.dirname(current_file))  # src/
project_root = os.path.dirname(src_dir)  # project root

# Add project root to path first (for test modules)
sys.path.insert(0, project_root)
# Add src to path (for src modules)
sys.path.insert(0, src_dir)

from connectors.topstepx_client import TopstepXClient
from connectors.market_data_adapter import MarketDataAdapter
from connectors.mgc_contract_specs import MGC_SPECS
from data.timeseries_store import TimeseriesStore
from data.trade_history import TradeHistory
from data.state_persistence import StatePersistence
from strategy.position_manager import PositionManager
from strategy.risk_manager import RiskManager
from execution.order_client import OrderClient
from test.statarb_strategy import StatArbCalculator, SpreadSignal
from test.statarb_contract_specs import GC_SPECS

logger = logging.getLogger(__name__)


@dataclass
class StatArbPosition:
    """Represents an active statistical arbitrage position"""
    entry_time: datetime
    entry_price_gc: float
    entry_price_mgc: float
    entry_spread: float
    entry_zscore: float
    beta: float
    contracts_gc: int
    contracts_mgc: int
    is_long_spread: bool  # True = long GC, short MGC
    order_id_gc: Optional[str] = None
    order_id_mgc: Optional[str] = None
    remaining_gc: Optional[int] = None
    remaining_mgc: Optional[int] = None
    partial_taken: bool = False
    best_spread: Optional[float] = None


class StatArbStrategy:
    """Main Statistical Arbitrage strategy implementation for production"""
    
    def __init__(self, config_path: str = "statarb_config.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
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
        
        # Strategy components
        self.symbol_a = self.config.get('instrument_a', 'GC')
        self.symbol_b = self.config.get('instrument_b', 'MGC')
        
        # Position and risk management
        self.position_manager = PositionManager(
            max_net_notional=self.config.get('max_net_notional', 10000.0),
            tick_values={
                self.symbol_a: GC_SPECS.tick_value,
                self.symbol_b: MGC_SPECS.tick_value
            },
            tick_sizes={
                self.symbol_a: GC_SPECS.tick_size,
                self.symbol_b: MGC_SPECS.tick_size
            },
            contract_multipliers={
                self.symbol_a: 100.0,  # GC = 100 oz per contract
                self.symbol_b: 10.0   # MGC = 10 oz per contract
            }
        )
        
        self.risk_manager = RiskManager(
            max_daily_loss=self.config.get('max_daily_loss', 900.0),
            trailing_drawdown_limit=self.config.get('trailing_drawdown_limit', 1800.0),
            max_net_notional=self.config.get('max_net_notional', 10000.0)
        )
        
        # Strategy calculator (from backtest)
        self.calculator = StatArbCalculator(
            z_entry=self.config.get('z_entry', 2.0),
            z_exit=self.config.get('z_exit', 0.6),
            lookback_periods=self.config.get('lookback_periods', 1440),
            min_lookback=self.config.get('min_lookback', 100),
            beta_lookback=self.config.get('beta_lookback', 500)
        )
        
        # Strategy parameters
        self.spread_stop_std = self.config.get('spread_stop_std', 1.5)
        self.time_stop_hours = self.config.get('time_stop_hours', 2.0)
        self.risk_per_trade = self.config.get('risk_per_trade', 100.0)
        self.recalc_interval = timedelta(seconds=self.config.get('recalc_interval_seconds', 30))
        self.spread_divergence_hours = self.config.get('spread_divergence_hours', 4.0)
        self.partial_profit_z = self.config.get('partial_profit_z', 1.0)
        self.partial_profit_fraction = self.config.get('partial_profit_fraction', 0.5)
        self.trailing_stop_std = self.config.get('trailing_stop_std', 1.0)
        
        # Dynamic position sizing parameters
        self.dynamic_sizing_enabled = self.config.get('dynamic_sizing_enabled', True)
        self.max_risk_multiplier = self.config.get('max_risk_multiplier', 5.0)  # Max 5x intended risk
        self.min_risk_multiplier = self.config.get('min_risk_multiplier', 0.3)  # Min 0.3x intended risk
        self.volatility_stop_adjustment = self.config.get('volatility_stop_adjustment', True)
        self.high_volatility_threshold = self.config.get('high_volatility_threshold', 10.0)  # $10 spread std
        self.high_volatility_stop_std = self.config.get('high_volatility_stop_std', 1.0)  # Tighter stop in high vol
        self.max_contracts_gc = self.config.get('max_contracts_gc', 3)  # Maximum GC contracts per trade
        
        # Execution
        self.order_client = OrderClient(self.api_client, dry_run=self.dry_run)
        
        # State
        self.running = False
        self.paused = False
        self.contract_id_a: Optional[str] = None  # GC contract ID
        self.contract_id_b: Optional[str] = None  # MGC contract ID
        self.current_price_a: Optional[float] = None  # GC price
        self.current_price_b: Optional[float] = None  # MGC price
        self.current_position: Optional[StatArbPosition] = None
        self.last_recalc_time = datetime.now(timezone.utc)
        self.data_lock = Lock()  # Thread-safe access to price data
        
        # Track pending entry orders to prevent multiple simultaneous entries
        self.pending_entry_orders: set = set()  # Set of order IDs for pending entry orders
        
        # Track order details for fill handling (order_id -> (symbol, side, quantity))
        self.order_details: Dict[str, Tuple[str, str, int]] = {}
        
        # Historical data for calculations (pandas DataFrames)
        self.df_gc: Optional[pd.DataFrame] = None
        self.df_mgc: Optional[pd.DataFrame] = None
        
        # Spread divergence tracking
        self.extreme_spread_start_time: Optional[datetime] = None  # When spread first became extreme
        self.last_extreme_direction: Optional[str] = None  # "LONG" or "SHORT" or None
        
        logger.info(f"StatArb Strategy initialized (dry_run={self.dry_run})")
        if not self.dry_run:
            logger.warning("=" * 60)
            logger.warning("LIVE TRADING MODE ENABLED - REAL ORDERS WILL BE PLACED!")
            logger.warning("=" * 60)
    
    def initialize(self) -> bool:
        """Initialize connection and fetch contract info"""
        logger.info("Initializing StatArb strategy...")
        
        # Authenticate
        if not self.api_client.authenticate():
            logger.error("Failed to authenticate with TopstepX")
            return False
        
        # Get accounts
        accounts = self.api_client.get_accounts()
        print(f"Accounts response: {accounts}")
        if not accounts:
            logger.error("No accounts found")
            return False
        
        # Log all available accounts for debugging
        logger.info(f"Found {len(accounts)} available account(s):")
        for acc in accounts:
            acc_id = acc.get('id') or acc.get('accountId')
            acc_name = acc.get('name', 'Unknown')
            is_sim = acc.get('simulated', False)
            acc_type = acc.get('type', 'Unknown')
            logger.info(f"  - {acc_name} (ID: {acc_id}, Simulated: {is_sim}, Type: {acc_type})")
        
        # Select account (same logic as other strategies)
        account = None
        prefer_practice = self.config.get('prefer_practice_account', True)
        specified_account_id = self.config.get('account_id')
        
        if specified_account_id:
            account = next((acc for acc in accounts if 
                          (acc.get('id') == specified_account_id or 
                           acc.get('accountId') == specified_account_id)), None)
        
        if not account and prefer_practice:
            # Look for practice accounts - check multiple indicators
            # Prioritize accounts with "PRAC" in name, then other practice indicators
            practice_accounts_with_prac = []
            practice_accounts_other = []
            
            for acc in accounts:
                name = acc.get('name', '').upper()
                is_simulated = acc.get('simulated', False)
                account_type = acc.get('type', '').upper()
                account_status = acc.get('status', '').upper()
                
                # Check if it's a practice account
                is_practice = (is_simulated or 
                              'PRAC' in name or 
                              'PRACTICE' in name or
                              'SIM' in name or
                              'SIMULATED' in name or
                              'DEMO' in name or
                              account_type == 'PRACTICE' or
                              account_status == 'PRACTICE')
                
                if is_practice:
                    # Prioritize accounts with "PRAC" or "PRACTICE" in name
                    if 'PRAC' in name or 'PRACTICE' in name:
                        practice_accounts_with_prac.append(acc)
                        logger.debug(f"Found practice account (PRAC in name): {acc.get('name')} (ID: {acc.get('id') or acc.get('accountId')})")
                    else:
                        practice_accounts_other.append(acc)
                        logger.debug(f"Found practice account (other indicator): {acc.get('name')} (ID: {acc.get('id') or acc.get('accountId')})")
            
            # Prefer accounts with "PRAC" in name
            if practice_accounts_with_prac:
                account = practice_accounts_with_prac[0]
                logger.info(f"Selected practice account (PRAC): {account.get('name')} (ID: {account.get('id') or account.get('accountId')})")
            elif practice_accounts_other:
                account = practice_accounts_other[0]
                logger.info(f"Selected practice account: {account.get('name')} (ID: {account.get('id') or account.get('accountId')})")
            else:
                logger.warning("No practice accounts found, will use first available account")
        
        if not account:
            account = accounts[0]
            logger.info(f"Using first available account: {account.get('name')} (ID: {account.get('id') or account.get('accountId')})")
        
        if account:
            account_id = account.get('id') or account.get('accountId')
            if account_id:
                self.api_client.set_account(account_id)
                logger.info(f"Using account: {account.get('name', 'Unknown')} (ID: {account_id})")
            else:
                logger.error("Selected account has no ID")
                return False
        else:
            logger.error("Could not select an account")
            return False
        
        # Search for contracts
        contracts_gc = self.api_client.search_contracts(self.symbol_a)
        contracts_mgc = self.api_client.search_contracts(self.symbol_b)
        
        if contracts_gc:
            self.contract_id_a = contracts_gc[0].get('contractId') or contracts_gc[0].get('id')
            logger.info(f"Contract {self.symbol_a}: {self.contract_id_a}")
        else:
            logger.error(f"Could not find contract for {self.symbol_a}")
            return False
        
        if contracts_mgc:
            self.contract_id_b = contracts_mgc[0].get('contractId') or contracts_mgc[0].get('id')
            logger.info(f"Contract {self.symbol_b}: {self.contract_id_b}")
        else:
            logger.error(f"Could not find contract for {self.symbol_b}")
            return False
        
        # Load historical data
        self._load_historical_data()
        
        # Setup real-time callbacks
        self._setup_realtime_callbacks()
        
        # Connect to SignalR
        self.api_client.connect_realtime(
            account_id=self.api_client.account_id,
            contract_ids=[self.contract_id_a, self.contract_id_b] if self.contract_id_a and self.contract_id_b else None
        )
        
        logger.info("StatArb Strategy initialized successfully")
        return True
    
    def _load_historical_data(self):
        """Load historical candles for spread calculation"""
        logger.info("Loading historical data for spread calculation...")
        
        # Fetch bars for both instruments
        # Use smaller limit to avoid timeout - API will return what's available
        # Statarb needs ~1440 bars (1 day of 1m bars), but request more to account for gaps
        requested_limit = 2000
        actual_limit = min(requested_limit, 1500)  # Reduce to avoid timeout
        
        logger.info(f"Fetching historical bars (limit={actual_limit}, may take longer for large requests)...")
        bars_gc = self.api_client.get_bars(
            contract_id=self.contract_id_a,
            interval="1m",
            limit=actual_limit  # Enough for spread history
        )
        
        bars_mgc = self.api_client.get_bars(
            contract_id=self.contract_id_b,
            interval="1m",
            limit=actual_limit
        )
        
        if bars_gc and bars_mgc:
            adapter = MarketDataAdapter()
            candles_gc = adapter.normalize_bars(bars_gc, self.symbol_a, "1m")
            candles_mgc = adapter.normalize_bars(bars_mgc, self.symbol_b, "1m")
            
            # Convert to DataFrames (similar to backtest format)
            self.df_gc = pd.DataFrame([{
                'timestamp': c.timestamp,
                'open': c.open,
                'high': c.high,
                'low': c.low,
                'close': c.close,
                'volume': c.volume
            } for c in candles_gc])
            
            self.df_mgc = pd.DataFrame([{
                'timestamp': c.timestamp,
                'open': c.open,
                'high': c.high,
                'low': c.low,
                'close': c.close,
                'volume': c.volume
            } for c in candles_mgc])
            
            # Align by timestamp
            merged = pd.merge(
                self.df_gc,
                self.df_mgc,
                on='timestamp',
                how='inner',
                suffixes=('_gc', '_mgc')
            )
            
            if len(merged) > 0:
                self.df_gc = merged[['timestamp', 'open_gc', 'high_gc', 'low_gc', 'close_gc', 'volume_gc']].copy()
                self.df_gc.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                
                self.df_mgc = merged[['timestamp', 'open_mgc', 'high_mgc', 'low_mgc', 'close_mgc', 'volume_mgc']].copy()
                self.df_mgc.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                
                logger.info(f"Loaded {len(self.df_gc)} aligned bars for spread calculation")
                
                # Initialize current prices
                if len(self.df_gc) > 0:
                    self.current_price_a = self.df_gc.iloc[-1]['close']
                    self.current_price_b = self.df_mgc.iloc[-1]['close']
                    self.position_manager.update_price(self.symbol_a, self.current_price_a)
                    self.position_manager.update_price(self.symbol_b, self.current_price_b)
                    logger.info(f"Initialized prices: {self.symbol_a}={self.current_price_a:.2f}, {self.symbol_b}={self.current_price_b:.2f}")
    
    def _setup_realtime_callbacks(self):
        """Setup callbacks for real-time market data updates"""
        symbol_id_a = f"F.US.{self.symbol_a}"
        symbol_id_b = f"F.US.{self.symbol_b}"
        
        def on_quote_update(data):
            """Handle real-time quote updates"""
            try:
                if isinstance(data, list):
                    if len(data) > 0 and isinstance(data[0], dict):
                        data = data[0]
                    else:
                        return
                
                if not isinstance(data, dict):
                    return
                
                symbol_id_data = data.get('symbol', '')
                last_price = data.get('lastPrice')
                
                if last_price is None:
                    return
                
                with self.data_lock:
                    if symbol_id_data == symbol_id_a:
                        self.current_price_a = float(last_price)
                        self.position_manager.update_price(self.symbol_a, self.current_price_a)
                    elif symbol_id_data == symbol_id_b:
                        self.current_price_b = float(last_price)
                        self.position_manager.update_price(self.symbol_b, self.current_price_b)
                    
            except Exception as e:
                logger.error(f"Error processing quote update: {e}")
        
        def on_market_trade_update(data):
            """Handle real-time trade updates"""
            try:
                if isinstance(data, list):
                    if len(data) > 0 and isinstance(data[0], dict):
                        data = data[0]
                    else:
                        return
                
                if not isinstance(data, dict):
                    return
                
                symbol_id_data = data.get('symbolId', '')
                price = data.get('price')
                
                if price is None:
                    return
                
                with self.data_lock:
                    if symbol_id_data == symbol_id_a:
                        self.current_price_a = float(price)
                        self.position_manager.update_price(self.symbol_a, self.current_price_a)
                    elif symbol_id_data == symbol_id_b:
                        self.current_price_b = float(price)
                        self.position_manager.update_price(self.symbol_b, self.current_price_b)
                    
            except Exception as e:
                logger.error(f"Error processing trade update: {e}")
        
        # Register callbacks
        self.api_client.register_realtime_callback("on_quote_update", on_quote_update)
        self.api_client.register_realtime_callback("on_market_trade_update", on_market_trade_update)
        
        logger.info("Registered real-time market data callbacks")
    
    def _update_historical_data(self):
        """Update historical data with latest bars"""
        # Fetch latest bars
        bars_gc = self.api_client.get_bars(
            contract_id=self.contract_id_a,
            interval="1m",
            limit=100  # Just get recent bars
        )
        
        bars_mgc = self.api_client.get_bars(
            contract_id=self.contract_id_b,
            interval="1m",
            limit=100
        )
        
        if bars_gc and bars_mgc:
            adapter = MarketDataAdapter()
            candles_gc = adapter.normalize_bars(bars_gc, self.symbol_a, "1m")
            candles_mgc = adapter.normalize_bars(bars_mgc, self.symbol_b, "1m")
            
            # Convert to DataFrames and merge
            df_gc_new = pd.DataFrame([{
                'timestamp': c.timestamp,
                'open': c.open,
                'high': c.high,
                'low': c.low,
                'close': c.close,
                'volume': c.volume
            } for c in candles_gc])
            
            df_mgc_new = pd.DataFrame([{
                'timestamp': c.timestamp,
                'open': c.open,
                'high': c.high,
                'low': c.low,
                'close': c.close,
                'volume': c.volume
            } for c in candles_mgc])
            
            # Merge and update
            merged = pd.merge(
                df_gc_new,
                df_mgc_new,
                on='timestamp',
                how='inner',
                suffixes=('_gc', '_mgc')
            )
            
            if len(merged) > 0 and self.df_gc is not None:
                # Remove duplicates and append
                all_timestamps = set(self.df_gc['timestamp'].tolist())
                new_rows = merged[~merged['timestamp'].isin(all_timestamps)]
                
                if len(new_rows) > 0:
                    df_gc_append = new_rows[['timestamp', 'open_gc', 'high_gc', 'low_gc', 'close_gc', 'volume_gc']].copy()
                    df_gc_append.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    
                    df_mgc_append = new_rows[['timestamp', 'open_mgc', 'high_mgc', 'low_mgc', 'close_mgc', 'volume_mgc']].copy()
                    df_mgc_append.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    
                    self.df_gc = pd.concat([self.df_gc, df_gc_append], ignore_index=True)
                    self.df_mgc = pd.concat([self.df_mgc, df_mgc_append], ignore_index=True)
                    
                    # Keep only last N bars (e.g., 2000)
                    if len(self.df_gc) > 2000:
                        self.df_gc = self.df_gc.tail(2000).reset_index(drop=True)
                        self.df_mgc = self.df_mgc.tail(2000).reset_index(drop=True)
    
    def _calculate_position_size(
        self,
        entry_price_a: float,
        entry_price_b: float,
        stop_spread: float,
        beta: float,
        is_long_spread: bool,
        spread_std: Optional[float] = None,
        entry_spread: Optional[float] = None
    ) -> Tuple[int, int]:
        """
        Calculate position sizes dynamically based on risk per trade and stop distance.
        
        Args:
            entry_price_a: GC entry price
            entry_price_b: MGC entry price
            stop_spread: Stop loss spread level
            beta: Hedge ratio
            is_long_spread: True if long spread
            spread_std: Standard deviation of spread (for volatility-based adjustments)
            entry_spread: Entry spread value (for calculating stop distance)
        
        Returns:
            (contracts_gc, contracts_mgc) maintaining 1:10 ratio
        """
        # If dynamic sizing disabled, use fixed sizing
        if not self.dynamic_sizing_enabled:
            contracts_gc = 1
            contracts_mgc = 10
            logger.debug("Dynamic sizing disabled, using fixed: 1 GC, 10 MGC")
            return contracts_gc, contracts_mgc
        
        # Calculate entry spread if not provided
        if entry_spread is None:
            entry_spread = entry_price_a - beta * entry_price_b
        
        # Calculate stop distance (how far spread can move before stop loss)
        stop_distance = abs(entry_spread - stop_spread)
        
        if stop_distance == 0:
            logger.warning("Stop distance is 0, using minimum position size")
            return 1, 10
        
        # Volatility-based stop adjustment
        effective_stop_std = self.spread_stop_std
        if self.volatility_stop_adjustment and spread_std is not None:
            if spread_std > self.high_volatility_threshold:
                # High volatility: use tighter stop
                effective_stop_std = self.high_volatility_stop_std
                logger.info(
                    f"High volatility detected (spread_std=${spread_std:.2f}), "
                    f"using tighter stop: {effective_stop_std} std (normal: {self.spread_stop_std})"
                )
                # Recalculate stop_distance with adjusted stop_std
                if is_long_spread:
                    adjusted_stop_spread = entry_spread - effective_stop_std * spread_std
                else:
                    adjusted_stop_spread = entry_spread + effective_stop_std * spread_std
                stop_distance = abs(entry_spread - adjusted_stop_spread)
        
        # Calculate risk per GC contract
        # For spread trade: if spread moves by stop_distance, GC moves approximately by that amount
        # GC contract = 100 oz, so risk = stop_distance × 100 oz
        risk_per_gc_contract = stop_distance * 100.0  # 100 oz per GC contract
        
        if risk_per_gc_contract <= 0:
            logger.warning("Risk per contract is 0 or negative, using minimum position size")
            return 1, 10
        
        # Calculate contracts needed to match risk_per_trade
        contracts_gc = int(self.risk_per_trade / risk_per_gc_contract)
        
        # Enforce minimum (can't trade 0 contracts)
        contracts_gc = max(1, contracts_gc)
        
        # Enforce maximum (hard cap at 2 GC / 10 MGC for safety)
        contracts_gc = min(contracts_gc, self.max_contracts_gc)
        contracts_gc = min(contracts_gc, 2)  # Hard cap at 2 GC
        
        # Maintain 1:10 hedge ratio
        contracts_mgc = contracts_gc * 10
        contracts_mgc = min(contracts_mgc, 10)  # Hard cap at 10 MGC (maintains 1:10 ratio)
        
        # Calculate actual risk
        actual_risk = risk_per_gc_contract * contracts_gc
        risk_multiplier = actual_risk / self.risk_per_trade
        
        # Check if risk is too high
        if risk_multiplier > self.max_risk_multiplier:
            logger.warning(
                f"⚠️ Calculated risk (${actual_risk:.2f}) exceeds max multiplier "
                f"({self.max_risk_multiplier}x = ${self.risk_per_trade * self.max_risk_multiplier:.2f}). "
                f"Using minimum position size to limit risk."
            )
            # Use minimum size to limit risk
            contracts_gc = 1
            contracts_mgc = 10
            actual_risk = risk_per_gc_contract * contracts_gc
            risk_multiplier = actual_risk / self.risk_per_trade
        
        # Log position sizing details
        spread_std_str = f"{spread_std:.2f}" if spread_std is not None else "N/A"
        logger.info(
            f"📊 Dynamic Position Sizing: "
            f"stop_distance=${stop_distance:.2f}, "
            f"spread_std=${spread_std_str}, "
            f"risk_per_contract=${risk_per_gc_contract:.2f}, "
            f"contracts_gc={contracts_gc}, contracts_mgc={contracts_mgc}, "
            f"actual_risk=${actual_risk:.2f} ({risk_multiplier:.2f}x intended ${self.risk_per_trade:.2f})"
        )
        
        # Warn if risk is significantly different from intended
        if risk_multiplier > 2.0:
            logger.warning(
                f"⚠️ Actual risk (${actual_risk:.2f}) is {risk_multiplier:.2f}x intended "
                f"(${self.risk_per_trade:.2f}) due to contract minimums. "
                f"Consider tighter stops or skipping high-volatility trades."
            )
        elif risk_multiplier < self.min_risk_multiplier:
            logger.info(
                f"ℹ️ Actual risk (${actual_risk:.2f}) is {risk_multiplier:.2f}x intended "
                f"(${self.risk_per_trade:.2f}). Could use more contracts, but at minimum size."
            )
        
        return contracts_gc, contracts_mgc
    
    def _check_order_rejected(self, order_id: str, symbol: str) -> bool:
        """
        Check if an order was immediately rejected after placement.
        Returns True if order is rejected, False otherwise.
        """
        if not order_id or self.dry_run:
            return False
        
        try:
            # Get recent orders to check status
            recent_orders = self.api_client.get_orders(
                self.api_client.account_id,
                start_timestamp=datetime.now(timezone.utc) - timedelta(minutes=5)
            )
            
            for order in recent_orders:
                order_id_str = str(order.get('id') or order.get('orderId', ''))
                if order_id_str == str(order_id):
                    status = order.get('status')
                    # Status 4 = Rejected
                    if status == 4:
                        rejection_reason = order.get('rejectionReason') or order.get('reason') or order.get('message') or 'Unknown'
                        logger.warning(
                            f"⚠️ {symbol} order {order_id} was rejected: {rejection_reason}"
                        )
                        return True
                    # Status 2 = Filled, 1 = Pending, 3 = Cancelled (but not rejected)
                    return False
            
            # Order not found in recent orders - might still be pending
            return False
        except Exception as e:
            logger.debug(f"Error checking order status for {symbol} order {order_id}: {e}")
            return False
    
    def _check_spread_divergence(self, signal: SpreadSignal) -> bool:
        """
        Check if spread has been extreme for too long (divergence filter)
        Returns True if entry should be blocked due to divergence
        """
        now = datetime.now(timezone.utc)
        current_direction = "LONG" if signal.signal == "LONG_SPREAD" else "SHORT"
        
        # Check if spread is currently extreme (z-score exceeds entry threshold)
        is_extreme = abs(signal.zscore) >= self.calculator.z_entry
        
        if is_extreme:
            # If this is a new extreme direction or continuation of same direction
            if self.last_extreme_direction != current_direction:
                # New extreme direction - reset timer
                self.extreme_spread_start_time = now
                self.last_extreme_direction = current_direction
                logger.debug(f"Spread became extreme ({current_direction}), starting divergence timer")
                return False  # Allow entry, just started being extreme
            elif self.extreme_spread_start_time is not None:
                # Same direction, check duration
                duration_hours = (now - self.extreme_spread_start_time).total_seconds() / 3600.0
                if duration_hours >= self.spread_divergence_hours:
                    logger.warning(
                        f"BLOCKING ENTRY: Spread has been extreme ({current_direction}) "
                        f"for {duration_hours:.2f} hours (limit: {self.spread_divergence_hours} hours). "
                        f"Z-score: {signal.zscore:.2f}"
                    )
                    return True  # Block entry - spread has diverged too long
                else:
                    logger.debug(
                        f"Spread extreme for {duration_hours:.2f} hours "
                        f"(limit: {self.spread_divergence_hours} hours) - allowing entry"
                    )
                    return False  # Still within limit
            else:
                # First time tracking - start timer
                self.extreme_spread_start_time = now
                self.last_extreme_direction = current_direction
                return False
        else:
            # Spread is not extreme - reset tracking
            if self.extreme_spread_start_time is not None:
                logger.debug("Spread returned to normal range, resetting divergence timer")
            self.extreme_spread_start_time = None
            self.last_extreme_direction = None
            return False  # Not extreme, no blocking needed
    
    def _enter_spread_position(self, signal: SpreadSignal) -> bool:
        """Enter a spread position (both legs simultaneously)"""
        # Check if we already have a position
        if self.current_position:
            logger.warning("Already in a position, cannot enter new one")
            return False
        
        # Calculate stop spread (will be adjusted if volatility-based adjustment is enabled)
        stop_spread = signal.spread_mean + (
            -self.spread_stop_std * signal.spread_std if signal.signal == "LONG_SPREAD"
            else self.spread_stop_std * signal.spread_std
        )
        
        # Calculate position sizes with dynamic sizing
        contracts_gc, contracts_mgc = self._calculate_position_size(
            signal.price_a,
            signal.price_b,
            stop_spread,
            signal.beta,
            signal.signal == "LONG_SPREAD",
            spread_std=signal.spread_std,
            entry_spread=signal.spread
        )
        
        # If dynamic sizing returned 0 contracts (risk too high), skip this trade
        if contracts_gc == 0 or contracts_mgc == 0:
            logger.warning(
                f"❌ Position sizing returned 0 contracts - risk too high. "
                f"Skipping trade. spread_std=${signal.spread_std:.2f}, "
                f"stop_distance would be ${abs(signal.spread - stop_spread):.2f}"
            )
            return False
        
        # Determine order sides
        if signal.signal == "LONG_SPREAD":
            side_gc = "BUY"
            side_mgc = "SELL"
        else:  # SHORT_SPREAD
            side_gc = "SELL"
            side_mgc = "BUY"
        
        # Use current prices for orders
        with self.data_lock:
            order_price_gc = self.current_price_a or signal.price_a
            order_price_mgc = self.current_price_b or signal.price_b
        
        # Place orders for both legs
        order_id_gc = self.order_client.place_limit_order(
            contract_id=self.contract_id_a,
            side=side_gc,
            quantity=contracts_gc,
            price=order_price_gc
        )
        
        order_id_mgc = self.order_client.place_limit_order(
            contract_id=self.contract_id_b,
            side=side_mgc,
            quantity=contracts_mgc,
            price=order_price_mgc
        )
        
        # Check if orders were immediately rejected (e.g., "Trading is not available for this instrument")
        if order_id_gc and not self.dry_run:
            gc_rejected = self._check_order_rejected(order_id_gc, "GC")
            if gc_rejected:
                logger.error(
                    f"❌ GC order {order_id_gc} was rejected immediately. "
                    f"GC trading may not be available on this account. Skipping spread trade."
                )
                if order_id_mgc:
                    self.order_client.cancel_order(order_id_mgc)
                return False
        
        if order_id_mgc and not self.dry_run:
            mgc_rejected = self._check_order_rejected(order_id_mgc, "MGC")
            if mgc_rejected:
                logger.error(
                    f"❌ MGC order {order_id_mgc} was rejected immediately. "
                    f"Not enough margin or trading not available. Skipping spread trade."
                )
                if order_id_gc:
                    self.order_client.cancel_order(order_id_gc)
                return False
        
        # CRITICAL: For spread trades, BOTH legs must be placed successfully
        # If one fails, cancel the other to prevent unhedged positions
        if not order_id_gc or not order_id_mgc:
            # One or both orders failed - cancel any that succeeded
            if order_id_gc:
                logger.warning(f"GC order placed but MGC order failed - cancelling GC order {order_id_gc}")
                self.order_client.cancel_order(order_id_gc)
            if order_id_mgc:
                logger.warning(f"MGC order placed but GC order failed - cancelling MGC order {order_id_mgc}")
                self.order_client.cancel_order(order_id_mgc)
            logger.error(
                f"❌ SPREAD ENTRY FAILED: Both legs must be placed. "
                f"GC order: {'SUCCESS' if order_id_gc else 'FAILED'}, "
                f"MGC order: {'SUCCESS' if order_id_mgc else 'FAILED'}"
            )
            return False
        
        # Both orders placed successfully - track them
        order_id_gc_str = str(order_id_gc)
        order_id_mgc_str = str(order_id_mgc)
        self.pending_entry_orders.add(order_id_gc_str)
        self.pending_entry_orders.add(order_id_mgc_str)
        self.order_details[order_id_gc_str] = (self.symbol_a, side_gc, contracts_gc)
        self.order_details[order_id_mgc_str] = (self.symbol_b, side_mgc, contracts_mgc)
        
        # Create position only if BOTH orders were placed successfully
        if order_id_gc and order_id_mgc:
            # Create position - ensure timezone-aware datetime
            if hasattr(signal.timestamp, 'to_pydatetime'):
                entry_time = signal.timestamp.to_pydatetime()
                # If timezone-aware, convert to UTC; if naive, assume UTC
                if entry_time.tzinfo is None:
                    entry_time = entry_time.replace(tzinfo=timezone.utc)
                else:
                    entry_time = entry_time.astimezone(timezone.utc)
            else:
                entry_time = datetime.now(timezone.utc)
            
            self.current_position = StatArbPosition(
                entry_time=entry_time,
                entry_price_gc=order_price_gc if order_id_gc else signal.price_a,
                entry_price_mgc=order_price_mgc if order_id_mgc else signal.price_b,
                entry_spread=signal.spread,
                entry_zscore=signal.zscore,
                beta=signal.beta,
                contracts_gc=contracts_gc,
                contracts_mgc=contracts_mgc,
                is_long_spread=(signal.signal == "LONG_SPREAD"),
                order_id_gc=order_id_gc,
                order_id_mgc=order_id_mgc
            )
            self.current_position.remaining_gc = contracts_gc
            self.current_position.remaining_mgc = contracts_mgc
            self.current_position.partial_taken = False
            self.current_position.best_spread = signal.spread
            
            logger.info(
                f"✅ PLACED BOTH ORDERS for {signal.signal} at z={signal.zscore:.2f} | "
                f"GC: {side_gc} {contracts_gc} @ {order_price_gc:.2f} (order: {order_id_gc}) | "
                f"MGC: {side_mgc} {contracts_mgc} @ {order_price_mgc:.2f} (order: {order_id_mgc}) | "
                f"Spread={signal.spread:.2f}"
            )
            return True
        else:
            # This should never happen now (we return False above), but keep as safety
            logger.error(f"Failed to create position: GC={order_id_gc}, MGC={order_id_mgc}")
            return False
    
    def _exit_spread_position(self, exit_reason: str) -> bool:
        """Exit current spread position (close both legs)"""
        if not self.current_position:
            return False
        
        pos = self.current_position
        remaining_gc = pos.remaining_gc if pos.remaining_gc is not None else pos.contracts_gc
        remaining_mgc = pos.remaining_mgc if pos.remaining_mgc is not None else pos.contracts_mgc
        if remaining_gc <= 0 or remaining_mgc <= 0:
            logger.warning("No remaining contracts to exit")
            return False
        
        # Determine exit sides (opposite of entry)
        if pos.is_long_spread:
            side_gc = "SELL"
            side_mgc = "BUY"
        else:
            side_gc = "BUY"
            side_mgc = "SELL"
        
        # Get current prices
        with self.data_lock:
            exit_price_gc = self.current_price_a or pos.entry_price_gc
            exit_price_mgc = self.current_price_b or pos.entry_price_mgc
        
        # Place exit orders
        exit_order_gc = self.order_client.place_limit_order(
            contract_id=self.contract_id_a,
            side=side_gc,
            quantity=remaining_gc,
            price=exit_price_gc
        )
        
        exit_order_mgc = self.order_client.place_limit_order(
            contract_id=self.contract_id_b,
            side=side_mgc,
            quantity=remaining_mgc,
            price=exit_price_mgc
        )
        
        # Track exit orders for fill handling
        if exit_order_gc:
            exit_order_gc_str = str(exit_order_gc)
            self.order_details[exit_order_gc_str] = (self.symbol_a, side_gc, remaining_gc)
        if exit_order_mgc:
            exit_order_mgc_str = str(exit_order_mgc)
            self.order_details[exit_order_mgc_str] = (self.symbol_b, side_mgc, remaining_mgc)
        
        if exit_order_gc and exit_order_mgc:
            logger.info(
                f"PLACED EXIT ORDERS: {exit_reason} | "
                f"GC: {side_gc} {remaining_gc} @ {exit_price_gc:.2f} | "
                f"MGC: {side_mgc} {remaining_mgc} @ {exit_price_mgc:.2f}"
            )
            pos.remaining_gc = 0
            pos.remaining_mgc = 0
            return True
        else:
            logger.error(f"Failed to place exit orders: GC={exit_order_gc}, MGC={exit_order_mgc}")
            return False

    def _partial_exit_position(self, exit_reason: str, fraction: float) -> bool:
        """Exit part of the current spread position (close both legs proportionally)"""
        if not self.current_position:
            return False
        
        pos = self.current_position
        remaining_gc = pos.remaining_gc if pos.remaining_gc is not None else pos.contracts_gc
        remaining_mgc = pos.remaining_mgc if pos.remaining_mgc is not None else pos.contracts_mgc
        
        if remaining_gc <= 1:
            logger.debug("Partial exit skipped: only 1 GC contract remaining")
            return False
        
        fraction = max(0.1, min(0.9, fraction))
        partial_gc = max(1, int(remaining_gc * fraction))
        partial_mgc = max(1, int(remaining_mgc * fraction))
        
        # Ensure we leave at least 1 GC and 10 MGC (hedge ratio) remaining
        if remaining_gc - partial_gc < 1:
            partial_gc = remaining_gc - 1
        if remaining_mgc - partial_mgc < 10:
            partial_mgc = remaining_mgc - 10
        
        if partial_gc <= 0 or partial_mgc <= 0:
            logger.debug("Partial exit skipped: not enough contracts to reduce")
            return False
        
        # Determine exit sides (opposite of entry)
        if pos.is_long_spread:
            side_gc = "SELL"
            side_mgc = "BUY"
        else:
            side_gc = "BUY"
            side_mgc = "SELL"
        
        # Get current prices
        with self.data_lock:
            exit_price_gc = self.current_price_a or pos.entry_price_gc
            exit_price_mgc = self.current_price_b or pos.entry_price_mgc
        
        # Place partial exit orders
        exit_order_gc = self.order_client.place_limit_order(
            contract_id=self.contract_id_a,
            side=side_gc,
            quantity=partial_gc,
            price=exit_price_gc
        )
        
        exit_order_mgc = self.order_client.place_limit_order(
            contract_id=self.contract_id_b,
            side=side_mgc,
            quantity=partial_mgc,
            price=exit_price_mgc
        )
        
        # Track exit orders for fill handling
        if exit_order_gc:
            exit_order_gc_str = str(exit_order_gc)
            self.order_details[exit_order_gc_str] = (self.symbol_a, side_gc, partial_gc)
        if exit_order_mgc:
            exit_order_mgc_str = str(exit_order_mgc)
            self.order_details[exit_order_mgc_str] = (self.symbol_b, side_mgc, partial_mgc)
        
        if exit_order_gc and exit_order_mgc:
            pos.remaining_gc = remaining_gc - partial_gc
            pos.remaining_mgc = remaining_mgc - partial_mgc
            pos.partial_taken = True
            logger.info(
                f"PARTIAL EXIT: {exit_reason} | "
                f"GC: {side_gc} {partial_gc} @ {exit_price_gc:.2f} | "
                f"MGC: {side_mgc} {partial_mgc} @ {exit_price_mgc:.2f} | "
                f"Remaining GC={pos.remaining_gc}, MGC={pos.remaining_mgc}"
            )
            return True
        
        logger.error(f"Failed to place partial exit orders: GC={exit_order_gc}, MGC={exit_order_mgc}")
        return False
    
    def _check_exit_conditions(self) -> Tuple[bool, str]:
        """Check if current position should be exited"""
        if not self.current_position:
            return False, ""
        
        # CRITICAL: Don't check exit conditions if entry orders haven't been filled yet
        # This prevents premature exits that clear the position before it's actually entered
        if self.pending_entry_orders:
            logger.debug(
                f"Exit check skipped: entry orders still pending ({len(self.pending_entry_orders)} orders)"
            )
            return False, ""
        
        pos = self.current_position
        
        # Check time stop - ensure both datetimes are timezone-aware
        now = datetime.now(timezone.utc)
        entry_time = pos.entry_time
        # If entry_time is naive, assume UTC; if aware, ensure UTC
        if entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)
        else:
            entry_time = entry_time.astimezone(timezone.utc)
        
        duration = (now - entry_time).total_seconds() / 3600
        if duration >= self.time_stop_hours:
            return True, "TIME_STOP"
        
        # Need current prices and spread calculation
        with self.data_lock:
            if not self.current_price_a or not self.current_price_b:
                return False, ""
            
            current_price_gc = self.current_price_a
            current_price_mgc = self.current_price_b
        
        # Calculate current spread and z-score
        current_spread = current_price_gc - pos.beta * current_price_mgc
        
        # Get spread history for z-score (need to recalculate with entry beta)
        if self.df_gc is None or len(self.df_gc) < self.calculator.min_lookback:
            return False, ""
        
        # Recalculate spread history with entry beta
        lookback_start = max(0, len(self.df_gc) - self.calculator.lookback_periods)
        spread_history_list = []
        for j in range(lookback_start, len(self.df_gc)):
            price_gc_j = self.df_gc.iloc[j]['close']
            price_mgc_j = self.df_mgc.iloc[j]['close']
            spread_j = price_gc_j - pos.beta * price_mgc_j
            spread_history_list.append(spread_j)
        spread_history = pd.Series(spread_history_list)
        
        current_zscore, spread_mean, spread_std = self.calculator.calculate_zscore(
            current_spread,
            spread_history
        )

        # Track best favorable spread for trailing stop
        if pos.best_spread is None:
            pos.best_spread = current_spread
        else:
            if pos.is_long_spread and current_spread > pos.best_spread:
                pos.best_spread = current_spread
            elif not pos.is_long_spread and current_spread < pos.best_spread:
                pos.best_spread = current_spread
        
        # Partial profit taking (only if we have enough size)
        if not pos.partial_taken and abs(current_zscore) <= self.partial_profit_z:
            if self._partial_exit_position("PARTIAL_PROFIT", self.partial_profit_fraction):
                # Reset best spread to start trailing from the partial profit level
                pos.best_spread = current_spread
            return False, ""
        
        # Trailing stop after partial profit
        if pos.partial_taken and spread_std > 0:
            trail_distance = self.trailing_stop_std * spread_std
            if pos.is_long_spread:
                if current_spread <= pos.best_spread - trail_distance:
                    return True, "TRAILING_SL"
            else:
                if current_spread >= pos.best_spread + trail_distance:
                    return True, "TRAILING_SL"
        
        # Check z-score exit
        if abs(current_zscore) < self.calculator.z_exit:
            return True, "Z_EXIT"
        
        # Check zero cross
        if pos.is_long_spread and current_zscore > 0:
            return True, "ZERO_CROSS"
        if not pos.is_long_spread and current_zscore < 0:
            return True, "ZERO_CROSS"
        
        # Check stop loss
        spread_move = current_spread - pos.entry_spread
        if pos.is_long_spread:
            if spread_move < -self.spread_stop_std * spread_std:
                return True, "SL"
        else:
            if spread_move > self.spread_stop_std * spread_std:
                return True, "SL"
        
        return False, ""
    
    def _check_order_fills(self):
        """Check if pending orders have been filled"""
        if not self.current_position or not self.api_client.account_id:
            # If no position but we have pending orders, check if they were cancelled/rejected
            if self.pending_entry_orders and not self.dry_run:
                # Check recent orders to see if any pending orders were cancelled/rejected
                try:
                    recent_orders = self.api_client.get_orders(
                        self.api_client.account_id,
                        start_timestamp=datetime.now(timezone.utc) - timedelta(hours=1)
                    )
                    for order in recent_orders:
                        order_id = str(order.get('id') or order.get('orderId', ''))
                        if order_id in self.pending_entry_orders:
                            status = order.get('status')
                            # Status 3 = Cancelled, 4 = Rejected, etc.
                            if status in [3, 4] or status != 2:  # Not filled
                                logger.warning(f"Pending order {order_id} was cancelled/rejected, removing from tracking")
                                self.pending_entry_orders.discard(order_id)
                except Exception as e:
                    logger.debug(f"Error checking pending orders status: {e}")
            return
        
        # Get recent orders to check for fills
        recent_orders = self.api_client.get_orders(
            self.api_client.account_id,
            start_timestamp=datetime.now(timezone.utc) - timedelta(hours=1)
        )
        
        gc_filled = False
        mgc_filled = False
        
        for order in recent_orders:
            order_id = str(order.get('id') or order.get('orderId', ''))
            status = order.get('status')  # 2 = Filled, 3 = Cancelled, 4 = Rejected
            
            # Handle any tracked order (entry or exit) - check order_details first
            if order_id in self.order_details and status == 2:  # Filled
                fill_price = order.get('filledPrice') or order.get('limitPrice')
                if fill_price:
                    fill_price_float = float(fill_price)
                    symbol, side, qty = self.order_details[order_id]
                    # CRITICAL: Notify position manager of fill (works for both entry and exit)
                    self.position_manager.on_fill(symbol, side, qty, fill_price_float)
                    logger.info(f"Order filled: {side} {qty} {symbol} @ {fill_price} (order: {order_id})")
                    # Remove from tracking after fill
                    self.order_details.pop(order_id, None)
                    if order_id in self.pending_entry_orders:
                        self.pending_entry_orders.discard(order_id)
            
            # Check if this is one of our position entry orders (for position tracking)
            if self.current_position and order_id == str(self.current_position.order_id_gc):
                if status == 2:  # Filled
                    fill_price = order.get('filledPrice') or order.get('limitPrice')
                    if fill_price:
                        fill_price_float = float(fill_price)
                        self.current_position.entry_price_gc = fill_price_float
                        gc_filled = True
                        # Remove from pending orders
                        self.pending_entry_orders.discard(order_id)
                        # Note: on_fill already called above for all tracked orders
                        logger.info(f"GC entry order filled: {order_id} @ {fill_price}")
                elif status in [3, 4]:  # Cancelled or Rejected
                    logger.warning(f"GC order {order_id} was cancelled/rejected (status: {status})")
                    self.pending_entry_orders.discard(order_id)
                    self.order_details.pop(order_id, None)
                    # CRITICAL: Cancel the other leg to prevent unhedged position
                    if self.current_position and self.current_position.order_id_mgc:
                        logger.warning("GC leg cancelled/rejected - cancelling MGC leg to prevent unhedged position")
                        self.order_client.cancel_order(self.current_position.order_id_mgc)
                    # Clear position and pending orders
                    if self.current_position:
                        logger.warning("One leg cancelled/rejected, clearing position and pending orders")
                        self.current_position = None
                        self.pending_entry_orders.clear()
                        self.order_details.clear()
            elif order_id == str(self.current_position.order_id_mgc):
                if status == 2:  # Filled
                    fill_price = order.get('filledPrice') or order.get('limitPrice')
                    if fill_price:
                        fill_price_float = float(fill_price)
                        self.current_position.entry_price_mgc = fill_price_float
                        mgc_filled = True
                        # Remove from pending orders
                        self.pending_entry_orders.discard(order_id)
                        # Note: on_fill already called above for all tracked orders
                        # Recalculate entry spread with actual fill prices
                        self.current_position.entry_spread = (
                            self.current_position.entry_price_gc - 
                            self.current_position.beta * self.current_position.entry_price_mgc
                        )
                        logger.info(f"MGC entry order filled: {order_id} @ {fill_price}")
                elif status in [3, 4]:  # Cancelled or Rejected
                    logger.warning(f"MGC order {order_id} was cancelled/rejected (status: {status})")
                    self.pending_entry_orders.discard(order_id)
                    self.order_details.pop(order_id, None)
                    # CRITICAL: Cancel the other leg to prevent unhedged position
                    if self.current_position and self.current_position.order_id_gc:
                        logger.warning("MGC leg cancelled/rejected - cancelling GC leg to prevent unhedged position")
                        self.order_client.cancel_order(self.current_position.order_id_gc)
                    # Clear position and pending orders
                    if self.current_position:
                        logger.warning("One leg cancelled/rejected, clearing position and pending orders")
                        self.current_position = None
                        self.pending_entry_orders.clear()
                        self.order_details.clear()
        
        # If both orders are filled, clear pending orders set (should already be empty, but ensure it)
        if gc_filled and mgc_filled:
            self.pending_entry_orders.clear()
            # Clean up order details for filled orders
            if str(self.current_position.order_id_gc) in self.order_details:
                self.order_details.pop(str(self.current_position.order_id_gc))
            if str(self.current_position.order_id_mgc) in self.order_details:
                self.order_details.pop(str(self.current_position.order_id_mgc))
            logger.info("✅ Both entry orders filled - spread position fully entered")
        elif gc_filled and not mgc_filled:
            # GC filled but MGC didn't - check if MGC order still pending or failed
            if self.current_position and self.current_position.order_id_mgc:
                mgc_order_id = str(self.current_position.order_id_mgc)
                # Check if MGC order is still pending (not filled, not cancelled)
                mgc_order_status = None
                for order in recent_orders:
                    if str(order.get('id') or order.get('orderId', '')) == mgc_order_id:
                        mgc_order_status = order.get('status')
                        break
                
                if mgc_order_status == 2:  # Filled (should have been caught above)
                    pass  # Will be handled next iteration
                elif mgc_order_status in [3, 4]:  # Cancelled/Rejected
                    logger.error("❌ GC filled but MGC was cancelled/rejected - UNHEDGED POSITION! Cancelling GC immediately.")
                    # Cancel GC position immediately to prevent unhedged exposure
                    if self.current_position:
                        self._exit_spread_position("EMERGENCY_UNHEDGED")
                        self.current_position = None
                else:
                    # MGC order still pending - wait a bit, but log warning
                    logger.warning(f"⚠️ GC filled but MGC order {mgc_order_id} still pending - monitoring...")
        elif mgc_filled and not gc_filled:
            # MGC filled but GC didn't - same check
            if self.current_position and self.current_position.order_id_gc:
                gc_order_id = str(self.current_position.order_id_gc)
                gc_order_status = None
                for order in recent_orders:
                    if str(order.get('id') or order.get('orderId', '')) == gc_order_id:
                        gc_order_status = order.get('status')
                        break
                
                if gc_order_status == 2:  # Filled (should have been caught above)
                    pass  # Will be handled next iteration
                elif gc_order_status in [3, 4]:  # Cancelled/Rejected
                    logger.error("❌ MGC filled but GC was cancelled/rejected - UNHEDGED POSITION! Cancelling MGC immediately.")
                    # Cancel MGC position immediately to prevent unhedged exposure
                    if self.current_position:
                        self._exit_spread_position("EMERGENCY_UNHEDGED")
                        self.current_position = None
                else:
                    # GC order still pending - wait a bit, but log warning
                    logger.warning(f"⚠️ MGC filled but GC order {gc_order_id} still pending - monitoring...")
    
    def _process_tick(self):
        """Process a single tick - called periodically"""
        if self.paused or not self.running:
            return
        
        # Continuous risk checks: enforce daily loss and trailing drawdown limits
        if self._update_risk_checks():
            return
        
        # Update historical data periodically
        now = datetime.now(timezone.utc)
        if now - self.last_recalc_time > self.recalc_interval:
            self._update_historical_data()
            self.last_recalc_time = now
        
        # Need both prices and sufficient data
        with self.data_lock:
            if not self.current_price_a or not self.current_price_b:
                return
            
            if self.df_gc is None or len(self.df_gc) < self.calculator.min_lookback:
                return
            
            current_price_gc = self.current_price_a
            current_price_mgc = self.current_price_b
        
        # Calculate current beta
        current_index = len(self.df_gc) - 1
        current_beta = self.calculator.calculate_beta(self.df_gc, self.df_mgc, current_index)
        
        # Recalculate spread history with current beta
        lookback_start = max(0, current_index - self.calculator.lookback_periods)
        spread_history_list = []
        for j in range(lookback_start, current_index):
            price_gc = self.df_gc.iloc[j]['close']
            price_mgc = self.df_mgc.iloc[j]['close']
            spread = price_gc - current_beta * price_mgc
            spread_history_list.append(spread)
        spread_history = pd.Series(spread_history_list)
        
        # Get current position status
        current_pos = None
        if self.current_position:
            current_pos = "LONG_SPREAD" if self.current_position.is_long_spread else "SHORT_SPREAD"
        
        # Process bar and get signal
        signal = self.calculator.process_bar(
            self.df_gc,
            self.df_mgc,
            current_index,
            current_pos,
            spread_history,
            fixed_beta=current_beta
        )
        
        if signal is None:
            # Log why signal is None for debugging
            if len(spread_history) < self.calculator.min_lookback:
                logger.debug(
                    f"⚠️ No signal: Insufficient spread history. "
                    f"Have {len(spread_history)} periods, need {self.calculator.min_lookback}"
                )
            else:
                logger.debug(f"⚠️ No signal: process_bar returned None (check data alignment)")
            return
        
        # Log signal details for diagnostics
        logger.info(
            f"📊 STATARB SIGNAL: spread={signal.spread:.2f}, zscore={signal.zscore:.3f}, "
            f"beta={signal.beta:.3f}, signal={signal.signal}, "
            f"is_entry={signal.is_entry}, is_exit={signal.is_exit}, "
            f"current_pos={current_pos}, z_entry_threshold=±{self.calculator.z_entry}"
        )
        
        # Handle entry signals (only if no position)
        if signal.is_entry and not self.current_position:
            logger.info(f"✅ ENTRY SIGNAL TRIGGERED: {signal.signal} (zscore={signal.zscore:.3f})")
            self._enter_spread_position(signal)
        elif signal.is_entry and self.current_position:
            logger.debug(f"⏸️ Entry signal ignored: Already in position")
        elif not signal.is_entry and abs(signal.zscore) < self.calculator.z_entry:
            logger.debug(
                f"⏸️ No entry: |zscore|={abs(signal.zscore):.3f} < entry_threshold={self.calculator.z_entry}"
            )
        
        # Monitor position for exits
        # CRITICAL: Only check exit conditions if both entry orders are filled
        # If there are pending entry orders, we haven't actually entered the position yet
        if self.current_position and not self.pending_entry_orders:
            should_exit, exit_reason = self._check_exit_conditions()
            if should_exit:
                self._exit_spread_position(exit_reason)
                # Clear position after exit and reset divergence tracking
                self.current_position = None
                self.pending_entry_orders.clear()  # Clear any pending orders when position is cleared
                self.extreme_spread_start_time = None
                self.last_extreme_direction = None
        elif self.current_position and self.pending_entry_orders:
            # Position exists but orders not filled yet - wait for fills before checking exits
            logger.debug(
                f"Position exists but waiting for entry order fills. "
                f"Pending orders: {len(self.pending_entry_orders)}"
            )
    
    def _update_risk_checks(self) -> bool:
        """Update risk checks and return True if should stop"""
        if not self.api_client.account_id:
            return False
        
        # Get current exposure
        exposure = self.position_manager.net_exposure_dollars()
        
        # Update risk manager with equity (preferred) or balance
        # Equity includes unrealized P&L, which is what we need for accurate risk tracking
        accounts = self.api_client.get_accounts()
        if accounts:
            selected_account = next(
                (acc for acc in accounts if 
                 (acc.get('id') == self.api_client.account_id or 
                  acc.get('accountId') == self.api_client.account_id)),
                None
            )
            if selected_account:
                # Try to get equity first (includes unrealized P&L)
                current_equity = selected_account.get('equity') or selected_account.get('netLiquidation')
                current_balance = selected_account.get('balance')
                
                if current_equity is not None:
                    # Use equity (preferred - includes unrealized P&L)
                    self.risk_manager.update_equity(float(current_equity))
                elif current_balance is not None:
                    # Fallback to balance if equity not available
                    self.risk_manager.update_balance(float(current_balance))
                    logger.debug("Using balance instead of equity (equity not available from API)")
        
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
        
        return False
    
    def emergency_flatten(self, reason: str = "Emergency"):
        """Emergency flatten all positions"""
        logger.critical(f"EMERGENCY FLATTEN: {reason}")
        
        # Exit current position
        if self.current_position:
            self._exit_spread_position(reason)
            self.current_position = None
        
        # Clear pending orders tracking
        self.pending_entry_orders.clear()
        
        # Close all positions via API
        if not self.dry_run and self.api_client.account_id:
            positions = self.api_client.get_positions(self.api_client.account_id)
            print(f"Positions response: {positions}")
            for position in positions:
                contract_id = position.get('contractId')
                if contract_id:
                    self.api_client.close_position(contract_id)
        
        self.position_manager.flatten_all()
        self.running = False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status"""
        with self.data_lock:
            current_price_gc = self.current_price_a
            current_price_mgc = self.current_price_b
        
        # Calculate current spread if we have prices
        current_spread = None
        current_zscore = None
        if current_price_gc and current_price_mgc and self.current_position:
            current_spread = current_price_gc - self.current_position.beta * current_price_mgc
            # Could calculate z-score here if needed
        
        return {
            "status": "running" if self.running else "stopped",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            "trading_enabled": not self.paused and self.running,
            "dry_run": self.dry_run,
            "symbol_a": self.symbol_a,
            "symbol_b": self.symbol_b,
            "current_price_a": current_price_gc,
            "current_price_b": current_price_mgc,
            "current_spread": current_spread,
            "current_zscore": current_zscore,
            "has_position": self.current_position is not None,
            "position": {
                "entry_time": self.current_position.entry_time.isoformat() if self.current_position else None,
                "entry_spread": self.current_position.entry_spread if self.current_position else None,
                "entry_zscore": self.current_position.entry_zscore if self.current_position else None,
                "is_long_spread": self.current_position.is_long_spread if self.current_position else None,
                "contracts_gc": self.current_position.contracts_gc if self.current_position else None,
                "contracts_mgc": self.current_position.contracts_mgc if self.current_position else None,
            } if self.current_position else None,
            "total_pnl": self.position_manager.get_total_pnl(),
            "daily_pnl": self.risk_manager.get_daily_pnl(),
            "drawdown": self.risk_manager.get_trailing_drawdown(),
        }
    
    def run(self):
        """Main trading loop"""
        if not self.initialize():
            logger.error("Failed to initialize strategy")
            return
        
        self.running = True
        logger.info("Starting StatArb trading loop...")
        
        try:
            while self.running:
                if self.paused:
                    time.sleep(1)
                    continue
                
                # CRITICAL: Reconcile positions with API (source of truth)
                # This ensures PositionManager knows about actual positions
                try:
                    api_positions = self.api_client.get_positions(self.api_client.account_id)
                    if api_positions:
                        self.position_manager.reconcile_with_api_positions(api_positions)
                except Exception as e:
                    logger.debug(f"Error reconciling positions: {e}")
                
                # Check for order fills
                self._check_order_fills()
                
                # Process tick (calculate spread, check signals)
                self._process_tick()
                
                # Risk checks
                if self._update_risk_checks():
                    break
                
                # Update historical data periodically
                if datetime.now(timezone.utc).minute % 30 == 0:  # Every 30 minutes
                    self._load_historical_data()
                
                time.sleep(5)  # Check every 5 seconds
        
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        except Exception as e:
            logger.error(f"Error in trading loop: {e}", exc_info=True)
        finally:
            logger.info("Shutting down...")
            self.emergency_flatten("Shutdown")
            self.api_client.disconnect()

