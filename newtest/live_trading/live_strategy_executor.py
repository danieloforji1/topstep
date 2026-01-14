"""
Live Trading Executor for Production Strategies
Runs Optimal Stopping, Multi-Timeframe, and Liquidity Provision strategies on TopstepX
"""
import os
import sys
import time
import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import pandas as pd
from dotenv import load_dotenv

# Add parent directories to path
# live_trading/ -> newtest/ -> topstep/ -> (src/ and newtest/)
live_trading_dir = os.path.dirname(__file__)
newtest_dir = os.path.dirname(live_trading_dir)
topstep_dir = os.path.dirname(newtest_dir)

# Add topstep directory (contains src/)
sys.path.insert(0, topstep_dir)
# Add newtest directory (contains strategies/ and framework/)
sys.path.insert(0, newtest_dir)

from src.connectors.topstepx_client import TopstepXClient
from src.connectors.market_data_adapter import MarketDataAdapter, Tick, Candle
# TimeseriesStore is optional for live trading - we'll manage data in memory
# from src.data.timeseries_store import TimeseriesStore
from strategies.optimal_stopping import OptimalStoppingStrategy
from strategies.multi_timeframe import MultiTimeframeStrategy
from strategies.liquidity_provision import LiquidityProvisionStrategy
from framework.base_strategy import Signal, MarketData, ExitReason

# Load .env file if it exists (try multiple locations)
env_paths = [
    os.path.join(os.path.dirname(__file__), '../../../.env'),
    os.path.join(os.path.dirname(__file__), '../../.env'),
    os.path.join(os.path.dirname(__file__), '../.env'),
    '.env'
]
env_loaded = False
for env_path in env_paths:
    try:
        if os.path.exists(env_path) and os.access(env_path, os.R_OK):
            load_dotenv(env_path, override=False)
            env_loaded = True
            break
    except (PermissionError, OSError):
        continue

if not env_loaded:
    # Try default location (won't fail if file doesn't exist)
    try:
        load_dotenv(dotenv_path=None, override=False)
    except Exception:
        pass  # .env file is optional

logger = logging.getLogger(__name__)


@dataclass
class LivePosition:
    """Track live position"""
    strategy_name: str
    entry_time: datetime
    entry_price: float
    contracts: int
    is_long: bool
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    order_id: Optional[str] = None
    signal: Optional[Signal] = None


class LiveStrategyExecutor:
    """
    Executes strategies in live trading environment
    
    Features:
    - Real-time market data feed
    - Multi-strategy execution
    - Order management
    - Risk monitoring
    - Performance tracking
    """
    
    def __init__(
        self,
        symbol: str = "MES",
        contract_id: Optional[str] = None,
        dry_run: bool = True,
        strategies: Optional[List[str]] = None
    ):
        """
        Initialize live strategy executor
        
        Args:
            symbol: Trading symbol (MES, MNQ, etc.)
            contract_id: Contract ID (auto-discovered if None)
            dry_run: If True, don't place real orders
            strategies: List of strategies to run ['optimal_stopping', 'multi_timeframe', 'liquidity_provision']
        """
        self.symbol = symbol
        self.contract_id = contract_id
        self.dry_run = dry_run
        
        # Initialize API client
        self.api_client = TopstepXClient(
            username=os.getenv('TOPSTEPX_USERNAME'),
            api_key=os.getenv('TOPSTEPX_API_KEY'),
            dry_run=dry_run
        )
        
        # Authenticate
        if not self.api_client.authenticate():
            raise Exception("Failed to authenticate with TopstepX")
        
        # Get account
        accounts = self.api_client.get_accounts()
        if not accounts:
            raise Exception("No accounts found")
        
        # Log all available accounts
        logger.info("="*80)
        logger.info("AVAILABLE ACCOUNTS")
        logger.info("="*80)
        for i, acc in enumerate(accounts):
            acc_id = acc.get('id') or acc.get('accountId')
            acc_name = acc.get('name', 'Unknown')
            is_sim = acc.get('simulated', acc.get('isSimulated', False))
            acc_type = acc.get('type', 'Unknown')
            acc_status = acc.get('status', 'Unknown')
            balance = acc.get('balance', acc.get('accountBalance', 0))
            equity = acc.get('equity', acc.get('accountEquity', 0))
            
            logger.info(f"  [{i+1}] {acc_name}")
            logger.info(f"      ID: {acc_id}")
            logger.info(f"      Type: {acc_type}")
            logger.info(f"      Simulated/Practice: {is_sim}")
            logger.info(f"      Status: {acc_status}")
            if balance:
                logger.info(f"      Balance: ${balance:,.2f}")
            if equity:
                logger.info(f"      Equity: ${equity:,.2f}")
            logger.info("")
        
        # Select account (prefer practice/simulated accounts)
        selected_account = None
        for acc in accounts:
            is_sim = acc.get('simulated', acc.get('isSimulated', False))
            if is_sim:
                selected_account = acc
                logger.info(f"✅ Selected PRACTICE account: {acc.get('name', 'Unknown')} (ID: {acc.get('id')})")
                break
        
        # If no practice account found, use first one
        if not selected_account:
            selected_account = accounts[0]
            logger.warning(f"⚠️  No practice account found, using first account: {selected_account.get('name', 'Unknown')} (ID: {selected_account.get('id')})")
        
        self.account_id = selected_account.get('id') or selected_account.get('accountId')
        self.api_client.account_id = self.account_id
        
        # Print selected account details
        logger.info("="*80)
        logger.info("SELECTED ACCOUNT DETAILS")
        logger.info("="*80)
        logger.info(f"Account ID: {self.account_id}")
        logger.info(f"Account Name: {selected_account.get('name', 'N/A')}")
        logger.info(f"Account Type: {selected_account.get('type', 'N/A')}")
        logger.info(f"Simulated/Practice: {selected_account.get('simulated', selected_account.get('isSimulated', 'N/A'))}")
        logger.info(f"Account Status: {selected_account.get('status', 'N/A')}")
        
        # Get balance from account info or API
        balance = selected_account.get('balance', selected_account.get('accountBalance'))
        equity = selected_account.get('equity', selected_account.get('accountEquity'))
        available_margin = selected_account.get('availableMargin', selected_account.get('availableMargin'))
        
        if balance is not None:
            logger.info(f"Balance: ${balance:,.2f}")
        if equity is not None:
            logger.info(f"Equity: ${equity:,.2f}")
        if available_margin is not None:
            logger.info(f"Available Margin: ${available_margin:,.2f}")
        
        # Try to get balance from API
        try:
            balance_info = self.api_client.get_account_balance(self.account_id)
            if balance_info:
                if 'balance' in balance_info:
                    logger.info(f"API Balance: ${balance_info.get('balance', 0):,.2f}")
                if 'equity' in balance_info:
                    logger.info(f"API Equity: ${balance_info.get('equity', 0):,.2f}")
        except Exception as e:
            logger.debug(f"Could not fetch balance from API: {e}")
        
        logger.info("="*80)
        
        # Find contract if not provided
        if not self.contract_id:
            self.contract_id = self._find_contract()
            if not self.contract_id:
                raise Exception(f"Could not find contract for {symbol}")
        
        logger.info(f"Using contract: {self.contract_id} for {symbol}")
        
        # Data stores (we manage data in memory for live trading)
        # self.timeseries_store = TimeseriesStore()  # Not needed for live trading
        self.market_data_adapter = MarketDataAdapter()
        
        # Strategy configurations (OPTIMIZED)
        self.strategy_configs = {
            'optimal_stopping': {
                'lookback_window': 100,
                'min_opportunities_seen': 20,  # REDUCED from 37 - allow earlier entries
                'score_threshold': 0.5,  # REDUCED from 0.7 - more reasonable threshold
                'momentum_weight': 0.4,
                'mean_reversion_weight': 0.3,
                'volatility_weight': 0.3,
                'atr_period': 14,
                'atr_multiplier_stop': 0.75,
                'atr_multiplier_target': 1.25,
                'risk_per_trade': 100.0,
                'max_hold_bars': 40,
                'signal_reversal_min_profit_pct': 0.01,
                'use_trailing_stop': True,
                'trailing_stop_atr_multiplier': 0.5,
                'trailing_stop_activation_pct': 0.001,
                'use_partial_profit': True,
                'partial_profit_pct': 0.5,
                'partial_profit_target_atr': 0.75
            },
            'multi_timeframe': {
                'convergence_threshold': 0.5,  # INCREASED from 0.2 - require stronger signals
                'divergence_threshold': 0.2,
                'lookback_period': 20,
                'momentum_period': 10,
                'mean_reversion_period': 20,
                'atr_period': 14,
                'atr_multiplier_stop': 1.0,
                'atr_multiplier_target': 1.5,
                'risk_per_trade': 100.0,
                'max_hold_bars': 30,
                'timeframe_weights': {'1m': 0.25, '5m': 0.35, '15m': 0.40},
                'min_confidence': 0.5,  # NEW: Minimum confidence required (50%)
                'use_trailing_stop': True,
                'trailing_stop_atr_multiplier': 0.5,
                'trailing_stop_activation_pct': 0.001,
                'use_partial_profit': True,
                'partial_profit_pct': 0.5,
                'partial_profit_target_atr': 0.75
            },
            'liquidity_provision': {
                'imbalance_lookback': 3,
                'imbalance_threshold': 0.06,  # REDUCED from 0.08 - slightly more sensitive
                'adverse_selection_threshold': 0.65,  # INCREASED from 0.55 - allow more trades
                'favorable_fill_threshold': 0.45,  # REDUCED from 0.55 - more permissive
                'spread_target_ticks': 4,
                'max_spread_ticks': 5,
                'atr_period': 14,
                'atr_multiplier_stop': 1.25,
                'risk_per_trade': 100.0,
                'max_hold_bars': 15,
                'cancel_on_reversal': True,
                'use_trailing_stop': True,
                'trailing_stop_atr_multiplier': 0.5,
                'trailing_stop_activation_pct': 0.001,
                'use_partial_profit': True,
                'partial_profit_pct': 0.5,
                'partial_profit_target_atr': 0.75,
                'confidence_scaling': True,
                'max_position_size': 5
            }
        }
        
        # Initialize strategies
        self.strategies = {}
        self.active_strategies = strategies or ['optimal_stopping', 'multi_timeframe', 'liquidity_provision']
        
        for strategy_name in self.active_strategies:
            if strategy_name == 'optimal_stopping':
                self.strategies[strategy_name] = OptimalStoppingStrategy(
                    self.strategy_configs[strategy_name]
                )
            elif strategy_name == 'multi_timeframe':
                # Multi-timeframe needs special setup
                self.strategies[strategy_name] = MultiTimeframeStrategy(
                    self.strategy_configs[strategy_name]
                )
            elif strategy_name == 'liquidity_provision':
                self.strategies[strategy_name] = LiquidityProvisionStrategy(
                    self.strategy_configs[strategy_name]
                )
        
        logger.info(f"Initialized strategies: {list(self.strategies.keys())}")
        
        # Position tracking
        self.positions: Dict[str, LivePosition] = {}  # strategy_name -> position
        self.historical_data: Dict[str, pd.DataFrame] = {}  # strategy_name -> dataframe
        
        # Cooldown tracking (prevent rapid re-entry after exit)
        self.exit_cooldowns: Dict[str, datetime] = {}  # strategy_name -> last exit time
        self.cooldown_seconds = 60  # Wait 60 seconds after exit before allowing new entry
        
        # Threading
        self.running = False
        self.data_lock = threading.Lock()
        
        # Market data
        self.current_price: Optional[float] = None
        self.current_bid: Optional[float] = None
        self.current_ask: Optional[float] = None
        self.latest_bar: Optional[Candle] = None
    
    def _find_contract(self) -> Optional[str]:
        """Find contract ID for symbol"""
        contracts = self.api_client.search_contracts(self.symbol)
        if contracts:
            # Get most recent/liquid contract
            return contracts[0].get('id')
        return None
    
    def start(self):
        """Start live trading"""
        logger.info("="*80)
        logger.info("STARTING LIVE TRADING")
        logger.info("="*80)
        logger.info(f"Symbol: {self.symbol}")
        logger.info(f"Contract ID: {self.contract_id}")
        logger.info(f"Strategies: {list(self.strategies.keys())}")
        logger.info(f"Dry Run: {self.dry_run}")
        logger.info("="*80)
        
        self.running = True
        
        # Connect to real-time data
        self.api_client.connect_realtime(
            account_id=self.account_id,
            contract_ids=[self.contract_id]
        )
        
        # Setup callbacks
        self._setup_callbacks()
        
        # Load initial historical data
        self._load_historical_data()
        
        # Do an immediate update to ensure we have fresh data
        logger.info("Performing initial data update...")
        self._update_historical_data()
        
        # Log data status
        for strategy_name in self.active_strategies:
            historical = self.historical_data.get(strategy_name)
            if historical is not None:
                logger.info(f"[{strategy_name}] Historical data loaded: {len(historical)} bars")
            else:
                logger.warning(f"[{strategy_name}] No historical data loaded!")
        
        # Start main loop
        self._main_loop()
    
    def _setup_callbacks(self):
        """Setup real-time data callbacks"""
        def on_quote_update(data):
            """Handle quote updates"""
            try:
                if isinstance(data, list) and len(data) > 0:
                    data = data[0]
                
                if not isinstance(data, dict):
                    return
                
                contract_id = data.get('contractId', '')
                if contract_id != self.contract_id:
                    return
                
                with self.data_lock:
                    self.current_price = data.get('lastPrice')
                    self.current_bid = data.get('bestBid')
                    self.current_ask = data.get('bestAsk')
                    
                    # Update latest bar if we have OHLC data
                    if 'open' in data and 'high' in data and 'low' in data and 'close' in data:
                        self.latest_bar = Candle(
                            symbol=self.symbol,
                            open=data.get('open'),
                            high=data.get('high'),
                            low=data.get('low'),
                            close=data.get('close'),
                            volume=data.get('volume', 0),
                            timestamp=datetime.now(timezone.utc),
                            interval="1m"  # Default to 1m
                        )
            
            except Exception as e:
                logger.error(f"Error in quote callback: {e}")
        
        self.api_client.register_realtime_callback("on_quote_update", on_quote_update)
        logger.info("Registered real-time callbacks")
    
    def _load_historical_data(self):
        """Load historical data for each strategy"""
        logger.info("Loading historical data...")
        
        # Get bars for different intervals
        intervals = {
            'optimal_stopping': '15m',
            'multi_timeframe': '15m',  # Primary interval
            'liquidity_provision': '5m'
        }
        
        for strategy_name in self.active_strategies:
            interval = intervals.get(strategy_name, '15m')
            
            # Fetch last 200 bars
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=7)  # Get last week
            
            bars = self.api_client.get_bars(
                contract_id=self.contract_id,
                interval=interval,
                start_time=start_time,
                end_time=end_time,
                limit=200
            )
            
            if bars:
                # Normalize bars using MarketDataAdapter (handles t/o/h/l/c/v format)
                candles = self.market_data_adapter.normalize_bars(bars, self.symbol, interval)
                
                # Convert to DataFrame with standard column names
                df = pd.DataFrame([{
                    'timestamp': c.timestamp,
                    'open': c.open,
                    'high': c.high,
                    'low': c.low,
                    'close': c.close,
                    'volume': c.volume
                } for c in candles])
                
                df = df.sort_values('timestamp').reset_index(drop=True)
                
                self.historical_data[strategy_name] = df
                logger.info(f"Loaded {len(df)} bars for {strategy_name} ({interval})")
            else:
                logger.warning(f"No historical data for {strategy_name}")
                self.historical_data[strategy_name] = pd.DataFrame()
        
        # For multi-timeframe, load all three intervals
        if 'multi_timeframe' in self.active_strategies:
            strategy = self.strategies['multi_timeframe']
            df_1m = None
            df_5m = None
            df_15m = None
            
            for tf in ['1m', '5m', '15m']:
                bars = self.api_client.get_bars(
                    contract_id=self.contract_id,
                    interval=tf,
                    start_time=start_time,
                    end_time=end_time,
                    limit=200
                )
                if bars:
                    # Normalize bars using MarketDataAdapter
                    candles = self.market_data_adapter.normalize_bars(bars, self.symbol, tf)
                    
                    # Convert to DataFrame with standard column names
                    df = pd.DataFrame([{
                        'timestamp': c.timestamp,
                        'open': c.open,
                        'high': c.high,
                        'low': c.low,
                        'close': c.close,
                        'volume': c.volume
                    } for c in candles])
                    
                    df = df.sort_values('timestamp').reset_index(drop=True)
                    
                    if tf == '1m':
                        df_1m = df
                    elif tf == '5m':
                        df_5m = df
                    elif tf == '15m':
                        df_15m = df
            
            # Set all timeframes at once
            if df_1m is not None and df_5m is not None and df_15m is not None:
                strategy.set_timeframe_data(df_1m=df_1m, df_5m=df_5m, df_15m=df_15m)
                logger.info("Multi-timeframe data loaded: 1m, 5m, 15m")
    
    def _main_loop(self):
        """Main trading loop"""
        logger.info("Starting main trading loop...")
        
        # Track last update time
        last_data_update = datetime.now()
        data_update_interval = timedelta(minutes=5)  # Update every 5 minutes
        
        while self.running:
            try:
                # Update historical data periodically
                if datetime.now() - last_data_update >= data_update_interval:
                    self._update_historical_data()
                    last_data_update = datetime.now()
                
                # Poll for latest bar if we don't have one (fallback)
                if self.latest_bar is None or self.current_price is None:
                    try:
                        bars = self.api_client.get_bars(
                            contract_id=self.contract_id,
                            interval="1m",
                            limit=1
                        )
                        if bars:
                            candles = self.market_data_adapter.normalize_bars(bars, self.symbol, "1m")
                            if candles:
                                self.latest_bar = candles[-1]
                                self.current_price = candles[-1].close
                                logger.debug(f"Polled latest price: {self.current_price:.2f}")
                    except Exception as e:
                        logger.debug(f"Error polling price: {e}")
                
                # Process each strategy
                for strategy_name, strategy in self.strategies.items():
                    if strategy_name not in self.active_strategies:
                        continue
                    
                    # Check if we have a position
                    has_position = strategy_name in self.positions
                    
                    if has_position:
                        # Check exit conditions
                        self._check_exit(strategy_name, strategy)
                    else:
                        # Check for entry signals
                        self._check_entry(strategy_name, strategy)
                
                # Sleep before next iteration
                time.sleep(1)  # Check every second
            
            except KeyboardInterrupt:
                logger.info("Received interrupt, shutting down...")
                self.stop()
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(5)  # Wait before retrying
    
    def _update_historical_data(self):
        """Update historical data with latest bars"""
        logger.debug("Updating historical data...")
        
        # Intervals for each strategy
        intervals = {
            'optimal_stopping': '15m',
            'multi_timeframe': '15m',  # Primary interval
            'liquidity_provision': '5m'
        }
        
        for strategy_name in self.active_strategies:
            interval = intervals.get(strategy_name, '15m')
            
            try:
                # Fetch latest bars (get last 50 bars to ensure we have recent data)
                bars = self.api_client.get_bars(
                    contract_id=self.contract_id,
                    interval=interval,
                    limit=50
                )
                
                if bars:
                    # Normalize bars using MarketDataAdapter
                    candles = self.market_data_adapter.normalize_bars(bars, self.symbol, interval)
                    
                    # Convert to DataFrame
                    df = pd.DataFrame([{
                        'timestamp': c.timestamp,
                        'open': c.open,
                        'high': c.high,
                        'low': c.low,
                        'close': c.close,
                        'volume': c.volume
                    } for c in candles])
                    
                    df = df.sort_values('timestamp').reset_index(drop=True)
                    
                    # Update historical data
                    self.historical_data[strategy_name] = df
                    logger.debug(f"Updated {strategy_name} data: {len(df)} bars ({interval})")
                    
                    # Update latest bar if this is the primary interval
                    if interval == '15m' and len(candles) > 0:
                        self.latest_bar = candles[-1]
                        if self.current_price is None:
                            self.current_price = candles[-1].close
                else:
                    logger.warning(f"No bars returned for {strategy_name} ({interval})")
            
            except Exception as e:
                logger.error(f"Error updating historical data for {strategy_name}: {e}")
        
        # For multi-timeframe, update all three intervals
        if 'multi_timeframe' in self.active_strategies:
            strategy = self.strategies['multi_timeframe']
            df_1m = None
            df_5m = None
            df_15m = None
            
            for tf in ['1m', '5m', '15m']:
                try:
                    bars = self.api_client.get_bars(
                        contract_id=self.contract_id,
                        interval=tf,
                        limit=50
                    )
                    if bars:
                        candles = self.market_data_adapter.normalize_bars(bars, self.symbol, tf)
                        df = pd.DataFrame([{
                            'timestamp': c.timestamp,
                            'open': c.open,
                            'high': c.high,
                            'low': c.low,
                            'close': c.close,
                            'volume': c.volume
                        } for c in candles])
                        df = df.sort_values('timestamp').reset_index(drop=True)
                        
                        if tf == '1m':
                            df_1m = df
                        elif tf == '5m':
                            df_5m = df
                        elif tf == '15m':
                            df_15m = df
                except Exception as e:
                    logger.error(f"Error updating {tf} data for multi-timeframe: {e}")
            
            # Update multi-timeframe strategy with all timeframes
            if df_1m is not None and df_5m is not None and df_15m is not None:
                strategy.set_timeframe_data(df_1m=df_1m, df_5m=df_5m, df_15m=df_15m)
                logger.debug("Updated multi-timeframe data: 1m, 5m, 15m")
    
    def _check_entry(self, strategy_name: str, strategy):
        """Check for entry signals"""
        if strategy_name in self.positions:
            return  # Already in position
        
        # Check cooldown period after exit
        if strategy_name in self.exit_cooldowns:
            last_exit = self.exit_cooldowns[strategy_name]
            time_since_exit = (datetime.now(timezone.utc) - last_exit).total_seconds()
            if time_since_exit < self.cooldown_seconds:
                # Still in cooldown, don't enter
                if datetime.now().second % 30 == 0:  # Log occasionally
                    logger.debug(f"[{strategy_name}] In cooldown: {int(self.cooldown_seconds - time_since_exit)}s remaining")
                return
        
        if self.current_price is None or self.latest_bar is None:
            return  # No market data
        
        # Get historical data
        historical = self.historical_data.get(strategy_name)
        if historical is None or len(historical) == 0:
            logger.debug(f"[{strategy_name}] No historical data available")
            return
        
        # Check if we have enough data (strategies need minimum bars)
        min_bars_required = {
            'optimal_stopping': 100,
            'multi_timeframe': 20,
            'liquidity_provision': 20
        }
        min_bars = min_bars_required.get(strategy_name, 20)
        if len(historical) < min_bars:
            logger.debug(f"[{strategy_name}] Insufficient data: {len(historical)} < {min_bars} bars")
            return
        
        # Create MarketData object
        market_data = MarketData(
            timestamp=datetime.now(timezone.utc),
            symbol=self.symbol,
            open=self.latest_bar.open if self.latest_bar else self.current_price,
            high=self.latest_bar.high if self.latest_bar else self.current_price,
            low=self.latest_bar.low if self.latest_bar else self.current_price,
            close=self.current_price,
            volume=self.latest_bar.volume if self.latest_bar else 0,
            bid=self.current_bid,
            ask=self.current_ask
        )
        
        # Generate signal
        try:
            signal = strategy.generate_signal(
                market_data=market_data,
                historical_data=historical,
                current_position=None
            )
            
            if signal:
                # Additional confidence check (double-check)
                min_confidence_required = self.strategy_configs.get(strategy_name, {}).get('min_confidence', 0.0)
                if signal.confidence < min_confidence_required:
                    logger.debug(f"[{strategy_name}] Signal rejected: confidence {signal.confidence:.2f} < {min_confidence_required:.2f}")
                    return
                
                logger.info(f"[{strategy_name}] ✅ Signal generated: {signal.direction} @ {signal.entry_price:.2f} (confidence: {signal.confidence:.2f}, signal_strength: {signal.metadata.get('normalized_signal', 0):.2f})")
                self._execute_entry(strategy_name, signal)
            else:
                # Log diagnostic info for why signal wasn't generated (only occasionally to avoid spam)
                if datetime.now().second % 30 == 0:  # Every 30 seconds
                    # Add strategy-specific diagnostics
                    if strategy_name == 'optimal_stopping':
                        # Log opportunity count and score info
                        if hasattr(strategy, 'opportunities_seen') and hasattr(strategy, 'opportunity_scores'):
                            opp_count = strategy.opportunities_seen
                            min_opp = self.strategy_configs.get(strategy_name, {}).get('min_opportunities_seen', 0)
                            score_thresh = self.strategy_configs.get(strategy_name, {}).get('score_threshold', 0.0)
                            if len(strategy.opportunity_scores) > 0:
                                last_score = strategy.opportunity_scores[-1] if strategy.opportunity_scores else 0
                                logger.debug(f"[{strategy_name}] No signal: opp_seen={opp_count}/{min_opp}, last_score={abs(last_score):.3f}/{score_thresh:.3f}")
                            else:
                                logger.debug(f"[{strategy_name}] No signal: opp_seen={opp_count}/{min_opp}, no scores yet")
                    elif strategy_name == 'liquidity_provision':
                        # Log imbalance and threshold info
                        if len(historical) >= 3:
                            try:
                                from strategies.liquidity_provision import estimate_order_flow_imbalance
                                from strategies.liquidity_provision import calculate_atr as calc_atr_lp
                                imbalance = estimate_order_flow_imbalance(historical, lookback=3)
                                atr = calc_atr_lp(historical, period=14)
                                imbalance_thresh = self.strategy_configs.get(strategy_name, {}).get('imbalance_threshold', 0.0)
                                logger.debug(f"[{strategy_name}] No signal: imbalance={imbalance:.3f} (need |{imbalance_thresh:.3f}|), atr={atr:.2f if atr else 'None'}")
                            except Exception as e:
                                logger.debug(f"[{strategy_name}] Diagnostic error: {e}")
                    else:
                        logger.debug(f"[{strategy_name}] No signal generated (checking conditions...)")
        except Exception as e:
            logger.error(f"[{strategy_name}] Error generating signal: {e}", exc_info=True)
    
    def _execute_entry(self, strategy_name: str, signal: Signal):
        """Execute entry order"""
        try:
            # Get strategy object
            strategy = self.strategies.get(strategy_name)
            if strategy is None:
                logger.error(f"[{strategy_name}] Strategy not found")
                return
            
            # Calculate position size
            historical = self.historical_data.get(strategy_name)
            if historical is None:
                logger.warning(f"[{strategy_name}] No historical data for position sizing")
                return
            
            market_data = MarketData(
                timestamp=datetime.now(timezone.utc),
                symbol=self.symbol,
                open=self.latest_bar.open if self.latest_bar else self.current_price,
                high=self.latest_bar.high if self.latest_bar else self.current_price,
                low=self.latest_bar.low if self.latest_bar else self.current_price,
                close=self.current_price,
                volume=self.latest_bar.volume if self.latest_bar else 0,
                bid=self.current_bid,
                ask=self.current_ask
            )
            
            # Get account balance for position sizing
            balance = self.api_client.get_account_balance(self.account_id)
            account_equity = balance.get('balance', 50000.0) if balance else 50000.0
            
            contracts = strategy.calculate_position_size(
                signal=signal,
                account_equity=account_equity,
                market_data=market_data,
                historical_data=historical
            )
            
            if contracts <= 0:
                logger.warning(f"[{strategy_name}] Invalid position size: {contracts}")
                return
            
            # Determine order type
            order_type = "Limit" if signal.metadata.get('order_type') == 'LIMIT' else "Market"
            side = "Buy" if signal.direction == "LONG" else "Sell"
            price = signal.entry_price if order_type == "Limit" else None
            
            # Place order
            logger.info(f"[{strategy_name}] Placing {order_type} order: {side} {contracts} contracts @ {price or 'MARKET'}")
            
            try:
                order_result = self.api_client.place_order(
                    contract_id=self.contract_id,
                    side=side,
                    quantity=contracts,
                    order_type=order_type,
                    price=price
                )
                
                if order_result:
                    # Check for success in various formats (API might return different structures)
                    success = order_result.get('success', True)  # Default to True if not specified
                    order_id = order_result.get('orderId') or order_result.get('id') or order_result.get('order_id')
                    
                    if success and order_id:
                        logger.info(f"[{strategy_name}] ✅ Order placed successfully: {side} {contracts} @ {price or 'MARKET'} (Order ID: {order_id})")
                        
                        # Track position
                        self.positions[strategy_name] = LivePosition(
                            strategy_name=strategy_name,
                            entry_time=datetime.now(timezone.utc),
                            entry_price=signal.entry_price,
                            contracts=contracts,
                            is_long=signal.direction == "LONG",
                            stop_loss=signal.stop_loss,
                            take_profit=signal.take_profit,
                            order_id=str(order_id),
                            signal=signal
                        )
                    else:
                        logger.error(f"[{strategy_name}] ❌ Order placement failed: success={success}, order_id={order_id}, result={order_result}")
                else:
                    logger.error(f"[{strategy_name}] ❌ Order placement returned None/empty result")
            except Exception as e:
                logger.error(f"[{strategy_name}] ❌ Exception placing order: {e}", exc_info=True)
        
        except Exception as e:
            logger.error(f"[{strategy_name}] Error executing entry: {e}", exc_info=True)
    
    def _check_exit(self, strategy_name: str, strategy):
        """Check exit conditions"""
        if strategy_name not in self.positions:
            return
        
        position = self.positions[strategy_name]
        
        if self.current_price is None or self.latest_bar is None:
            return
        
        # Create MarketData
        market_data = MarketData(
            timestamp=datetime.now(timezone.utc),
            symbol=self.symbol,
            open=self.latest_bar.open,
            high=self.latest_bar.high,
            low=self.latest_bar.low,
            close=self.current_price,
            volume=self.latest_bar.volume,
            bid=self.current_bid,
            ask=self.current_ask
        )
        
        # Check exit
        historical = self.historical_data.get(strategy_name)
        if historical is None:
            return
        
        # Check minimum hold time (prevent immediate exits)
        time_in_position = (datetime.now(timezone.utc) - position.entry_time).total_seconds()
        min_hold_seconds = 10  # Minimum 10 seconds before allowing exit (except stop loss)
        
        # Create a position object for the strategy
        from framework.backtest_engine import Position
        strategy_position = Position(
            entry_time=position.entry_time,
            entry_price=position.entry_price,
            contracts=position.contracts,
            is_long=position.is_long,
            stop_loss=position.stop_loss,
            take_profit=position.take_profit,
            signal=position.signal,
            entry_bar_index=len(historical) - 1  # Approximate
        )
        
        # Only check exit if we've held for minimum time (except for stop loss)
        if time_in_position < min_hold_seconds:
            # Still check stop loss immediately (safety)
            if position.stop_loss:
                if position.is_long and market_data.low <= position.stop_loss:
                    exit_reason = ExitReason(
                        reason="STOP_LOSS",
                        timestamp=market_data.timestamp,
                        metadata={'exit_price': position.stop_loss}
                    )
                    logger.info(f"[{strategy_name}] Exit signal: {exit_reason.reason}")
                    self._execute_exit(strategy_name, exit_reason)
                    return
                elif not position.is_long and market_data.high >= position.stop_loss:
                    exit_reason = ExitReason(
                        reason="STOP_LOSS",
                        timestamp=market_data.timestamp,
                        metadata={'exit_price': position.stop_loss}
                    )
                    logger.info(f"[{strategy_name}] Exit signal: {exit_reason.reason}")
                    self._execute_exit(strategy_name, exit_reason)
                    return
            
            # Don't check other exit conditions yet
            return
        
        exit_reason = strategy.check_exit(
            position=strategy_position,
            market_data=market_data,
            historical_data=historical
        )
        
        if exit_reason:
            logger.info(f"[{strategy_name}] Exit signal: {exit_reason.reason}")
            self._execute_exit(strategy_name, exit_reason)
    
    def _execute_exit(self, strategy_name: str, exit_reason):
        """Execute exit order"""
        if strategy_name not in self.positions:
            return
        
        position = self.positions[strategy_name]
        
        try:
            # Place market order to exit
            side = "Sell" if position.is_long else "Buy"
            
            order_result = self.api_client.place_order(
                contract_id=self.contract_id,
                side=side,
                quantity=position.contracts,
                order_type="Market"
            )
            
            if order_result and order_result.get('success'):
                logger.info(f"[{strategy_name}] Exit order placed: {side} {position.contracts} @ MARKET")
                
                # Calculate P&L
                exit_price = self.current_price
                if position.is_long:
                    pnl = (exit_price - position.entry_price) * position.contracts * 5.0  # MES tick value
                else:
                    pnl = (position.entry_price - exit_price) * position.contracts * 5.0
                
                logger.info(f"[{strategy_name}] Position closed: P&L = ${pnl:.2f}")
                
                # Remove position
                del self.positions[strategy_name]
                
                # Set cooldown period (prevent immediate re-entry)
                self.exit_cooldowns[strategy_name] = datetime.now(timezone.utc)
                logger.info(f"[{strategy_name}] Cooldown activated: {self.cooldown_seconds}s before next entry allowed")
            else:
                logger.error(f"[{strategy_name}] Failed to place exit order")
        
        except Exception as e:
            logger.error(f"[{strategy_name}] Error executing exit: {e}", exc_info=True)
    
    def stop(self):
        """Stop live trading"""
        logger.info("Stopping live trading...")
        self.running = False
        
        # Close all positions
        for strategy_name in list(self.positions.keys()):
            logger.info(f"Closing position for {strategy_name}...")
            # Place exit orders
            # (Implementation would close positions)
        
        # Disconnect
        if self.api_client:
            # Disconnect SignalR if connected
            pass
        
        logger.info("Live trading stopped")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Live Trading Executor')
    parser.add_argument('--symbol', default='MES', help='Trading symbol')
    parser.add_argument('--contract-id', default=None, help='Contract ID (auto-discover if not provided)')
    parser.add_argument('--dry-run', action='store_true', default=True, help='Dry run mode (default: True)')
    parser.add_argument('--live', action='store_true', help='Live trading mode (overrides dry-run)')
    parser.add_argument('--strategies', nargs='+', 
                       choices=['optimal_stopping', 'multi_timeframe', 'liquidity_provision'],
                       default=['optimal_stopping', 'multi_timeframe', 'liquidity_provision'],
                       help='Strategies to run')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('live_trading.log'),
            logging.StreamHandler()
        ]
    )
    
    dry_run = not args.live if args.live else args.dry_run
    
    if not dry_run:
        logger.warning("="*80)
        logger.warning("LIVE TRADING MODE - REAL ORDERS WILL BE PLACED")
        logger.warning("="*80)
        response = input("Are you sure you want to continue? (yes/no): ")
        if response.lower() != 'yes':
            logger.info("Aborted by user")
            return
    
    # Create executor
    executor = LiveStrategyExecutor(
        symbol=args.symbol,
        contract_id=args.contract_id,
        dry_run=dry_run,
        strategies=args.strategies
    )
    
    # Start trading
    try:
        executor.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        executor.stop()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        executor.stop()


if __name__ == "__main__":
    main()

