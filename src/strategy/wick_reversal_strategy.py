"""
Wick Reversal Strategy - Production Implementation
Trades reversals after "fat" hourly wicks with pullback confirmation
"""
import os
import sys
import time
import logging
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass
from threading import Lock
from collections import deque

# Get absolute paths
current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(os.path.dirname(current_file))  # src/
project_root = os.path.dirname(src_dir)  # project root

# Add project root to path first
sys.path.insert(0, project_root)
# Add src to path
sys.path.insert(0, src_dir)

from connectors.topstepx_client import TopstepXClient
from connectors.market_data_adapter import MarketDataAdapter
from data.timeseries_store import TimeseriesStore
from data.trade_history import TradeHistory
from data.state_persistence import StatePersistence
from strategy.position_manager import PositionManager
from strategy.risk_manager import RiskManager
from execution.order_client import OrderClient

logger = logging.getLogger(__name__)


@dataclass
class FatWick:
    """Represents a detected fat wick on hourly timeframe"""
    timestamp: datetime
    high: float
    low: float
    open: float
    close: float
    upper_wick: float  # Distance from high to body
    lower_wick: float  # Distance from body to low
    total_wick: float  # Sum of upper + lower wick
    wick_percentage: float  # Total wick / range * 100
    wick_direction: str  # "UPPER" (long upper wick), "LOWER" (long lower wick), "BOTH"
    wick_midpoint: float  # 50% retracement level
    wick_extreme: float  # The extreme price (high for upper wick, low for lower wick)
    is_active: bool = True  # Whether we're still waiting for pullback


@dataclass
class StructureBreak:
    """Represents a structure break on short timeframe"""
    timestamp: datetime
    price: float
    direction: str  # "BULLISH" or "BEARISH"
    confirmation: bool  # Whether structure break is confirmed


@dataclass
class WickReversalPosition:
    """Represents an active wick reversal position"""
    entry_time: datetime
    entry_price: float
    direction: str  # "LONG" or "SHORT"
    stop_loss: float
    take_profit: float
    contracts: int
    wick: FatWick  # The fat wick that triggered this trade
    order_id: Optional[str] = None
    risk_amount: float = 0.0  # Dollar amount at risk
    stop_order_id: Optional[str] = None  # Stop loss order ID
    contracts_exited: int = 0  # Number of contracts already exited (for partial profit taking)
    highest_profit: float = 0.0  # Highest unrealized profit reached (for trailing stop)


class WickReversalStrategy:
    """Main Wick Reversal Strategy implementation"""
    
    def __init__(self, config_path: str = "wick_reversal_config.yaml"):
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
        
        # Strategy components - support multiple instruments
        instruments_config = self.config.get('instruments', None)
        if instruments_config:
            # Support both list and single value for backward compatibility
            if isinstance(instruments_config, list):
                self.instruments = instruments_config
            else:
                self.instruments = [instruments_config]
        else:
            # Fallback to single instrument for backward compatibility
            single_instrument = self.config.get('instrument', 'MES')
            self.instruments = [single_instrument] if single_instrument else ['MES']
        
        # Get tick values and tick sizes for all instruments
        tick_values_map = {
            'MES': 1.25,  # $1.25 per tick (0.25 tick size)
            'MNQ': 0.50,  # $0.50 per tick (0.25 tick size)
            'MGC': 1.0,   # $1.00 per tick (0.10 tick size)
            'GC': 10.0,   # $10.00 per tick (0.10 tick size)
            'MYM': 0.50,  # $0.50 per tick (1.0 tick size) - Micro Dow
            'M2K': 0.50,  # $0.50 per tick (0.10 tick size) - Micro Russell 2000
            'M6E': 1.25,  # $1.25 per tick (0.0001 tick size) - Micro Euro
            'M6B': 0.625, # $0.625 per tick (0.0001 tick size) - Micro British Pound
        }
        tick_sizes_map = {
            'MES': 0.25,
            'MNQ': 0.25,
            'MGC': 0.10,
            'GC': 0.10,
            'MYM': 1.0,   # Micro Dow
            'M2K': 0.10,  # Micro Russell 2000
            'M6E': 0.0001, # Micro Euro (very small tick)
            'M6B': 0.0001, # Micro British Pound (very small tick)
        }
        tick_values = {symbol: tick_values_map.get(symbol, 1.25) for symbol in self.instruments}
        self.tick_sizes = {symbol: tick_sizes_map.get(symbol, 0.25) for symbol in self.instruments}
        
        # Position and risk management
        self.position_manager = PositionManager(
            max_net_notional=self.config.get('max_net_notional', 10000.0),
            tick_values=tick_values
        )
        
        self.risk_manager = RiskManager(
            max_daily_loss=self.config.get('max_daily_loss', 900.0),
            trailing_drawdown_limit=self.config.get('trailing_drawdown_limit', 1800.0),
            max_net_notional=self.config.get('max_net_notional', 10000.0)
        )
        
        # Strategy parameters
        self.wick_threshold = self.config.get('wick_threshold', 0.5)  # Min wick % of range
        self.pullback_tolerance = self.config.get('pullback_tolerance', 0.05)  # 5% tolerance (can be absolute or relative)
        self.pullback_tolerance_absolute = self.config.get('pullback_tolerance_absolute', None)  # Absolute tolerance in price points
        self.pullback_tolerance_wick_relative = self.config.get('pullback_tolerance_wick_relative', False)  # Use wick size instead of price
        self.stop_fraction = self.config.get('stop_fraction', 0.5)  # Stop beyond extreme by 50% of wick
        self.target_multiple = self.config.get('target_multiple', 1.0)  # 1:1 R:R
        self.risk_per_trade = self.config.get('risk_per_trade', 100.0)  # $100 risk per trade
        self.max_wick_age_hours = self.config.get('max_wick_age_hours', 24.0)  # Expire wicks after 24h
        self.max_position_hours = self.config.get('max_position_hours', 48.0)  # Max time to hold position
        self.partial_profit_enabled = self.config.get('partial_profit_enabled', True)  # Enable partial profit taking
        self.partial_profit_ratio = self.config.get('partial_profit_ratio', 0.5)  # Take 50% profit at target
        self.trailing_stop_enabled = self.config.get('trailing_stop_enabled', False)  # Enable trailing stop
        self.trailing_stop_distance = self.config.get('trailing_stop_distance', 0.3)  # Trailing stop distance (fraction of stop distance)
        
        # Trend filter parameters
        self.trend_filter_enabled = self.config.get('trend_filter_enabled', True)
        self.trend_ema_period = self.config.get('trend_ema_period', 50)  # EMA period for trend filter
        
        # Volume confirmation
        self.volume_confirmation_enabled = self.config.get('volume_confirmation_enabled', True)
        self.volume_multiplier = self.config.get('volume_multiplier', 1.2)  # Volume must be 1.2x average
        
        # Structure detection improvements
        self.structure_min_move = self.config.get('structure_min_move', 0.5)  # Minimum move size for structure (in price points)
        self.structure_swing_lookback = self.config.get('structure_swing_lookback', 5)  # Bars to look back for swing highs/lows
        self.entry_min_conditions = int(self.config.get('entry_min_conditions', 2))  # Allow 2-of-3 gates by default
        
        # Timeframes
        self.long_timeframe = self.config.get('long_timeframe', '1h')  # For fat wick detection
        self.short_timeframe = self.config.get('short_timeframe', '1m')  # For structure breaks
        self.structure_lookback = self.config.get('structure_lookback', 10)  # Bars to check for structure
        
        # Execution
        self.order_client = OrderClient(self.api_client, dry_run=self.dry_run)
        
        # State - multi-instrument support
        self.running = False
        self.paused = False
        self.contract_ids: Dict[str, str] = {}  # symbol -> contract_id
        self.current_prices: Dict[str, float] = {}  # symbol -> current_price
        self.positions: Dict[str, Optional[WickReversalPosition]] = {}  # symbol -> position
        self.data_lock = Lock()
        
        # Historical data (DataFrames) - per instrument
        self.df_hourly: Dict[str, pd.DataFrame] = {}  # symbol -> hourly bars
        self.df_minute: Dict[str, pd.DataFrame] = {}  # symbol -> 1-minute bars
        
        # Active fat wicks waiting for pullback - per instrument
        self.active_wicks: Dict[str, List[FatWick]] = {}  # symbol -> list of wicks
        
        # Pending entry orders tracking - per instrument
        self.pending_entry_orders: Dict[str, set] = {}  # symbol -> set of order_ids
        self.skip_reasons: Dict[str, str] = {}  # symbol -> latest reason for no-entry
        
        # API throttling / caching
        self.orders_poll_interval_seconds = int(self.config.get('orders_poll_interval_seconds', 20))
        self.positions_poll_interval_seconds = int(self.config.get('positions_poll_interval_seconds', 20))
        self.accounts_poll_interval_seconds = int(self.config.get('accounts_poll_interval_seconds', 30))
        self._last_orders_fetch: Optional[datetime] = None
        self._last_positions_fetch: Optional[datetime] = None
        self._last_accounts_fetch: Optional[datetime] = None
        self._cached_recent_orders: List[Dict[str, Any]] = []
        self._cached_positions: List[Dict[str, Any]] = []
        self._cached_accounts: List[Dict[str, Any]] = []
        
        # Observability / reload guards
        self.heartbeat_interval_seconds = int(self.config.get('heartbeat_interval_seconds', 60))
        self._last_heartbeat: Optional[datetime] = None
        self._last_history_reload_window: Optional[Tuple[int, int, int, int, int]] = None
        
        # Initialize per-instrument state
        for symbol in self.instruments:
            self.positions[symbol] = None
            self.active_wicks[symbol] = []
            self.pending_entry_orders[symbol] = set()
            self.skip_reasons[symbol] = "initializing"
        
        logger.info(f"Wick Reversal Strategy initialized (dry_run={self.dry_run})")
        if not self.dry_run:
            logger.warning("=" * 60)
            logger.warning("LIVE TRADING MODE ENABLED - REAL ORDERS WILL BE PLACED!")
            logger.warning("=" * 60)
    
    def initialize(self) -> bool:
        """Initialize connection and fetch contract info"""
        logger.info("Initializing Wick Reversal strategy...")
        
        # Authenticate
        if not self.api_client.authenticate():
            logger.error("Failed to authenticate with TopstepX")
            return False
        
        # Get accounts
        accounts = self.api_client.get_accounts()
        if not accounts:
            logger.error("No accounts found")
            return False
        
        # Select account (prefer practice)
        account = None
        prefer_practice = self.config.get('prefer_practice_account', True)
        specified_account_id = self.config.get('account_id')
        
        if specified_account_id:
            account = next((acc for acc in accounts if 
                          (acc.get('id') == specified_account_id or 
                           acc.get('accountId') == specified_account_id)), None)
        
        if not account and prefer_practice:
            for acc in accounts:
                name = acc.get('name', '').upper()
                is_simulated = acc.get('simulated', False)
                if is_simulated or 'PRAC' in name or 'PRACTICE' in name:
                    account = acc
                    break
        
        if not account and accounts:
            account = accounts[0]
        
        if not account:
            logger.error("No account available")
            return False
        
        account_id = account.get('id') or account.get('accountId')
        self.api_client.set_account(account_id)
        logger.info(f"Selected account: {account.get('name')} (ID: {account_id})")
        
        # Find contracts for all instruments
        all_contract_ids = []
        for symbol in self.instruments:
            contracts = self.api_client.search_contracts(symbol)
            if contracts:
                contract_id = contracts[0].get('contractId') or contracts[0].get('id')
                self.contract_ids[symbol] = contract_id
                all_contract_ids.append(contract_id)
                logger.info(f"Found contract: {symbol} (ID: {contract_id})")
            else:
                logger.error(f"Contract not found for symbol: {symbol}")
                return False
        
        # Load historical data for all instruments
        self._load_historical_data()
        
        # Setup real-time callbacks
        self._setup_realtime_callbacks()
        
        # Connect to SignalR for real-time updates
        self.api_client.connect_realtime(
            account_id=self.api_client.account_id,
            contract_ids=all_contract_ids if all_contract_ids else None
        )
        
        logger.info(f"Strategy initialized successfully for {len(self.instruments)} instrument(s): {', '.join(self.instruments)}")
        return True
    
    def _load_historical_data(self):
        """Load historical candles for both timeframes for all instruments"""
        logger.info("Loading historical data for all instruments...")
        
        adapter = MarketDataAdapter()
        
        for symbol in self.instruments:
            contract_id = self.contract_ids.get(symbol)
            if not contract_id:
                logger.warning(f"No contract ID for {symbol}, skipping")
                continue
            
            # Load hourly bars for wick detection
            bars_hourly = self.api_client.get_bars(
                contract_id=contract_id,
                interval=self.long_timeframe,
                limit=200  # ~8 days of hourly bars
            )
            
            # Load 1-minute bars for structure breaks
            bars_minute = self.api_client.get_bars(
                contract_id=contract_id,
                interval=self.short_timeframe,
                limit=500  # ~8 hours of 1-minute bars
            )
            
            if bars_hourly:
                candles_hourly = adapter.normalize_bars(bars_hourly, symbol, self.long_timeframe)
                df_hourly = pd.DataFrame([{
                    'timestamp': c.timestamp,
                    'open': c.open,
                    'high': c.high,
                    'low': c.low,
                    'close': c.close,
                    'volume': c.volume
                } for c in candles_hourly])
                df_hourly = df_hourly.sort_values('timestamp').reset_index(drop=True)
                self.df_hourly[symbol] = df_hourly
                logger.info(f"Loaded {len(df_hourly)} hourly bars for {symbol}")
                
                # Detect fat wicks in historical data
                self._detect_fat_wicks_historical(symbol)
            
            if bars_minute:
                candles_minute = adapter.normalize_bars(bars_minute, symbol, self.short_timeframe)
                df_minute = pd.DataFrame([{
                    'timestamp': c.timestamp,
                    'open': c.open,
                    'high': c.high,
                    'low': c.low,
                    'close': c.close,
                    'volume': c.volume
                } for c in candles_minute])
                df_minute = df_minute.sort_values('timestamp').reset_index(drop=True)
                self.df_minute[symbol] = df_minute
                logger.info(f"Loaded {len(df_minute)} 1-minute bars for {symbol}")
                
                # Initialize current price
                if candles_minute:
                    self.current_prices[symbol] = candles_minute[-1].close
                    self.position_manager.update_price(symbol, candles_minute[-1].close)
                    logger.info(f"Initialized {symbol} price: {candles_minute[-1].close:.2f}")
    
    def _detect_fat_wicks_historical(self, symbol: str):
        """Detect fat wicks in historical hourly data for a specific instrument"""
        df_hourly = self.df_hourly.get(symbol)
        if df_hourly is None or len(df_hourly) < 2:
            return
        
        for idx in range(1, len(df_hourly)):
            row = df_hourly.iloc[idx]
            wick = self._analyze_wick(row)
            if wick:
                self.active_wicks[symbol].append(wick)
                logger.info(
                    f"[{symbol}] Detected fat wick: {wick.wick_direction} wick, "
                    f"{wick.wick_percentage:.1f}% of range, "
                    f"midpoint={wick.wick_midpoint:.2f}"
                )
    
    def _analyze_wick(self, bar: pd.Series) -> Optional[FatWick]:
        """Analyze a bar to see if it has a fat wick"""
        high = bar['high']
        low = bar['low']
        open_price = bar['open']
        close_price = bar['close']
        
        # Volume confirmation (if enabled)
        if self.volume_confirmation_enabled and 'volume' in bar:
            volume = bar['volume']
            # Check if volume is above average (would need historical context)
            # For now, just check if volume exists and is > 0
            if volume <= 0:
                return None
        
        # Calculate body
        body_top = max(open_price, close_price)
        body_bottom = min(open_price, close_price)
        
        # Calculate wicks
        upper_wick = high - body_top
        lower_wick = body_bottom - low
        total_wick = upper_wick + lower_wick
        total_range = high - low
        
        if total_range == 0:
            return None
        
        # Calculate wick percentage
        wick_percentage = (total_wick / total_range) * 100
        
        # Check if wick is "fat" enough
        if wick_percentage < (self.wick_threshold * 100):
            return None
        
        # Determine wick direction
        if upper_wick > lower_wick * 1.5:  # Upper wick is significantly larger
            wick_direction = "UPPER"
            wick_extreme = high
            # Midpoint is 50% retracement from high to body top
            wick_midpoint = body_top + (upper_wick * 0.5)
        elif lower_wick > upper_wick * 1.5:  # Lower wick is significantly larger
            wick_direction = "LOWER"
            wick_extreme = low
            # Midpoint is 50% retracement from body bottom to low
            wick_midpoint = body_bottom - (lower_wick * 0.5)
        else:  # Both wicks are significant
            wick_direction = "BOTH"
            # Use the larger wick for direction
            if upper_wick >= lower_wick:
                wick_extreme = high
                wick_midpoint = body_top + (upper_wick * 0.5)
            else:
                wick_extreme = low
                wick_midpoint = body_bottom - (lower_wick * 0.5)
        
        return FatWick(
            timestamp=bar['timestamp'],
            high=high,
            low=low,
            open=open_price,
            close=close_price,
            upper_wick=upper_wick,
            lower_wick=lower_wick,
            total_wick=total_wick,
            wick_percentage=wick_percentage,
            wick_direction=wick_direction,
            wick_midpoint=wick_midpoint,
            wick_extreme=wick_extreme,
            is_active=True
        )
    
    def _check_pullback(self, symbol: str, price: float) -> Optional[FatWick]:
        """Check if price has pulled back into any active wick's midpoint for a specific instrument"""
        active_wicks = self.active_wicks.get(symbol, [])
        for wick in active_wicks:
            if not wick.is_active:
                continue
            
            # Calculate tolerance - prefer absolute, then wick-relative, then price-relative
            if self.pullback_tolerance_absolute is not None:
                tolerance = self.pullback_tolerance_absolute
            elif self.pullback_tolerance_wick_relative:
                # Use wick size as tolerance (e.g., 5% of wick size)
                tolerance = max(wick.upper_wick, wick.lower_wick) * self.pullback_tolerance
            else:
                # Original: percentage of price (less ideal but backward compatible)
                tolerance = wick.wick_midpoint * self.pullback_tolerance
            
            if abs(price - wick.wick_midpoint) <= tolerance:
                return wick
        
        return None
    
    def _check_trend_filter(self, symbol: str, direction: str) -> bool:
        """Check if trend filter allows this trade direction"""
        if not self.trend_filter_enabled:
            return True
        
        df_minute = self.df_minute.get(symbol)
        if df_minute is None or len(df_minute) < self.trend_ema_period:
            return True  # Not enough data, allow trade
        
        # Calculate EMA
        closes = df_minute['close'].values
        ema = pd.Series(closes).ewm(span=self.trend_ema_period, adjust=False).mean().iloc[-1]
        current_price = closes[-1]
        
        # For LONG: price should be above EMA (bullish trend) OR price should be below EMA but wick suggests reversal
        # For SHORT: price should be below EMA (bearish trend) OR price should be above EMA but wick suggests reversal
        # Actually, for mean reversion, we want to trade AGAINST the trend, so:
        # LONG when price is below EMA (oversold)
        # SHORT when price is above EMA (overbought)
        if direction == "LONG":
            return current_price <= ema  # Price below EMA = oversold, good for long
        else:  # SHORT
            return current_price >= ema  # Price above EMA = overbought, good for short
    
    def _detect_structure_break(self, symbol: str, wick: FatWick, apply_trend_filter: bool = True) -> Optional[StructureBreak]:
        """Detect structure break on short timeframe against wick direction for a specific instrument"""
        df_minute = self.df_minute.get(symbol)
        if df_minute is None or len(df_minute) < self.structure_lookback:
            return None
        
        # Get recent bars
        recent_bars = df_minute.tail(self.structure_lookback)
        
        # Determine what structure break we're looking for
        # If wick has long upper tail (bearish), look for bearish structure (lower highs/lows)
        # If wick has long lower tail (bullish), look for bullish structure (higher highs/lows)
        
        structure_break = None
        if wick.wick_direction in ["UPPER", "BOTH"] and wick.upper_wick > wick.lower_wick:
            # Long upper wick = bearish, look for bearish structure
            structure_break = self._check_bearish_structure(recent_bars)
        elif wick.wick_direction in ["LOWER", "BOTH"] and wick.lower_wick > wick.upper_wick:
            # Long lower wick = bullish, look for bullish structure
            structure_break = self._check_bullish_structure(recent_bars)
        
        # Apply trend filter if structure break found
        if apply_trend_filter and structure_break and structure_break.confirmation:
            direction = "LONG" if structure_break.direction == "BULLISH" else "SHORT"
            if not self._check_trend_filter(symbol, direction):
                logger.debug(f"[{symbol}] Trend filter blocked {direction} trade")
                return None
        
        return structure_break
    
    def _find_swing_highs(self, bars: pd.DataFrame, lookback: int = 5) -> List[Tuple[int, float]]:
        """Find swing highs (local maxima)"""
        swing_highs = []
        highs = bars['high'].values
        
        for i in range(lookback, len(highs) - lookback):
            is_swing_high = True
            # Check if this is higher than surrounding bars
            for j in range(i - lookback, i + lookback + 1):
                if j != i and highs[j] >= highs[i]:
                    is_swing_high = False
                    break
            if is_swing_high:
                swing_highs.append((i, highs[i]))
        
        return swing_highs
    
    def _find_swing_lows(self, bars: pd.DataFrame, lookback: int = 5) -> List[Tuple[int, float]]:
        """Find swing lows (local minima)"""
        swing_lows = []
        lows = bars['low'].values
        
        for i in range(lookback, len(lows) - lookback):
            is_swing_low = True
            # Check if this is lower than surrounding bars
            for j in range(i - lookback, i + lookback + 1):
                if j != i and lows[j] <= lows[i]:
                    is_swing_low = False
                    break
            if is_swing_low:
                swing_lows.append((i, lows[i]))
        
        return swing_lows
    
    def _check_bearish_structure(self, bars: pd.DataFrame) -> Optional[StructureBreak]:
        """Check for bearish structure using swing highs and lows"""
        if len(bars) < self.structure_swing_lookback * 2 + 1:
            return None
        
        # Find swing highs and lows
        swing_highs = self._find_swing_highs(bars, self.structure_swing_lookback)
        swing_lows = self._find_swing_lows(bars, self.structure_swing_lookback)
        
        # Need at least 2 swing highs and 2 swing lows
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return None
        
        # Check for lower highs (most recent must be lower than previous)
        recent_highs = sorted(swing_highs, key=lambda x: x[0])[-2:]  # Last 2 swing highs
        if len(recent_highs) >= 2:
            if recent_highs[-1][1] >= recent_highs[-2][1]:
                return None  # Not lower highs
        
        # Check for lower lows (most recent must be lower than previous)
        recent_lows = sorted(swing_lows, key=lambda x: x[0])[-2:]  # Last 2 swing lows
        if len(recent_lows) >= 2:
            if recent_lows[-1][1] >= recent_lows[-2][1]:
                return None  # Not lower lows
        
        # Check minimum move size
        if self.structure_min_move > 0:
            high_move = abs(recent_highs[-1][1] - recent_highs[-2][1]) if len(recent_highs) >= 2 else 0
            low_move = abs(recent_lows[-1][1] - recent_lows[-2][1]) if len(recent_lows) >= 2 else 0
            if high_move < self.structure_min_move and low_move < self.structure_min_move:
                return None  # Moves too small
        
        # Volume confirmation (if enabled)
        if self.volume_confirmation_enabled and 'volume' in bars.columns:
            recent_volumes = bars['volume'].tail(self.structure_swing_lookback * 2).values
            avg_volume = np.mean(recent_volumes[:-self.structure_swing_lookback]) if len(recent_volumes) > self.structure_swing_lookback else np.mean(recent_volumes)
            current_volume = recent_volumes[-1] if len(recent_volumes) > 0 else 0
            if avg_volume > 0 and current_volume < avg_volume * self.volume_multiplier:
                return None  # Volume not confirming
        
        current_price = bars['close'].iloc[-1]
        return StructureBreak(
            timestamp=bars['timestamp'].iloc[-1],
            price=current_price,
            direction="BEARISH",
            confirmation=True
        )
    
    def _check_bullish_structure(self, bars: pd.DataFrame) -> Optional[StructureBreak]:
        """Check for bullish structure using swing highs and lows"""
        if len(bars) < self.structure_swing_lookback * 2 + 1:
            return None
        
        # Find swing highs and lows
        swing_highs = self._find_swing_highs(bars, self.structure_swing_lookback)
        swing_lows = self._find_swing_lows(bars, self.structure_swing_lookback)
        
        # Need at least 2 swing highs and 2 swing lows
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return None
        
        # Check for higher highs (most recent must be higher than previous)
        recent_highs = sorted(swing_highs, key=lambda x: x[0])[-2:]  # Last 2 swing highs
        if len(recent_highs) >= 2:
            if recent_highs[-1][1] <= recent_highs[-2][1]:
                return None  # Not higher highs
        
        # Check for higher lows (most recent must be higher than previous)
        recent_lows = sorted(swing_lows, key=lambda x: x[0])[-2:]  # Last 2 swing lows
        if len(recent_lows) >= 2:
            if recent_lows[-1][1] <= recent_lows[-2][1]:
                return None  # Not higher lows
        
        # Check minimum move size
        if self.structure_min_move > 0:
            high_move = abs(recent_highs[-1][1] - recent_highs[-2][1]) if len(recent_highs) >= 2 else 0
            low_move = abs(recent_lows[-1][1] - recent_lows[-2][1]) if len(recent_lows) >= 2 else 0
            if high_move < self.structure_min_move and low_move < self.structure_min_move:
                return None  # Moves too small
        
        # Volume confirmation (if enabled)
        if self.volume_confirmation_enabled and 'volume' in bars.columns:
            recent_volumes = bars['volume'].tail(self.structure_swing_lookback * 2).values
            avg_volume = np.mean(recent_volumes[:-self.structure_swing_lookback]) if len(recent_volumes) > self.structure_swing_lookback else np.mean(recent_volumes)
            current_volume = recent_volumes[-1] if len(recent_volumes) > 0 else 0
            if avg_volume > 0 and current_volume < avg_volume * self.volume_multiplier:
                return None  # Volume not confirming
        
        current_price = bars['close'].iloc[-1]
        return StructureBreak(
            timestamp=bars['timestamp'].iloc[-1],
            price=current_price,
            direction="BULLISH",
            confirmation=True
        )
    
    def _calculate_position_size(self, symbol: str, stop_distance: float) -> int:
        """Calculate position size based on risk per trade and stop distance for a specific instrument"""
        if stop_distance == 0:
            return 1
        
        # Get tick size for this specific instrument
        tick_size = self.tick_sizes.get(symbol, 0.25)
        ticks_at_risk = stop_distance / tick_size
        
        if ticks_at_risk == 0:
            return 1
        
        # Get tick value for this symbol
        tick_value = self.position_manager.tick_values.get(symbol, 1.25)
        
        # Calculate contracts: risk_amount / (ticks_at_risk * tick_value)
        contracts = int(self.risk_per_trade / (ticks_at_risk * tick_value))
        contracts = max(1, min(contracts, self.config.get('max_contracts', 5)))
        
        return contracts
    
    def _enter_position(self, symbol: str, wick: FatWick, structure_break: StructureBreak) -> bool:
        """Enter a position based on wick reversal signal for a specific instrument"""
        if self.positions.get(symbol):
            logger.warning(f"[{symbol}] Already in a position, cannot enter new one")
            return False
        
        if self.pending_entry_orders.get(symbol):
            logger.warning(f"[{symbol}] Pending entry orders exist, cannot enter: {self.pending_entry_orders[symbol]}")
            return False
        
        # Determine direction based on wick
        # Long upper wick = bearish signal (SHORT)
        # Long lower wick = bullish signal (LONG)
        if wick.wick_direction in ["UPPER", "BOTH"] and wick.upper_wick > wick.lower_wick:
            direction = "SHORT"
            entry_price = structure_break.price
            # Stop beyond the upper extreme
            stop_loss = wick.wick_extreme + (wick.upper_wick * self.stop_fraction)
        elif wick.wick_direction in ["LOWER", "BOTH"] and wick.lower_wick > wick.upper_wick:
            direction = "LONG"
            entry_price = structure_break.price
            # Stop beyond the lower extreme
            stop_loss = wick.wick_extreme - (wick.lower_wick * self.stop_fraction)
        else:
            logger.warning(f"Unclear wick direction: {wick.wick_direction}")
            return False
        
        # Calculate stop distance
        stop_distance = abs(entry_price - stop_loss)
        
        # Calculate position size
        contracts = self._calculate_position_size(symbol, stop_distance)
        
        # Calculate take profit
        take_profit = entry_price + (stop_distance * self.target_multiple * (1 if direction == "LONG" else -1))
        
        # Calculate risk amount
        tick_size = self.tick_sizes.get(symbol, 0.25)
        tick_value = self.position_manager.tick_values.get(symbol, 1.25)
        ticks_at_risk = stop_distance / tick_size
        risk_amount = ticks_at_risk * tick_value * contracts
        
        # Place order
        side = "BUY" if direction == "LONG" else "SELL"
        contract_id = self.contract_ids.get(symbol)
        if not contract_id:
            logger.error(f"[{symbol}] No contract ID available")
            return False
        
        order_id = self.order_client.place_limit_order(
            contract_id=contract_id,
            side=side,
            quantity=contracts,
            price=entry_price
        )
        
        if order_id:
            self.pending_entry_orders[symbol].add(str(order_id))
            
            self.positions[symbol] = WickReversalPosition(
                entry_time=datetime.now(timezone.utc),
                entry_price=entry_price,
                direction=direction,
                stop_loss=stop_loss,
                take_profit=take_profit,
                contracts=contracts,
                wick=wick,
                order_id=order_id,
                risk_amount=risk_amount
            )
            
            # Mark wick as used
            wick.is_active = False
            
            logger.info(
                f"[{symbol}] ENTERED {direction} position: {contracts} contracts @ {entry_price:.2f} | "
                f"Stop: {stop_loss:.2f} | Target: {take_profit:.2f} | "
                f"Risk: ${risk_amount:.2f} | Wick: {wick.wick_direction}"
            )
            return True
        else:
            logger.error(f"[{symbol}] Failed to place entry order")
            return False
    
    def _check_exit_conditions(self, symbol: str) -> Tuple[bool, str, Optional[float]]:
        """Check if current position should be exited for a specific instrument
        
        Returns:
            (should_exit, exit_reason, exit_price)
        """
        pos = self.positions.get(symbol)
        if not pos:
            return False, "", None
        
        # Don't check exits if entry order hasn't filled
        if self.pending_entry_orders.get(symbol):
            return False, "", None
        
        # Get current price
        with self.data_lock:
            current_price = self.current_prices.get(symbol)
            if not current_price:
                return False, "", None
        
        # Check time-based exit
        if self.max_position_hours > 0:
            age_hours = (datetime.now(timezone.utc) - pos.entry_time).total_seconds() / 3600.0
            if age_hours > self.max_position_hours:
                return True, "TIME_EXIT", current_price
        
        # Calculate unrealized P&L for trailing stop
        if pos.direction == "LONG":
            unrealized_pnl_pct = ((current_price - pos.entry_price) / pos.entry_price) * 100
        else:  # SHORT
            unrealized_pnl_pct = ((pos.entry_price - current_price) / pos.entry_price) * 100
        
        # Update highest profit for trailing stop
        if unrealized_pnl_pct > pos.highest_profit:
            pos.highest_profit = unrealized_pnl_pct
        
        # Check trailing stop
        if self.trailing_stop_enabled and pos.highest_profit > 0:
            stop_distance = abs(pos.entry_price - pos.stop_loss)
            trailing_distance = stop_distance * self.trailing_stop_distance
            if pos.direction == "LONG":
                trailing_stop = pos.entry_price + (pos.highest_profit / 100 * pos.entry_price) - trailing_distance
                if current_price <= trailing_stop:
                    return True, "TRAILING_STOP", current_price
            else:  # SHORT
                trailing_stop = pos.entry_price - (pos.highest_profit / 100 * pos.entry_price) + trailing_distance
                if current_price >= trailing_stop:
                    return True, "TRAILING_STOP", current_price
        
        # Check partial profit taking
        if self.partial_profit_enabled and pos.contracts_exited == 0:
            if pos.direction == "LONG":
                if current_price >= pos.take_profit:
                    return True, "PARTIAL_PROFIT", pos.take_profit
            else:  # SHORT
                if current_price <= pos.take_profit:
                    return True, "PARTIAL_PROFIT", pos.take_profit
        
        # Check stop loss (stop order should handle this, but check as backup)
        if pos.direction == "LONG":
            if current_price <= pos.stop_loss:
                return True, "STOP_LOSS", pos.stop_loss
        else:  # SHORT
            if current_price >= pos.stop_loss:
                return True, "STOP_LOSS", pos.stop_loss
        
        # Check full take profit (if partial profit already taken)
        if pos.contracts_exited > 0:
            if pos.direction == "LONG":
                if current_price >= pos.take_profit:
                    return True, "TAKE_PROFIT", pos.take_profit
            else:  # SHORT
                if current_price <= pos.take_profit:
                    return True, "TAKE_PROFIT", pos.take_profit
        
        return False, "", None
    
    def _exit_position(self, symbol: str, exit_reason: str, exit_price: Optional[float] = None, partial: bool = False) -> bool:
        """Exit current position for a specific instrument
        
        Args:
            symbol: Symbol to exit
            exit_reason: Reason for exit
            exit_price: Optional exit price (uses current price if None)
            partial: If True, only exit partial position (for profit taking)
        """
        pos = self.positions.get(symbol)
        if not pos:
            return False
        
        # Determine exit side (opposite of entry)
        side = "SELL" if pos.direction == "LONG" else "BUY"
        
        # Calculate contracts to exit
        if partial and self.partial_profit_enabled:
            contracts_to_exit = int(pos.contracts * self.partial_profit_ratio)
            contracts_to_exit = max(1, min(contracts_to_exit, pos.contracts - pos.contracts_exited))
        else:
            contracts_to_exit = pos.contracts - pos.contracts_exited
        
        if contracts_to_exit <= 0:
            return False
        
        # Get exit price
        if exit_price is None:
            with self.data_lock:
                exit_price = self.current_prices.get(symbol) or pos.entry_price
        
        contract_id = self.contract_ids.get(symbol)
        if not contract_id:
            logger.error(f"[{symbol}] No contract ID available")
            return False
        
        # For stop loss, use stop order. For other exits, try limit then market
        exit_order_id = None
        if exit_reason == "STOP_LOSS" and not pos.stop_order_id:
            # Place stop order for stop loss
            stop_side = "SELL" if pos.direction == "LONG" else "BUY"
            exit_order_id = self.order_client.place_stop_order(
                contract_id=contract_id,
                side=stop_side,
                quantity=contracts_to_exit,
                stop_price=pos.stop_loss
            )
            if exit_order_id:
                pos.stop_order_id = exit_order_id
                logger.info(f"[{symbol}] Placed stop order: {exit_order_id} @ {pos.stop_loss:.2f}")
        else:
            # Try limit order first
            exit_order_id = self.order_client.place_limit_order(
                contract_id=contract_id,
                side=side,
                quantity=contracts_to_exit,
                price=exit_price
            )
            
            # If limit order fails, use market order as fallback
            if not exit_order_id:
                logger.warning(f"[{symbol}] Limit order failed, trying market order")
                exit_order_id = self.order_client.place_market_order(
                    contract_id=contract_id,
                    side=side,
                    quantity=contracts_to_exit
                )
        
        if exit_order_id:
            if partial:
                pos.contracts_exited += contracts_to_exit
                logger.info(
                    f"[{symbol}] PARTIAL EXIT: {exit_reason} | "
                    f"{side} {contracts_to_exit}/{pos.contracts} @ {exit_price:.2f} "
                    f"(Remaining: {pos.contracts - pos.contracts_exited})"
                )
            else:
                logger.info(
                    f"[{symbol}] EXITED position: {exit_reason} | "
                    f"{side} {contracts_to_exit} @ {exit_price:.2f}"
                )
                self.positions[symbol] = None
            return True
        else:
            logger.error(f"[{symbol}] Failed to place exit order")
            return False
    
    def _infer_direction_from_wick(self, wick: FatWick) -> Optional[str]:
        """Infer directional bias from wick anatomy."""
        if wick.wick_direction in ["UPPER", "BOTH"] and wick.upper_wick > wick.lower_wick:
            return "SHORT"
        if wick.wick_direction in ["LOWER", "BOTH"] and wick.lower_wick > wick.upper_wick:
            return "LONG"
        return None
    
    def _get_recent_orders_cached(self) -> List[Dict[str, Any]]:
        """Fetch recent orders with throttle caching to reduce API 429s."""
        now = datetime.now(timezone.utc)
        if (
            self._last_orders_fetch and
            (now - self._last_orders_fetch).total_seconds() < self.orders_poll_interval_seconds
        ):
            return self._cached_recent_orders
        
        recent_orders = self.api_client.get_orders(
            self.api_client.account_id,
            start_timestamp=now - timedelta(hours=1)
        )
        self._cached_recent_orders = recent_orders or []
        self._last_orders_fetch = now
        return self._cached_recent_orders
    
    def _get_positions_cached(self) -> List[Dict[str, Any]]:
        """Fetch open positions with throttle caching."""
        now = datetime.now(timezone.utc)
        if (
            self._last_positions_fetch and
            (now - self._last_positions_fetch).total_seconds() < self.positions_poll_interval_seconds
        ):
            return self._cached_positions
        
        positions = self.api_client.get_positions(self.api_client.account_id)
        self._cached_positions = positions or []
        self._last_positions_fetch = now
        return self._cached_positions
    
    def _get_accounts_cached(self) -> List[Dict[str, Any]]:
        """Fetch accounts with throttle caching."""
        now = datetime.now(timezone.utc)
        if (
            self._last_accounts_fetch and
            (now - self._last_accounts_fetch).total_seconds() < self.accounts_poll_interval_seconds
        ):
            return self._cached_accounts
        
        accounts = self.api_client.get_accounts()
        self._cached_accounts = accounts or []
        self._last_accounts_fetch = now
        return self._cached_accounts
    
    def _emit_heartbeat(self):
        """Periodic per-symbol diagnostics for skipped entries."""
        now = datetime.now(timezone.utc)
        if (
            self._last_heartbeat and
            (now - self._last_heartbeat).total_seconds() < self.heartbeat_interval_seconds
        ):
            return
        
        snapshots = []
        for symbol in self.instruments:
            snapshots.append(
                f"{symbol}:price={self.current_prices.get(symbol)},"
                f"wicks={len(self.active_wicks.get(symbol, []))},"
                f"pos={self.positions.get(symbol) is not None},"
                f"pending={len(self.pending_entry_orders.get(symbol, set()))},"
                f"reason={self.skip_reasons.get(symbol, 'n/a')}"
            )
        logger.info("Heartbeat | " + " | ".join(snapshots))
        self._last_heartbeat = now
    
    def _check_order_fills(self):
        """Check if pending orders have been filled for all instruments"""
        if not self.api_client.account_id:
            return
        
        has_pending = any(len(self.pending_entry_orders.get(symbol, set())) > 0 for symbol in self.instruments)
        if not has_pending:
            return
        
        recent_orders = self._get_recent_orders_cached()
        
        for symbol in self.instruments:
            pos = self.positions.get(symbol)
            if not pos:
                continue
            
            pending_orders = self.pending_entry_orders.get(symbol, set())
            if not pending_orders:
                continue
            
            for order in recent_orders:
                order_id = str(order.get('id') or order.get('orderId', ''))
                status = order.get('status')  # 2 = Filled
                
                if order_id in pending_orders and status == 2:
                    fill_price = order.get('filledPrice') or order.get('limitPrice')
                    if fill_price:
                        pos.entry_price = float(fill_price)
                        self.pending_entry_orders[symbol].discard(order_id)
                        logger.info(f"[{symbol}] Entry order filled: {order_id} @ {fill_price}")
    
    def _cleanup_expired_wicks(self):
        """Remove wicks that are too old for all instruments"""
        now = datetime.now(timezone.utc)
        
        for symbol in self.instruments:
            active_wicks = self.active_wicks.get(symbol, [])
            expired = []
            
            for wick in active_wicks:
                age_hours = (now - wick.timestamp).total_seconds() / 3600.0
                if age_hours > self.max_wick_age_hours:
                    expired.append(wick)
            
            for wick in expired:
                self.active_wicks[symbol].remove(wick)
                logger.debug(f"[{symbol}] Expired wick removed: {wick.timestamp}")
    
    def _process_tick(self):
        """Process a single tick - main strategy logic for all instruments"""
        if self.paused or not self.running:
            return
        
        # Cleanup expired wicks for all instruments
        self._cleanup_expired_wicks()
        
        # Process each instrument independently
        for symbol in self.instruments:
            # Get current price
            with self.data_lock:
                current_price = self.current_prices.get(symbol)
                if not current_price:
                    self.skip_reasons[symbol] = "no-live-price"
                    continue
            
            # Check for pullback into active wicks
            pos = self.positions.get(symbol)
            pending_orders = self.pending_entry_orders.get(symbol, set())
            
            if pos:
                self.skip_reasons[symbol] = "position-open"
            elif pending_orders:
                self.skip_reasons[symbol] = "entry-order-pending"
            
            if not pos and not pending_orders:
                pulled_back_wick = self._check_pullback(symbol, current_price)
                
                if pulled_back_wick:
                    # Check for structure break
                    structure_break = self._detect_structure_break(symbol, pulled_back_wick, apply_trend_filter=False)
                    structure_ok = structure_break is not None and structure_break.confirmation
                    
                    direction = self._infer_direction_from_wick(pulled_back_wick)
                    structure_direction = ("LONG" if structure_break.direction == "BULLISH" else "SHORT") if structure_ok and structure_break else None
                    if direction and structure_direction and direction != structure_direction:
                        self.skip_reasons[symbol] = (
                            f"blocked (direction-mismatch wick={direction} structure={structure_direction})"
                        )
                        continue
                    
                    trend_ok = self._check_trend_filter(symbol, direction) if direction else False
                    conditions_met = 1 + int(structure_ok) + int(trend_ok)  # pullback + structure + trend
                    
                    if direction and conditions_met >= self.entry_min_conditions:
                        if not structure_ok:
                            structure_break = StructureBreak(
                                timestamp=datetime.now(timezone.utc),
                                price=current_price,
                                direction="BULLISH" if direction == "LONG" else "BEARISH",
                                confirmation=False
                            )
                        self._enter_position(symbol, pulled_back_wick, structure_break)
                        self.skip_reasons[symbol] = (
                            f"entered ({conditions_met}/3: pullback=True, structure={structure_ok}, trend={trend_ok})"
                        )
                    else:
                        self.skip_reasons[symbol] = (
                            f"blocked ({conditions_met}/3 < {self.entry_min_conditions} or unknown-direction) | "
                            f"pullback=True structure={structure_ok} trend={trend_ok}"
                        )
                else:
                    self.skip_reasons[symbol] = "no-pullback-to-wick-midpoint"
            
            # Monitor position for exits
            if pos and not pending_orders:
                should_exit, exit_reason, exit_price = self._check_exit_conditions(symbol)
                if should_exit:
                    # Handle partial profit taking
                    if exit_reason == "PARTIAL_PROFIT":
                        self._exit_position(symbol, exit_reason, exit_price, partial=True)
                    else:
                        self._exit_position(symbol, exit_reason, exit_price, partial=False)
    
    def _update_risk_checks(self) -> bool:
        """Update risk checks and return True if should stop"""
        if not self.api_client.account_id:
            return False
        
        # Reconcile positions with API (source of truth)
        try:
            api_positions = self._get_positions_cached()
            if api_positions is not None:
                self.position_manager.reconcile_with_api_positions(api_positions)
                logger.debug(f"Reconciled {len(api_positions)} positions with API")
        except Exception as e:
            logger.debug(f"Error reconciling positions: {e}")
        
        exposure = self.position_manager.net_exposure_dollars()
        
        # Update risk manager balance
        accounts = self._get_accounts_cached()
        if accounts:
            selected_account = next(
                (acc for acc in accounts if 
                 (acc.get('id') == self.api_client.account_id or 
                  acc.get('accountId') == self.api_client.account_id)),
                None
            )
            if selected_account:
                current_balance = selected_account.get('balance')
                if current_balance is not None:
                    self.risk_manager.update_balance(current_balance)
        
        daily_pnl = self.risk_manager.get_daily_pnl()
        drawdown = self.risk_manager.get_trailing_drawdown()
        
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
        """Emergency flatten all positions for all instruments"""
        logger.critical(f"EMERGENCY FLATTEN: {reason}")
        
        # Exit all positions
        for symbol in self.instruments:
            if self.positions.get(symbol):
                self._exit_position(symbol, reason)
        
        # Clear all pending orders
        for symbol in self.instruments:
            self.pending_entry_orders[symbol].clear()
        
        if not self.dry_run and self.api_client.account_id:
            positions = self.api_client.get_positions(self.api_client.account_id)
            for position in positions:
                contract_id = position.get('contractId')
                if contract_id:
                    self.api_client.close_position(contract_id)
        
        self.position_manager.flatten_all()
        self.running = False
    
    def _setup_realtime_callbacks(self):
        """Setup callbacks for real-time market data for all instruments"""
        def on_quote_update(quote_data):
            """Handle quote updates (includes OHLC data for bars)"""
            try:
                # Handle list format
                if isinstance(quote_data, list):
                    if len(quote_data) > 0 and isinstance(quote_data[0], dict):
                        quote_data = quote_data[0]
                    else:
                        return
                
                if not isinstance(quote_data, dict):
                    return
                
                # Determine which symbol this quote is for
                contract_id = quote_data.get('contractId') or quote_data.get('contract_id') or quote_data.get('symbolId')
                symbol = None
                for sym, cid in self.contract_ids.items():
                    if str(cid) == str(contract_id):
                        symbol = sym
                        break
                
                if not symbol:
                    return
                
                with self.data_lock:
                    # Update current price from various possible fields
                    price = quote_data.get('lastPrice') or quote_data.get('last') or quote_data.get('close') or quote_data.get('price')
                    if price:
                        self.current_prices[symbol] = float(price)
                        self.position_manager.update_price(symbol, float(price))
                    
                    # Check if this quote contains OHLC data (new bar)
                    if all(key in quote_data for key in ['open', 'high', 'low', 'close']):
                        # This might be a new bar - check if it's hourly
                        # Note: We'll need to track the last bar timestamp to detect new hourly bars
                        # For now, we'll rely on periodic historical data reloading for wick detection
                        pass
                        
            except Exception as e:
                logger.error(f"Error in quote update callback: {e}", exc_info=True)
        
        def on_market_trade_update(trade_data):
            """Handle market trade updates"""
            try:
                # Handle list format
                if isinstance(trade_data, list):
                    if len(trade_data) > 0 and isinstance(trade_data[0], dict):
                        trade_data = trade_data[0]
                    else:
                        return
                
                if not isinstance(trade_data, dict):
                    return
                
                # Determine which symbol this trade is for
                symbol_id = trade_data.get('symbolId', '')
                price = trade_data.get('price')
                
                if not price:
                    return
                
                # Find symbol by matching contract IDs
                symbol = None
                for sym, cid in self.contract_ids.items():
                    if str(cid) == str(symbol_id) or str(symbol_id) in str(cid):
                        symbol = sym
                        break
                
                if not symbol:
                    return
                
                with self.data_lock:
                    self.current_prices[symbol] = float(price)
                    self.position_manager.update_price(symbol, float(price))
                    
            except Exception as e:
                logger.error(f"Error in trade update callback: {e}", exc_info=True)
        
        # Register callbacks using the correct method name
        self.api_client.register_realtime_callback("on_quote_update", on_quote_update)
        self.api_client.register_realtime_callback("on_market_trade_update", on_market_trade_update)
        logger.info("Registered real-time callbacks for all instruments")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status for all instruments"""
        with self.data_lock:
            pass  # Lock acquired for thread safety
        
        instruments_status = {}
        for symbol in self.instruments:
            pos = self.positions.get(symbol)
            instruments_status[symbol] = {
                "current_price": self.current_prices.get(symbol),
                "has_position": pos is not None,
                "position": {
                    "direction": pos.direction if pos else None,
                    "entry_price": pos.entry_price if pos else None,
                    "stop_loss": pos.stop_loss if pos else None,
                    "take_profit": pos.take_profit if pos else None,
                    "contracts": pos.contracts if pos else None,
                } if pos else None,
                "active_wicks": len(self.active_wicks.get(symbol, [])),
                "pending_orders": len(self.pending_entry_orders.get(symbol, set())),
                "skip_reason": self.skip_reasons.get(symbol, "n/a"),
            }
        
        return {
            "status": "running" if self.running else "stopped",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trading_enabled": not self.paused and self.running,
            "dry_run": self.dry_run,
            "instruments": self.instruments,
            "instruments_status": instruments_status,
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
        logger.info("Starting Wick Reversal trading loop...")
        
        try:
            while self.running:
                if self.paused:
                    time.sleep(1)
                    continue
                
                # Check for order fills
                self._check_order_fills()
                
                # Process tick
                self._process_tick()
                
                # Risk checks
                if self._update_risk_checks():
                    break
                
                # Reload historical data once per 30-minute window
                now = datetime.now(timezone.utc)
                window_key = (now.year, now.month, now.day, now.hour, now.minute // 30)
                if now.minute % 30 == 0 and self._last_history_reload_window != window_key:
                    self._load_historical_data()
                    self._last_history_reload_window = window_key
                
                # Emit periodic diagnostics
                self._emit_heartbeat()
                
                time.sleep(5)  # Check every 5 seconds
        
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        except Exception as e:
            logger.error(f"Error in trading loop: {e}", exc_info=True)
        finally:
            logger.info("Shutting down...")
            self.emergency_flatten("Shutdown")
            self.api_client.disconnect()

