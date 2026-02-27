"""
Asian Range Breakout Strategy for MGC Gold Futures - Production Implementation
Implements the exact strategy from arbts.txt specification for live trading
"""
import os
import sys
import time as time_module
import logging
import yaml
import pandas as pd
import pytz
from datetime import datetime, time, timedelta
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from connectors.topstepx_client import TopstepXClient
from connectors.market_data_adapter import MarketDataAdapter
from connectors.mgc_contract_specs import MGC_SPECS
from data.timeseries_store import TimeseriesStore
from data.trade_history import TradeHistory
from data.state_persistence import StatePersistence
from data.analytics_store import AnalyticsStore
from strategy.position_manager import PositionManager
from strategy.risk_manager import RiskManager
from execution.order_client import OrderClient
from execution.fill_handler import FillHandler
from observability.metrics import MetricsExporter

logger = logging.getLogger(__name__)


@dataclass
class AsianRange:
    """Represents the Asian session range for a trading day"""
    date: datetime
    asian_high: float
    asian_low: float
    range_size: float
    asian_high_time: datetime
    asian_low_time: datetime


@dataclass
class PendingOrder:
    """Represents a pending OCO order"""
    buy_stop_price: float
    sell_stop_price: float
    asian_range: AsianRange
    order_time: datetime  # 3 AM ET
    buy_stop_order_id: Optional[str] = None
    sell_stop_order_id: Optional[str] = None


@dataclass
class ActiveTrade:
    """Represents an active trade"""
    entry_time: datetime
    entry_price: float
    stop_loss: float
    take_profit: float
    contracts: int
    is_long: bool
    asian_range: AsianRange
    breakeven_moved: bool = False
    partial_close_done: bool = False
    remaining_contracts: int = 0
    partial_close_price: Optional[float] = None
    stop_loss_after_partial: Optional[float] = None


class AsianRangeCalculator:
    """Calculates Asian session ranges"""
    
    # Time windows (ET timezone)
    ASIAN_START_HOUR = 20  # 8:00 PM ET
    ASIAN_END_HOUR = 2  # 2:00 AM ET (next day)
    LONDON_OPEN_HOUR = 3  # 3:00 AM ET
    NY_CLOSE_HOUR = 12  # 12:00 PM ET
    END_OF_DAY_HOUR = 17  # 5:00 PM ET
    
    def __init__(self, et_timezone: str = "America/New_York"):
        self.et_tz = pytz.timezone(et_timezone)
    
    def calculate_asian_range(self, candles: list, trading_date: datetime) -> Optional[AsianRange]:
        """
        Calculate Asian range for a specific date
        
        Asian session: 8 PM previous day to 2 AM current day
        
        Args:
            candles: List of candle objects with timestamp, high, low
            trading_date: Trading date (the day we're trading)
            
        Returns:
            AsianRange object or None if insufficient data
        """
        # Asian session is 8 PM previous day to 2 AM current day
        prev_day = trading_date - timedelta(days=1)
        asian_start = self.et_tz.localize(
            datetime.combine(prev_day.date(), time(20, 0))
        )
        asian_end = self.et_tz.localize(
            datetime.combine(trading_date.date(), time(2, 0))
        )
        
        # Filter candles for Asian session
        asian_candles = []
        for candle in candles:
            candle_time = candle.timestamp
            # Handle timezone conversion
            if candle_time.tzinfo is None:
                candle_time = self.et_tz.localize(candle_time)
            elif candle_time.tzinfo != self.et_tz:
                candle_time = candle_time.astimezone(self.et_tz)
            
            if asian_start <= candle_time < asian_end:
                asian_candles.append(candle)
        
        if len(asian_candles) == 0:
            return None
        
        # Find high and low during Asian session
        asian_high = max(c.high for c in asian_candles)
        asian_low = min(c.low for c in asian_candles)
        
        # Find timestamps of high and low
        high_candle = max(asian_candles, key=lambda c: c.high)
        low_candle = min(asian_candles, key=lambda c: c.low)
        
        asian_high_time = high_candle.timestamp
        asian_low_time = low_candle.timestamp
        
        range_size = asian_high - asian_low
        
        return AsianRange(
            date=trading_date,
            asian_high=asian_high,
            asian_low=asian_low,
            range_size=range_size,
            asian_high_time=asian_high_time,
            asian_low_time=asian_low_time
        )
    
    def create_pending_orders(self, asian_range: AsianRange, order_time: datetime) -> PendingOrder:
        """
        Create OCO pending orders at London open (3 AM ET)
        
        Buy Stop: Asian High + 1 tick
        Sell Stop: Asian Low - 1 tick
        """
        asian_high = MGC_SPECS.round_to_tick(asian_range.asian_high)
        asian_low = MGC_SPECS.round_to_tick(asian_range.asian_low)
        
        buy_stop_price = asian_high + MGC_SPECS.tick_size
        sell_stop_price = asian_low - MGC_SPECS.tick_size
        
        return PendingOrder(
            buy_stop_price=buy_stop_price,
            sell_stop_price=sell_stop_price,
            asian_range=asian_range,
            order_time=order_time
        )


class AsianRangeStrategy:
    """Main Asian Range Breakout strategy implementation for production"""
    
    def __init__(self, config_path: str = "asian_range_config.yaml"):
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
        self.analytics_store = AnalyticsStore()
        
        # Strategy components
        self.symbol = self.config.get('instrument', 'MGC')
        
        # Include multipliers for common symbols that might appear from other strategies on same account
        self.position_manager = PositionManager(
            max_net_notional=self.config.get('max_net_notional', 5000.0),
            tick_values={
                self.symbol: MGC_SPECS.tick_value,
                'MES': 5.0,  # MES tick value
                'MNQ': 2.0,  # MNQ tick value
                'GC': 10.0   # GC tick value
            },
            tick_sizes={
                self.symbol: MGC_SPECS.tick_size,  # MGC tick size = 0.10
                'MES': 0.25,  # MES tick size
                'MNQ': 0.25,  # MNQ tick size
                'GC': 0.10    # GC tick size
            },
            contract_multipliers={
                self.symbol: 10.0,  # CRITICAL: MGC = 10 oz per contract (not 100!)
                'MES': 5.0,   # MES contract multiplier
                'MNQ': 2.0,   # MNQ contract multiplier
                'GC': 100.0   # GC contract multiplier
            }
        )
        
        self.risk_manager = RiskManager(
            max_daily_loss=self.config.get('max_daily_loss', 900.0),
            trailing_drawdown_limit=self.config.get('trailing_drawdown_limit', 1800.0),
            max_net_notional=self.config.get('max_net_notional', 5000.0)
        )
        
        # Strategy parameters
        self.contracts_per_trade = self.config.get('contracts_per_trade', 5)
        self.tp_multiplier = self.config.get('tp_multiplier', 1.5)
        self.sl_buffer_ticks = self.config.get('sl_buffer_ticks', 1)
        self.partial_close_percent = self.config.get('partial_close_percent', 0.75)
        
        # Strategy components
        self.range_calculator = AsianRangeCalculator()
        
        # Execution
        self.order_client = OrderClient(self.api_client, dry_run=self.dry_run)
        
        # State
        self.running = False
        self.paused = False
        self.contract_id: Optional[str] = None
        self.current_price: Optional[float] = None
        self.pending_order: Optional[PendingOrder] = None
        self.active_trade: Optional[ActiveTrade] = None
        self.current_asian_range: Optional[AsianRange] = None
        self.last_asian_range_date: Optional[datetime] = None
        
        logger.info(f"Asian Range Strategy initialized (dry_run={self.dry_run})")
        if not self.dry_run:
            logger.warning("=" * 60)
            logger.warning("LIVE TRADING MODE ENABLED - REAL ORDERS WILL BE PLACED!")
            logger.warning("=" * 60)
    
    def initialize(self) -> bool:
        """Initialize connection and fetch contract info"""
        logger.info("Initializing Asian Range strategy...")
        
        # Authenticate
        if not self.api_client.authenticate():
            logger.error("Failed to authenticate with TopstepX")
            return False
        
        # Get accounts
        accounts = self.api_client.get_accounts()
        if not accounts:
            logger.error("No accounts found")
            return False
        
        # Select account (prioritize practice accounts)
        account = None
        prefer_practice = self.config.get('prefer_practice_account', True)
        specified_account_id = self.config.get('account_id')
        
        if specified_account_id:
            # Use specified account ID
            account = next((acc for acc in accounts if 
                          (acc.get('id') == specified_account_id or 
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
                logger.info(f"Using account: {account.get('name', 'Unknown')} (ID: {account_id})")
            else:
                logger.error("Selected account has no ID")
                return False
        else:
            logger.error("Could not select an account")
            return False
        
        # Search for MGC contract
        contracts = self.api_client.search_contracts(self.symbol)
        if contracts:
            self.contract_id = contracts[0].get('contractId') or contracts[0].get('id')
            logger.info(f"Contract: {self.symbol} -> {self.contract_id}")
        else:
            logger.error(f"Could not find contract for {self.symbol}")
            return False
        
        # Load historical data for Asian range calculation
        self._load_historical_data()
        
        # Setup real-time callbacks
        self._setup_realtime_callbacks()
        
        # Connect to SignalR
        self.api_client.connect_realtime(
            account_id=self.api_client.account_id,
            contract_ids=[self.contract_id] if self.contract_id else None
        )
        
        logger.info("Asian Range Strategy initialized successfully")
        return True
    
    def _load_historical_data(self):
        """Load historical candles for Asian range calculation"""
        logger.info("Loading historical data for Asian range calculation...")
        
        # Get bars from API (need data from previous day 8 PM to current day 2 AM)
        # Fetch enough bars to cover Asian session
        bars = self.api_client.get_bars(
            contract_id=self.contract_id,
            interval="1m",
            limit=500  # Enough for ~8 hours of 1-minute bars
        )
        
        if bars:
            adapter = MarketDataAdapter()
            candles = adapter.normalize_bars(bars, self.symbol, "1m")
            self.timeseries_store.store_candles(candles)
            logger.info(f"Loaded {len(candles)} historical candles")
            
            # Initialize current_price from most recent candle
            if candles and not self.current_price:
                self.current_price = candles[-1].close
                self.position_manager.update_price(self.symbol, self.current_price)
                logger.info(f"Initialized {self.symbol} price: {self.current_price:.2f}")
    
    def _setup_realtime_callbacks(self):
        """Setup callbacks for real-time market data updates"""
        symbol_id = f"F.US.{self.symbol}"
        
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
                
                if symbol_id_data == symbol_id or not self.current_price:
                    self.current_price = float(last_price)
                    self.position_manager.update_price(self.symbol, self.current_price)
                    logger.debug(f"Updated {self.symbol} price: {self.current_price:.2f}")
                    
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
                
                if symbol_id_data == symbol_id or not self.current_price:
                    self.current_price = float(price)
                    self.position_manager.update_price(self.symbol, self.current_price)
                    
            except Exception as e:
                logger.error(f"Error processing trade update: {e}")
        
        # Register callbacks
        self.api_client.register_realtime_callback("on_quote_update", on_quote_update)
        self.api_client.register_realtime_callback("on_market_trade_update", on_market_trade_update)
        
        logger.info("Registered real-time market data callbacks")
    
    def _calculate_asian_range(self) -> Optional[AsianRange]:
        """Calculate Asian range for current trading day"""
        now_et = datetime.now(self.range_calculator.et_tz)
        trading_date = now_et.date()
        
        # Check if we already calculated range for today
        if self.last_asian_range_date and self.last_asian_range_date.date() == trading_date:
            return self.current_asian_range
        
        # Get candles from timeseries store
        candles = self.timeseries_store.get_candles(
            self.symbol,
            interval="1m",
            limit=500
        )
        
        if len(candles) < 100:
            logger.warning("Insufficient data for Asian range calculation")
            return None
        
        # Calculate Asian range
        asian_range = self.range_calculator.calculate_asian_range(
            candles,
            datetime.combine(trading_date, time(0, 0))
        )
        
        if asian_range:
            self.current_asian_range = asian_range
            self.last_asian_range_date = now_et
            logger.info(f"Asian Range calculated: High={asian_range.asian_high:.2f}, "
                       f"Low={asian_range.asian_low:.2f}, Size={asian_range.range_size:.2f}")
        
        return asian_range
    
    def _place_oco_orders(self, pending_order: PendingOrder):
        """Place OCO orders (Buy Stop and Sell Stop)"""
        if not self.contract_id:
            return
        
        # Place Buy Stop order
        buy_order_id = self.order_client.place_limit_order(
            contract_id=self.contract_id,
            side="BUY",
            quantity=self.contracts_per_trade,
            price=pending_order.buy_stop_price
        )
        
        # Place Sell Stop order
        sell_order_id = self.order_client.place_limit_order(
            contract_id=self.contract_id,
            side="SELL",
            quantity=self.contracts_per_trade,
            price=pending_order.sell_stop_price
        )
        
        if buy_order_id:
            pending_order.buy_stop_order_id = buy_order_id
        if sell_order_id:
            pending_order.sell_stop_order_id = sell_order_id
        
        logger.info(f"Placed OCO orders: Buy Stop @ {pending_order.buy_stop_price:.2f} "
                   f"(ID: {buy_order_id}), Sell Stop @ {pending_order.sell_stop_price:.2f} (ID: {sell_order_id})")
    
    def _cancel_oco_orders(self, pending_order: PendingOrder):
        """Cancel OCO orders"""
        if pending_order.buy_stop_order_id:
            self.order_client.cancel_order(pending_order.buy_stop_order_id)
        if pending_order.sell_stop_order_id:
            self.order_client.cancel_order(pending_order.sell_stop_order_id)
        logger.info("Canceled OCO orders")
    
    def _check_order_fills(self):
        """Check if pending orders have been filled"""
        if not self.pending_order or not self.api_client.account_id:
            return
        
        # Get recent orders to check for fills
        recent_orders = self.api_client.get_orders(
            self.api_client.account_id,
            start_timestamp=datetime.now() - timedelta(hours=1)
        )
        
        for order in recent_orders:
            order_id = str(order.get('id') or order.get('orderId', ''))
            status = order.get('status')  # 2 = Filled
            
            if status == 2:
                # Check if this is one of our pending orders
                if order_id == self.pending_order.buy_stop_order_id:
                    # Buy stop filled - enter long
                    fill_price = order.get('filledPrice') or order.get('limitPrice')
                    self._enter_trade(fill_price, is_long=True)
                    self._cancel_oco_orders(self.pending_order)
                    self.pending_order = None
                    break
                elif order_id == self.pending_order.sell_stop_order_id:
                    # Sell stop filled - enter short
                    fill_price = order.get('filledPrice') or order.get('limitPrice')
                    self._enter_trade(fill_price, is_long=False)
                    self._cancel_oco_orders(self.pending_order)
                    self.pending_order = None
                    break
    
    def _enter_trade(self, entry_price: float, is_long: bool):
        """Enter a new trade"""
        if not self.current_asian_range:
            logger.error("Cannot enter trade: No Asian range available")
            return
        
        # Calculate stop loss and take profit
        asian_range = self.current_asian_range
        buffer_distance = self.sl_buffer_ticks * MGC_SPECS.tick_size
        
        if is_long:
            stop_loss = MGC_SPECS.round_to_tick(asian_range.asian_low - buffer_distance)
        else:
            stop_loss = MGC_SPECS.round_to_tick(asian_range.asian_high + buffer_distance)
        
        range_size = asian_range.range_size
        tp_distance = range_size * self.tp_multiplier
        
        if is_long:
            take_profit = MGC_SPECS.round_to_tick(entry_price + tp_distance)
        else:
            take_profit = MGC_SPECS.round_to_tick(entry_price - tp_distance)
        
        # Create active trade
        self.active_trade = ActiveTrade(
            entry_time=datetime.now(self.range_calculator.et_tz),
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            contracts=self.contracts_per_trade,
            is_long=is_long,
            asian_range=asian_range,
            remaining_contracts=self.contracts_per_trade
        )
        
        logger.info(f"Entered {'LONG' if is_long else 'SHORT'} trade: "
                   f"Entry={entry_price:.2f}, SL={stop_loss:.2f}, TP={take_profit:.2f}, "
                   f"Contracts={self.contracts_per_trade}")
    
    def _manage_active_trade(self):
        """Manage active trade (check SL, TP, break-even, partial close)"""
        if not self.active_trade or not self.current_price:
            return
        
        trade = self.active_trade
        now_et = datetime.now(self.range_calculator.et_tz)
        
        # Check stop loss
        if (trade.is_long and self.current_price <= trade.stop_loss) or \
           (not trade.is_long and self.current_price >= trade.stop_loss):
            self._exit_trade(trade.stop_loss, "SL")
            return
        
        # Check take profit
        if (trade.is_long and self.current_price >= trade.take_profit) or \
           (not trade.is_long and self.current_price <= trade.take_profit):
            self._exit_trade(trade.take_profit, "TP")
            return
        
        # Check break-even move (+1R profit)
        if not trade.breakeven_moved:
            range_size = trade.asian_range.range_size
            if trade.is_long:
                breakeven_price = trade.entry_price + range_size
                if self.current_price >= breakeven_price:
                    trade.stop_loss = trade.entry_price
                    trade.breakeven_moved = True
                    logger.info(f"Moved stop loss to break-even: {trade.stop_loss:.2f}")
            else:
                breakeven_price = trade.entry_price - range_size
                if self.current_price <= breakeven_price:
                    trade.stop_loss = trade.entry_price
                    trade.breakeven_moved = True
                    logger.info(f"Moved stop loss to break-even: {trade.stop_loss:.2f}")
        
        # Check partial close at 12 PM ET
        if not trade.partial_close_done and now_et.hour >= self.range_calculator.NY_CLOSE_HOUR:
            self._partial_close_trade()
        
        # Check end of day exit (5 PM ET)
        if now_et.hour >= self.range_calculator.END_OF_DAY_HOUR:
            self._exit_trade(self.current_price, "EndOfDay")
    
    def _partial_close_trade(self):
        """Partial close at 12 PM ET"""
        if not self.active_trade:
            return
        
        trade = self.active_trade
        partial_contracts = int(trade.contracts * self.partial_close_percent)
        remaining_contracts = trade.contracts - partial_contracts
        
        if remaining_contracts <= 0:
            # Close entire position
            self._exit_trade(self.current_price, "PartialClose")
            return
        
        # Place market order to close partial position
        close_side = "SELL" if trade.is_long else "BUY"
        # Note: In production, you'd place a market order here
        # For now, we'll simulate the partial close
        
        trade.partial_close_done = True
        trade.partial_close_price = self.current_price
        trade.remaining_contracts = remaining_contracts
        
        # Move stop to 50% profit level
        if trade.is_long:
            profit_made = trade.partial_close_price - trade.entry_price
            new_stop_loss = trade.entry_price + (profit_made * 0.5)
        else:
            profit_made = trade.entry_price - trade.partial_close_price
            new_stop_loss = trade.entry_price - (profit_made * 0.5)
        
        trade.stop_loss_after_partial = MGC_SPECS.round_to_tick(new_stop_loss)
        trade.stop_loss = trade.stop_loss_after_partial
        
        logger.info(f"Partial close: Closed {partial_contracts} contracts @ {trade.partial_close_price:.2f}, "
                   f"Remaining {remaining_contracts} contracts, Stop moved to {trade.stop_loss:.2f}")
    
    def _exit_trade(self, exit_price: float, reason: str):
        """Exit active trade"""
        if not self.active_trade:
            return
        
        trade = self.active_trade
        
        # Calculate PnL
        pnl = MGC_SPECS.calculate_pnl(
            trade.entry_price,
            exit_price,
            trade.contracts if not trade.partial_close_done else trade.remaining_contracts,
            trade.is_long
        )
        
        # Add partial close PnL if applicable
        if trade.partial_close_done and trade.partial_close_price:
            partial_pnl = MGC_SPECS.calculate_pnl(
                trade.entry_price,
                trade.partial_close_price,
                trade.contracts - trade.remaining_contracts,
                trade.is_long
            )
            pnl += partial_pnl
        
        logger.info(f"Exited trade: {reason}, Exit={exit_price:.2f}, PnL=${pnl:.2f}")
        
        # Record trade
        self.trade_history.record_trade(
            symbol=self.symbol,
            side="BUY" if trade.is_long else "SELL",
            quantity=trade.contracts,
            entry_price=trade.entry_price,
            exit_price=exit_price,
            pnl=pnl,
            entry_time=trade.entry_time,
            exit_time=datetime.now(self.range_calculator.et_tz)
        )
        
        self.active_trade = None
    
    def _check_time_windows(self):
        """Check time windows and place/cancel orders accordingly"""
        now_et = datetime.now(self.range_calculator.et_tz)
        current_hour = now_et.hour
        current_minute = now_et.minute
        
        # Check if it's 3 AM ET (London open) - time to place OCO orders
        if current_hour == self.range_calculator.LONDON_OPEN_HOUR and current_minute == 0:
            if not self.pending_order:
                # Calculate Asian range
                asian_range = self._calculate_asian_range()
                if asian_range:
                    # Create pending orders
                    order_time = now_et.replace(minute=0, second=0, microsecond=0)
                    self.pending_order = self.range_calculator.create_pending_orders(
                        asian_range, order_time
                    )
                    # Place OCO orders
                    self._place_oco_orders(self.pending_order)
        
        # Check if we're past 5 PM ET - cancel any pending orders
        if current_hour >= self.range_calculator.END_OF_DAY_HOUR:
            if self.pending_order:
                self._cancel_oco_orders(self.pending_order)
                self.pending_order = None
    
    def _update_risk_checks(self) -> bool:
        """Perform risk checks, return True if should stop"""
        # Reconcile positions with API
        if not self.dry_run and self.api_client.account_id:
            try:
                api_positions = self.api_client.get_positions(self.api_client.account_id)
                if api_positions:
                    self.position_manager.reconcile_with_api_positions(api_positions)
            except Exception as e:
                logger.debug(f"Error reconciling positions: {e}")
        
        total_pnl = self.position_manager.get_total_pnl()
        exposure = self.position_manager.net_exposure_dollars()
        
        # Update risk manager balance
        accounts = self.api_client.get_accounts()
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
        
        # Cancel pending orders
        if self.pending_order:
            self._cancel_oco_orders(self.pending_order)
            self.pending_order = None
        
        # Close active trade
        if self.active_trade and self.current_price:
            self._exit_trade(self.current_price, reason)
        
        # Close all positions via API
        if not self.dry_run and self.api_client.account_id:
            positions = self.api_client.get_positions(self.api_client.account_id)
            for position in positions:
                contract_id = position.get('contractId')
                if contract_id:
                    self.api_client.close_position(contract_id)
        
        self.position_manager.flatten_all()
        self.running = False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status"""
        return {
            "status": "running" if self.running else "stopped",
            "timestamp": datetime.now().isoformat(),
            "trading_enabled": not self.paused and self.running,
            "dry_run": self.dry_run,
            "symbol": self.symbol,
            "current_price": self.current_price,
            "has_pending_order": self.pending_order is not None,
            "has_active_trade": self.active_trade is not None,
            "asian_range": {
                "high": self.current_asian_range.asian_high if self.current_asian_range else None,
                "low": self.current_asian_range.asian_low if self.current_asian_range else None,
                "size": self.current_asian_range.range_size if self.current_asian_range else None,
            } if self.current_asian_range else None,
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
        logger.info("Starting Asian Range trading loop...")
        
        try:
            while self.running:
                if self.paused:
                    time_module.sleep(1)
                    continue
                
                # Check time windows and place orders
                self._check_time_windows()
                
                # Check for order fills
                self._check_order_fills()
                
                # Manage active trade
                self._manage_active_trade()
                
                # Risk checks
                if self._update_risk_checks():
                    break
                
                # Update historical data periodically
                if datetime.now().minute % 30 == 0:  # Every 30 minutes
                    self._load_historical_data()
                
                time_module.sleep(5)  # Check every 5 seconds
        
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        except Exception as e:
            logger.error(f"Error in trading loop: {e}", exc_info=True)
        finally:
            logger.info("Shutting down...")
            self.emergency_flatten("Shutdown")
            self.api_client.disconnect()

