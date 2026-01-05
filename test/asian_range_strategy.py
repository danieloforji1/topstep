"""
Asian Range Breakout Strategy for MGC Gold Futures
Implements the exact strategy from arbts.txt specification
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, time
import pytz

from mgc_contract_specs import MGC_SPECS


@dataclass
class AsianRange:
    """Represents the Asian session range for a trading day"""
    date: pd.Timestamp
    asian_high: float
    asian_low: float
    range_size: float
    asian_high_time: pd.Timestamp
    asian_low_time: pd.Timestamp


@dataclass
class PendingOrder:
    """Represents a pending OCO order"""
    buy_stop_price: float
    sell_stop_price: float
    asian_range: AsianRange
    order_time: pd.Timestamp  # 3 AM ET


class AsianRangeCalculator:
    """Calculates Asian session ranges"""
    
    # Time windows (ET timezone)
    ASIAN_START_HOUR = 20  # 8:00 PM ET
    ASIAN_START_MINUTE = 0
    ASIAN_END_HOUR = 2  # 2:00 AM ET (next day)
    ASIAN_END_MINUTE = 0
    
    LONDON_OPEN_HOUR = 3  # 3:00 AM ET
    LONDON_OPEN_MINUTE = 0
    
    NY_CLOSE_HOUR = 12  # 12:00 PM ET
    NY_CLOSE_MINUTE = 0
    
    def __init__(self, et_timezone: str = "America/New_York"):
        self.et_tz = pytz.timezone(et_timezone)
    
    def is_asian_session(self, timestamp: pd.Timestamp) -> bool:
        """
        Check if timestamp is within Asian session (8 PM - 2 AM ET)
        
        Args:
            timestamp: Timestamp to check (must be in ET)
            
        Returns:
            True if within Asian session
        """
        hour = timestamp.hour
        
        # Asian session: 20:00 (8 PM) to 23:59, or 00:00 to 01:59 (2 AM)
        return hour >= 20 or hour < 2
    
    def is_london_open(self, timestamp: pd.Timestamp) -> bool:
        """Check if timestamp is at or after London open (3 AM ET)"""
        hour = timestamp.hour
        minute = timestamp.minute
        return hour > 3 or (hour == 3 and minute >= 0)
    
    def is_before_ny_close(self, timestamp: pd.Timestamp) -> bool:
        """Check if timestamp is before NY close (12 PM ET)"""
        hour = timestamp.hour
        minute = timestamp.minute
        return hour < 12 or (hour == 12 and minute < 0)
    
    def calculate_asian_range(self, df: pd.DataFrame, date: pd.Timestamp) -> Optional[AsianRange]:
        """
        Calculate Asian range for a specific date
        
        Asian session: 8 PM previous day to 2 AM current day
        
        Args:
            df: DataFrame with OHLC data and ET timestamps
            date: Trading date (the day we're trading)
            
        Returns:
            AsianRange object or None if insufficient data
        """
        # Asian session is 8 PM previous day to 2 AM current day
        # So for date 2025-01-15, Asian session is 2025-01-14 20:00 to 2025-01-15 02:00
        
        # Get date part only (remove time)
        if isinstance(date, pd.Timestamp):
            trading_date = date.date()
        else:
            trading_date = date
        
        prev_day = pd.Timestamp.combine(trading_date, time(0, 0)) - pd.Timedelta(days=1)
        asian_start = pd.Timestamp.combine(prev_day.date(), time(20, 0))
        asian_end = pd.Timestamp.combine(trading_date, time(2, 0))
        
        # Localize to ET timezone
        if df['timestamp'].dt.tz is not None:
            asian_start = asian_start.tz_localize(self.et_tz)
            asian_end = asian_end.tz_localize(self.et_tz)
        
        # Filter data for Asian session
        mask = (df['timestamp'] >= asian_start) & (df['timestamp'] < asian_end)
        asian_data = df[mask]
        
        if len(asian_data) == 0:
            return None
        
        # Find high and low during Asian session
        asian_high = asian_data['high'].max()
        asian_low = asian_data['low'].min()
        
        # Find timestamps of high and low
        high_idx = asian_data['high'].idxmax()
        low_idx = asian_data['low'].idxmin()
        
        asian_high_time = asian_data.loc[high_idx, 'timestamp']
        asian_low_time = asian_data.loc[low_idx, 'timestamp']
        
        range_size = asian_high - asian_low
        
        return AsianRange(
            date=date,
            asian_high=asian_high,
            asian_low=asian_low,
            range_size=range_size,
            asian_high_time=asian_high_time,
            asian_low_time=asian_low_time
        )
    
    def create_pending_orders(self, asian_range: AsianRange, order_time: pd.Timestamp) -> PendingOrder:
        """
        Create OCO pending orders at London open (3 AM ET)
        
        Buy Stop: Asian High + 1 tick
        Sell Stop: Asian Low - 1 tick
        
        Args:
            asian_range: Calculated Asian range
            order_time: Time to place orders (3 AM ET)
            
        Returns:
            PendingOrder object
        """
        # Round to ticks
        asian_high = MGC_SPECS.round_to_tick(asian_range.asian_high)
        asian_low = MGC_SPECS.round_to_tick(asian_range.asian_low)
        
        # Buy stop: Asian High + 1 tick
        buy_stop_price = asian_high + MGC_SPECS.tick_size
        
        # Sell stop: Asian Low - 1 tick
        sell_stop_price = asian_low - MGC_SPECS.tick_size
        
        return PendingOrder(
            buy_stop_price=buy_stop_price,
            sell_stop_price=sell_stop_price,
            asian_range=asian_range,
            order_time=order_time
        )


class PositionManager:
    """Manages position with break-even and take profit logic"""
    
    def __init__(self, tp_multiplier: float = 1.5, sl_buffer_ticks: int = 3):
        """
        Initialize position manager
        
        Args:
            tp_multiplier: Take profit multiplier (1x to 2x range size)
            sl_buffer_ticks: Stop loss buffer in ticks (places SL further from range to avoid wicks)
        """
        self.tp_multiplier = tp_multiplier
        self.sl_buffer_ticks = sl_buffer_ticks
    
    def calculate_stop_loss(self, entry_price: float, asian_range: AsianRange, is_long: bool) -> float:
        """
        Calculate stop loss (opposite side of Asian range with buffer)
        
        Long trade → stop below Asian Low (with buffer to avoid wicks)
        Short trade → stop above Asian High (with buffer to avoid wicks)
        
        Args:
            entry_price: Entry price
            asian_range: Asian range
            is_long: True for long, False for short
            
        Returns:
            Stop loss price
        """
        # Calculate buffer distance
        buffer_distance = self.sl_buffer_ticks * MGC_SPECS.tick_size
        
        if is_long:
            # Long: stop below Asian Low with buffer
            # Place stop at Asian Low - buffer (gives room for wicks/manipulation)
            stop_loss = asian_range.asian_low - buffer_distance
        else:
            # Short: stop above Asian High with buffer
            # Place stop at Asian High + buffer (gives room for wicks/manipulation)
            stop_loss = asian_range.asian_high + buffer_distance
        
        return MGC_SPECS.round_to_tick(stop_loss)
    
    def calculate_take_profit(self, entry_price: float, asian_range: AsianRange, is_long: bool) -> float:
        """
        Calculate take profit (1x to 2x range size)
        
        Args:
            entry_price: Entry price
            asian_range: Asian range
            is_long: True for long, False for short
            
        Returns:
            Take profit price
        """
        range_size = asian_range.range_size
        tp_distance = range_size * self.tp_multiplier
        
        if is_long:
            take_profit = entry_price + tp_distance
        else:
            take_profit = entry_price - tp_distance
        
        return MGC_SPECS.round_to_tick(take_profit)
    
    def calculate_risk_amount(self, entry_price: float, stop_loss: float, contracts: int, is_long: bool) -> float:
        """Calculate risk amount (R)"""
        return MGC_SPECS.calculate_risk_amount(entry_price, stop_loss, contracts, is_long)
    
    def should_move_to_breakeven(self, entry_price: float, current_price: float, stop_loss: float, 
                                 asian_range: AsianRange, is_long: bool) -> Tuple[bool, float]:
        """
        Check if stop should move to break-even (+1R profit)
        
        Returns:
            (should_move, new_stop_price)
        """
        # Calculate 1R profit level
        range_size = asian_range.range_size
        
        if is_long:
            breakeven_price = entry_price + range_size  # +1R
            if current_price >= breakeven_price:
                return True, entry_price  # Move SL to entry
        else:
            breakeven_price = entry_price - range_size  # +1R
            if current_price <= breakeven_price:
                return True, entry_price  # Move SL to entry
        
        return False, stop_loss

