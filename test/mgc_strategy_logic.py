"""
MGC Liquidity Sweep Strategy Logic
Implements the core strategy components from arbts.txt
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SwingPoint:
    """Represents a swing high or low"""
    index: int
    price: float
    timestamp: pd.Timestamp
    is_high: bool  # True for swing high, False for swing low


@dataclass
class LiquiditySweep:
    """Represents a detected liquidity sweep"""
    index: int
    swing_point: SwingPoint
    sweep_low: float  # The actual low of the sweep candle
    sweep_high: float  # The actual high of the sweep candle
    candle_close: float
    timestamp: pd.Timestamp


class SwingDetector:
    """Detects swing highs and lows using pivot length"""
    
    def __init__(self, pivot_length: int = 5):
        self.pivot_length = pivot_length
    
    def detect_swings(self, df: pd.DataFrame) -> Tuple[List[SwingPoint], List[SwingPoint]]:
        """
        Detect swing highs and lows
        
        Args:
            df: DataFrame with high, low, close columns
            
        Returns:
            (swing_highs, swing_lows) lists
        """
        swing_highs = []
        swing_lows = []
        
        for i in range(self.pivot_length, len(df) - self.pivot_length):
            # Check for swing high
            is_swing_high = True
            center_high = df.iloc[i]['high']
            
            for j in range(i - self.pivot_length, i + self.pivot_length + 1):
                if j != i and df.iloc[j]['high'] >= center_high:
                    is_swing_high = False
                    break
            
            if is_swing_high:
                swing_highs.append(SwingPoint(
                    index=i,
                    price=center_high,
                    timestamp=df.iloc[i]['timestamp'],
                    is_high=True
                ))
            
            # Check for swing low
            is_swing_low = True
            center_low = df.iloc[i]['low']
            
            for j in range(i - self.pivot_length, i + self.pivot_length + 1):
                if j != i and df.iloc[j]['low'] <= center_low:
                    is_swing_low = False
                    break
            
            if is_swing_low:
                swing_lows.append(SwingPoint(
                    index=i,
                    price=center_low,
                    timestamp=df.iloc[i]['timestamp'],
                    is_high=False
                ))
        
        return swing_highs, swing_lows


class LiquiditySweepDetector:
    """Detects liquidity sweeps"""
    
    def __init__(self, pivot_length: int = 5):
        self.swing_detector = SwingDetector(pivot_length)
    
    def detect_sweeps(
        self,
        df: pd.DataFrame,
        swing_points: Optional[List[SwingPoint]] = None
    ) -> List[LiquiditySweep]:
        """
        Detect liquidity sweeps in the data
        
        For longs: wick breaks swing low but closes above it
        For shorts: wick breaks swing high but closes below it
        
        Args:
            df: DataFrame with OHLC data
            swing_points: Pre-computed swing points (optional)
            
        Returns:
            List of detected liquidity sweeps
        """
        if swing_points is None:
            swing_highs, swing_lows = self.swing_detector.detect_swings(df)
        else:
            swing_highs = [s for s in swing_points if s.is_high]
            swing_lows = [s for s in swing_points if not s.is_high]
        
        sweeps = []
        
        # Detect long sweeps (sweep of swing low)
        for swing_low in swing_lows:
            # Look for candles after the swing low that sweep it
            for i in range(swing_low.index + 1, len(df)):
                candle = df.iloc[i]
                
                # Check if wick breaks swing low but closes above it
                if candle['low'] < swing_low.price and candle['close'] > swing_low.price:
                    sweeps.append(LiquiditySweep(
                        index=i,
                        swing_point=swing_low,
                        sweep_low=candle['low'],
                        sweep_high=candle['high'],
                        candle_close=candle['close'],
                        timestamp=candle['timestamp']
                    ))
                    break  # Only take first sweep of each swing point
        
        # Detect short sweeps (sweep of swing high)
        for swing_high in swing_highs:
            # Look for candles after the swing high that sweep it
            for i in range(swing_high.index + 1, len(df)):
                candle = df.iloc[i]
                
                # Check if wick breaks swing high but closes below it
                if candle['high'] > swing_high.price and candle['close'] < swing_high.price:
                    sweeps.append(LiquiditySweep(
                        index=i,
                        swing_point=swing_high,
                        sweep_low=candle['low'],
                        sweep_high=candle['high'],
                        candle_close=candle['close'],
                        timestamp=candle['timestamp']
                    ))
                    break  # Only take first sweep of each swing point
        
        return sweeps


class ConfirmationCandleDetector:
    """Detects confirmation candles after liquidity sweeps"""
    
    def __init__(self, body_ratio: float = 0.5):
        """
        Initialize confirmation candle detector
        
        Args:
            body_ratio: Minimum body ratio for confirmation (default 0.5 = 50%)
        """
        self.body_ratio = body_ratio
    
    def is_confirmation_candle(self, candle: pd.Series, is_long: bool) -> bool:
        """
        Check if candle is a confirmation candle
        
        Confirmation requires:
        - Body > 50% of candle range
        - Candle is bullish (for longs) or bearish (for shorts)
        
        Args:
            candle: Series with open, high, low, close
            is_long: True for long setup, False for short
            
        Returns:
            True if confirmation candle
        """
        body = abs(candle['close'] - candle['open'])
        range_ = candle['high'] - candle['low']
        
        if range_ == 0:
            return False
        
        body_ratio = body / range_
        
        if body_ratio <= self.body_ratio:
            return False
        
        # Check direction
        is_bullish = candle['close'] > candle['open']
        
        if is_long:
            return is_bullish
        else:
            return not is_bullish
    
    def find_confirmation(
        self,
        df: pd.DataFrame,
        sweep_index: int,
        is_long: bool
    ) -> Optional[int]:
        """
        Find confirmation candle after a liquidity sweep
        
        Args:
            df: DataFrame with OHLC data
            sweep_index: Index of the sweep candle
            is_long: True for long setup, False for short
            
        Returns:
            Index of confirmation candle, or None if not found
        """
        # Look at next candle after sweep
        if sweep_index + 1 >= len(df):
            return None
        
        next_candle = df.iloc[sweep_index + 1]
        
        if self.is_confirmation_candle(next_candle, is_long):
            return sweep_index + 1
        
        return None


class TrendFilter:
    """50 EMA trend filter on 15m timeframe"""
    
    def __init__(self, period: int = 50):
        self.period = period
    
    def calculate_ema(self, df: pd.DataFrame) -> pd.Series:
        """Calculate 50 EMA"""
        return df['close'].ewm(span=self.period, adjust=False).mean()
    
    def is_uptrend(self, df: pd.DataFrame, index: int) -> bool:
        """Check if price is above 50 EMA (uptrend)"""
        if index < self.period:
            return False  # Not enough data
        
        ema = self.calculate_ema(df)
        return df.iloc[index]['close'] > ema.iloc[index]
    
    def is_downtrend(self, df: pd.DataFrame, index: int) -> bool:
        """Check if price is below 50 EMA (downtrend)"""
        if index < self.period:
            return False  # Not enough data
        
        ema = self.calculate_ema(df)
        return df.iloc[index]['close'] < ema.iloc[index]


class PositionSizer:
    """Calculates position size based on fixed risk per trade"""
    
    def __init__(
        self, 
        contracts_per_trade: Optional[int] = None,
        risk_per_trade: float = 50.0, 
        atr_period: int = 5, 
        atr_multiplier: float = 5.0
    ):
        """
        Initialize position sizer
        
        Args:
            contracts_per_trade: Fixed number of contracts per trade (None = use risk-based sizing)
            risk_per_trade: Dollar amount to risk per trade (only used if contracts_per_trade is None)
            atr_period: ATR period for stop sizing (default 5)
            atr_multiplier: ATR multiplier for stop distance (default 5.0)
        """
        self.contracts_per_trade = contracts_per_trade
        self.risk_per_trade = risk_per_trade
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
    
    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate ATR"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        return true_range.rolling(window=self.atr_period).mean()
    
    def calculate_position_size(
        self,
        df: pd.DataFrame,
        entry_index: int,
        stop_price: float,
        is_long: bool
    ) -> int:
        """
        Calculate number of contracts
        
        If contracts_per_trade is set, returns that fixed amount.
        Otherwise, calculates based on risk_per_trade.
        
        Args:
            df: DataFrame with OHLC data
            entry_index: Index of entry candle
            stop_price: Stop loss price
            is_long: True for long, False for short
            
        Returns:
            Number of contracts
        """
        # If fixed contracts specified, use that
        if self.contracts_per_trade is not None and self.contracts_per_trade > 0:
            return self.contracts_per_trade
        
        # Otherwise, use risk-based sizing
        entry_price = df.iloc[entry_index]['close']
        
        # Calculate stop distance in price
        if is_long:
            stop_distance = entry_price - stop_price
        else:
            stop_distance = stop_price - entry_price
        
        # Import contract specs
        from mgc_contract_specs import MGC_SPECS
        
        # Convert to ticks (MGC tick size is 0.10)
        stop_ticks = MGC_SPECS.price_to_ticks(stop_distance)
        
        # Calculate ATR-based stop
        atr = self.calculate_atr(df)
        if entry_index < len(atr) and not pd.isna(atr.iloc[entry_index]):
            # Convert ATR to ticks
            atr_price = atr.iloc[entry_index] * self.atr_multiplier
            atr_stop_ticks = MGC_SPECS.price_to_ticks(atr_price)
        else:
            atr_stop_ticks = stop_ticks
        
        # Use max of ATR stop and actual stop
        final_stop_ticks = max(atr_stop_ticks, stop_ticks)
        
        # Ensure minimum stop (at least 1 tick)
        if final_stop_ticks < 1:
            final_stop_ticks = 1
        
        # Calculate contracts: risk_per_trade / (stop_ticks * tick_value)
        # Example: $50 risk / (5 ticks * $1/tick) = 10 contracts
        contracts = int(self.risk_per_trade / (final_stop_ticks * MGC_SPECS.tick_value))
        
        # Minimum 1 contract
        return max(1, contracts)

