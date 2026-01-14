"""
Optimal Stopping Theory Strategy
Uses optimal stopping theory to select best entry/exit points
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
from framework.base_strategy import BaseStrategy, Signal, ExitReason, MarketData


def calculate_atr(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    """
    Calculate Average True Range (ATR) from DataFrame
    
    Args:
        df: DataFrame with columns: high, low, close
        period: ATR period (default: 14)
        
    Returns:
        ATR value or None if insufficient data
    """
    if len(df) < period + 1:
        return None
    
    # Calculate True Range
    df = df.copy()
    df['prev_close'] = df['close'].shift(1)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['prev_close'])
    df['tr3'] = abs(df['low'] - df['prev_close'])
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Calculate ATR as simple moving average of true ranges
    atr = df['true_range'].tail(period).mean()
    
    return atr if not pd.isna(atr) else None


class OptimalStoppingStrategy(BaseStrategy):
    """
    Optimal Stopping Theory Strategy
    
    Uses the "Secretary Problem" variant:
    - Wait until you've seen 37% of opportunities
    - Then take the next one that's better than all previous
    
    This strategy:
    1. Scores each potential entry
    2. Maintains a "best so far" threshold
    3. Only enters when current score > threshold AND > 37% of opportunities seen
    4. Uses dynamic programming for optimal exit timing
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Optimal Stopping strategy
        
        Config parameters:
            lookback_window: Window for opportunity counting (default: 100)
            min_opportunities_seen: Minimum opportunities before entry (default: 37)
            score_threshold: Minimum score to consider entry (default: 0.6)
            momentum_weight: Weight for momentum factor (default: 0.4)
            mean_reversion_weight: Weight for mean reversion factor (default: 0.3)
            volatility_weight: Weight for volatility factor (default: 0.3)
            atr_period: ATR period for volatility (default: 14)
            atr_multiplier_stop: ATR multiplier for stop loss (default: 1.5)
            atr_multiplier_target: ATR multiplier for take profit (default: 2.0)
            risk_per_trade: Dollar risk per trade (default: 100.0)
            max_hold_bars: Maximum bars to hold position (default: 50)
        """
        super().__init__(config)
        
        self.lookback_window = config.get('lookback_window', 100)
        self.min_opportunities_seen = config.get('min_opportunities_seen', 37)
        self.score_threshold = config.get('score_threshold', 0.6)
        
        # Factor weights
        self.momentum_weight = config.get('momentum_weight', 0.4)
        self.mean_reversion_weight = config.get('mean_reversion_weight', 0.3)
        self.volatility_weight = config.get('volatility_weight', 0.3)
        
        # ATR settings
        self.atr_period = config.get('atr_period', 14)
        self.atr_multiplier_stop = config.get('atr_multiplier_stop', 1.5)
        self.atr_multiplier_target = config.get('atr_multiplier_target', 2.0)
        self.risk_per_trade = config.get('risk_per_trade', 100.0)
        self.max_hold_bars = config.get('max_hold_bars', 50)
        
        # Signal reversal exit settings
        self.signal_reversal_threshold = config.get('signal_reversal_threshold', 0.8)  # Score threshold for reversal
        self.signal_reversal_min_profit_pct = config.get('signal_reversal_min_profit_pct', 0.005)  # Min profit % to exit on reversal
        self.signal_reversal_min_hold_pct = config.get('signal_reversal_min_hold_pct', 0.25)  # Min hold time % before reversal exit
        
        # Trailing stop settings
        self.use_trailing_stop = config.get('use_trailing_stop', False)
        self.trailing_stop_atr_multiplier = config.get('trailing_stop_atr_multiplier', 0.5)
        self.trailing_stop_activation_pct = config.get('trailing_stop_activation_pct', 0.001)  # Activate after 0.1% profit (more aggressive)
        
        # Partial profit taking settings
        self.use_partial_profit = config.get('use_partial_profit', False)
        self.partial_profit_pct = config.get('partial_profit_pct', 0.5)  # Take 50% at first target
        self.partial_profit_target_atr = config.get('partial_profit_target_atr', 0.75)  # Take profit at 0.75x ATR
        
        # State
        self.opportunities_seen = 0
        self.best_score_so_far = -float('inf')
        self.entry_bar_index = None
        self.opportunity_scores: List[float] = []
        
        # Trailing stop state
        self.highest_price_since_entry: Optional[float] = None
        self.lowest_price_since_entry: Optional[float] = None
        self.initial_stop_loss: Optional[float] = None
        self.trailing_stop_activated: bool = False
    
    def get_required_data(self) -> List[str]:
        """Optimal stopping needs OHLCV data"""
        return ["OHLCV"]
    
    def _calculate_momentum_score(self, df: pd.DataFrame, current_index: int) -> float:
        """Calculate momentum score (-1 to +1)"""
        if current_index < 5:
            return 0.0
        
        # Short-term momentum (5 bars)
        short_ma = df.iloc[current_index-5:current_index]['close'].mean()
        current_price = df.iloc[current_index]['close']
        
        # Medium-term momentum (20 bars)
        if current_index >= 20:
            medium_ma = df.iloc[current_index-20:current_index]['close'].mean()
            momentum_pct = (current_price - medium_ma) / medium_ma
        else:
            momentum_pct = (current_price - short_ma) / short_ma
        
        # Normalize to -1 to +1
        # For futures, typical moves are 0.1-1%, so scale appropriately
        # Multiply by 50 to make 1% move = 0.5 score, then tanh to bound
        momentum_score = np.tanh(momentum_pct * 50)
        
        return momentum_score
    
    def _calculate_mean_reversion_score(self, df: pd.DataFrame, current_index: int) -> float:
        """Calculate mean reversion score (-1 to +1)"""
        if current_index < 20:
            return 0.0
        
        # Calculate VWAP or simple moving average
        window = min(20, current_index)
        ma = df.iloc[current_index-window:current_index]['close'].mean()
        current_price = df.iloc[current_index]['close']
        
        # Deviation from mean
        deviation_pct = (current_price - ma) / ma
        
        # Mean reversion: negative deviation = positive score (buy opportunity)
        # Positive deviation = negative score (sell opportunity)
        # Scale similarly to momentum
        mean_rev_score = -np.tanh(deviation_pct * 50)
        
        return mean_rev_score
    
    def _calculate_volatility_score(self, df: pd.DataFrame, current_index: int) -> float:
        """Calculate volatility score (-1 to +1)"""
        if current_index < self.atr_period:
            return 0.0
        
        # Calculate ATR
        atr = calculate_atr(
            df.iloc[current_index-self.atr_period:current_index],
            period=self.atr_period
        )
        
        if atr is None or atr == 0:
            return 0.0
        
        # Calculate historical ATR for comparison
        if current_index >= self.atr_period * 2:
            historical_atr = calculate_atr(
                df.iloc[current_index-self.atr_period*2:current_index-self.atr_period],
                period=self.atr_period
            )
            if historical_atr and historical_atr > 0:
                vol_ratio = atr / historical_atr
                # Low volatility = good for mean reversion
                # High volatility = good for momentum
                # Scale to -0.5 to +0.5 range
                vol_score = np.tanh((1.0 - vol_ratio) * 2) * 0.5
                return vol_score
        
        return 0.0
    
    def _calculate_entry_score(self, df: pd.DataFrame, current_index: int) -> float:
        """
        Calculate composite entry score
        
        Returns:
            Score from -1 to +1 (positive = long, negative = short)
        """
        momentum = self._calculate_momentum_score(df, current_index)
        mean_rev = self._calculate_mean_reversion_score(df, current_index)
        volatility = self._calculate_volatility_score(df, current_index)
        
        # Weighted combination
        raw_score = (
            self.momentum_weight * momentum +
            self.mean_reversion_weight * mean_rev +
            self.volatility_weight * volatility
        )
        
        # Use percentile-based scoring: compare to historical scores in lookback window
        # This makes the score relative to recent market conditions
        if len(self.opportunity_scores) >= 20:  # Need some history
            historical_scores = np.array(self.opportunity_scores[-50:])  # Last 50 scores
            if len(historical_scores) > 0:
                # Calculate percentile of current score relative to history
                abs_historical = np.abs(historical_scores)
                abs_current = abs(raw_score)
                
                if abs_historical.max() > 0:
                    percentile = np.sum(abs_historical <= abs_current) / len(abs_historical)
                    # Convert percentile (0-1) to score (-1 to +1)
                    # High percentile (e.g., 0.8) means current score is stronger than 80% of recent scores
                    # Scale to -1 to +1 range, with sign preserved
                    score = np.sign(raw_score) * (percentile * 2.0 - 1.0)  # Maps 0.5 percentile to 0, 1.0 to +1, 0.0 to -1
                else:
                    score = raw_score
            else:
                score = raw_score
        else:
            # Not enough history, use raw score but scale it up
            score = raw_score * 10.0  # Scale up for early bars
        
        # Add additional scaling based on strength of signals
        # If all components agree, boost the score
        component_signs = [np.sign(momentum), np.sign(mean_rev), np.sign(volatility)]
        if len(set(component_signs)) == 1 and component_signs[0] != 0:
            # All components agree on direction, boost by 1.2x
            score = score * 1.2
        
        # Clamp to -1 to +1 range
        score = np.clip(score, -1.0, 1.0)
        
        return score
    
    def generate_signal(
        self,
        market_data: MarketData,
        historical_data: pd.DataFrame,
        current_position: Optional[Any] = None
    ) -> Optional[Signal]:
        """
        Generate trading signal using optimal stopping theory
        """
        if current_position is not None:
            return None  # Already in position
        
        # Find current index
        current_index = len(historical_data) - 1
        if current_index < self.lookback_window:
            return None  # Not enough data
        
        # Calculate entry score
        score = self._calculate_entry_score(historical_data, current_index)
        
        # Track opportunities
        self.opportunities_seen += 1
        self.opportunity_scores.append(score)
        
        # Keep only recent opportunities
        if len(self.opportunity_scores) > self.lookback_window:
            self.opportunity_scores.pop(0)
            self.opportunities_seen = len(self.opportunity_scores)
        
        # Update best score so far (use rolling window of recent scores)
        # Only consider scores from the lookback window
        # Reset best score periodically to avoid it getting stuck too high
        if len(self.opportunity_scores) > 0:
            # Reset best score every 50 opportunities to allow new entries
            if self.opportunities_seen % 50 == 0:
                self.best_score_so_far = -float('inf')
            
            recent_scores = self.opportunity_scores[-min(30, len(self.opportunity_scores)):]  # Last 30 scores
            if recent_scores:
                best_recent = max(recent_scores, key=abs)
                # Only update if significantly better (avoid constantly raising the bar)
                if abs(best_recent) > abs(self.best_score_so_far) * 1.05:
                    self.best_score_so_far = best_recent
        
        # Optimal stopping rule:
        # 1. Must have seen at least min_opportunities_seen
        # 2. Current score must exceed threshold
        # 3. Current score must be better than best so far (or close)
        
        opportunities_pct = self.opportunities_seen / self.lookback_window if self.lookback_window > 0 else 0
        
        # Check entry conditions
        if abs(score) < self.score_threshold:
            return None  # Score too low
        
        if opportunities_pct < (self.min_opportunities_seen / self.lookback_window):
            return None  # Haven't seen enough opportunities yet
        
        # Check if this is in the top percentile of recent scores
        # Instead of comparing to absolute best, use percentile ranking
        if len(self.opportunity_scores) >= 20:  # Need some history
            recent_scores_abs = [abs(s) for s in self.opportunity_scores[-30:]]  # Last 30 scores
            if recent_scores_abs:
                # Calculate what percentile the current score is
                percentile = sum(1 for s in recent_scores_abs if s <= abs(score)) / len(recent_scores_abs)
                
                # Only enter if score is in top percentile of recent scores
                # RELAXED: Start at 50% percentile, lower as we see more opportunities
                min_percentile = 0.5  # REDUCED from 0.7 - more permissive
                if opportunities_pct > 0.4:  # After seeing 40% of opportunities
                    min_percentile = 0.4  # Lower the bar
                if opportunities_pct > 0.6:  # After seeing 60% of opportunities
                    min_percentile = 0.35  # Lower the bar more
                if opportunities_pct > 0.8:  # After seeing 80% of opportunities
                    min_percentile = 0.3  # Lower the bar even more
                
                if percentile < min_percentile:
                    return None  # Not in top percentile
        else:
            # Not enough history, use simple best score comparison
            if abs(self.best_score_so_far) > 0 and abs(score) < abs(self.best_score_so_far) * 0.8:
                return None
        
        # Generate signal
        direction = "LONG" if score > 0 else "SHORT"
        
        # Calculate ATR for stop/target
        if current_index >= self.atr_period:
            atr = calculate_atr(
                historical_data.iloc[current_index-self.atr_period:current_index],
                period=self.atr_period
            )
        else:
            atr = None
        
        if atr and atr > 0:
            # Use tighter stops/targets - ATR multipliers should be smaller
            # For MES, typical ATR is 10-20 points, so 1.0x ATR = 10-20 points stop
            # We want tighter stops and achievable targets to improve risk/reward
            if direction == "LONG":
                stop_loss = market_data.close - (atr * self.atr_multiplier_stop)
                take_profit = market_data.close + (atr * self.atr_multiplier_target)
            else:
                stop_loss = market_data.close + (atr * self.atr_multiplier_stop)
                take_profit = market_data.close - (atr * self.atr_multiplier_target)
            
            # Ensure minimum stop distance (at least 5 points for MES)
            min_stop_distance = 5.0
            if direction == "LONG":
                if (market_data.close - stop_loss) < min_stop_distance:
                    stop_loss = market_data.close - min_stop_distance
                # Ensure target is at least 1.2x the stop distance (minimum R:R)
                min_target_distance = (market_data.close - stop_loss) * 1.2
                if (take_profit - market_data.close) < min_target_distance:
                    take_profit = market_data.close + min_target_distance
            else:
                if (stop_loss - market_data.close) < min_stop_distance:
                    stop_loss = market_data.close + min_stop_distance
                # Ensure target is at least 1.2x the stop distance (minimum R:R)
                min_target_distance = (stop_loss - market_data.close) * 1.2
                if (market_data.close - take_profit) < min_target_distance:
                    take_profit = market_data.close - min_target_distance
        else:
            # Fallback: use tighter percentage-based stops
            if direction == "LONG":
                stop_loss = market_data.close * 0.995  # 0.5% stop
                take_profit = market_data.close * 1.01  # 1% target
            else:
                stop_loss = market_data.close * 1.005  # 0.5% stop
                take_profit = market_data.close * 0.99  # 1% target
        
        # Reset best score after entry
        self.best_score_so_far = -float('inf')
        self.entry_bar_index = current_index
        
        # Reset trailing stop state for new position
        self.highest_price_since_entry = None
        self.lowest_price_since_entry = None
        self.initial_stop_loss = None
        self.trailing_stop_activated = False
        
        return Signal(
            timestamp=market_data.timestamp,
            direction=direction,
            entry_price=market_data.close,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=abs(score),
            metadata={
                'score': score,
                'opportunities_seen': self.opportunities_seen,
                'momentum': self._calculate_momentum_score(historical_data, current_index),
                'mean_reversion': self._calculate_mean_reversion_score(historical_data, current_index),
                'volatility': self._calculate_volatility_score(historical_data, current_index)
            }
        )
    
    def calculate_position_size(
        self,
        signal: Signal,
        account_equity: float,
        market_data: MarketData,
        historical_data: pd.DataFrame
    ) -> int:
        """Calculate position size based on risk"""
        if signal.stop_loss is None:
            return 1
        
        # Calculate risk per contract
        risk_per_contract = abs(market_data.close - signal.stop_loss)
        
        if risk_per_contract == 0:
            return 1
        
        # Calculate contracts based on risk per trade
        # For futures, need to convert price risk to dollar risk
        # Assuming tick_value = 5.0 and tick_size = 0.25
        tick_size = 0.25
        tick_value = 5.0
        risk_ticks = risk_per_contract / tick_size
        risk_dollars = risk_ticks * tick_value
        
        if risk_dollars == 0:
            return 1
        
        contracts = int(self.risk_per_trade / risk_dollars)
        return max(1, min(contracts, 10))  # Limit to 10 contracts
    
    def check_exit(
        self,
        position: Any,
        market_data: MarketData,
        historical_data: pd.DataFrame
    ) -> Optional[ExitReason]:
        """Check if position should be exited using optimal stopping"""
        if position is None or self.entry_bar_index is None:
            return None
        
        current_index = len(historical_data) - 1
        bars_held = current_index - self.entry_bar_index
        
        # Initialize trailing stop tracking
        if self.highest_price_since_entry is None:
            self.highest_price_since_entry = market_data.high if position.is_long else market_data.low
            self.lowest_price_since_entry = market_data.low if position.is_long else market_data.high
            self.initial_stop_loss = position.stop_loss
            self.trailing_stop_activated = False
        
        # Update highest/lowest price since entry
        # For LONG: track highest price (best profit point)
        # For SHORT: track lowest price (best profit point)
        if position.is_long:
            self.highest_price_since_entry = max(self.highest_price_since_entry, market_data.high)
            self.lowest_price_since_entry = min(self.lowest_price_since_entry, market_data.low)
        else:  # SHORT
            # For shorts, lowest price = best profit, highest price = worst
            self.lowest_price_since_entry = min(self.lowest_price_since_entry, market_data.low)
            self.highest_price_since_entry = max(self.highest_price_since_entry, market_data.high)
        
        # Apply trailing stop if enabled
        if self.use_trailing_stop:
            # Calculate current profit percentage
            if position.is_long:
                current_profit_pct = (market_data.close - position.entry_price) / position.entry_price
                # Activate trailing stop after profit threshold
                if current_profit_pct >= self.trailing_stop_activation_pct:
                    self.trailing_stop_activated = True
                    # Calculate ATR for trailing stop distance
                    if current_index >= self.atr_period:
                        atr = calculate_atr(
                            historical_data.iloc[current_index-self.atr_period:current_index],
                            period=self.atr_period
                        )
                        if atr and atr > 0:
                            # Trail stop behind highest price (best profit point)
                            trailing_stop = self.highest_price_since_entry - (atr * self.trailing_stop_atr_multiplier)
                            # Only move stop up, never down (protect profits)
                            if trailing_stop > position.stop_loss:
                                position.stop_loss = trailing_stop
            else:  # SHORT
                current_profit_pct = (position.entry_price - market_data.close) / position.entry_price
                # Activate trailing stop after profit threshold
                if current_profit_pct >= self.trailing_stop_activation_pct:
                    self.trailing_stop_activated = True
                    # Calculate ATR for trailing stop distance
                    if current_index >= self.atr_period:
                        atr = calculate_atr(
                            historical_data.iloc[current_index-self.atr_period:current_index],
                            period=self.atr_period
                        )
                        if atr and atr > 0:
                            # Trail stop behind lowest price (best profit point for shorts)
                            trailing_stop = self.lowest_price_since_entry + (atr * self.trailing_stop_atr_multiplier)
                            # Only move stop down (lower price = better for shorts), never up
                            # For shorts, stop_loss is above entry, so lower = better
                            if trailing_stop < position.stop_loss:
                                position.stop_loss = trailing_stop
        
        # Check stop loss FIRST (risk management priority)
        if position.stop_loss:
            is_trailing_stop = self.use_trailing_stop and self.trailing_stop_activated and position.stop_loss != self.initial_stop_loss
            if position.is_long and market_data.low <= position.stop_loss:
                return ExitReason(
                    reason="TRAILING_STOP" if is_trailing_stop else "STOP_LOSS",
                    timestamp=market_data.timestamp
                )
            elif not position.is_long and market_data.high >= position.stop_loss:
                return ExitReason(
                    reason="TRAILING_STOP" if is_trailing_stop else "STOP_LOSS",
                    timestamp=market_data.timestamp
                )
        
        # Check for partial profit taking (before full take profit and other exits)
        # This allows us to lock in profits early
        if self.use_partial_profit and not position.partial_profit_taken:
            # Use ATR-based partial profit with tighter target
            # Calculate partial profit target based on ATR (more adaptive)
            if current_index >= self.atr_period:
                atr = calculate_atr(
                    historical_data.iloc[current_index-self.atr_period:current_index],
                    period=self.atr_period
                )
                if atr and atr > 0:
                    # Use smaller ATR multiplier for partial profit (0.4x ATR = ~5-8 points for MES)
                    partial_target_atr_mult = 0.4
                    if position.is_long:
                        partial_target = position.entry_price + (atr * partial_target_atr_mult)
                        # Ensure minimum distance (at least 3 points)
                        if (partial_target - position.entry_price) < 3.0:
                            partial_target = position.entry_price + 3.0
                        if market_data.high >= partial_target:
                            return ExitReason(
                                reason="PARTIAL_PROFIT",
                                timestamp=market_data.timestamp,
                                metadata={
                                    'partial_pct': self.partial_profit_pct,
                                    'target_price': partial_target
                                }
                            )
                    else:  # SHORT
                        partial_target = position.entry_price - (atr * partial_target_atr_mult)
                        # Ensure minimum distance (at least 3 points)
                        if (position.entry_price - partial_target) < 3.0:
                            partial_target = position.entry_price - 3.0
                        if market_data.low <= partial_target:
                            return ExitReason(
                                reason="PARTIAL_PROFIT",
                                timestamp=market_data.timestamp,
                                metadata={
                                    'partial_pct': self.partial_profit_pct,
                                    'target_price': partial_target
                                }
                            )
        
        # Check full take profit
        if position.take_profit:
            if position.is_long and market_data.high >= position.take_profit:
                return ExitReason(
                    reason="TAKE_PROFIT",
                    timestamp=market_data.timestamp
                )
            elif not position.is_long and market_data.low <= position.take_profit:
                return ExitReason(
                    reason="TAKE_PROFIT",
                    timestamp=market_data.timestamp
                )
        
        # Time-based exit (max hold) - check after profit opportunities
        if bars_held >= self.max_hold_bars:
            return ExitReason(
                reason="TIME_STOP",
                timestamp=market_data.timestamp,
                metadata={'bars_held': bars_held}
            )
        
        # Optimal exit: if score reverses significantly
        # Only exit on reversal if we're significantly in profit AND reversal is very strong
        # This allows stops/targets to work properly
        if current_index >= self.lookback_window:
            current_score = self._calculate_entry_score(historical_data, current_index)
            
            # Calculate current P&L
            if position.is_long:
                current_pnl_pct = (market_data.close - position.entry_price) / position.entry_price
            else:
                current_pnl_pct = (position.entry_price - market_data.close) / position.entry_price
            
            # Only exit on reversal if:
            # 1. We're in profit (configurable threshold), AND
            # 2. Reversal is strong (configurable threshold), AND
            # 3. We've held for minimum time (configurable % of max hold)
            should_exit_reversal = False
            
            if position.is_long:
                # Strong reversal signal (more negative)
                if current_score < -self.signal_reversal_threshold:
                    # Exit only if in profit and held for minimum time
                    if current_pnl_pct > self.signal_reversal_min_profit_pct and bars_held >= self.max_hold_bars * self.signal_reversal_min_hold_pct:
                        should_exit_reversal = True
            else:
                # Strong reversal signal (more positive)
                if current_score > self.signal_reversal_threshold:
                    # Exit only if in profit and held for minimum time
                    if current_pnl_pct > self.signal_reversal_min_profit_pct and bars_held >= self.max_hold_bars * self.signal_reversal_min_hold_pct:
                        should_exit_reversal = True
            
            if should_exit_reversal:
                return ExitReason(
                    reason="SIGNAL_REVERSAL",
                    timestamp=market_data.timestamp,
                    metadata={'reversal_score': current_score, 'bars_held': bars_held, 'pnl_pct': current_pnl_pct}
                )
        
        return None
    
    def reset(self):
        """Reset strategy state"""
        self.opportunities_seen = 0
        self.best_score_so_far = -float('inf')
        self.entry_bar_index = None
        self.opportunity_scores = []
        
        # Reset trailing stop state
        self.highest_price_since_entry = None
        self.lowest_price_since_entry = None
        self.initial_stop_loss = None
        self.trailing_stop_activated = False

