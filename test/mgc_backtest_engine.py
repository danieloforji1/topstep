"""
MGC Liquidity Sweep Backtest Engine
Main backtest engine that executes the strategy
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

import logging
from mgc_strategy_logic import (
    SwingDetector,
    LiquiditySweepDetector,
    ConfirmationCandleDetector,
    TrendFilter,
    PositionSizer,
    LiquiditySweep
)
from mgc_contract_specs import MGC_SPECS

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a completed trade"""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    contracts: int
    is_long: bool
    pnl: float
    pnl_pct: float
    risk_reward: float
    duration_minutes: int
    exit_reason: str  # "TP", "SL", "OppositeLiquidity", "EndOfData"


class MGCBacktestEngine:
    """Main backtest engine for MGC Liquidity Sweep Strategy"""
    
    def __init__(
        self,
        df_5m: pd.DataFrame,
        df_15m: pd.DataFrame,
        risk_reward: float = 1.5,
        pivot_length: int = 5,
        contracts_per_trade: Optional[int] = None,
        risk_per_trade: Optional[float] = None,
        atr_period: int = 5,
        atr_multiplier: float = 5.0,
        ema_period: int = 50,
        confirmation_body_ratio: float = 0.5,
        initial_equity: float = 10000.0
    ):
        """
        Initialize backtest engine
        
        Args:
            df_5m: 5-minute OHLCV DataFrame
            df_15m: 15-minute OHLCV DataFrame (for trend filter)
            risk_reward: Risk:Reward ratio (default 1.5)
            pivot_length: Pivot length for swing detection (default 5)
            contracts_per_trade: Fixed number of contracts per trade (None = use risk-based sizing)
            risk_per_trade: Dollar amount to risk per trade (only used if contracts_per_trade is None)
            atr_period: ATR period for stop sizing (default 5)
            atr_multiplier: ATR multiplier for stop distance (default 5.0)
            ema_period: EMA period for trend filter (default 50)
            confirmation_body_ratio: Minimum body ratio for confirmation candle (default 0.5)
            initial_equity: Starting capital (default 10000.0)
        """
        self.df_5m = df_5m.copy()
        self.df_15m = df_15m.copy()
        self.risk_reward = risk_reward
        self.pivot_length = pivot_length
        self.contracts_per_trade = contracts_per_trade
        self.risk_per_trade = risk_per_trade if risk_per_trade is not None else 50.0
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.ema_period = ema_period
        self.confirmation_body_ratio = confirmation_body_ratio
        self.initial_equity = initial_equity
        
        # Initialize components
        self.swing_detector = SwingDetector(pivot_length)
        self.sweep_detector = LiquiditySweepDetector(pivot_length)
        self.confirmation_detector = ConfirmationCandleDetector(confirmation_body_ratio)
        self.trend_filter = TrendFilter(period=ema_period)
        self.position_sizer = PositionSizer(
            contracts_per_trade=contracts_per_trade,
            risk_per_trade=self.risk_per_trade,
            atr_period=atr_period,
            atr_multiplier=atr_multiplier
        )
        
        # Results
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.current_position: Optional[Dict[str, Any]] = None
        
        # Pre-compute swing points and sweeps for efficiency
        self._precompute_swings_and_sweeps()
    
    def _precompute_swings_and_sweeps(self):
        """Pre-compute swing points and liquidity sweeps"""
        logger.info("Pre-computing swing points...")
        self.swing_highs, self.swing_lows = self.swing_detector.detect_swings(self.df_5m)
        logger.info(f"Found {len(self.swing_highs)} swing highs and {len(self.swing_lows)} swing lows")
        
        logger.info("Pre-computing liquidity sweeps...")
        all_swings = self.swing_highs + self.swing_lows
        self.sweeps = self.sweep_detector.detect_sweeps(self.df_5m, all_swings)
        logger.info(f"Found {len(self.sweeps)} liquidity sweeps")
    
    def _get_15m_trend(self, timestamp: pd.Timestamp) -> Optional[bool]:
        """
        Get 15m trend direction at given timestamp
        
        Returns:
            True for uptrend, False for downtrend, None if not enough data
        """
        # Find corresponding 15m candle
        # Find the 15m candle that contains this timestamp
        for i in range(len(self.df_15m)):
            if self.df_15m.iloc[i]['timestamp'] <= timestamp:
                if i + 1 < len(self.df_15m):
                    if self.df_15m.iloc[i + 1]['timestamp'] > timestamp:
                        # This 15m candle contains the timestamp
                        if i >= 50:  # Need at least 50 candles for EMA
                            return self.trend_filter.is_uptrend(self.df_15m, i)
                elif i == len(self.df_15m) - 1:
                    # Last candle
                    if i >= 50:
                        return self.trend_filter.is_uptrend(self.df_15m, i)
        
        return None
    
    def _find_opposite_liquidity_zone(
        self,
        current_index: int,
        is_long: bool
    ) -> Optional[float]:
        """
        Find opposite liquidity zone for exit
        
        For longs: find next swing high
        For shorts: find next swing low
        
        Args:
            current_index: Current candle index
            is_long: True for long position
            
        Returns:
            Price of opposite liquidity zone, or None
        """
        if is_long:
            # Find next swing high after current index
            for swing_high in self.swing_highs:
                if swing_high.index > current_index:
                    return swing_high.price
        else:
            # Find next swing low after current index
            for swing_low in self.swing_lows:
                if swing_low.index > current_index:
                    return swing_low.price
        
        return None
    
    def _check_exit_conditions(
        self,
        current_index: int,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        is_long: bool
    ) -> tuple[bool, str, float]:
        """
        Check if exit conditions are met
        
        Returns:
            (should_exit, exit_reason, exit_price)
        """
        if current_index >= len(self.df_5m):
            # End of data
            candle = self.df_5m.iloc[-1]
            exit_price = candle['close']
            return True, "EndOfData", exit_price
        
        candle = self.df_5m.iloc[current_index]
        
        # Round prices to ticks for realistic execution
        stop_loss = MGC_SPECS.round_to_tick(stop_loss)
        take_profit = MGC_SPECS.round_to_tick(take_profit)
        
        # Check stop loss
        if is_long:
            if candle['low'] <= stop_loss:
                return True, "SL", stop_loss
        else:
            if candle['high'] >= stop_loss:
                return True, "SL", stop_loss
        
        # Check take profit
        if is_long:
            if candle['high'] >= take_profit:
                return True, "TP", take_profit
        else:
            if candle['low'] <= take_profit:
                return True, "TP", take_profit
        
        # Check opposite liquidity zone
        opposite_zone = self._find_opposite_liquidity_zone(current_index, is_long)
        if opposite_zone:
            opposite_zone = MGC_SPECS.round_to_tick(opposite_zone)
            if is_long:
                if candle['high'] >= opposite_zone:
                    return True, "OppositeLiquidity", opposite_zone
            else:
                if candle['low'] <= opposite_zone:
                    return True, "OppositeLiquidity", opposite_zone
        
        return False, "", 0.0
    
    def _enter_long(
        self,
        entry_index: int,
        entry_price: float,
        stop_loss: float
    ) -> Optional[Dict[str, Any]]:
        """Enter long position"""
        # Ensure prices are rounded to ticks
        entry_price = MGC_SPECS.round_to_tick(entry_price)
        stop_loss = MGC_SPECS.round_to_tick(stop_loss)
        
        # Calculate position size
        contracts = self.position_sizer.calculate_position_size(
            self.df_5m,
            entry_index,
            stop_loss,
            is_long=True
        )
        
        # Calculate take profit in ticks, then convert back to price
        risk_ticks = MGC_SPECS.price_to_ticks(entry_price - stop_loss)
        reward_ticks = risk_ticks * self.risk_reward
        take_profit = entry_price + MGC_SPECS.ticks_to_price(reward_ticks)
        take_profit = MGC_SPECS.round_to_tick(take_profit)
        
        return {
            'entry_index': entry_index,
            'entry_time': self.df_5m.iloc[entry_index]['timestamp'],
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'contracts': contracts,
            'is_long': True
        }
    
    def _enter_short(
        self,
        entry_index: int,
        entry_price: float,
        stop_loss: float
    ) -> Optional[Dict[str, Any]]:
        """Enter short position"""
        # Ensure prices are rounded to ticks
        entry_price = MGC_SPECS.round_to_tick(entry_price)
        stop_loss = MGC_SPECS.round_to_tick(stop_loss)
        
        # Calculate position size
        contracts = self.position_sizer.calculate_position_size(
            self.df_5m,
            entry_index,
            stop_loss,
            is_long=False
        )
        
        # Calculate take profit in ticks, then convert back to price
        risk_ticks = MGC_SPECS.price_to_ticks(stop_loss - entry_price)
        reward_ticks = risk_ticks * self.risk_reward
        take_profit = entry_price - MGC_SPECS.ticks_to_price(reward_ticks)
        take_profit = MGC_SPECS.round_to_tick(take_profit)
        
        return {
            'entry_index': entry_index,
            'entry_time': self.df_5m.iloc[entry_index]['timestamp'],
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'contracts': contracts,
            'is_long': False
        }
    
    def _exit_position(self, exit_index: int, exit_price: float, exit_reason: str):
        """Exit current position and record trade"""
        if not self.current_position:
            return
        
        pos = self.current_position
        
        # Round prices to nearest tick for realistic execution
        entry_price = MGC_SPECS.round_to_tick(pos['entry_price'])
        exit_price = MGC_SPECS.round_to_tick(exit_price)
        stop_price = MGC_SPECS.round_to_tick(pos['stop_loss'])
        
        # Calculate PnL using proper futures contract calculations
        pnl = MGC_SPECS.calculate_pnl(
            entry_price=entry_price,
            exit_price=exit_price,
            contracts=pos['contracts'],
            is_long=pos['is_long']
        )
        
        # Calculate risk amount (what we would lose if stop hit)
        risk_amount = MGC_SPECS.calculate_risk_amount(
            entry_price=entry_price,
            stop_price=stop_price,
            contracts=pos['contracts'],
            is_long=pos['is_long']
        )
        
        # Calculate PnL percentage
        pnl_pct = (pnl / risk_amount * 100) if risk_amount > 0 else 0
        
        # Calculate risk:reward using ticks
        risk_ticks = MGC_SPECS.price_to_ticks(abs(entry_price - stop_price))
        reward_ticks = MGC_SPECS.price_to_ticks(abs(exit_price - entry_price))
        rr = reward_ticks / risk_ticks if risk_ticks > 0 else 0
        
        # Calculate duration
        exit_time = self.df_5m.iloc[exit_index]['timestamp']
        duration = (exit_time - pos['entry_time']).total_seconds() / 60
        
        trade = Trade(
            entry_time=pos['entry_time'],
            exit_time=exit_time,
            entry_price=entry_price,  # Use rounded price
            exit_price=exit_price,  # Use rounded price
            stop_loss=stop_price,  # Use rounded price
            take_profit=MGC_SPECS.round_to_tick(pos['take_profit']),
            contracts=pos['contracts'],
            is_long=pos['is_long'],
            pnl=pnl,
            pnl_pct=pnl_pct,
            risk_reward=rr,
            duration_minutes=int(duration),
            exit_reason=exit_reason
        )
        
        self.trades.append(trade)
        self.current_position = None
    
    def run_backtest(self):
        """Run the complete backtest"""
        logger.info("Starting backtest...")
        
        equity = self.initial_equity
        self.equity_curve = [equity]
        
        # Process each sweep
        for sweep in self.sweeps:
            # Skip if we're already in a position
            if self.current_position:
                continue
            
            # Determine if this is a long or short setup
            is_long = not sweep.swing_point.is_high  # Long if sweep of swing low
            
            # Check trend filter on 15m
            trend = self._get_15m_trend(sweep.timestamp)
            if trend is None:
                continue
            
            if is_long and not trend:  # Long setup but downtrend
                continue
            if not is_long and trend:  # Short setup but uptrend
                continue
            
            # Find confirmation candle
            conf_index = self.confirmation_detector.find_confirmation(
                self.df_5m,
                sweep.index,
                is_long
            )
            
            if conf_index is None:
                continue
            
            # Entry on break of confirmation candle high/low
            conf_candle = self.df_5m.iloc[conf_index]
            
            if is_long:
                entry_trigger = MGC_SPECS.round_to_tick(conf_candle['high'])
                stop_loss = MGC_SPECS.round_to_tick(sweep.sweep_low)
            else:
                entry_trigger = MGC_SPECS.round_to_tick(conf_candle['low'])
                stop_loss = MGC_SPECS.round_to_tick(sweep.sweep_high)
            
            # Wait for break of confirmation candle high/low
            entry_index = None
            entry_price = None
            
            for i in range(conf_index + 1, len(self.df_5m)):
                candle = self.df_5m.iloc[i]
                
                if is_long:
                    if candle['high'] > entry_trigger:
                        entry_index = i
                        entry_price = entry_trigger  # Enter at the break level
                        break
                else:
                    if candle['low'] < entry_trigger:
                        entry_index = i
                        entry_price = entry_trigger  # Enter at the break level
                        break
            
            if entry_index is None:
                continue  # Break never occurred
            
            # Round entry price to tick
            entry_price = MGC_SPECS.round_to_tick(entry_price)
            
            # Enter position
            if is_long:
                self.current_position = self._enter_long(entry_index, entry_price, stop_loss)
            else:
                self.current_position = self._enter_short(entry_index, entry_price, stop_loss)
            
            if not self.current_position:
                continue
            
            # Monitor position for exit
            entry_idx = entry_index + 1  # Start monitoring from next candle
            
            for i in range(entry_idx, len(self.df_5m)):
                should_exit, exit_reason, exit_price = self._check_exit_conditions(
                    i,
                    self.current_position['entry_price'],
                    self.current_position['stop_loss'],
                    self.current_position['take_profit'],
                    self.current_position['is_long']
                )
                
                if should_exit:
                    self._exit_position(i, exit_price, exit_reason)
                    
                    # Update equity
                    if self.trades:
                        equity += self.trades[-1].pnl
                        self.equity_curve.append(equity)
                    break
            
            # If position still open at end of data
            if self.current_position:
                last_candle = self.df_5m.iloc[-1]
                exit_price = MGC_SPECS.round_to_tick(last_candle['close'])
                self._exit_position(len(self.df_5m) - 1, exit_price, "EndOfData")
                if self.trades:
                    equity += self.trades[-1].pnl
                    self.equity_curve.append(equity)
        
        logger.info(f"Backtest complete. Total trades: {len(self.trades)}")
    
    def get_results(self) -> Dict[str, Any]:
        """Get backtest results summary"""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'avg_rr': 0,
                'expectancy': 0
            }
        
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl < 0]
        
        total_pnl = sum(t.pnl for t in self.trades)
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        
        win_rate = len(winning_trades) / len(self.trades) * 100 if self.trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Calculate max drawdown
        peak = self.equity_curve[0]
        max_dd = 0
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100
            if dd > max_dd:
                max_dd = dd
        
        avg_rr = np.mean([t.risk_reward for t in self.trades]) if self.trades else 0
        
        # Expectancy = (Win Rate × Avg Win) - (Loss Rate × Avg Loss)
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        expectancy = (win_rate / 100 * avg_win) - ((100 - win_rate) / 100 * abs(avg_loss))
        
        return {
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_dd,
            'avg_rr': avg_rr,
            'expectancy': expectancy,
            'initial_equity': self.equity_curve[0] if self.equity_curve else 0,
            'final_equity': self.equity_curve[-1] if self.equity_curve else 0,
            'total_return_pct': ((self.equity_curve[-1] - self.equity_curve[0]) / self.equity_curve[0] * 100) if len(self.equity_curve) > 1 else 0
        }


# Import logger
import logging
logger = logging.getLogger(__name__)

