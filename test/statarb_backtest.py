"""
Statistical Arbitrage Strategy Backtest Engine
Implements pair trading between MGC and GC futures
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from statarb_contract_specs import MGC_SPECS, GC_SPECS
from statarb_strategy import StatArbCalculator, SpreadSignal

logger = logging.getLogger(__name__)


@dataclass
class StatArbTrade:
    """Represents a completed statistical arbitrage trade"""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_spread: float
    exit_spread: float
    entry_zscore: float
    exit_zscore: float
    entry_price_a: float  # GC entry price
    entry_price_b: float  # MGC entry price
    exit_price_a: float  # GC exit price
    exit_price_b: float  # MGC exit price
    beta: float
    contracts_a: int  # GC contracts
    contracts_b: int  # MGC contracts
    is_long_spread: bool  # True = long A, short B
    pnl: float
    pnl_pct: float
    duration_minutes: int
    exit_reason: str  # "Z_EXIT", "SL", "TIME_STOP", "ZERO_CROSS"
    max_adverse_spread: float  # Worst spread move against position
    max_favorable_spread: float  # Best spread move in favor of position


class StatArbBacktestEngine:
    """Backtest engine for Statistical Arbitrage strategy"""
    
    def __init__(
        self,
        df_gc: pd.DataFrame,
        df_mgc: pd.DataFrame,
        z_entry: float = 2.0,
        z_exit: float = 0.6,
        spread_stop_std: float = 3.0,
        time_stop_hours: float = 2.0,
        lookback_periods: int = 1440,  # 1 day of 1-minute bars
        risk_per_trade: float = 100.0,  # Dollar risk per trade
        initial_equity: float = 10000.0,
        min_lookback: int = 100
    ):
        """
        Initialize StatArb backtest engine
        
        Args:
            df_gc: GC (full gold) DataFrame with columns: timestamp, open, high, low, close, volume
            df_mgc: MGC (micro gold) DataFrame with columns: timestamp, open, high, low, close, volume
            z_entry: Z-score entry threshold (default 2.0)
            z_exit: Z-score exit threshold (default 0.6)
            spread_stop_std: Stop loss in standard deviations (default 3.0)
            time_stop_hours: Maximum hold time in hours (default 2.0)
            lookback_periods: Rolling window for mean/std calculation
            risk_per_trade: Dollar amount to risk per trade
            initial_equity: Starting capital
            min_lookback: Minimum periods needed before trading
        """
        self.df_gc = df_gc.copy()
        self.df_mgc = df_mgc.copy()
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.spread_stop_std = spread_stop_std
        self.time_stop_hours = time_stop_hours
        self.risk_per_trade = risk_per_trade
        self.initial_equity = initial_equity
        
        # Initialize strategy calculator
        self.calculator = StatArbCalculator(
            z_entry=z_entry,
            z_exit=z_exit,
            lookback_periods=lookback_periods,
            min_lookback=min_lookback
        )
        
        # Align dataframes by timestamp
        self._align_dataframes()
        
        # Pre-compute spread history
        self._precompute_spread_history()
        
        # Results
        self.trades: List[StatArbTrade] = []
        self.equity_curve: List[float] = []
        self.current_position: Optional[Dict[str, Any]] = None
    
    def _align_dataframes(self):
        """Align GC and MGC dataframes by timestamp"""
        logger.info("Aligning GC and MGC dataframes...")
        
        # Ensure timestamps are datetime
        self.df_gc['timestamp'] = pd.to_datetime(self.df_gc['timestamp'])
        self.df_mgc['timestamp'] = pd.to_datetime(self.df_mgc['timestamp'])
        
        # Merge on timestamp (inner join to keep only matching timestamps)
        merged = pd.merge(
            self.df_gc,
            self.df_mgc,
            on='timestamp',
            how='inner',
            suffixes=('_gc', '_mgc')
        )
        
        if len(merged) == 0:
            raise ValueError("No matching timestamps between GC and MGC dataframes")
        
        # Split back into aligned dataframes
        self.df_gc = merged[['timestamp', 'open_gc', 'high_gc', 'low_gc', 'close_gc', 'volume_gc']].copy()
        self.df_gc.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        self.df_mgc = merged[['timestamp', 'open_mgc', 'high_mgc', 'low_mgc', 'close_mgc', 'volume_mgc']].copy()
        self.df_mgc.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        logger.info(f"Aligned dataframes: {len(self.df_gc)} bars")
    
    def _precompute_spread_history(self):
        """DEPRECATED: Spread history is now calculated dynamically with rolling beta"""
        # This method is kept for compatibility but not used
        # Spread history is recalculated for each bar with current beta
        # Initialize empty series for compatibility
        self.spread_series = pd.Series([])
    
    def _calculate_position_size(
        self,
        entry_price_a: float,
        entry_price_b: float,
        stop_spread: float,
        beta: float,
        is_long_spread: bool
    ) -> Tuple[int, int]:
        """
        Calculate position sizes for both legs using proper risk management
        
        For spread trading: risk comes from spread movement
        Position sizing should ensure we risk exactly risk_per_trade
        
        Args:
            entry_price_a: GC entry price
            entry_price_b: MGC entry price
            stop_spread: Stop loss spread value
            beta: Hedge ratio (from regression, ~10 for GC/MGC)
            is_long_spread: True if long spread (long A, short B)
            
        Returns:
            (contracts_a, contracts_b) tuple
        """
        # Calculate entry spread
        entry_spread = entry_price_a - beta * entry_price_b
        
        # Calculate spread move that would hit stop (in spread units)
        spread_move_at_stop = abs(entry_spread - stop_spread)
        
        if spread_move_at_stop == 0:
            # No risk defined, use minimum position
            # For GC/MGC: 1 GC contract = 10 MGC contracts (contract size ratio, not beta)
            return 1, 10
        
        # For spread PnL calculation:
        # Spread is in per-ounce price terms: spread = GC_price - beta * MGC_price
        # When spread moves by X (in price terms), the dollar impact depends on contract sizes:
        # - 1 GC contract = 100 oz, so $1 move = $100
        # - 1 MGC contract = 10 oz, so $1 move = $10
        # 
        # For a hedged position (1 GC long, 10 MGC short):
        # If spread increases by $1:
        # - GC might increase by ~$1 → +$100
        # - MGC might increase by ~$1 → -$100 (10 contracts * $10)
        # Net ≈ $0 (hedged)
        # 
        # But if spread moves against us by spread_move_at_stop:
        # Risk ≈ spread_move_at_stop * $100 (from GC leg, MGC hedges most of it)
        # Actually, for proper hedge, the risk is much smaller
        
        # More accurate: For 1 GC + 10 MGC hedge:
        # If spread moves by $1 against us, actual risk is small because positions are hedged
        # But we'll use a conservative estimate: assume risk is from spread volatility
        
        # Risk per spread point: approximately $100 (from GC leg)
        # But since we hedge with MGC, actual risk is lower
        # Use a conservative multiplier
        risk_per_spread_point = 100.0  # $100 per $1 spread move (from GC contract)
        total_risk = spread_move_at_stop * risk_per_spread_point
        
        if total_risk == 0:
            return 1, 10  # Default: 1 GC, 10 MGC
        
        # Calculate number of GC contracts to risk exactly risk_per_trade
        contracts_gc = max(1, int(self.risk_per_trade / total_risk))
        
        # Cap at reasonable maximum (e.g., 10 contracts)
        contracts_gc = min(contracts_gc, 10)
        
        # Hedge with MGC: 1 GC contract = 10 MGC contracts (contract size ratio)
        # This is NOT beta - beta is for price relationship, this is for contract hedging
        contracts_mgc = max(1, contracts_gc * 10)
        
        return contracts_gc, contracts_mgc
    
    def _check_exit_conditions(
        self,
        current_index: int,
        entry_index: int,
        entry_spread: float,
        entry_zscore: float,
        current_spread: float,
        current_zscore: float,
        spread_std: float,
        is_long_spread: bool
    ) -> Tuple[bool, str]:
        """
        Check if exit conditions are met
        
        Returns:
            (should_exit, exit_reason)
        """
        # Check time stop
        entry_time = self.df_gc.iloc[entry_index]['timestamp']
        current_time = self.df_gc.iloc[current_index]['timestamp']
        duration = (current_time - entry_time).total_seconds() / 3600  # hours
        
        if duration >= self.time_stop_hours:
            return True, "TIME_STOP"
        
        # Check z-score exit
        if abs(current_zscore) < self.z_exit:
            return True, "Z_EXIT"
        
        # Check zero cross
        if is_long_spread and current_zscore > 0:
            return True, "ZERO_CROSS"
        if not is_long_spread and current_zscore < 0:
            return True, "ZERO_CROSS"
        
        # Check stop loss (spread moved against us by stop_std standard deviations)
        spread_move = current_spread - entry_spread
        if is_long_spread:
            # Long spread: lose if spread goes down
            if spread_move < -self.spread_stop_std * spread_std:
                return True, "SL"
        else:
            # Short spread: lose if spread goes up
            if spread_move > self.spread_stop_std * spread_std:
                return True, "SL"
        
        return False, ""
    
    def _exit_position(self, exit_index: int, exit_reason: str):
        """Exit current position and record trade"""
        if not self.current_position:
            return
        
        pos = self.current_position
        
        # Get exit prices
        exit_price_gc = self.df_gc.iloc[exit_index]['close']
        exit_price_mgc = self.df_mgc.iloc[exit_index]['close']
        exit_time = self.df_gc.iloc[exit_index]['timestamp']
        
        # Calculate exit spread and z-score
        exit_beta = pos['beta']  # Use entry beta (locked in)
        exit_spread = exit_price_gc - exit_beta * exit_price_mgc
        
        # Recalculate spread history with entry beta for consistency
        lookback_start = max(0, exit_index - self.calculator.lookback_periods)
        spread_history_list = []
        for j in range(lookback_start, exit_index):
            price_gc_j = self.df_gc.iloc[j]['close']
            price_mgc_j = self.df_mgc.iloc[j]['close']
            spread_j = price_gc_j - exit_beta * price_mgc_j
            spread_history_list.append(spread_j)
        spread_history = pd.Series(spread_history_list)
        exit_zscore, _, spread_std = self.calculator.calculate_zscore(exit_spread, spread_history)
        
        # Calculate PnL for both legs
        pnl_gc = GC_SPECS.calculate_pnl(
            pos['entry_price_gc'],
            exit_price_gc,
            pos['contracts_gc'],
            pos['is_long_spread']  # Long GC if long spread
        )
        
        pnl_mgc = MGC_SPECS.calculate_pnl(
            pos['entry_price_mgc'],
            exit_price_mgc,
            pos['contracts_mgc'],
            not pos['is_long_spread']  # Short MGC if long spread
        )
        
        total_pnl = pnl_gc + pnl_mgc
        
        # Calculate risk amount
        risk_amount = self.risk_per_trade  # We sized based on this
        pnl_pct = (total_pnl / risk_amount * 100) if risk_amount > 0 else 0
        
        # Calculate duration
        duration = (exit_time - pos['entry_time']).total_seconds() / 60
        
        # Calculate max adverse/favorable spread moves
        max_adverse = pos.get('max_adverse_spread', 0.0)
        max_favorable = pos.get('max_favorable_spread', 0.0)
        
        trade = StatArbTrade(
            entry_time=pos['entry_time'],
            exit_time=exit_time,
            entry_spread=pos['entry_spread'],
            exit_spread=exit_spread,
            entry_zscore=pos['entry_zscore'],
            exit_zscore=exit_zscore,
            entry_price_a=pos['entry_price_gc'],
            entry_price_b=pos['entry_price_mgc'],
            exit_price_a=exit_price_gc,
            exit_price_b=exit_price_mgc,
            beta=pos['beta'],
            contracts_a=pos['contracts_gc'],
            contracts_b=pos['contracts_mgc'],
            is_long_spread=pos['is_long_spread'],
            pnl=total_pnl,
            pnl_pct=pnl_pct,
            duration_minutes=int(duration),
            exit_reason=exit_reason,
            max_adverse_spread=max_adverse,
            max_favorable_spread=max_favorable
        )
        
        self.trades.append(trade)
        self.current_position = None
    
    def run_backtest(self):
        """Run the complete backtest"""
        logger.info("Starting Statistical Arbitrage backtest...")
        logger.info(f"Z-entry: {self.z_entry}, Z-exit: {self.z_exit}")
        logger.info(f"Spread stop: {self.spread_stop_std} std, Time stop: {self.time_stop_hours} hours")
        
        equity = self.initial_equity
        self.equity_curve = [equity]
        
        # Process each bar
        # For realism: Calculate beta dynamically for each bar, then recalculate
        # spread history with that same beta to ensure consistency
        
        for i in range(self.calculator.min_lookback, len(self.df_gc)):
            # Calculate current beta dynamically (rolling regression)
            current_beta = self.calculator.calculate_beta(self.df_gc, self.df_mgc, i)
            
            # Recalculate spread history with CURRENT beta for consistency
            # This ensures both current spread and history use the same beta
            lookback_start = max(0, i - self.calculator.lookback_periods)
            spread_history_list = []
            for j in range(lookback_start, i):
                price_gc = self.df_gc.iloc[j]['close']
                price_mgc = self.df_mgc.iloc[j]['close']
                # Use CURRENT beta for all history (ensures consistency)
                spread = price_gc - current_beta * price_mgc
                spread_history_list.append(spread)
            spread_history = pd.Series(spread_history_list)
            
            # Get current position status
            current_pos = None
            if self.current_position:
                current_pos = "LONG_SPREAD" if self.current_position['is_long_spread'] else "SHORT_SPREAD"
            
            # Process bar and get signal
            # Pass fixed_beta=current_beta to ensure current spread uses same beta as history
            signal = self.calculator.process_bar(
                self.df_gc,
                self.df_mgc,
                i,
                current_pos,
                spread_history,
                fixed_beta=current_beta  # Use same beta as spread_history
            )
            
            if signal is None:
                continue
            
            # Handle entry signals - ONLY if no position exists
            # Note: We don't use signal-based exits here - we monitor positions separately below
            if signal.is_entry and self.current_position is None:
                # Enter position
                entry_price_gc = signal.price_a
                entry_price_mgc = signal.price_b
                
                # Calculate position sizes
                # For stop loss, calculate spread that would trigger stop
                stop_spread = signal.spread_mean + (
                    -self.spread_stop_std * signal.spread_std if signal.signal == "LONG_SPREAD"
                    else self.spread_stop_std * signal.spread_std
                )
                
                contracts_gc, contracts_mgc = self._calculate_position_size(
                    entry_price_gc,
                    entry_price_mgc,
                    stop_spread,
                    signal.beta,
                    signal.signal == "LONG_SPREAD"
                )
                
                self.current_position = {
                    'entry_index': i,
                    'entry_time': signal.timestamp,
                    'entry_price_gc': entry_price_gc,
                    'entry_price_mgc': entry_price_mgc,
                    'entry_spread': signal.spread,
                    'entry_zscore': signal.zscore,
                    'beta': signal.beta,
                    'contracts_gc': contracts_gc,
                    'contracts_mgc': contracts_mgc,
                    'is_long_spread': signal.signal == "LONG_SPREAD",
                    'max_adverse_spread': 0.0,
                    'max_favorable_spread': 0.0
                }
                
                logger.info(
                    f"ENTRY: {signal.signal} at z={signal.zscore:.2f} | "
                    f"GC={entry_price_gc:.2f}, MGC={entry_price_mgc:.2f} | "
                    f"Spread={signal.spread:.2f} | "
                    f"Contracts: GC={contracts_gc}, MGC={contracts_mgc}"
                )
            
            # Monitor position for exits
            if self.current_position:
                entry_idx = self.current_position['entry_index']
                
                # Get current spread and z-score
                # Use the beta from entry (locked in) for spread calculation
                current_price_gc = self.df_gc.iloc[i]['close']
                current_price_mgc = self.df_mgc.iloc[i]['close']
                # Use entry beta for consistency during position hold
                entry_beta = self.current_position['beta']
                current_spread = current_price_gc - entry_beta * current_price_mgc
                
                # For z-score during position monitoring, recalculate spread history with entry beta
                # This ensures consistency: entry spread and monitoring spread use the same beta
                spread_history_for_zscore_list = []
                for j in range(lookback_start, i):
                    price_gc_j = self.df_gc.iloc[j]['close']
                    price_mgc_j = self.df_mgc.iloc[j]['close']
                    # Use entry beta for all history (locked in at entry)
                    spread_j = price_gc_j - entry_beta * price_mgc_j
                    spread_history_for_zscore_list.append(spread_j)
                spread_history_for_zscore = pd.Series(spread_history_for_zscore_list)
                
                current_zscore, _, spread_std = self.calculator.calculate_zscore(
                    current_spread, 
                    spread_history_for_zscore
                )
                
                # Track max adverse/favorable moves
                spread_move = current_spread - self.current_position['entry_spread']
                if self.current_position['is_long_spread']:
                    # Long spread: adverse is down, favorable is up
                    if spread_move < self.current_position['max_adverse_spread']:
                        self.current_position['max_adverse_spread'] = spread_move
                    if spread_move > self.current_position['max_favorable_spread']:
                        self.current_position['max_favorable_spread'] = spread_move
                else:
                    # Short spread: adverse is up, favorable is down
                    if spread_move > self.current_position['max_adverse_spread']:
                        self.current_position['max_adverse_spread'] = spread_move
                    if spread_move < self.current_position['max_favorable_spread']:
                        self.current_position['max_favorable_spread'] = spread_move
                
                # Check exit conditions
                should_exit, exit_reason = self._check_exit_conditions(
                    i,
                    entry_idx,
                    self.current_position['entry_spread'],
                    self.current_position['entry_zscore'],
                    current_spread,
                    current_zscore,
                    spread_std,
                    self.current_position['is_long_spread']
                )
                
                if should_exit:
                    self._exit_position(i, exit_reason)
                    if self.trades:
                        equity += self.trades[-1].pnl
                        self.equity_curve.append(equity)
                        logger.info(
                            f"EXIT: {exit_reason} | PnL=${self.trades[-1].pnl:.2f} | "
                            f"Z-score: {current_zscore:.2f}"
                        )
        
        # Close any remaining position at end of data
        if self.current_position:
            self._exit_position(len(self.df_gc) - 1, "END_OF_DATA")
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
                'avg_pnl': 0,
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
        
        avg_pnl = np.mean([t.pnl for t in self.trades]) if self.trades else 0
        
        # Expectancy
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
            'avg_pnl': avg_pnl,
            'expectancy': expectancy,
            'initial_equity': self.equity_curve[0] if self.equity_curve else 0,
            'final_equity': self.equity_curve[-1] if self.equity_curve else 0,
            'total_return_pct': ((self.equity_curve[-1] - self.equity_curve[0]) / self.equity_curve[0] * 100) if len(self.equity_curve) > 1 else 0,
            'avg_duration_minutes': np.mean([t.duration_minutes for t in self.trades]) if self.trades else 0
        }

