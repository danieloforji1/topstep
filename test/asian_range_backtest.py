"""
Asian Range Breakout Strategy Backtest Engine
Implements the exact strategy from arbts.txt
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime, time
import pytz
import logging

from mgc_contract_specs import MGC_SPECS
from asian_range_strategy import (
    AsianRangeCalculator,
    PositionManager,
    AsianRange,
    PendingOrder
)

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
    exit_reason: str  # "TP", "SL", "BE", "TimeExit" (12PM), "PartialClose"
    asian_range_size: float
    tp_multiplier: float
    # Partial close fields
    partial_close_price: Optional[float] = None
    partial_close_contracts: int = 0
    remaining_contracts: int = 0
    final_exit_price: Optional[float] = None
    final_exit_time: Optional[pd.Timestamp] = None
    final_exit_reason: Optional[str] = None


class AsianRangeBacktestEngine:
    """Backtest engine for Asian Range Breakout Strategy"""
    
    def __init__(
        self,
        df_1m: pd.DataFrame,
        contracts_per_trade: int = 3,
        tp_multiplier: float = 1.5,
        sl_buffer_ticks: int = 3,
        partial_close_percent: float = 0.75,
        initial_equity: float = 10000.0,
        et_timezone: str = "America/New_York"
    ):
        """
        Initialize backtest engine
        
        Args:
            df_1m: 1-minute OHLCV DataFrame with ET timestamps
            contracts_per_trade: Fixed number of contracts per trade
            tp_multiplier: Take profit multiplier (1.0 to 2.0)
            sl_buffer_ticks: Stop loss buffer in ticks (default 3)
            partial_close_percent: Percentage to close at 12 PM (0.75 = 75%, remaining stays open)
            initial_equity: Starting capital
            et_timezone: Eastern timezone string
        """
        self.df_1m = df_1m.copy()
        self.contracts_per_trade = contracts_per_trade
        self.tp_multiplier = tp_multiplier
        self.sl_buffer_ticks = sl_buffer_ticks
        self.partial_close_percent = partial_close_percent
        self.initial_equity = initial_equity
        
        # Initialize components
        self.range_calculator = AsianRangeCalculator(et_timezone)
        self.position_manager = PositionManager(tp_multiplier, sl_buffer_ticks)
        
        # Results
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.asian_ranges: List[AsianRange] = []
        
        # Ensure timestamps are in ET
        self._ensure_et_timezone()
    
    def _ensure_et_timezone(self):
        """Ensure all timestamps are in ET timezone"""
        if self.df_1m['timestamp'].dt.tz is None:
            # Assume UTC if no timezone
            self.df_1m['timestamp'] = pd.to_datetime(self.df_1m['timestamp']).dt.tz_localize('UTC')
        
        # Convert to ET
        et_tz = pytz.timezone("America/New_York")
        self.df_1m['timestamp'] = self.df_1m['timestamp'].dt.tz_convert(et_tz)
    
    def get_trading_dates(self) -> List[pd.Timestamp]:
        """Get list of unique trading dates"""
        dates = self.df_1m['timestamp'].dt.date.unique()
        return [pd.Timestamp.combine(d, time(0, 0)).tz_localize(self.range_calculator.et_tz) for d in dates]
    
    def process_trading_day(self, date: pd.Timestamp) -> Optional[Trade]:
        """
        Process a single trading day according to strategy rules
        
        Args:
            date: Trading date
            
        Returns:
            Trade if one was taken, None otherwise
        """
        # Get date string for logging
        if isinstance(date, pd.Timestamp):
            date_str = date.strftime('%Y-%m-%d')
            trading_date = date.date()
        else:
            date_str = str(date)
            trading_date = date
        
        # Step 1: Calculate Asian Range (8 PM previous day to 2 AM current day)
        asian_range = self.range_calculator.calculate_asian_range(self.df_1m, date)
        if asian_range is None:
            logger.info(f"[{date_str}] NO TRADE: Insufficient data to calculate Asian range (8 PM - 2 AM ET)")
            return None
        
        self.asian_ranges.append(asian_range)
        
        # Step 2: Get data from 3 AM to end of trading day (5 PM ET - futures market close)
        london_open = pd.Timestamp.combine(trading_date, time(3, 0))
        ny_close = pd.Timestamp.combine(trading_date, time(12, 0))
        end_of_trading_day = pd.Timestamp.combine(trading_date, time(17, 0))  # 5 PM ET - end of trading day
        
        # Localize to ET if needed
        if self.df_1m['timestamp'].dt.tz is not None:
            london_open = london_open.tz_localize(self.range_calculator.et_tz)
            ny_close = ny_close.tz_localize(self.range_calculator.et_tz)
            end_of_trading_day = end_of_trading_day.tz_localize(self.range_calculator.et_tz)
        
        # Get data from 3 AM to end of trading day (for partial close monitoring after 12 PM)
        mask = (self.df_1m['timestamp'] >= london_open) & (self.df_1m['timestamp'] <= end_of_trading_day)
        trading_data = self.df_1m[mask].copy()
        
        if len(trading_data) == 0:
            logger.info(f"[{date_str}] NO TRADE: No trading data available from 3 AM to 12 PM ET")
            return None
        
        # Step 3: Create pending orders at 3 AM
        pending_order = self.range_calculator.create_pending_orders(asian_range, london_open)
        
        logger.info(f"[{date_str}] Asian Range: High={asian_range.asian_high:.2f}, Low={asian_range.asian_low:.2f}, "
                   f"Range={asian_range.range_size:.2f} | Buy Stop={pending_order.buy_stop_price:.2f}, "
                   f"Sell Stop={pending_order.sell_stop_price:.2f}")
        
        # Step 4: Check for order fills and manage position
        trade = self._simulate_trading_day(trading_data, pending_order, asian_range, ny_close, end_of_trading_day, date_str)
        
        if trade is None:
            logger.info(f"[{date_str}] NO TRADE: Price stayed within Asian range (no breakout occurred)")
        
        return trade
    
    def _simulate_trading_day(
        self,
        trading_data: pd.DataFrame,
        pending_order: PendingOrder,
        asian_range: AsianRange,
        ny_close: pd.Timestamp,
        end_of_trading_day: pd.Timestamp,
        date_str: str = ""
    ) -> Optional[Trade]:
        """
        Simulate trading for the day
        
        Args:
            trading_data: Data from 3 AM to 12 PM
            pending_order: Pending OCO orders
            asian_range: Asian range
            ny_close: NY close time (12 PM)
            
        Returns:
            Trade if executed, None otherwise
        """
        entry_index = None
        entry_price = None
        is_long = None
        stop_loss = None
        take_profit = None
        
        # Check for order fills
        for idx, row in trading_data.iterrows():
            candle = row
            
            # Check if buy stop triggered
            if entry_index is None and candle['high'] >= pending_order.buy_stop_price:
                entry_index = idx
                entry_price = pending_order.buy_stop_price
                is_long = True
                stop_loss = self.position_manager.calculate_stop_loss(entry_price, asian_range, True)
                take_profit = self.position_manager.calculate_take_profit(entry_price, asian_range, True)
                entry_time = candle['timestamp']
                logger.info(f"[{date_str}] ENTRY: LONG at {entry_price:.2f} (broke above Asian High) | "
                           f"SL={stop_loss:.2f}, TP={take_profit:.2f}")
                break
            
            # Check if sell stop triggered
            if entry_index is None and candle['low'] <= pending_order.sell_stop_price:
                entry_index = idx
                entry_price = pending_order.sell_stop_price
                is_long = False
                stop_loss = self.position_manager.calculate_stop_loss(entry_price, asian_range, False)
                take_profit = self.position_manager.calculate_take_profit(entry_price, asian_range, False)
                entry_time = candle['timestamp']
                logger.info(f"[{date_str}] ENTRY: SHORT at {entry_price:.2f} (broke below Asian Low) | "
                           f"SL={stop_loss:.2f}, TP={take_profit:.2f}")
                break
        
        # If no order filled, no trade
        if entry_index is None:
            return None
        
        entry_time = trading_data.loc[entry_index, 'timestamp']
        
        # Step 5: Monitor position for exits
        # Continue from entry_index
        remaining_data = trading_data.loc[trading_data.index >= entry_index]
        current_stop_loss = stop_loss
        moved_to_breakeven = False
        
        for idx, row in remaining_data.iterrows():
            candle = row
            current_time = candle['timestamp']
            
            # Check break-even rule (move SL to entry at +1R)
            if not moved_to_breakeven:
                should_move, new_stop = self.position_manager.should_move_to_breakeven(
                    entry_price, candle['close'], current_stop_loss, asian_range, is_long
                )
                if should_move:
                    current_stop_loss = new_stop
                    moved_to_breakeven = True
                    logger.debug(f"Moved to break-even at {current_time}")
            
            # Check stop loss
            if is_long:
                if candle['low'] <= current_stop_loss:
                    exit_price = current_stop_loss
                    exit_reason = "SL"
                    trade = self._create_trade(
                        entry_time, current_time, entry_price, exit_price,
                        stop_loss, take_profit, is_long, exit_reason, asian_range
                    )
                    logger.info(f"[{date_str}] EXIT: Stop Loss hit at {exit_price:.2f} | PnL=${trade.pnl:.2f}")
                    return trade
            else:
                if candle['high'] >= current_stop_loss:
                    exit_price = current_stop_loss
                    exit_reason = "SL"
                    trade = self._create_trade(
                        entry_time, current_time, entry_price, exit_price,
                        stop_loss, take_profit, is_long, exit_reason, asian_range
                    )
                    logger.info(f"[{date_str}] EXIT: Stop Loss hit at {exit_price:.2f} | PnL=${trade.pnl:.2f}")
                    return trade
            
            # Check take profit
            if is_long:
                if candle['high'] >= take_profit:
                    exit_price = take_profit
                    exit_reason = "TP"
                    trade = self._create_trade(
                        entry_time, current_time, entry_price, exit_price,
                        stop_loss, take_profit, is_long, exit_reason, asian_range
                    )
                    logger.info(f"[{date_str}] EXIT: Take Profit hit at {exit_price:.2f} | PnL=${trade.pnl:.2f}")
                    return trade
            else:
                if candle['low'] <= take_profit:
                    exit_price = take_profit
                    exit_reason = "TP"
                    trade = self._create_trade(
                        entry_time, current_time, entry_price, exit_price,
                        stop_loss, take_profit, is_long, exit_reason, asian_range
                    )
                    logger.info(f"[{date_str}] EXIT: Take Profit hit at {exit_price:.2f} | PnL=${trade.pnl:.2f}")
                    return trade
            
            # Check hard exit time (12 PM) - partial close
            if current_time >= ny_close:
                partial_close_price = candle['close']
                
                # Calculate partial close
                partial_close_contracts = int(self.contracts_per_trade * self.partial_close_percent)
                remaining_contracts = self.contracts_per_trade - partial_close_contracts
                
                if remaining_contracts > 0:
                    # Partial close: close most, keep remainder with BE stop
                    logger.info(f"[{date_str}] PARTIAL CLOSE: Closing {partial_close_contracts}/{self.contracts_per_trade} contracts at {partial_close_price:.2f} (12 PM ET)")
                    
                    # Calculate PnL from partial close
                    partial_pnl = MGC_SPECS.calculate_pnl(entry_price, partial_close_price, partial_close_contracts, is_long)
                    
                    # Move stop to 50% profit level (halfway between entry and partial close)
                    # This gives more buffer for flash drawdowns while still protecting profits
                    if is_long:
                        # For long: entry < partial_close, so 50% profit = entry + (partial_close - entry) * 0.5
                        profit_made = partial_close_price - entry_price
                        new_stop_loss = entry_price + (profit_made * 0.5)
                    else:
                        # For short: entry > partial_close, so 50% profit = entry - (entry - partial_close) * 0.5
                        profit_made = entry_price - partial_close_price
                        new_stop_loss = entry_price - (profit_made * 0.5)
                    
                    new_stop_loss = MGC_SPECS.round_to_tick(new_stop_loss)
                    
                    # Continue monitoring remaining position after 12 PM
                    # Get data AFTER the partial close candle (skip the 12 PM candle itself)
                    remaining_data_after_12pm = trading_data.loc[trading_data.index > idx].copy()
                    
                    if len(remaining_data_after_12pm) == 0:
                        # No more data after 12 PM, close remaining at partial close price
                        final_exit_price = partial_close_price
                        final_exit_reason = "EndOfData"
                        final_pnl = 0  # No additional PnL since exit at same price
                        total_pnl = partial_pnl
                        trade = self._create_trade_with_partial(
                            entry_time, current_time, current_time, entry_price,
                            partial_close_price, final_exit_price, stop_loss, take_profit,
                            is_long, partial_close_contracts, remaining_contracts,
                            total_pnl, "PartialClose", final_exit_reason, asian_range
                        )
                        logger.info(f"[{date_str}] FINAL EXIT: No data after 12 PM, remaining {remaining_contracts} contracts closed at {final_exit_price:.2f} | Total PnL=${total_pnl:.2f}")
                        return trade
                    
                    logger.info(f"[{date_str}] Remaining {remaining_contracts} contracts: Stop moved to 50% profit level at {new_stop_loss:.2f} (entry: {entry_price:.2f}, partial close: {partial_close_price:.2f})")
                    
                    for idx2, row2 in remaining_data_after_12pm.iterrows():
                        candle2 = row2
                        current_time2 = candle2['timestamp']
                        
                        # Check end of trading day (5 PM ET) - hard exit for remaining position
                        if current_time2 >= end_of_trading_day:
                            final_exit_price = candle2['close']
                            final_exit_reason = "EndOfDay"
                            final_pnl = MGC_SPECS.calculate_pnl(partial_close_price, final_exit_price, remaining_contracts, is_long)
                            total_pnl = partial_pnl + final_pnl
                            trade = self._create_trade_with_partial(
                                entry_time, current_time, current_time2, entry_price,
                                partial_close_price, final_exit_price, stop_loss, take_profit,
                                is_long, partial_close_contracts, remaining_contracts,
                                total_pnl, "PartialClose", final_exit_reason, asian_range
                            )
                            logger.info(f"[{date_str}] FINAL EXIT: Remaining {remaining_contracts} contracts closed at end of trading day (5 PM ET), price={final_exit_price:.2f} | Total PnL=${total_pnl:.2f}")
                            return trade
                        
                        # Check stop loss (now at break-even)
                        if is_long:
                            if candle2['low'] <= new_stop_loss:
                                final_exit_price = new_stop_loss
                                final_exit_reason = "SL"
                                final_pnl = MGC_SPECS.calculate_pnl(partial_close_price, final_exit_price, remaining_contracts, is_long)
                                total_pnl = partial_pnl + final_pnl
                                trade = self._create_trade_with_partial(
                                    entry_time, current_time, current_time2, entry_price, 
                                    partial_close_price, final_exit_price, stop_loss, take_profit,
                                    is_long, partial_close_contracts, remaining_contracts,
                                    total_pnl, "PartialClose", final_exit_reason, asian_range
                                )
                                logger.info(f"[{date_str}] FINAL EXIT: Remaining {remaining_contracts} contracts hit BE stop at {final_exit_price:.2f} | Total PnL=${total_pnl:.2f}")
                                return trade
                        else:
                            if candle2['high'] >= new_stop_loss:
                                final_exit_price = new_stop_loss
                                final_exit_reason = "SL"
                                final_pnl = MGC_SPECS.calculate_pnl(partial_close_price, final_exit_price, remaining_contracts, is_long)
                                total_pnl = partial_pnl + final_pnl
                                trade = self._create_trade_with_partial(
                                    entry_time, current_time, current_time2, entry_price,
                                    partial_close_price, final_exit_price, stop_loss, take_profit,
                                    is_long, partial_close_contracts, remaining_contracts,
                                    total_pnl, "PartialClose", final_exit_reason, asian_range
                                )
                                logger.info(f"[{date_str}] FINAL EXIT: Remaining {remaining_contracts} contracts hit BE stop at {final_exit_price:.2f} | Total PnL=${total_pnl:.2f}")
                                return trade
                        
                        # Check take profit for remaining position
                        if is_long:
                            if candle2['high'] >= take_profit:
                                final_exit_price = take_profit
                                final_exit_reason = "TP"
                                final_pnl = MGC_SPECS.calculate_pnl(partial_close_price, final_exit_price, remaining_contracts, is_long)
                                total_pnl = partial_pnl + final_pnl
                                trade = self._create_trade_with_partial(
                                    entry_time, current_time, current_time2, entry_price,
                                    partial_close_price, final_exit_price, stop_loss, take_profit,
                                    is_long, partial_close_contracts, remaining_contracts,
                                    total_pnl, "PartialClose", final_exit_reason, asian_range
                                )
                                logger.info(f"[{date_str}] FINAL EXIT: Remaining {remaining_contracts} contracts hit TP at {final_exit_price:.2f} | Total PnL=${total_pnl:.2f}")
                                return trade
                        else:
                            if candle2['low'] <= take_profit:
                                final_exit_price = take_profit
                                final_exit_reason = "TP"
                                final_pnl = MGC_SPECS.calculate_pnl(partial_close_price, final_exit_price, remaining_contracts, is_long)
                                total_pnl = partial_pnl + final_pnl
                                trade = self._create_trade_with_partial(
                                    entry_time, current_time, current_time2, entry_price,
                                    partial_close_price, final_exit_price, stop_loss, take_profit,
                                    is_long, partial_close_contracts, remaining_contracts,
                                    total_pnl, "PartialClose", final_exit_reason, asian_range
                                )
                                logger.info(f"[{date_str}] FINAL EXIT: Remaining {remaining_contracts} contracts hit TP at {final_exit_price:.2f} | Total PnL=${total_pnl:.2f}")
                                return trade
                    
                    # If we reach here, remaining position still open at end of data
                    # Close it at last price (end of trading day)
                    last_candle2 = remaining_data_after_12pm.iloc[-1] if len(remaining_data_after_12pm) > 0 else candle
                    final_exit_price = last_candle2['close']
                    final_exit_reason = "EndOfDay"
                    final_pnl = MGC_SPECS.calculate_pnl(partial_close_price, final_exit_price, remaining_contracts, is_long)
                    total_pnl = partial_pnl + final_pnl
                    trade = self._create_trade_with_partial(
                        entry_time, current_time, last_candle2['timestamp'], entry_price,
                        partial_close_price, final_exit_price, stop_loss, take_profit,
                        is_long, partial_close_contracts, remaining_contracts,
                        total_pnl, "PartialClose", final_exit_reason, asian_range
                    )
                    logger.info(f"[{date_str}] FINAL EXIT: Remaining {remaining_contracts} contracts closed at end of trading day, price={final_exit_price:.2f} | Total PnL=${total_pnl:.2f}")
                    return trade
                else:
                    # No remaining contracts (partial_close_percent = 1.0), full exit
                    exit_price = partial_close_price
                    exit_reason = "TimeExit"
                    trade = self._create_trade(
                        entry_time, current_time, entry_price, exit_price,
                        stop_loss, take_profit, is_long, exit_reason, asian_range
                    )
                    logger.info(f"[{date_str}] EXIT: Time exit at 12 PM ET, price={exit_price:.2f} | PnL=${trade.pnl:.2f}")
                    return trade
        
        # Should not reach here, but if we do, exit at last price
        last_candle = remaining_data.iloc[-1]
        exit_price = last_candle['close']
        exit_reason = "TimeExit"
        trade = self._create_trade(
            entry_time, last_candle['timestamp'], entry_price, exit_price,
            stop_loss, take_profit, is_long, exit_reason, asian_range
        )
        logger.info(f"[{date_str}] EXIT: End of data, price={exit_price:.2f} | PnL=${trade.pnl:.2f}")
        return trade
    
    def _create_trade_with_partial(
        self,
        entry_time: pd.Timestamp,
        partial_close_time: pd.Timestamp,
        final_exit_time: pd.Timestamp,
        entry_price: float,
        partial_close_price: float,
        final_exit_price: float,
        stop_loss: float,
        take_profit: float,
        is_long: bool,
        partial_close_contracts: int,
        remaining_contracts: int,
        total_pnl: float,
        exit_reason: str,
        final_exit_reason: str,
        asian_range: AsianRange
    ) -> Trade:
        """Create a trade with partial close"""
        # Calculate risk amount based on original position
        risk_amount = self.position_manager.calculate_risk_amount(
            entry_price, stop_loss, self.contracts_per_trade, is_long
        )
        pnl_pct = (total_pnl / risk_amount * 100) if risk_amount > 0 else 0
        
        # Calculate risk:reward
        risk_ticks = MGC_SPECS.price_to_ticks(abs(entry_price - stop_loss))
        reward_ticks = MGC_SPECS.price_to_ticks(abs(final_exit_price - entry_price))
        rr = reward_ticks / risk_ticks if risk_ticks > 0 else 0
        
        # Calculate duration (from entry to final exit)
        duration = (final_exit_time - entry_time).total_seconds() / 60
        
        return Trade(
            entry_time=entry_time,
            exit_time=partial_close_time,  # First exit time
            entry_price=entry_price,
            exit_price=partial_close_price,  # Partial close price
            stop_loss=stop_loss,
            take_profit=take_profit,
            contracts=self.contracts_per_trade,
            is_long=is_long,
            pnl=total_pnl,
            pnl_pct=pnl_pct,
            risk_reward=rr,
            duration_minutes=int(duration),
            exit_reason=exit_reason,
            asian_range_size=asian_range.range_size,
            tp_multiplier=self.tp_multiplier,
            partial_close_price=partial_close_price,
            partial_close_contracts=partial_close_contracts,
            remaining_contracts=remaining_contracts,
            final_exit_price=final_exit_price,
            final_exit_time=final_exit_time,
            final_exit_reason=final_exit_reason
        )
    
    def _create_trade(
        self,
        entry_time: pd.Timestamp,
        exit_time: pd.Timestamp,
        entry_price: float,
        exit_price: float,
        stop_loss: float,
        take_profit: float,
        is_long: bool,
        exit_reason: str,
        asian_range: AsianRange
    ) -> Trade:
        """Create and return a Trade object"""
        # Round prices
        entry_price = MGC_SPECS.round_to_tick(entry_price)
        exit_price = MGC_SPECS.round_to_tick(exit_price)
        stop_loss = MGC_SPECS.round_to_tick(stop_loss)
        
        # Calculate PnL
        pnl = MGC_SPECS.calculate_pnl(entry_price, exit_price, self.contracts_per_trade, is_long)
        
        # Calculate risk amount
        risk_amount = self.position_manager.calculate_risk_amount(
            entry_price, stop_loss, self.contracts_per_trade, is_long
        )
        pnl_pct = (pnl / risk_amount * 100) if risk_amount > 0 else 0
        
        # Calculate risk:reward
        risk_ticks = MGC_SPECS.price_to_ticks(abs(entry_price - stop_loss))
        reward_ticks = MGC_SPECS.price_to_ticks(abs(exit_price - entry_price))
        rr = reward_ticks / risk_ticks if risk_ticks > 0 else 0
        
        # Calculate duration
        duration = (exit_time - entry_time).total_seconds() / 60
        
        return Trade(
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=entry_price,
            exit_price=exit_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            contracts=self.contracts_per_trade,
            is_long=is_long,
            pnl=pnl,
            pnl_pct=pnl_pct,
            risk_reward=rr,
            duration_minutes=int(duration),
            exit_reason=exit_reason,
            asian_range_size=asian_range.range_size,
            tp_multiplier=self.tp_multiplier
        )
    
    def run_backtest(self):
        """Run the complete backtest"""
        logger.info("Starting Asian Range Breakout backtest...")
        
        equity = self.initial_equity
        self.equity_curve = [equity]
        
        # Get all trading dates
        trading_dates = self.get_trading_dates()
        logger.info(f"Processing {len(trading_dates)} trading days")
        
        # Process each trading day
        for date in trading_dates:
            trade = self.process_trading_day(date)
            
            if trade:
                self.trades.append(trade)
                equity += trade.pnl
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
        
        # Expectancy
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        expectancy = (win_rate / 100 * avg_win) - ((100 - win_rate) / 100 * abs(avg_loss))
        
        # Average Asian range size
        avg_range_size = np.mean([r.range_size for r in self.asian_ranges]) if self.asian_ranges else 0
        
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
            'total_return_pct': ((self.equity_curve[-1] - self.equity_curve[0]) / self.equity_curve[0] * 100) if len(self.equity_curve) > 1 else 0,
            'avg_asian_range_size': avg_range_size,
            'days_with_trades': len(self.trades),
            'total_days': len(self.asian_ranges)
        }

