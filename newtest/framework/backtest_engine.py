"""
Unified Backtest Engine
Core execution engine for running strategy backtests
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import logging

from .base_strategy import BaseStrategy, Signal, ExitReason, MarketData

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents an open position"""
    entry_time: datetime
    entry_price: float
    contracts: int
    is_long: bool
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    signal: Optional[Signal] = None
    metadata: Dict[str, Any] = None
    # Partial profit taking support
    contracts_remaining: Optional[int] = None  # For partial exits
    partial_profit_taken: bool = False  # Track if partial profit was taken
    entry_bar_index: Optional[int] = None  # Bar index when position was entered
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.contracts_remaining is None:
            self.contracts_remaining = self.contracts
    
    @property
    def notional_value(self) -> float:
        """Calculate notional value of position"""
        return abs(self.entry_price * self.contracts_remaining)


@dataclass
class Trade:
    """Represents a completed trade"""
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    contracts: int
    is_long: bool
    pnl: float
    pnl_pct: float
    duration_minutes: int
    exit_reason: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BacktestEngine:
    """
    Unified backtest engine that works with any BaseStrategy implementation
    
    Handles:
    - Bar-by-bar simulation
    - Order execution (market/limit)
    - Position tracking
    - P&L calculation
    - Slippage and commission modeling
    """
    
    def __init__(
        self,
        strategy: BaseStrategy,
        df: pd.DataFrame,
        initial_equity: float = 50000.0,
        commission_per_contract: float = 2.50,  # $2.50 per contract round trip
        slippage_ticks: float = 1.0,  # 1 tick slippage for market orders
        tick_size: float = 0.25,  # Default tick size (can be overridden)
        tick_value: float = 5.0,  # Default tick value (can be overridden)
        symbol: str = "MES"
    ):
        """
        Initialize backtest engine
        
        Args:
            strategy: Strategy instance implementing BaseStrategy
            df: OHLCV DataFrame with columns: timestamp, open, high, low, close, volume
            initial_equity: Starting capital
            commission_per_contract: Commission per contract per round trip
            slippage_ticks: Slippage in ticks for market orders
            tick_size: Price tick size
            tick_value: Dollar value per tick
            symbol: Trading symbol
        """
        self.strategy = strategy
        self.df = df.copy()
        self.initial_equity = initial_equity
        self.commission_per_contract = commission_per_contract
        self.slippage_ticks = slippage_ticks
        self.tick_size = tick_size
        self.tick_value = tick_value
        self.symbol = symbol
        
        # Validate DataFrame
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_columns if col not in self.df.columns]
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.df['timestamp']):
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        # Sort by timestamp
        self.df = self.df.sort_values('timestamp').reset_index(drop=True)
        
        # Results
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.current_position: Optional[Position] = None
        
        # Account state
        self.equity = initial_equity
        self.cash = initial_equity
        self.realized_pnl = 0.0
        
        # Statistics
        self.total_commission = 0.0
        self.total_slippage = 0.0
    
    def round_to_tick(self, price: float) -> float:
        """Round price to nearest tick"""
        return round(price / self.tick_size) * self.tick_size
    
    def price_to_ticks(self, price: float) -> int:
        """Convert price difference to ticks"""
        return int(round(price / self.tick_size))
    
    def calculate_pnl(
        self,
        entry_price: float,
        exit_price: float,
        contracts: int,
        is_long: bool
    ) -> float:
        """
        Calculate P&L for a trade
        
        Args:
            entry_price: Entry price
            exit_price: Exit price
            contracts: Number of contracts
            is_long: True for long, False for short
            
        Returns:
            P&L in dollars
        """
        if is_long:
            price_diff = exit_price - entry_price
        else:
            price_diff = entry_price - exit_price
        
        ticks = self.price_to_ticks(price_diff)
        return ticks * self.tick_value * contracts
    
    def apply_slippage(self, price: float, is_long: bool, is_market_order: bool = True) -> float:
        """
        Apply slippage to order price
        
        Args:
            price: Base price
            is_long: True for long (buy), False for short (sell)
            is_market_order: True for market order, False for limit
            
        Returns:
            Execution price with slippage
        """
        if not is_market_order:
            return price  # No slippage on limit orders
        
        slippage = self.slippage_ticks * self.tick_size
        if is_long:
            return self.round_to_tick(price + slippage)  # Buy at higher price
        else:
            return self.round_to_tick(price - slippage)  # Sell at lower price
    
    def apply_commission(self, contracts: int) -> float:
        """Calculate commission for trade"""
        return contracts * self.commission_per_contract
    
    def enter_position(self, signal: Signal, market_data: MarketData, bar_index: int) -> bool:
        """
        Enter a new position based on signal
        
        Args:
            signal: Trading signal
            market_data: Current market data
            bar_index: Current bar index
            
        Returns:
            True if position entered, False otherwise
        """
        if self.current_position is not None:
            return False  # Already in position
        
        if signal.direction == "FLAT":
            return False
        
        # Calculate position size
        contracts = self.strategy.calculate_position_size(
            signal=signal,
            account_equity=self.equity,
            market_data=market_data,
            historical_data=self.df.iloc[:bar_index+1]
        )
        
        if contracts <= 0:
            return False
        
        # Determine entry price (use signal price or current close)
        entry_price = signal.entry_price if signal.entry_price else market_data.close
        
        # Check if this is a limit order
        is_limit_order = signal.metadata.get('order_type') == 'LIMIT'
        is_long = signal.direction == "LONG"
        
        if is_limit_order:
            # For limit orders, check if price touched the limit during the bar
            limit_price = entry_price
            if is_long:
                # BID (buy limit): fill if low touched or went below limit
                if market_data.low <= limit_price:
                    execution_price = limit_price  # Filled at limit
                else:
                    return False  # Limit order not filled
            else:
                # ASK (sell limit): fill if high touched or went above limit
                if market_data.high >= limit_price:
                    execution_price = limit_price  # Filled at limit
                else:
                    return False  # Limit order not filled
        else:
            # Market order: apply slippage
            execution_price = self.apply_slippage(entry_price, is_long, is_market_order=True)
        
        # Apply commission
        commission = self.apply_commission(contracts)
        
        # Check if we have enough capital (simplified - assume margin requirements)
        # For futures, margin is typically 5-10% of notional
        notional = execution_price * contracts
        margin_required = notional * 0.05  # 5% margin requirement
        
        if self.cash < margin_required:
            logger.warning(f"Insufficient capital: need ${margin_required:.2f}, have ${self.cash:.2f}")
            return False
        
        # Create position
        self.current_position = Position(
            entry_time=market_data.timestamp,
            entry_price=execution_price,
            contracts=contracts,
            is_long=is_long,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            signal=signal,
            entry_bar_index=bar_index
        )
        
        # Update account
        self.cash -= commission
        self.total_commission += commission
        self.total_slippage += abs(execution_price - entry_price) * contracts * self.tick_value / self.tick_size
        
        logger.debug(f"Entered {signal.direction} position: {contracts} contracts @ ${execution_price:.2f}")
        
        return True
    
    def exit_position(self, exit_reason: ExitReason, market_data: MarketData, bar_index: int) -> Optional[Trade]:
        """
        Exit current position
        
        Args:
            exit_reason: Reason for exit
            market_data: Current market data
            bar_index: Current bar index
            
        Returns:
            Trade object if position exited, None otherwise
        """
        if self.current_position is None:
            return None
        
        pos = self.current_position
        
        # Determine exit price
        # Check if stop loss or take profit was hit
        exit_price = market_data.close
        
        if exit_reason.reason == "STOP_LOSS" and pos.stop_loss:
            exit_price = pos.stop_loss
        elif exit_reason.reason == "TAKE_PROFIT" and pos.take_profit:
            exit_price = pos.take_profit
        else:
            # Use current bar's high/low if stop/target was hit
            if pos.is_long:
                if pos.stop_loss and market_data.low <= pos.stop_loss:
                    exit_price = pos.stop_loss
                elif pos.take_profit and market_data.high >= pos.take_profit:
                    exit_price = pos.take_profit
            else:
                if pos.stop_loss and market_data.high >= pos.stop_loss:
                    exit_price = pos.stop_loss
                elif pos.take_profit and market_data.low <= pos.take_profit:
                    exit_price = pos.take_profit
        
        # Apply slippage
        execution_price = self.apply_slippage(exit_price, not pos.is_long, is_market_order=True)
        
        # Calculate P&L
        pnl = self.calculate_pnl(
            entry_price=pos.entry_price,
            exit_price=execution_price,
            contracts=pos.contracts,
            is_long=pos.is_long
        )
        
        # Apply commission
        commission = self.apply_commission(pos.contracts)
        
        # Calculate duration
        duration = (market_data.timestamp - pos.entry_time).total_seconds() / 60
        
        # Create trade record
        trade = Trade(
            entry_time=pos.entry_time,
            exit_time=market_data.timestamp,
            entry_price=pos.entry_price,
            exit_price=execution_price,
            contracts=pos.contracts,
            is_long=pos.is_long,
            pnl=pnl - commission,  # Net P&L after commission
            pnl_pct=(pnl - commission) / (pos.entry_price * pos.contracts) * 100 if pos.entry_price * pos.contracts > 0 else 0,
            duration_minutes=int(duration),
            exit_reason=exit_reason.reason,
            stop_loss=pos.stop_loss,
            take_profit=pos.take_profit,
            metadata=exit_reason.metadata
        )
        
        # Update account
        self.cash += pnl - commission
        self.realized_pnl += pnl - commission
        self.total_commission += commission
        self.equity = self.cash  # Simplified - assume no unrealized P&L
        
        # Update equity curve
        self.equity_curve.append(self.equity)
        
        # Clear position
        self.current_position = None
        
        logger.debug(f"Exited position: {trade.exit_reason}, P&L: ${trade.pnl:.2f}")
        
        return trade
    
    def exit_partial_position(
        self,
        contracts_to_exit: int,
        exit_price: float,
        market_data: MarketData,
        bar_index: int
    ) -> Optional[Trade]:
        """
        Exit partial position (for partial profit taking)
        
        Args:
            contracts_to_exit: Number of contracts to exit
            exit_price: Exit price
            market_data: Current market data
            bar_index: Current bar index
            
        Returns:
            Trade object for the partial exit
        """
        if self.current_position is None:
            return None
        
        pos = self.current_position
        
        if contracts_to_exit >= pos.contracts:
            # If trying to exit all, use full exit
            return None
        
        # Apply slippage
        execution_price = self.apply_slippage(exit_price, not pos.is_long, is_market_order=True)
        
        # Calculate P&L for partial exit
        pnl = self.calculate_pnl(
            entry_price=pos.entry_price,
            exit_price=execution_price,
            contracts=contracts_to_exit,
            is_long=pos.is_long
        )
        
        # Apply commission
        commission = self.apply_commission(contracts_to_exit)
        
        # Calculate duration
        duration = (market_data.timestamp - pos.entry_time).total_seconds() / 60
        
        # Create trade record for partial exit
        trade = Trade(
            entry_time=pos.entry_time,
            exit_time=market_data.timestamp,
            entry_price=pos.entry_price,
            exit_price=execution_price,
            contracts=contracts_to_exit,
            is_long=pos.is_long,
            pnl=pnl - commission,
            pnl_pct=(pnl - commission) / (pos.entry_price * contracts_to_exit) * 100 if pos.entry_price * contracts_to_exit > 0 else 0,
            duration_minutes=int(duration),
            exit_reason="PARTIAL_PROFIT",
            stop_loss=pos.stop_loss,
            take_profit=pos.take_profit,
            metadata={'partial_exit': True, 'contracts_remaining': pos.contracts - contracts_to_exit}
        )
        
        # Update account (partial exit)
        self.cash += pnl - commission
        self.realized_pnl += pnl - commission
        self.total_commission += commission
        
        logger.debug(f"Partial exit: {contracts_to_exit} contracts @ ${execution_price:.2f}, P&L: ${pnl - commission:.2f}")
        
        return trade
    
    def run(self) -> Dict[str, Any]:
        """
        Run the backtest
        
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Starting backtest for {self.strategy.name}")
        logger.info(f"Data points: {len(self.df)}, Initial equity: ${self.initial_equity:.2f}")
        
        # Reset strategy
        self.strategy.reset()
        
        # Initialize equity curve
        self.equity_curve = [self.initial_equity]
        
        # Process each bar
        for i in range(len(self.df)):
            row = self.df.iloc[i]
            
            # Create MarketData object
            market_data = MarketData(
                timestamp=row['timestamp'],
                symbol=self.symbol,
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume']
            )
            
            # Get historical data up to current bar
            historical_data = self.df.iloc[:i+1]
            
            # Call strategy's on_bar hook
            self.strategy.on_bar(market_data, historical_data)
            
            # Check for exit first (if in position)
            if self.current_position is not None:
                exit_reason = self.strategy.check_exit(
                    position=self.current_position,
                    market_data=market_data,
                    historical_data=historical_data
                )
                
                if exit_reason:
                    # Handle partial profit taking
                    if exit_reason.reason == "PARTIAL_PROFIT":
                        partial_pct = exit_reason.metadata.get('partial_pct', 0.5)
                        target_price = exit_reason.metadata.get('target_price', market_data.close)
                        
                        # Calculate contracts to exit
                        contracts_to_exit = max(1, int(self.current_position.contracts * partial_pct))
                        contracts_remaining = self.current_position.contracts - contracts_to_exit
                        
                        if contracts_remaining > 0:
                            # Exit partial position
                            partial_trade = self.exit_partial_position(
                                contracts_to_exit=contracts_to_exit,
                                exit_price=target_price,
                                market_data=market_data,
                                bar_index=i
                            )
                            if partial_trade:
                                self.trades.append(partial_trade)
                            
                            # Update position with remaining contracts
                            self.current_position.contracts = contracts_remaining
                            self.current_position.contracts_remaining = contracts_remaining
                            self.current_position.partial_profit_taken = True
                        else:
                            # Exit full position if remaining would be 0
                            trade = self.exit_position(exit_reason, market_data, i)
                            if trade:
                                self.trades.append(trade)
                    else:
                        # Full exit
                        trade = self.exit_position(exit_reason, market_data, i)
                        if trade:
                            self.trades.append(trade)
            
            # Generate new signal (if not in position)
            if self.current_position is None:
                signal = self.strategy.generate_signal(
                    market_data=market_data,
                    historical_data=historical_data,
                    current_position=None
                )
                
                if signal and signal.direction != "FLAT":
                    self.enter_position(signal, market_data, i)
            
            # Update equity curve (include unrealized P&L if in position)
            if self.current_position:
                # Calculate unrealized P&L (use contracts_remaining if partial exit occurred)
                contracts_for_pnl = self.current_position.contracts_remaining if self.current_position.contracts_remaining else self.current_position.contracts
                unrealized_pnl = self.calculate_pnl(
                    entry_price=self.current_position.entry_price,
                    exit_price=market_data.close,
                    contracts=contracts_for_pnl,
                    is_long=self.current_position.is_long
                )
                self.equity = self.cash + unrealized_pnl
            else:
                self.equity = self.cash
            
            self.equity_curve.append(self.equity)
        
        # Close any remaining position at end
        if self.current_position is not None:
            last_row = self.df.iloc[-1]
            market_data = MarketData(
                timestamp=last_row['timestamp'],
                symbol=self.symbol,
                open=last_row['open'],
                high=last_row['high'],
                low=last_row['low'],
                close=last_row['close'],
                volume=last_row['volume']
            )
            exit_reason = ExitReason(
                reason="END_OF_DATA",
                timestamp=market_data.timestamp
            )
            trade = self.exit_position(exit_reason, market_data, len(self.df) - 1)
            if trade:
                self.trades.append(trade)
        
        logger.info(f"Backtest complete: {len(self.trades)} trades, Final equity: ${self.equity:.2f}")
        
        return {
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'initial_equity': self.initial_equity,
            'final_equity': self.equity,
            'total_pnl': self.realized_pnl,
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage,
            'num_trades': len(self.trades)
        }

