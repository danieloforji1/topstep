"""
Strategy Runner
Orchestrates backtests and manages strategy execution
"""
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from .base_strategy import BaseStrategy
from .backtest_engine import BacktestEngine
from .performance_analyzer import PerformanceAnalyzer
from .data_manager import DataManager

logger = logging.getLogger(__name__)


class StrategyRunner:
    """
    Orchestrates strategy backtests
    """
    
    def __init__(
        self,
        data_manager: Optional[DataManager] = None,
        initial_equity: float = 50000.0,
        commission_per_contract: float = 2.50,
        slippage_ticks: float = 1.0
    ):
        """
        Initialize strategy runner
        
        Args:
            data_manager: DataManager instance (creates new if None)
            initial_equity: Starting equity
            commission_per_contract: Commission per contract
            slippage_ticks: Slippage in ticks
        """
        self.data_manager = data_manager or DataManager()
        self.initial_equity = initial_equity
        self.commission_per_contract = commission_per_contract
        self.slippage_ticks = slippage_ticks
    
    def backtest_strategy(
        self,
        strategy: BaseStrategy,
        symbol: str,
        interval: str = "15m",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        tick_size: float = 0.25,
        tick_value: float = 5.0,
        contract_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run backtest for a single strategy
        
        Args:
            strategy: Strategy instance
            symbol: Trading symbol
            interval: Bar interval
            start_date: Start date
            end_date: End date
            tick_size: Price tick size
            tick_value: Dollar value per tick
            contract_id: Optional contract ID
            
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Running backtest for {strategy.name} on {symbol}")
        
        # Fetch data
        df = self.data_manager.fetch_bars(
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            contract_id=contract_id
        )
        
        if df.empty:
            raise Exception(f"No data fetched for {symbol}")
        
        # Create backtest engine
        engine = BacktestEngine(
            strategy=strategy,
            df=df,
            initial_equity=self.initial_equity,
            commission_per_contract=self.commission_per_contract,
            slippage_ticks=self.slippage_ticks,
            tick_size=tick_size,
            tick_value=tick_value,
            symbol=symbol
        )
        
        # Run backtest
        results = engine.run()
        
        # Analyze performance
        analyzer = PerformanceAnalyzer(
            trades=results['trades'],
            equity_curve=results['equity_curve'],
            initial_equity=self.initial_equity
        )
        
        # Add performance metrics to results
        results['metrics'] = analyzer.get_metrics_dict()
        results['report'] = analyzer.generate_report()
        results['analyzer'] = analyzer
        
        return results
    
    def compare_strategies(
        self,
        strategies: List[BaseStrategy],
        symbol: str,
        interval: str = "15m",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run backtests for multiple strategies and compare
        
        Args:
            strategies: List of strategy instances
            symbol: Trading symbol
            interval: Bar interval
            start_date: Start date
            end_date: End date
            **kwargs: Additional arguments for backtest_strategy
            
        Returns:
            Dictionary with comparison results
        """
        logger.info(f"Comparing {len(strategies)} strategies on {symbol}")
        
        results = {}
        for strategy in strategies:
            try:
                result = self.backtest_strategy(
                    strategy=strategy,
                    symbol=symbol,
                    interval=interval,
                    start_date=start_date,
                    end_date=end_date,
                    **kwargs
                )
                results[strategy.name] = result
            except Exception as e:
                logger.error(f"Error backtesting {strategy.name}: {e}")
                results[strategy.name] = {'error': str(e)}
        
        return results

