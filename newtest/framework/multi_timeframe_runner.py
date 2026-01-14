"""
Multi-Timeframe Strategy Runner
Handles fetching and aligning multiple timeframe data for strategies
"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import pandas as pd

from .strategy_runner import StrategyRunner
from .backtest_engine import BacktestEngine
from .performance_analyzer import PerformanceAnalyzer
from .data_manager import DataManager

# Import strategy - handle both relative and absolute imports
try:
    from ..strategies.multi_timeframe import MultiTimeframeStrategy
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
    from strategies.multi_timeframe import MultiTimeframeStrategy

logger = logging.getLogger(__name__)


class MultiTimeframeRunner:
    """
    Specialized runner for multi-timeframe strategies
    
    Fetches data from multiple timeframes (1m, 5m, 15m) and aligns them
    for the strategy to use.
    """
    
    def __init__(
        self,
        data_manager: Optional[DataManager] = None,
        initial_equity: float = 50000.0,
        commission_per_contract: float = 2.50,
        slippage_ticks: float = 1.0
    ):
        """
        Initialize multi-timeframe runner
        
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
    
    def run_backtest(
        self,
        strategy: MultiTimeframeStrategy,
        symbol: str,
        primary_interval: str = "15m",  # Primary timeframe for execution
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        tick_size: float = 0.25,
        tick_value: float = 5.0,
        contract_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run backtest for multi-timeframe strategy
        
        Args:
            strategy: MultiTimeframeStrategy instance
            symbol: Trading symbol
            primary_interval: Primary timeframe for execution (default: 15m)
            start_date: Start date
            end_date: End date
            tick_size: Price tick size
            tick_value: Dollar value per tick
            contract_id: Optional contract ID
            
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Running multi-timeframe backtest for {strategy.name} on {symbol}")
        
        # Fetch data for all timeframes
        logger.info("Fetching 1-minute data...")
        df_1m = self.data_manager.fetch_bars(
            symbol=symbol,
            interval="1m",
            start_date=start_date,
            end_date=end_date,
            contract_id=contract_id
        )
        
        logger.info("Fetching 5-minute data...")
        df_5m = self.data_manager.fetch_bars(
            symbol=symbol,
            interval="5m",
            start_date=start_date,
            end_date=end_date,
            contract_id=contract_id
        )
        
        logger.info("Fetching 15-minute data...")
        df_15m = self.data_manager.fetch_bars(
            symbol=symbol,
            interval="15m",
            start_date=start_date,
            end_date=end_date,
            contract_id=contract_id
        )
        
        if df_1m.empty or df_5m.empty or df_15m.empty:
            raise Exception(f"Failed to fetch data: 1m={len(df_1m)}, 5m={len(df_5m)}, 15m={len(df_15m)}")
        
        logger.info(f"Fetched data: 1m={len(df_1m)} bars, 5m={len(df_5m)} bars, 15m={len(df_15m)} bars")
        
        # Set timeframe data in strategy
        strategy.set_timeframe_data(df_1m, df_5m, df_15m)
        
        # Use primary interval for execution (typically 15m)
        primary_df = df_15m if primary_interval == "15m" else (df_5m if primary_interval == "5m" else df_1m)
        
        if primary_df.empty:
            raise Exception(f"No data for primary interval {primary_interval}")
        
        logger.info(f"Using {primary_interval} as primary timeframe ({len(primary_df)} bars)")
        
        # Create backtest engine
        engine = BacktestEngine(
            strategy=strategy,
            df=primary_df,
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

