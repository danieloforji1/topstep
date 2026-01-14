"""
Unified Backtesting Framework
Core components for strategy backtesting and comparison
"""

from .base_strategy import BaseStrategy, Signal, ExitReason
from .backtest_engine import BacktestEngine, Trade, Position
from .performance_analyzer import PerformanceAnalyzer
from .data_manager import DataManager
from .strategy_runner import StrategyRunner

__all__ = [
    'BaseStrategy',
    'Signal',
    'ExitReason',
    'BacktestEngine',
    'Trade',
    'Position',
    'PerformanceAnalyzer',
    'DataManager',
    'StrategyRunner'
]

