# Unified Strategy Backtesting Framework

This framework provides a unified interface for backtesting and comparing multiple trading strategies.

## Structure

```
newtest/
├── framework/          # Core backtesting components
│   ├── base_strategy.py       # Abstract strategy interface
│   ├── backtest_engine.py     # Core execution engine
│   ├── performance_analyzer.py # Performance metrics
│   ├── data_manager.py        # Data fetching/caching
│   └── strategy_runner.py     # Orchestration
├── strategies/        # Strategy implementations
│   └── calendar_spread.py     # Calendar Spread strategy
├── comparison/        # Strategy comparison tools
├── configs/          # Strategy configuration files
└── results/          # Backtest results
```

## Quick Start

### 1. Basic Backtest

```python
from framework import BaseStrategy, StrategyRunner, DataManager
from strategies.calendar_spread import CalendarSpreadStrategy
from datetime import datetime

# Initialize
data_manager = DataManager()
runner = StrategyRunner(data_manager=data_manager)

# Create strategy
config = {
    'z_entry': 2.0,
    'z_exit': 0.5,
    'risk_per_trade': 100.0
}
strategy = CalendarSpreadStrategy(config)

# Run backtest
results = runner.backtest_strategy(
    strategy=strategy,
    symbol="MES",
    interval="15m",
    start_date=datetime(2025, 1, 1),
    end_date=datetime(2025, 11, 14)
)

# Print results
print(results['report'])
print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['metrics']['max_drawdown']:.2f}%")
```

### 2. Compare Multiple Strategies

```python
from framework import StrategyRunner
from strategies.calendar_spread import CalendarSpreadStrategy

# Create multiple strategies with different configs
strategies = [
    CalendarSpreadStrategy({'z_entry': 2.0, 'z_exit': 0.5}),
    CalendarSpreadStrategy({'z_entry': 2.5, 'z_exit': 0.3}),
]

# Compare
results = runner.compare_strategies(
    strategies=strategies,
    symbol="MES",
    interval="15m",
    start_date=datetime(2025, 1, 1),
    end_date=datetime(2025, 11, 14)
)

# Print comparison
for name, result in results.items():
    if 'error' not in result:
        print(f"{name}: Sharpe={result['metrics']['sharpe_ratio']:.2f}")
```

## Creating a New Strategy

1. Inherit from `BaseStrategy`:

```python
from framework.base_strategy import BaseStrategy, Signal, ExitReason, MarketData

class MyStrategy(BaseStrategy):
    def get_required_data(self) -> List[str]:
        return ["OHLCV"]
    
    def generate_signal(self, market_data, historical_data, current_position):
        # Your signal logic here
        if some_condition:
            return Signal(
                timestamp=market_data.timestamp,
                direction="LONG",
                entry_price=market_data.close,
                stop_loss=market_data.close - 10,
                take_profit=market_data.close + 20
            )
        return None
    
    def calculate_position_size(self, signal, account_equity, market_data, historical_data):
        # Your position sizing logic
        return 1
    
    def check_exit(self, position, market_data, historical_data):
        # Your exit logic
        return None
```

2. Use it in backtests:

```python
strategy = MyStrategy(config={})
results = runner.backtest_strategy(strategy, "MES", "15m", ...)
```

## Performance Metrics

All strategies are evaluated using standardized metrics:

- **Sharpe Ratio**: Risk-adjusted return
- **Sortino Ratio**: Downside risk-adjusted return
- **Max Drawdown**: Maximum peak-to-trough decline
- **Win Rate**: Percentage of winning trades
- **Profit Factor**: Gross profit / Gross loss
- **Calmar Ratio**: Annual return / Max drawdown
- **Consistency Score**: Measure of return consistency

## Next Steps

1. Implement remaining strategies (Optimal Stopping, Multi-Timeframe, etc.)
2. Build comparison dashboard
3. Add parameter optimization
4. Integrate best strategies into live trading

