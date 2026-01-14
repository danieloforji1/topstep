"""
Example: Run Optimal Stopping Strategy Backtest
"""
import sys
import os
import logging
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

from framework.strategy_runner import StrategyRunner
from framework.data_manager import DataManager
from strategies.optimal_stopping import OptimalStoppingStrategy
from dotenv import load_dotenv

# Try to load .env, but don't fail if it doesn't exist
try:
    load_dotenv()
except Exception:
    pass  # .env file is optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Run optimal stopping strategy backtest"""
    
    # Initialize
    data_manager = DataManager(cache_dir="newtest/results/cache")
    runner = StrategyRunner(
        data_manager=data_manager,
        initial_equity=50000.0
    )
    
    # Create strategy
    config = {
        'lookback_window': 100,
        'min_opportunities_seen': 37,
        'score_threshold': 0.6,
        'momentum_weight': 0.4,
        'mean_reversion_weight': 0.3,
        'volatility_weight': 0.3,
        'atr_period': 14,
        'atr_multiplier_stop': 1.5,
        'atr_multiplier_target': 2.0,
        'risk_per_trade': 100.0,
        'max_hold_bars': 50
    }
    
    strategy = OptimalStoppingStrategy(config)
    
    # Run backtest
    results = runner.backtest_strategy(
        strategy=strategy,
        symbol="MES",
        interval="15m",
        start_date=datetime(2025, 1, 1),
        end_date=datetime(2025, 11, 14),
        tick_size=0.25,
        tick_value=5.0
    )
    
    # Print results
    print("\n" + "="*80)
    print("OPTIMAL STOPPING STRATEGY BACKTEST RESULTS")
    print("="*80)
    print(results['report'])
    
    # Print key metrics
    metrics = results['metrics']
    print(f"\nKey Metrics:")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {metrics['max_drawdown']:.2f}%")
    print(f"  Win Rate: {metrics['win_rate']:.2f}%")
    print(f"  Total Return: {metrics['total_return']:.2f}%")
    print(f"  Total Trades: {metrics['num_trades']}")
    print(f"  Consistency Score: {metrics['consistency_score']:.2f}")
    
    # Save results
    import json
    results_file = "newtest/results/backtests/optimal_stopping_results.json"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    # Convert to JSON-serializable format
    results_dict = {
        'metrics': metrics,
        'num_trades': len(results['trades']),
        'initial_equity': results['initial_equity'],
        'final_equity': results['final_equity']
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()

