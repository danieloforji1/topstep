"""
Liquidity Provision Strategy Optimization
Grid search and walk-forward optimization
"""
import sys
import os
import logging
from datetime import datetime, timedelta, timezone
import argparse
import pandas as pd
import numpy as np
from itertools import product
import time

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

from framework.strategy_runner import StrategyRunner
from framework.data_manager import DataManager
from strategies.liquidity_provision import LiquidityProvisionStrategy
from dotenv import load_dotenv

# Try to load .env
try:
    load_dotenv()
except Exception:
    pass

logging.basicConfig(
    level=logging.WARNING,  # Reduce logging noise
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def run_single_backtest(
    runner: StrategyRunner,
    symbol: str,
    config: dict,
    start_date: datetime,
    end_date: datetime,
    interval: str = "5m"
) -> dict:
    """Run a single backtest with given config"""
    try:
        strategy = LiquidityProvisionStrategy(config)
        results = runner.backtest_strategy(
            strategy=strategy,
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            tick_size=0.25,
            tick_value=5.0
        )
        
        metrics = results['metrics']
        return {
            'sharpe_ratio': metrics['sharpe_ratio'],
            'total_return': metrics['total_return'],
            'max_drawdown': metrics['max_drawdown'],
            'win_rate': metrics['win_rate'],
            'profit_factor': metrics['profit_factor'],
            'num_trades': metrics['num_trades'],
            'consistency_score': metrics['consistency_score'],
            'success': True
        }
    except Exception as e:
        logger.warning(f"Backtest failed: {e}")
        return {'success': False, 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description='Optimize Liquidity Provision Strategy')
    parser.add_argument('--symbol', default='MES', help='Trading symbol')
    parser.add_argument('--start-date', type=str, default='2025-12-12', help='Start date')
    parser.add_argument('--end-date', type=str, default=None, help='End date')
    parser.add_argument('--interval', default='5m', help='Bar interval')
    
    args = parser.parse_args()
    
    # Parse dates
    start_date = datetime.fromisoformat(args.start_date).replace(tzinfo=timezone.utc)
    if args.end_date:
        end_date = datetime.fromisoformat(args.end_date).replace(tzinfo=timezone.utc)
    else:
        end_date = datetime.now(timezone.utc) - timedelta(days=1)
    
    # Initialize
    data_manager = DataManager(cache_dir="newtest/results/cache")
    runner = StrategyRunner(
        data_manager=data_manager,
        initial_equity=50000.0,
        commission_per_contract=2.50,
        slippage_ticks=0.5  # Lower for limit orders
    )
    
    # Base config
    base_config = {
        'atr_period': 14,
        'atr_multiplier_stop': 1.0,
        'risk_per_trade': 100.0,
        'cancel_on_reversal': True
    }
    
    # Parameter grid (REDUCED for faster optimization)
    param_grid = {
        'imbalance_lookback': [3, 5, 7],
        'imbalance_threshold': [0.1, 0.15, 0.2],  # Lower range
        'adverse_selection_threshold': [0.5, 0.6, 0.7],  # Higher range
        'favorable_fill_threshold': [0.4, 0.5, 0.6],  # Lower range
        'spread_target_ticks': [2, 3, 4],  # Higher targets for better R:R
        'max_hold_bars': [15, 20, 25],
        'atr_multiplier_stop': [0.75, 1.0, 1.25]  # Tighter stops
    }
    
    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))
    
    print(f"Testing {len(combinations)} parameter combinations...")
    print(f"Symbol: {args.symbol}, Date Range: {start_date.date()} to {end_date.date()}")
    print("="*80)
    
    results = []
    for i, combo in enumerate(combinations):
        config = base_config.copy()
        config.update(dict(zip(param_names, combo)))
        
        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{len(combinations)}")
        
        result = run_single_backtest(
            runner=runner,
            symbol=args.symbol,
            config=config,
            start_date=start_date,
            end_date=end_date,
            interval=args.interval
        )
        
        if result.get('success'):
            row = dict(zip(param_names, combo))
            row.update(result)
            results.append(row)
        
        # Rate limiting
        time.sleep(0.1)
    
    if not results:
        print("No successful backtests!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Sort by Sharpe ratio
    df = df.sort_values('sharpe_ratio', ascending=False)
    
    # Print top 20
    print("\n" + "="*80)
    print("TOP 20 PARAMETER COMBINATIONS (by Sharpe Ratio)")
    print("="*80)
    print(df.head(20).to_string())
    
    # Save results
    results_dir = "newtest/results/backtests"
    os.makedirs(results_dir, exist_ok=True)
    
    date_str = f"{start_date.date()}_{end_date.date()}"
    output_file = os.path.join(results_dir, f"liquidity_provision_optimization_{args.symbol}_{date_str}.csv")
    df.to_csv(output_file, index=False)
    
    print(f"\nFull results saved to: {output_file}")
    
    # Print summary stats
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Total combinations tested: {len(combinations)}")
    print(f"Successful backtests: {len(results)}")
    print(f"Best Sharpe Ratio: {df['sharpe_ratio'].max():.2f}")
    print(f"Best Total Return: {df['total_return'].max():.2f}%")
    print(f"Best Win Rate: {df['win_rate'].max():.2f}%")
    print(f"Max Trades: {df['num_trades'].max()}")
    
    # Best config
    best = df.iloc[0]
    print(f"\nBest Configuration:")
    for param in param_names:
        print(f"  {param}: {best[param]}")


if __name__ == "__main__":
    main()

