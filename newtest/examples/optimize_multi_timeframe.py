"""
Multi-Timeframe Convergence Strategy Optimization
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

from framework.multi_timeframe_runner import MultiTimeframeRunner
from framework.data_manager import DataManager
from strategies.multi_timeframe import MultiTimeframeStrategy
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
    runner: MultiTimeframeRunner,
    symbol: str,
    config: dict,
    start_date: datetime,
    end_date: datetime,
    primary_interval: str = "15m"
) -> dict:
    """Run a single backtest with given config"""
    try:
        strategy = MultiTimeframeStrategy(config)
        results = runner.run_backtest(
            strategy=strategy,
            symbol=symbol,
            primary_interval=primary_interval,
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
    parser = argparse.ArgumentParser(description='Optimize Multi-Timeframe Strategy')
    parser.add_argument('--symbol', default='MES', help='Trading symbol')
    parser.add_argument('--start-date', type=str, default='2026-01-01', help='Start date')
    parser.add_argument('--end-date', type=str, default=None, help='End date')
    parser.add_argument('--primary-interval', default='15m', choices=['1m', '5m', '15m'])
    
    args = parser.parse_args()
    
    # Parse dates
    start_date = datetime.fromisoformat(args.start_date).replace(tzinfo=timezone.utc)
    if args.end_date:
        end_date = datetime.fromisoformat(args.end_date).replace(tzinfo=timezone.utc)
    else:
        end_date = datetime.now(timezone.utc) - timedelta(days=1)
    
    # Initialize
    data_manager = DataManager(cache_dir="newtest/results/cache")
    runner = MultiTimeframeRunner(
        data_manager=data_manager,
        initial_equity=50000.0,
        commission_per_contract=2.50,
        slippage_ticks=1.0
    )
    
    # Base config (PRODUCTION-READY - Best optimized parameters)
    base_config = {
        'lookback_period': 20,
        'momentum_period': 10,
        'mean_reversion_period': 20,
        'atr_period': 14,
        'atr_multiplier_stop': 1.0,    # Best: 1.0
        'atr_multiplier_target': 1.5,   # Best: 1.5
        'risk_per_trade': 100.0,
        'max_hold_bars': 30,            # Best: 30
        'timeframe_weights': {
            '1m': 0.25,
            '5m': 0.35,
            '15m': 0.40
        },
        # Profit capture features (PRODUCTION-READY - Enabled by default)
        'use_trailing_stop': True,
        'trailing_stop_atr_multiplier': 0.5,
        'trailing_stop_activation_pct': 0.001,  # Activate after 0.1% profit
        'use_partial_profit': True,
        'partial_profit_pct': 0.5,  # Take 50% at first target
        'partial_profit_target_atr': 0.75  # Take profit at 0.75x ATR
    }
    
    # Parameter grid (REDUCED for faster optimization)
    # Focus on middle-range thresholds for better trade-off
    param_grid = {
        'convergence_threshold': [0.2, 0.25, 0.3, 0.35],  # Middle range for balance
        'divergence_threshold': [0.2, 0.3],
        'atr_multiplier_stop': [1.0, 1.5],
        'atr_multiplier_target': [1.5, 2.0],
        'max_hold_bars': [30, 40, 50]
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
            primary_interval=args.primary_interval
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
    output_file = os.path.join(results_dir, f"multi_timeframe_optimization_{args.symbol}_{date_str}.csv")
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

