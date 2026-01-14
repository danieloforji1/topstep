"""
Compare Trailing Stop vs Partial Profit Taking
Run backtests with different exit methods and compare results
"""
import sys
import os
import pandas as pd
from datetime import datetime
import logging

logging.basicConfig(level=logging.WARNING)  # Reduce noise

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
from framework.data_manager import DataManager
from framework.strategy_runner import StrategyRunner
from strategies.optimal_stopping import OptimalStoppingStrategy

def run_test(config_name, config):
    """Run a single backtest with given config"""
    print(f"\n{'='*80}")
    print(f"Testing: {config_name}")
    print(f"{'='*80}")
    
    # Load data
    data_manager = DataManager()
    df = data_manager.fetch_bars(
        symbol="MES",
        interval="15m",
        start_date=datetime(2025, 12, 7),
        end_date=datetime(2026, 1, 6)
    )
    
    # Create strategy with fresh config
    strategy = OptimalStoppingStrategy(config)
    
    # Verify config is applied
    print(f"  Config: trailing_stop={strategy.use_trailing_stop}, partial_profit={strategy.use_partial_profit}")
    
    # Run backtest
    runner = StrategyRunner(
        data_manager=data_manager,
        initial_equity=50000.0,
        commission_per_contract=2.50,
        slippage_ticks=1.0
    )
    
    result = runner.backtest_strategy(
        strategy=strategy,
        symbol="MES",
        interval="15m",
        start_date=datetime(2025, 12, 7),
        end_date=datetime(2026, 1, 6),
        tick_size=0.25,
        tick_value=5.0
    )
    
    metrics = result['metrics']
    trades = result['trades']
    
    # Analyze exit reasons
    if trades:
        exit_reasons = {}
        for trade in trades:
            reason = trade.exit_reason
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
    else:
        exit_reasons = {}
    
    print(f"  Total Return: {metrics['total_return']:.2f}%")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"  Total Trades: {len(trades)}")
    print(f"  Win Rate: {metrics['win_rate']:.2f}%")
    print(f"  Average Win: ${metrics['average_win']:.2f}")
    print(f"  Average Loss: ${metrics['average_loss']:.2f}")
    print(f"  Max Drawdown: {metrics['max_drawdown']:.2f}%")
    print(f"  Exit Reasons: {exit_reasons}")
    
    return {
        'name': config_name,
        'return': metrics['total_return'],
        'sharpe': metrics['sharpe_ratio'],
        'profit_factor': metrics['profit_factor'],
        'trades': len(trades),
        'win_rate': metrics['win_rate'],
        'avg_win': metrics['average_win'],
        'avg_loss': metrics['average_loss'],
        'max_dd': metrics['max_drawdown'],
        'exit_reasons': exit_reasons
    }

def main():
    base_config = {
        'lookback_window': 100,
        'min_opportunities_seen': 37,
        'score_threshold': 0.7,
        'momentum_weight': 0.4,
        'mean_reversion_weight': 0.3,
        'volatility_weight': 0.3,
        'atr_period': 14,
        'atr_multiplier_stop': 0.75,
        'atr_multiplier_target': 1.25,
        'risk_per_trade': 100.0,
        'max_hold_bars': 40,
    }
    
    results = []
    
    # Test 1: Baseline (no trailing, no partial)
    config1 = base_config.copy()
    config1['use_trailing_stop'] = False
    config1['use_partial_profit'] = False
    results.append(run_test("1. Baseline", config1))
    
    # Test 2: Trailing Stop Only (more aggressive)
    config2 = base_config.copy()
    config2['use_trailing_stop'] = True
    config2['trailing_stop_atr_multiplier'] = 0.3  # Tighter trailing stop
    config2['trailing_stop_activation_pct'] = 0.0005  # Activate after 0.05% profit
    config2['use_partial_profit'] = False
    results.append(run_test("2. Trailing Stop Only", config2))
    
    # Test 3: Partial Profit Only (more aggressive target)
    config3 = base_config.copy()
    config3['use_trailing_stop'] = False
    config3['use_partial_profit'] = True
    config3['partial_profit_pct'] = 0.5
    config3['partial_profit_target_atr'] = 0.3  # Very tight target (3-4 points)
    results.append(run_test("3. Partial Profit Only", config3))
    
    # Test 4: Both (aggressive settings)
    config4 = base_config.copy()
    config4['use_trailing_stop'] = True
    config4['trailing_stop_atr_multiplier'] = 0.3
    config4['trailing_stop_activation_pct'] = 0.0005
    config4['use_partial_profit'] = True
    config4['partial_profit_pct'] = 0.5
    config4['partial_profit_target_atr'] = 0.3
    results.append(run_test("4. Both (Aggressive)", config4))
    
    # Print comparison table
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Method':<25} {'Return':<10} {'Sharpe':<10} {'Profit Factor':<15} {'Trades':<10} {'Win Rate':<10}")
    print("-"*80)
    
    for r in results:
        print(f"{r['name']:<25} {r['return']:>8.2f}% {r['sharpe']:>8.2f} {r['profit_factor']:>13.2f} {r['trades']:>8} {r['win_rate']:>8.2f}%")
        print(f"  Exit Reasons: {r['exit_reasons']}")
    
    # Find best
    best = max(results, key=lambda x: x['return'])
    print(f"\n{'='*80}")
    print(f"BEST METHOD: {best['name']}")
    print(f"{'='*80}")
    print(f"  Return: {best['return']:.2f}%")
    print(f"  Sharpe: {best['sharpe']:.2f}")
    print(f"  Profit Factor: {best['profit_factor']:.2f}")
    print(f"  Trades: {best['trades']}")
    print(f"  Win Rate: {best['win_rate']:.2f}%")
    print(f"  Avg Win: ${best['avg_win']:.2f}")
    print(f"  Avg Loss: ${best['avg_loss']:.2f}")
    print(f"  Max DD: {best['max_dd']:.2f}%")

if __name__ == "__main__":
    main()
