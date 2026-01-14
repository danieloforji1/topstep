"""
Test different TIME_STOP and SIGNAL_REVERSAL thresholds
"""
import sys
import os
import pandas as pd
from datetime import datetime
import logging

logging.basicConfig(level=logging.WARNING)

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
    
    # Create strategy
    strategy = OptimalStoppingStrategy(config)
    
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
        'use_trailing_stop': False,
        'use_partial_profit': False,
    }
    
    results = []
    
    # Test 1: Baseline (current optimized)
    config1 = base_config.copy()
    config1['max_hold_bars'] = 40
    config1['signal_reversal_threshold'] = 0.8
    config1['signal_reversal_min_profit_pct'] = 0.005
    config1['signal_reversal_min_hold_pct'] = 0.25
    results.append(run_test("1. Baseline (40 bars, 0.8 reversal, 0.5% profit)", config1))
    
    # Test 2: Longer hold time
    config2 = base_config.copy()
    config2['max_hold_bars'] = 60
    config2['signal_reversal_threshold'] = 0.8
    config2['signal_reversal_min_profit_pct'] = 0.005
    config2['signal_reversal_min_hold_pct'] = 0.25
    results.append(run_test("2. Longer Hold (60 bars)", config2))
    
    # Test 3: Shorter hold time
    config3 = base_config.copy()
    config3['max_hold_bars'] = 30
    config3['signal_reversal_threshold'] = 0.8
    config3['signal_reversal_min_profit_pct'] = 0.005
    config3['signal_reversal_min_hold_pct'] = 0.25
    results.append(run_test("3. Shorter Hold (30 bars)", config3))
    
    # Test 4: More sensitive reversal (lower threshold)
    config4 = base_config.copy()
    config4['max_hold_bars'] = 40
    config4['signal_reversal_threshold'] = 0.6  # More sensitive
    config4['signal_reversal_min_profit_pct'] = 0.005
    config4['signal_reversal_min_hold_pct'] = 0.25
    results.append(run_test("4. Sensitive Reversal (0.6 threshold)", config4))
    
    # Test 5: Less sensitive reversal (higher threshold)
    config5 = base_config.copy()
    config5['max_hold_bars'] = 40
    config5['signal_reversal_threshold'] = 0.9  # Less sensitive
    config5['signal_reversal_min_profit_pct'] = 0.005
    config5['signal_reversal_min_hold_pct'] = 0.25
    results.append(run_test("5. Less Sensitive Reversal (0.9 threshold)", config5))
    
    # Test 6: Lower profit requirement for reversal exit
    config6 = base_config.copy()
    config6['max_hold_bars'] = 40
    config6['signal_reversal_threshold'] = 0.8
    config6['signal_reversal_min_profit_pct'] = 0.002  # Lower profit requirement
    config6['signal_reversal_min_hold_pct'] = 0.25
    results.append(run_test("6. Lower Profit Req (0.2% profit)", config6))
    
    # Test 7: Higher profit requirement for reversal exit
    config7 = base_config.copy()
    config7['max_hold_bars'] = 40
    config7['signal_reversal_threshold'] = 0.8
    config7['signal_reversal_min_profit_pct'] = 0.01  # Higher profit requirement (1%)
    config7['signal_reversal_min_hold_pct'] = 0.25
    results.append(run_test("7. Higher Profit Req (1% profit)", config7))
    
    # Test 8: Shorter min hold time for reversal
    config8 = base_config.copy()
    config8['max_hold_bars'] = 40
    config8['signal_reversal_threshold'] = 0.8
    config8['signal_reversal_min_profit_pct'] = 0.005
    config8['signal_reversal_min_hold_pct'] = 0.15  # Shorter min hold
    results.append(run_test("8. Shorter Min Hold (15% of max)", config8))
    
    # Test 9: Longer min hold time for reversal
    config9 = base_config.copy()
    config9['max_hold_bars'] = 40
    config9['signal_reversal_threshold'] = 0.8
    config9['signal_reversal_min_profit_pct'] = 0.005
    config9['signal_reversal_min_hold_pct'] = 0.35  # Longer min hold
    results.append(run_test("9. Longer Min Hold (35% of max)", config9))
    
    # Test 10: Best combination (longer hold, sensitive reversal, lower profit req)
    config10 = base_config.copy()
    config10['max_hold_bars'] = 50
    config10['signal_reversal_threshold'] = 0.7
    config10['signal_reversal_min_profit_pct'] = 0.003
    config10['signal_reversal_min_hold_pct'] = 0.20
    results.append(run_test("10. Best Combo (50 bars, 0.7 reversal, 0.3% profit, 20% hold)", config10))
    
    # Print comparison table
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Method':<50} {'Return':<10} {'Sharpe':<10} {'Profit Factor':<15} {'Trades':<10}")
    print("-"*80)
    
    for r in results:
        print(f"{r['name']:<50} {r['return']:>8.2f}% {r['sharpe']:>8.2f} {r['profit_factor']:>13.2f} {r['trades']:>8}")
    
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
    print(f"  Exit Reasons: {best['exit_reasons']}")

if __name__ == "__main__":
    main()

