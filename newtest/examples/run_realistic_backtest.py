"""
Realistic Backtest Runner
Production-grade backtesting with all realistic assumptions
"""
import sys
import os
import logging
from datetime import datetime
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

from comprehensive_backtest import ComprehensiveBacktest
from framework.data_manager import DataManager
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
    """Run realistic backtest with comprehensive analysis"""
    
    parser = argparse.ArgumentParser(description='Realistic Backtest Runner')
    parser.add_argument('--strategy', choices=['optimal_stopping', 'multi_timeframe'], required=True,
                       help='Strategy to backtest')
    parser.add_argument('--symbol', default='MES', help='Trading symbol')
    parser.add_argument('--interval', default='15m', help='Bar interval')
    parser.add_argument('--start-date', type=str, default='2025-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2025-11-14', help='End date (YYYY-MM-DD)')
    parser.add_argument('--mode', choices=['single', 'walk-forward', 'optimize', 'all'], default='all',
                       help='Backtest mode')
    
    args = parser.parse_args()
    
    # Initialize
    data_manager = DataManager(cache_dir="newtest/results/cache")
    backtest = ComprehensiveBacktest(
        data_manager=data_manager,
        initial_equity=50000.0,
        commission_per_contract=2.50,  # Realistic Topstep commission
        slippage_ticks=1.0  # 1 tick slippage for market orders
    )
    
    start_date = datetime.fromisoformat(args.start_date)
    end_date = datetime.fromisoformat(args.end_date)
    
    # Strategy configurations
    if args.strategy == 'optimal_stopping':
        base_config = {
            'lookback_window': 100,
            'min_opportunities_seen': 37,
            'score_threshold': 0.7,  # Optimized: 0.7
            'momentum_weight': 0.4,
            'mean_reversion_weight': 0.3,
            'volatility_weight': 0.3,
            'atr_period': 14,
            'atr_multiplier_stop': 0.75,  # Optimized: 0.75 (tighter stop)
            'atr_multiplier_target': 1.25,  # Optimized: 1.25 (achievable target)
            'risk_per_trade': 100.0,
            'max_hold_bars': 40,  # Optimized: 40 (allows stops/targets to work)
            # Signal reversal exit settings (optimized)
            'signal_reversal_threshold': 0.8,  # Reversal score threshold (0.8 = strong reversal)
            'signal_reversal_min_profit_pct': 0.01,  # Min 1% profit to exit on reversal (optimized: was 0.5%)
            'signal_reversal_min_hold_pct': 0.25,  # Min 25% of max hold time before reversal exit
            # Trailing stop settings (disabled for now)
            'use_trailing_stop': False,
            'trailing_stop_atr_multiplier': 0.5,
            'trailing_stop_activation_pct': 0.001,
            # Partial profit taking settings (disabled for now)
            'use_partial_profit': False,
            'partial_profit_pct': 0.5,
            'partial_profit_target_atr': 0.5
        }
    elif args.strategy == 'multi_timeframe':
        base_config = {
            'convergence_threshold': 0.2,  # Best: 0.2
            'divergence_threshold': 0.2,   # Best: 0.2
            'lookback_period': 20,
            'momentum_period': 10,
            'mean_reversion_period': 20,
            'atr_period': 14,
            'atr_multiplier_stop': 1.0,    # Best: 1.0
            'atr_multiplier_target': 1.5,  # Best: 1.5
            'risk_per_trade': 100.0,
            'max_hold_bars': 30,            # Best: 30
            'timeframe_weights': {
                '1m': 0.25,
                '5m': 0.35,
                '15m': 0.40
            },
            # Profit capture features (PRODUCTION-READY)
            'use_trailing_stop': True,
            'trailing_stop_atr_multiplier': 0.5,
            'trailing_stop_activation_pct': 0.001,
            'use_partial_profit': True,
            'partial_profit_pct': 0.5,
            'partial_profit_target_atr': 0.75
        }
    
    logger.info(f"Running realistic backtest for {args.strategy}")
    logger.info(f"Symbol: {args.symbol}, Interval: {args.interval}")
    logger.info(f"Date Range: {start_date.date()} to {end_date.date()}")
    logger.info(f"Mode: {args.mode}")
    logger.info("")
    
    if args.mode in ['single', 'all']:
        logger.info("="*80)
        logger.info("SINGLE BACKTEST")
        logger.info("="*80)
        
        from framework.strategy_runner import StrategyRunner
        from framework.multi_timeframe_runner import MultiTimeframeRunner
        from strategies.optimal_stopping import OptimalStoppingStrategy
        from strategies.multi_timeframe import MultiTimeframeStrategy
        
        if args.strategy == 'optimal_stopping':
            runner = StrategyRunner(data_manager=data_manager, initial_equity=50000.0)
            strategy = OptimalStoppingStrategy(base_config)
            results = runner.backtest_strategy(
                strategy=strategy,
                symbol=args.symbol,
                interval=args.interval,
                start_date=start_date,
                end_date=end_date,
                tick_size=0.25,
                tick_value=5.0
            )
        elif args.strategy == 'multi_timeframe':
            runner = MultiTimeframeRunner(data_manager=data_manager, initial_equity=50000.0)
            strategy = MultiTimeframeStrategy(base_config)
            results = runner.run_backtest(
                strategy=strategy,
                symbol=args.symbol,
                primary_interval=args.interval,
                start_date=start_date,
                end_date=end_date,
                tick_size=0.25,
                tick_value=5.0
            )
        
        # Generate detailed report
        report = backtest.generate_detailed_report(
            results,
            args.strategy,
            f"newtest/results/backtests/{args.strategy}_report.txt"
        )
        print(report)
        
        # Export trades
        backtest.export_trades_csv(
            results['trades'],
            f"newtest/results/backtests/{args.strategy}_trades.csv"
        )
    
    if args.mode in ['walk-forward', 'all']:
        logger.info("\n" + "="*80)
        logger.info("WALK-FORWARD ANALYSIS")
        logger.info("="*80)
        
        wf_results = backtest.run_walk_forward(
            args.strategy,
            base_config,
            args.symbol,
            args.interval,
            start_date,
            end_date,
            train_ratio=0.6,
            step_size_days=30
        )
        
        print("\n" + "="*80)
        print("WALK-FORWARD SUMMARY")
        print("="*80)
        print(f"Iterations: {wf_results['iterations']}")
        print(f"Average Sharpe: {wf_results['avg_sharpe']:.2f} ± {wf_results['std_sharpe']:.2f}")
        print(f"Sharpe Range: [{wf_results['min_sharpe']:.2f}, {wf_results['max_sharpe']:.2f}]")
        print(f"Average Return: {wf_results['avg_return']:.2f}% ± {wf_results['std_return']:.2f}%")
        print(f"Average Max DD: {wf_results['avg_max_dd']:.2f}%")
        print(f"Average Win Rate: {wf_results['avg_win_rate']:.2f}%")
        print(f"Total Trades: {wf_results['total_trades']}")
        print(f"Positive Iterations: {wf_results['positive_iterations']}/{wf_results['iterations']}")
        print(f"Negative Iterations: {wf_results['negative_iterations']}/{wf_results['iterations']}")
        print(f"Success Rate: {wf_results['positive_iterations']/wf_results['iterations']*100:.1f}%")
    
    if args.mode in ['optimize', 'all']:
        logger.info("\n" + "="*80)
        logger.info("PARAMETER OPTIMIZATION")
        logger.info("="*80)
        
        if args.strategy == 'optimal_stopping':
            # Expanded parameter ranges for comprehensive optimization
            param_grid = {
                'score_threshold': [0.5, 0.6, 0.7],
                'atr_multiplier_stop': [0.5, 0.75, 1.0, 1.25],  # Tighter stops
                'atr_multiplier_target': [0.75, 1.0, 1.25, 1.5],  # Achievable targets
                'max_hold_bars': [20, 30, 40, 50, 60],  # Different hold times
                'signal_reversal_threshold': [0.6, 0.7, 0.8, 0.9],  # Reversal sensitivity
                'signal_reversal_min_profit_pct': [0.002, 0.005, 0.01],  # Min profit to exit on reversal
                'signal_reversal_min_hold_pct': [0.15, 0.25, 0.35]  # Min hold time before reversal exit
            }
        elif args.strategy == 'multi_timeframe':
            param_grid = {
                'convergence_threshold': [0.2, 0.3, 0.4],
                'divergence_threshold': [0.2, 0.3],
                'atr_multiplier_stop': [1.0, 1.5],
                'atr_multiplier_target': [1.5, 2.0],
                'max_hold_bars': [30, 40, 50]
            }
        
        results_df = backtest.run_parameter_sweep(
            args.strategy,
            base_config,
            param_grid,
            args.symbol,
            args.interval,
            start_date,
            end_date
        )
        
        # Sort by Sharpe ratio
        results_df = results_df.sort_values('sharpe_ratio', ascending=False)
        
        print("\n" + "="*80)
        print("TOP 10 PARAMETER COMBINATIONS (by Sharpe Ratio)")
        print("="*80)
        print(results_df.head(10).to_string())
        
        # Save full results
        results_df.to_csv(
            f"newtest/results/backtests/{args.strategy}_optimization.csv",
            index=False
        )
        logger.info(f"\nFull optimization results saved to: newtest/results/backtests/{args.strategy}_optimization.csv")
    
    logger.info("\n" + "="*80)
    logger.info("BACKTEST COMPLETE")
    logger.info("="*80)
    logger.info(f"Results saved to: newtest/results/backtests/")


if __name__ == "__main__":
    main()

