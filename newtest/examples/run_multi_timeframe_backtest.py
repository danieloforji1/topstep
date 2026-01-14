"""
Multi-Timeframe Convergence Strategy Backtest Runner
"""
import sys
import os
import logging
from datetime import datetime, timedelta, timezone
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

from framework.multi_timeframe_runner import MultiTimeframeRunner
from framework.data_manager import DataManager
from strategies.multi_timeframe import MultiTimeframeStrategy
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
    """Run multi-timeframe convergence backtest"""
    
    parser = argparse.ArgumentParser(description='Multi-Timeframe Convergence Backtest')
    parser.add_argument('--symbol', default='MES', help='Trading symbol')
    parser.add_argument('--start-date', type=str, default='2025-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None, help='End date (YYYY-MM-DD)')
    parser.add_argument('--primary-interval', default='15m', choices=['1m', '5m', '15m'],
                       help='Primary execution timeframe')
    
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
    
    # Strategy configuration (PRODUCTION-READY - OPTIMIZED)
    base_config = {
        'convergence_threshold': 0.2,  # Best: 0.2 (generates good trade frequency)
        'divergence_threshold': 0.2,   # Best: 0.2 (balanced exit sensitivity)
        'lookback_period': 20,
        'momentum_period': 10,
        'mean_reversion_period': 20,
        'atr_period': 14,
        'atr_multiplier_stop': 1.0,    # Best: 1.0 (tighter stops)
        'atr_multiplier_target': 1.5,  # Best: 1.5 (achievable targets)
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
        'trailing_stop_activation_pct': 0.001,  # Activate after 0.1% profit
        'use_partial_profit': True,
        'partial_profit_pct': 0.5,  # Take 50% at first target
        'partial_profit_target_atr': 0.75  # Take profit at 0.75x ATR
    }
    
    # Create strategy
    strategy = MultiTimeframeStrategy(base_config)
    
    logger.info("="*80)
    logger.info("MULTI-TIMEFRAME CONVERGENCE BACKTEST")
    logger.info("="*80)
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Primary Interval: {args.primary_interval}")
    logger.info(f"Date Range: {start_date.date()} to {end_date.date()}")
    logger.info(f"Config: {base_config}")
    logger.info("="*80)
    
    # Run backtest
    try:
        results = runner.run_backtest(
            strategy=strategy,
            symbol=args.symbol,
            primary_interval=args.primary_interval,
            start_date=start_date,
            end_date=end_date,
            tick_size=0.25,
            tick_value=5.0
        )
        
        # Print results
        print("\n" + "="*80)
        print("BACKTEST RESULTS")
        print("="*80)
        print(results['report'])
        print("="*80)
        
        # Print trade summary
        trades = results['trades']
        if trades:
            print(f"\nTotal Trades: {len(trades)}")
            winning_trades = [t for t in trades if t.pnl > 0]
            losing_trades = [t for t in trades if t.pnl <= 0]
            print(f"Winning: {len(winning_trades)}, Losing: {len(losing_trades)}")
            print(f"Win Rate: {len(winning_trades)/len(trades)*100:.2f}%")
            
            # Exit reasons
            exit_reasons = {}
            for trade in trades:
                reason = trade.exit_reason
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
            print(f"\nExit Reasons:")
            for reason, count in sorted(exit_reasons.items(), key=lambda x: x[1], reverse=True):
                print(f"  {reason}: {count}")
        
        # Save results
        import json
        results_dir = "newtest/results/backtests"
        os.makedirs(results_dir, exist_ok=True)
        
        date_str = f"{start_date.date()}_{end_date.date()}"
        results_file = os.path.join(results_dir, f"multi_timeframe_{args.symbol}_{date_str}.json")
        
        # Convert to JSON-serializable format
        results_json = {
            'symbol': args.symbol,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'config': base_config,
            'metrics': results['metrics'],
            'initial_equity': results['initial_equity'],
            'final_equity': results['final_equity'],
            'num_trades': len(trades),
            'trades': [
                {
                    'entry_time': t.entry_time.isoformat(),
                    'exit_time': t.exit_time.isoformat(),
                    'entry_price': t.entry_price,
                    'exit_price': t.exit_price,
                    'contracts': t.contracts,
                    'is_long': t.is_long,
                    'pnl': t.pnl,
                    'pnl_pct': t.pnl_pct,
                    'duration_minutes': t.duration_minutes,
                    'exit_reason': t.exit_reason
                }
                for t in trades
            ]
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_json, f, indent=2, default=str)
        
        logger.info(f"\nResults saved to: {results_file}")
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

