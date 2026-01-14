"""
Comprehensive Backtest Framework
Production-grade backtesting with walk-forward analysis, parameter optimization, and detailed reporting
"""
import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

from framework.strategy_runner import StrategyRunner
from framework.multi_timeframe_runner import MultiTimeframeRunner
from framework.data_manager import DataManager
from framework.performance_analyzer import PerformanceAnalyzer
from strategies.optimal_stopping import OptimalStoppingStrategy
from strategies.multi_timeframe import MultiTimeframeStrategy
from strategies.liquidity_provision import LiquidityProvisionStrategy
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


class ComprehensiveBacktest:
    """
    Comprehensive backtesting framework with:
    - Walk-forward analysis
    - Parameter optimization
    - Multiple date ranges
    - Detailed reporting
    - Trade-by-trade analysis
    """
    
    def __init__(
        self,
        data_manager: DataManager,
        initial_equity: float = 50000.0,
        commission_per_contract: float = 2.50,
        slippage_ticks: float = 1.0
    ):
        self.data_manager = data_manager
        self.initial_equity = initial_equity
        self.commission_per_contract = commission_per_contract
        self.slippage_ticks = slippage_ticks
        self.results_dir = "newtest/results/backtests"
        os.makedirs(self.results_dir, exist_ok=True)
    
    def run_walk_forward(
        self,
        strategy_name: str,
        strategy_config: Dict[str, Any],
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: datetime,
        train_ratio: float = 0.6,
        step_size_days: int = 30
    ) -> Dict[str, Any]:
        """
        Run walk-forward analysis
        
        Args:
            strategy_name: Name of strategy ('optimal_stopping', 'multi_timeframe', or 'liquidity_provision')
            strategy_config: Strategy configuration
            symbol: Trading symbol
            interval: Bar interval
            start_date: Start date
            end_date: End date
            train_ratio: Ratio of data for training (default 0.6)
            step_size_days: Days to step forward each iteration
            
        Returns:
            Dictionary with walk-forward results
        """
        logger.info(f"Starting walk-forward analysis for {strategy_name}")
        
        total_days = (end_date - start_date).days
        train_days = int(total_days * train_ratio)
        
        all_results = []
        current_start = start_date
        
        iteration = 0
        while current_start + timedelta(days=train_days) < end_date:
            train_end = current_start + timedelta(days=train_days)
            test_start = train_end
            test_end = min(test_start + timedelta(days=step_size_days), end_date)
            
            if test_end <= test_start:
                break
            
            iteration += 1
            logger.info(f"\n{'='*80}")
            logger.info(f"Walk-Forward Iteration {iteration}")
            logger.info(f"Train: {current_start.date()} to {train_end.date()}")
            logger.info(f"Test: {test_start.date()} to {test_end.date()}")
            logger.info(f"{'='*80}")
            
            # Run backtest on test period
            try:
                if strategy_name == 'optimal_stopping':
                    runner = StrategyRunner(
                        data_manager=self.data_manager,
                        initial_equity=self.initial_equity,
                        commission_per_contract=self.commission_per_contract,
                        slippage_ticks=self.slippage_ticks
                    )
                    strategy = OptimalStoppingStrategy(strategy_config)
                    result = runner.backtest_strategy(
                        strategy=strategy,
                        symbol=symbol,
                        interval=interval,
                        start_date=test_start,
                        end_date=test_end,
                        tick_size=0.25,
                        tick_value=5.0
                    )
                elif strategy_name == 'multi_timeframe':
                    runner = MultiTimeframeRunner(
                        data_manager=self.data_manager,
                        initial_equity=self.initial_equity,
                        commission_per_contract=self.commission_per_contract,
                        slippage_ticks=self.slippage_ticks
                    )
                    strategy = MultiTimeframeStrategy(strategy_config)
                    result = runner.run_backtest(
                        strategy=strategy,
                        symbol=symbol,
                        primary_interval=interval,
                        start_date=test_start,
                        end_date=test_end,
                        tick_size=0.25,
                        tick_value=5.0
                    )
                elif strategy_name == 'liquidity_provision':
                    runner = StrategyRunner(
                        data_manager=self.data_manager,
                        initial_equity=self.initial_equity,
                        commission_per_contract=self.commission_per_contract,
                        slippage_ticks=0.5  # Lower for limit orders
                    )
                    strategy = LiquidityProvisionStrategy(strategy_config)
                    result = runner.backtest_strategy(
                        strategy=strategy,
                        symbol=symbol,
                        interval=interval,
                        start_date=test_start,
                        end_date=test_end,
                        tick_size=0.25,
                        tick_value=5.0
                    )
                else:
                    raise ValueError(f"Unknown strategy: {strategy_name}")
                
                result['iteration'] = iteration
                result['train_start'] = current_start.isoformat()
                result['train_end'] = train_end.isoformat()
                result['test_start'] = test_start.isoformat()
                result['test_end'] = test_end.isoformat()
                all_results.append(result)
                
                # Print summary
                metrics = result['metrics']
                logger.info(f"Test Results:")
                logger.info(f"  Sharpe: {metrics['sharpe_ratio']:.2f}")
                logger.info(f"  Return: {metrics['total_return']:.2f}%")
                logger.info(f"  Max DD: {metrics['max_drawdown']:.2f}%")
                logger.info(f"  Win Rate: {metrics['win_rate']:.2f}%")
                logger.info(f"  Trades: {metrics['num_trades']}")
                
            except Exception as e:
                logger.error(f"Error in iteration {iteration}: {e}")
                continue
            
            # Step forward
            current_start += timedelta(days=step_size_days)
        
        # Aggregate results
        return self._aggregate_walk_forward_results(all_results)
    
    def _aggregate_walk_forward_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Aggregate walk-forward results"""
        if not results:
            return {}
        
        sharpe_ratios = [r['metrics']['sharpe_ratio'] for r in results]
        returns = [r['metrics']['total_return'] for r in results]
        max_dds = [r['metrics']['max_drawdown'] for r in results]
        win_rates = [r['metrics']['win_rate'] for r in results]
        num_trades = [r['metrics']['num_trades'] for r in results]
        
        return {
            'iterations': len(results),
            'avg_sharpe': np.mean(sharpe_ratios),
            'std_sharpe': np.std(sharpe_ratios),
            'min_sharpe': np.min(sharpe_ratios),
            'max_sharpe': np.max(sharpe_ratios),
            'avg_return': np.mean(returns),
            'std_return': np.std(returns),
            'avg_max_dd': np.mean(max_dds),
            'avg_win_rate': np.mean(win_rates),
            'total_trades': sum(num_trades),
            'avg_trades_per_iteration': np.mean(num_trades),
            'positive_iterations': sum(1 for r in returns if r > 0),
            'negative_iterations': sum(1 for r in returns if r < 0),
            'detailed_results': results
        }
    
    def run_parameter_sweep(
        self,
        strategy_name: str,
        base_config: Dict[str, Any],
        param_grid: Dict[str, List[Any]],
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Run parameter optimization sweep
        
        Args:
            strategy_name: Strategy name
            base_config: Base configuration
            param_grid: Dictionary of parameter names to lists of values to test
            symbol: Trading symbol
            interval: Bar interval
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with results for each parameter combination
        """
        logger.info(f"Starting parameter sweep for {strategy_name}")
        
        # Generate all parameter combinations
        from itertools import product
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))
        
        results = []
        
        for i, combo in enumerate(combinations):
            config = base_config.copy()
            config.update(dict(zip(param_names, combo)))
            
            logger.info(f"\nTesting combination {i+1}/{len(combinations)}: {dict(zip(param_names, combo))}")
            
            try:
                if strategy_name == 'optimal_stopping':
                    runner = StrategyRunner(
                        data_manager=self.data_manager,
                        initial_equity=self.initial_equity,
                        commission_per_contract=self.commission_per_contract,
                        slippage_ticks=self.slippage_ticks
                    )
                    strategy = OptimalStoppingStrategy(config)
                    result = runner.backtest_strategy(
                        strategy=strategy,
                        symbol=symbol,
                        interval=interval,
                        start_date=start_date,
                        end_date=end_date,
                        tick_size=0.25,
                        tick_value=5.0
                    )
                elif strategy_name == 'multi_timeframe':
                    runner = MultiTimeframeRunner(
                        data_manager=self.data_manager,
                        initial_equity=self.initial_equity,
                        commission_per_contract=self.commission_per_contract,
                        slippage_ticks=self.slippage_ticks
                    )
                    strategy = MultiTimeframeStrategy(config)
                    result = runner.run_backtest(
                        strategy=strategy,
                        symbol=symbol,
                        primary_interval=interval,
                        start_date=start_date,
                        end_date=end_date,
                        tick_size=0.25,
                        tick_value=5.0
                    )
                elif strategy_name == 'liquidity_provision':
                    runner = StrategyRunner(
                        data_manager=self.data_manager,
                        initial_equity=self.initial_equity,
                        commission_per_contract=self.commission_per_contract,
                        slippage_ticks=0.5
                    )
                    strategy = LiquidityProvisionStrategy(config)
                    result = runner.backtest_strategy(
                        strategy=strategy,
                        symbol=symbol,
                        interval=interval,
                        start_date=start_date,
                        end_date=end_date,
                        tick_size=0.25,
                        tick_value=5.0
                    )
                else:
                    raise ValueError(f"Unknown strategy: {strategy_name}")
                
                metrics = result['metrics']
                row = dict(zip(param_names, combo))
                row.update({
                    'sharpe_ratio': metrics['sharpe_ratio'],
                    'total_return': metrics['total_return'],
                    'max_drawdown': metrics['max_drawdown'],
                    'win_rate': metrics['win_rate'],
                    'profit_factor': metrics['profit_factor'],
                    'num_trades': metrics['num_trades'],
                    'consistency_score': metrics['consistency_score']
                })
                results.append(row)
                
            except Exception as e:
                logger.error(f"Error testing combination {combo}: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def generate_detailed_report(
        self,
        results: Dict[str, Any],
        strategy_name: str,
        output_file: Optional[str] = None
    ) -> str:
        """Generate comprehensive backtest report"""
        metrics = results['metrics']
        trades = results.get('trades', [])
        
        report = []
        report.append("="*80)
        report.append(f"COMPREHENSIVE BACKTEST REPORT: {strategy_name.upper()}")
        report.append("="*80)
        report.append("")
        
        # Performance Summary
        report.append("PERFORMANCE SUMMARY")
        report.append("-"*80)
        report.append(f"Initial Equity: ${results['initial_equity']:,.2f}")
        report.append(f"Final Equity: ${results['final_equity']:,.2f}")
        report.append(f"Total Return: {metrics['total_return']:.2f}%")
        report.append(f"Annualized Return: {metrics['annualized_return']:.2f}%")
        report.append(f"Total P&L: ${results['final_equity'] - results['initial_equity']:,.2f}")
        report.append("")
        
        # Risk Metrics
        report.append("RISK METRICS")
        report.append("-"*80)
        report.append(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        report.append(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
        report.append(f"Calmar Ratio: {metrics['calmar_ratio']:.2f}")
        report.append(f"Maximum Drawdown: {metrics['max_drawdown']:.2f}% (${metrics['max_drawdown_dollar']:,.2f})")
        report.append("")
        
        # Trade Statistics
        report.append("TRADE STATISTICS")
        report.append("-"*80)
        report.append(f"Total Trades: {metrics['num_trades']}")
        report.append(f"Winning Trades: {metrics['winning_trades']} ({metrics['win_rate']:.2f}%)")
        report.append(f"Losing Trades: {metrics['losing_trades']} ({100 - metrics['win_rate']:.2f}%)")
        report.append(f"Profit Factor: {metrics['profit_factor']:.2f}")
        report.append(f"Expectancy: ${metrics['expectancy']:.2f} per trade")
        report.append("")
        
        # Trade Analysis
        if trades:
            report.append("TRADE ANALYSIS")
            report.append("-"*80)
            report.append(f"Average Win: ${metrics['average_win']:.2f}")
            report.append(f"Average Loss: ${metrics['average_loss']:.2f}")
            report.append(f"Largest Win: ${metrics['largest_win']:.2f}")
            report.append(f"Largest Loss: ${metrics['largest_loss']:.2f}")
            report.append("")
            
            # Monthly breakdown
            trades_df = pd.DataFrame([
                {
                    'date': trade.exit_time,
                    'pnl': trade.pnl
                }
                for trade in trades
            ])
            trades_df['year_month'] = pd.to_datetime(trades_df['date']).dt.to_period('M')
            monthly_pnl = trades_df.groupby('year_month')['pnl'].sum()
            
            report.append("MONTHLY PERFORMANCE")
            report.append("-"*80)
            for period, pnl in monthly_pnl.items():
                report.append(f"{period}: ${pnl:,.2f}")
            report.append("")
            
            # Drawdown periods
            equity_curve = results.get('equity_curve', [])
            if equity_curve:
                equity_series = pd.Series(equity_curve)
                running_max = equity_series.expanding().max()
                drawdown = (equity_series - running_max) / running_max * 100
                
                report.append("DRAWDOWN ANALYSIS")
                report.append("-"*80)
                report.append(f"Maximum Drawdown: {drawdown.min():.2f}%")
                report.append(f"Average Drawdown: {drawdown[drawdown < 0].mean():.2f}%")
                report.append(f"Number of Drawdown Periods: {len(drawdown[drawdown < -5])}")  # >5% drawdowns
                report.append("")
        
        # Consistency
        report.append("CONSISTENCY METRICS")
        report.append("-"*80)
        report.append(f"Consistency Score: {metrics['consistency_score']:.2f}")
        report.append(f"Average Daily Return: ${metrics['average_daily_return']:.2f}")
        report.append("")
        
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            logger.info(f"Report saved to {output_file}")
        
        return report_text
    
    def export_trades_csv(self, trades: List[Any], filename: str):
        """Export trades to CSV"""
        if not trades:
            return
        
        data = []
        for trade in trades:
            data.append({
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'contracts': trade.contracts,
                'is_long': trade.is_long,
                'pnl': trade.pnl,
                'pnl_pct': trade.pnl_pct,
                'duration_minutes': trade.duration_minutes,
                'exit_reason': trade.exit_reason,
                'stop_loss': trade.stop_loss,
                'take_profit': trade.take_profit
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        logger.info(f"Trades exported to {filename}")


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='Comprehensive Backtest Framework')
    parser.add_argument('--strategy', choices=['optimal_stopping', 'multi_timeframe', 'liquidity_provision'], required=True)
    parser.add_argument('--symbol', default='MES')
    parser.add_argument('--interval', default='15m')
    parser.add_argument('--start-date', type=str, default='2025-01-01')
    parser.add_argument('--end-date', type=str, default='2025-11-14')
    parser.add_argument('--mode', choices=['single', 'walk-forward', 'optimize'], default='single')
    
    args = parser.parse_args()
    
    # Initialize
    data_manager = DataManager(cache_dir="newtest/results/cache")
    backtest = ComprehensiveBacktest(
        data_manager=data_manager,
        initial_equity=50000.0
    )
    
    start_date = datetime.fromisoformat(args.start_date)
    end_date = datetime.fromisoformat(args.end_date)
    
    if args.strategy == 'optimal_stopping':
        base_config = {
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
    elif args.strategy == 'multi_timeframe':
        base_config = {
            'convergence_threshold': 0.2,  # Best: 0.2
            'divergence_threshold': 0.2,   # Best: 0.2
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
            # Profit capture features (PRODUCTION-READY)
            'use_trailing_stop': True,
            'trailing_stop_atr_multiplier': 0.5,
            'trailing_stop_activation_pct': 0.001,
            'use_partial_profit': True,
            'partial_profit_pct': 0.5,
            'partial_profit_target_atr': 0.75
        }
    elif args.strategy == 'liquidity_provision':
        base_config = {
            'imbalance_lookback': 3,
            'imbalance_threshold': 0.08,  # Slightly lower for more trades
            'adverse_selection_threshold': 0.55,  # Slightly higher
            'favorable_fill_threshold': 0.55,  # Slightly lower
            'spread_target_ticks': 4,
            'max_spread_ticks': 5,
            'atr_period': 14,
            'atr_multiplier_stop': 1.25,
            'risk_per_trade': 100.0,
            'max_hold_bars': 15,
            'cancel_on_reversal': True,
            # Profit enhancement features
            'use_trailing_stop': True,
            'trailing_stop_atr_multiplier': 0.5,
            'trailing_stop_activation_pct': 0.001,
            'use_partial_profit': True,
            'partial_profit_pct': 0.5,
            'partial_profit_target_atr': 0.75,
            'confidence_scaling': True,
            'max_position_size': 5
        }
    
    if args.mode == 'single':
        # Single backtest
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
        elif args.strategy == 'liquidity_provision':
            runner = StrategyRunner(data_manager=data_manager, initial_equity=50000.0, slippage_ticks=0.5)
            strategy = LiquidityProvisionStrategy(base_config)
            results = runner.backtest_strategy(
                strategy=strategy,
                symbol=args.symbol,
                interval=args.interval,
                start_date=start_date,
                end_date=end_date,
                tick_size=0.25,
                tick_value=5.0
            )
        
        # Generate report
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
    
    elif args.mode == 'walk-forward':
        # Walk-forward analysis
        wf_results = backtest.run_walk_forward(
            args.strategy,
            base_config,
            args.symbol,
            args.interval,
            start_date,
            end_date
        )
        
        print("\n" + "="*80)
        print("WALK-FORWARD ANALYSIS RESULTS")
        print("="*80)
        print(f"Iterations: {wf_results['iterations']}")
        print(f"Average Sharpe: {wf_results['avg_sharpe']:.2f} ± {wf_results['std_sharpe']:.2f}")
        print(f"Average Return: {wf_results['avg_return']:.2f}% ± {wf_results['std_return']:.2f}%")
        print(f"Average Max DD: {wf_results['avg_max_dd']:.2f}%")
        print(f"Positive Iterations: {wf_results['positive_iterations']}/{wf_results['iterations']}")
        print(f"Negative Iterations: {wf_results['negative_iterations']}/{wf_results['iterations']}")
        
        # Save results
        with open(f"newtest/results/backtests/{args.strategy}_walkforward.json", 'w') as f:
            json.dump(wf_results, f, indent=2, default=str)
    
    elif args.mode == 'optimize':
        # Parameter optimization
        if args.strategy == 'optimal_stopping':
            param_grid = {
                'score_threshold': [0.5, 0.6, 0.7],
                'atr_multiplier_stop': [1.0, 1.5, 2.0],
                'atr_multiplier_target': [1.5, 2.0, 2.5]
            }
        elif args.strategy == 'multi_timeframe':
            param_grid = {
                'convergence_threshold': [0.2, 0.3, 0.4],
                'divergence_threshold': [0.2, 0.3],
                'atr_multiplier_stop': [1.0, 1.5],
                'atr_multiplier_target': [1.5, 2.0],
                'max_hold_bars': [30, 40, 50]
            }
        elif args.strategy == 'liquidity_provision':
            param_grid = {
                'imbalance_threshold': [0.2, 0.3, 0.4],
                'adverse_selection_threshold': [0.3, 0.4, 0.5],
                'favorable_fill_threshold': [0.5, 0.6, 0.7],
                'spread_target_ticks': [1, 2, 3],
                'max_hold_bars': [15, 20, 25]
            }
        else:
            raise ValueError(f"Unknown strategy: {args.strategy}")
        
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
        print("PARAMETER OPTIMIZATION RESULTS")
        print("="*80)
        print(results_df.head(10).to_string())
        
        # Save results
        results_df.to_csv(f"newtest/results/backtests/{args.strategy}_optimization.csv", index=False)


if __name__ == "__main__":
    main()

