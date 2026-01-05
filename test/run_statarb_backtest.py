"""
Main Script to Run Statistical Arbitrage Strategy Backtest
"""
import os
import sys
import logging
from datetime import datetime
import yaml
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from statarb_data_fetcher import StatArbDataFetcher
from statarb_backtest import StatArbBacktestEngine, StatArbTrade
from mgc_performance import PerformanceAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def load_config(config_path: str = "test/statarb_config.yaml") -> dict:
    """Load configuration from YAML file"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {config_path}")
        return config
    else:
        logger.warning(f"Config file not found at {config_path}, using defaults")
        return {}


def format_statarb_report(results: dict, trades: list, z_entry: float = 2.0, z_exit: float = 0.6, 
                          spread_stop: float = 3.0, time_stop: float = 2.0) -> str:
    """Generate formatted report for StatArb backtest"""
    report = []
    report.append("=" * 80)
    report.append("STATISTICAL ARBITRAGE STRATEGY BACKTEST RESULTS")
    report.append("=" * 80)
    report.append("")
    
    # Strategy Parameters
    report.append("STRATEGY PARAMETERS:")
    report.append(f"  Z-Entry Threshold: ±{z_entry}")
    report.append(f"  Z-Exit Threshold: {z_exit}")
    report.append(f"  Spread Stop: {spread_stop} std deviations")
    report.append(f"  Time Stop: {time_stop} hours")
    report.append("")
    
    # Performance Summary
    report.append("PERFORMANCE SUMMARY:")
    report.append(f"  Total Trades: {results['total_trades']}")
    report.append(f"  Winning Trades: {results['winning_trades']}")
    report.append(f"  Losing Trades: {results['losing_trades']}")
    report.append(f"  Win Rate: {results['win_rate']:.2f}%")
    report.append("")
    
    # PnL Metrics
    report.append("PROFIT & LOSS:")
    report.append(f"  Total PnL: ${results['total_pnl']:,.2f}")
    report.append(f"  Gross Profit: ${results['gross_profit']:,.2f}")
    report.append(f"  Gross Loss: ${results['gross_loss']:,.2f}")
    report.append(f"  Profit Factor: {results['profit_factor']:.2f}")
    report.append(f"  Average PnL per Trade: ${results['avg_pnl']:,.2f}")
    report.append(f"  Expectancy: ${results['expectancy']:,.2f}")
    report.append("")
    
    # Equity Metrics
    report.append("EQUITY CURVE:")
    report.append(f"  Initial Equity: ${results['initial_equity']:,.2f}")
    report.append(f"  Final Equity: ${results['final_equity']:,.2f}")
    report.append(f"  Total Return: {results['total_return_pct']:.2f}%")
    report.append(f"  Maximum Drawdown: {results['max_drawdown']:.2f}%")
    report.append("")
    
    # Trade Statistics
    if trades:
        avg_duration = results.get('avg_duration_minutes', 0)
        report.append("TRADE STATISTICS:")
        report.append(f"  Average Duration: {avg_duration:.1f} minutes")
        
        # Exit reasons breakdown
        exit_reasons = {}
        for trade in trades:
            reason = trade.exit_reason
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        
        report.append("  Exit Reasons:")
        for reason, count in exit_reasons.items():
            pct = (count / len(trades)) * 100
            report.append(f"    {reason}: {count} ({pct:.1f}%)")
        report.append("")
        
        # Long vs Short spread performance
        long_trades = [t for t in trades if t.is_long_spread]
        short_trades = [t for t in trades if not t.is_long_spread]
        
        if long_trades:
            long_pnl = sum(t.pnl for t in long_trades)
            long_win_rate = len([t for t in long_trades if t.pnl > 0]) / len(long_trades) * 100
            report.append(f"  Long Spread Trades: {len(long_trades)}")
            report.append(f"    Total PnL: ${long_pnl:,.2f}")
            report.append(f"    Win Rate: {long_win_rate:.1f}%")
        
        if short_trades:
            short_pnl = sum(t.pnl for t in short_trades)
            short_win_rate = len([t for t in short_trades if t.pnl > 0]) / len(short_trades) * 100
            report.append(f"  Short Spread Trades: {len(short_trades)}")
            report.append(f"    Total PnL: ${short_pnl:,.2f}")
            report.append(f"    Win Rate: {short_win_rate:.1f}%")
        report.append("")
    
    report.append("=" * 80)
    return "\n".join(report)


def main():
    """Main backtest execution"""
    # Load config file
    config_path = os.path.join(os.path.dirname(__file__), 'statarb_config.yaml')
    config = load_config(config_path)
    
    # Extract config values with defaults
    start_date = config.get('start_date', '2025-01-01')
    end_date = config.get('end_date', '2025-11-14')
    interval = config.get('interval', '1m')
    use_cached = config.get('use_cached_data', False)
    cache_dir = config.get('cache_dir', 'test/cache')
    output_dir = config.get('output_dir', 'test/results')
    
    z_entry = config.get('z_entry', 2.0)
    z_exit = config.get('z_exit', 0.6)
    spread_stop_std = config.get('spread_stop_std', 3.0)
    time_stop_hours = config.get('time_stop_hours', 2.0)
    risk_per_trade = config.get('risk_per_trade', 100.0)
    initial_equity = config.get('initial_equity', 10000.0)
    lookback_periods = config.get('lookback_periods', 1440)
    min_lookback = config.get('min_lookback', 100)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("STATISTICAL ARBITRAGE STRATEGY BACKTEST")
    logger.info("=" * 80)
    logger.info(f"Config File: {config_path}")
    logger.info(f"Start Date: {start_date}")
    logger.info(f"End Date: {end_date}")
    logger.info(f"Z-Entry: ±{z_entry}")
    logger.info(f"Z-Exit: {z_exit}")
    logger.info(f"Spread Stop: {spread_stop_std} std")
    logger.info(f"Time Stop: {time_stop_hours} hours")
    logger.info(f"Risk Per Trade: ${risk_per_trade:.2f}")
    logger.info(f"Initial Equity: ${initial_equity:,.2f}")
    logger.info(f"Bar Interval: {interval}")
    logger.info("")
    
    # Step 1: Fetch Data
    logger.info("Step 1: Fetching GC and MGC historical data from TopstepX...")
    fetcher = StatArbDataFetcher()
    
    if not fetcher.authenticate():
        logger.error("Failed to authenticate with TopstepX")
        return
    
    df_gc, df_mgc = fetcher.fetch_for_backtest(
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        use_cached=use_cached,
        cache_dir=cache_dir
    )
    
    if df_gc.empty or df_mgc.empty:
        logger.error("Failed to fetch data or no data available")
        return
    
    logger.info(f"Data loaded: {len(df_gc)} GC bars, {len(df_mgc)} MGC bars")
    logger.info("")
    
    # Step 2: Run Backtest
    logger.info("Step 2: Running backtest...")
    engine = StatArbBacktestEngine(
        df_gc=df_gc,
        df_mgc=df_mgc,
        z_entry=z_entry,
        z_exit=z_exit,
        spread_stop_std=spread_stop_std,
        time_stop_hours=time_stop_hours,
        risk_per_trade=risk_per_trade,
        initial_equity=initial_equity,
        lookback_periods=lookback_periods,
        min_lookback=min_lookback
    )
    
    engine.run_backtest()
    logger.info("")
    
    # Step 3: Analyze Results
    logger.info("Step 3: Analyzing results...")
    results = engine.get_results()
    
    # Update report to use actual config values
    report = format_statarb_report(results, engine.trades, z_entry, z_exit, spread_stop_std, time_stop_hours)
    print("\n" + report)
    
    # Save report
    report_file = os.path.join(
        output_dir,
        f"statarb_backtest_{start_date}_{end_date}.txt"
    )
    with open(report_file, 'w') as f:
        f.write(report)
    logger.info(f"Report saved to {report_file}")
    
    # Export trades
    if engine.trades:
        trades_file = os.path.join(
            output_dir,
            f"statarb_trades_{start_date}_{end_date}.csv"
        )
        
        trades_data = {
            'entry_time': [t.entry_time for t in engine.trades],
            'exit_time': [t.exit_time for t in engine.trades],
            'entry_spread': [t.entry_spread for t in engine.trades],
            'exit_spread': [t.exit_spread for t in engine.trades],
            'entry_zscore': [t.entry_zscore for t in engine.trades],
            'exit_zscore': [t.exit_zscore for t in engine.trades],
            'entry_price_gc': [t.entry_price_a for t in engine.trades],
            'entry_price_mgc': [t.entry_price_b for t in engine.trades],
            'exit_price_gc': [t.exit_price_a for t in engine.trades],
            'exit_price_mgc': [t.exit_price_b for t in engine.trades],
            'beta': [t.beta for t in engine.trades],
            'contracts_gc': [t.contracts_a for t in engine.trades],
            'contracts_mgc': [t.contracts_b for t in engine.trades],
            'is_long_spread': [t.is_long_spread for t in engine.trades],
            'pnl': [t.pnl for t in engine.trades],
            'pnl_pct': [t.pnl_pct for t in engine.trades],
            'duration_minutes': [t.duration_minutes for t in engine.trades],
            'exit_reason': [t.exit_reason for t in engine.trades]
        }
        
        df_trades = pd.DataFrame(trades_data)
        df_trades.to_csv(trades_file, index=False)
        logger.info(f"Trades exported to {trades_file}")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("BACKTEST COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

