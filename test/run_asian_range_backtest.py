"""
Main Script to Run Asian Range Breakout Strategy Backtest
"""
import os
import sys
import logging
from datetime import datetime
import argparse
import yaml
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mgc_backtest_data import MGCDataFetcher
from asian_range_backtest import AsianRangeBacktestEngine, Trade
from mgc_performance import PerformanceAnalyzer
from asian_range_visualizer import AsianRangeVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def load_config(config_path: str = "test/backtest_config.yaml") -> dict:
    """Load configuration from YAML file"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {config_path}")
        return config
    else:
        logger.warning(f"Config file not found at {config_path}, using defaults")
        return {}


def main():
    """Main backtest execution"""
    # Load config file
    config_path = os.path.join(os.path.dirname(__file__), 'backtest_config.yaml')
    config = load_config(config_path)
    
    parser = argparse.ArgumentParser(description='Asian Range Breakout Strategy Backtest')
    parser.add_argument('--config', type=str, default=config_path,
                       help='Path to config file')
    parser.add_argument('--start-date', type=str, default=config.get('start_date', '2025-01-01'),
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=config.get('end_date', '2025-11-14'),
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--contracts-per-trade', type=int, default=config.get('contracts_per_trade', 3),
                       help='Fixed number of contracts per trade')
    parser.add_argument('--tp-multiplier', type=float, default=config.get('tp_multiplier', 1.5),
                       help='Take profit multiplier (1.0 to 2.0)')
    parser.add_argument('--sl-buffer-ticks', type=int, default=config.get('sl_buffer_ticks', 3),
                       help='Stop loss buffer in ticks (default: 3)')
    parser.add_argument('--partial-close-percent', type=float, default=config.get('partial_close_percent', 0.75),
                       help='Percentage to close at 12 PM (0.75 = 75%%, remaining stays open)')
    parser.add_argument('--use-cached', action='store_true', default=config.get('use_cached_data', False),
                       help='Use cached data if available')
    parser.add_argument('--cache-dir', type=str, default=config.get('cache_dir', 'test/cache'),
                       help='Directory for cached data')
    parser.add_argument('--output-dir', type=str, default=config.get('output_dir', 'test/results'),
                       help='Directory for output files')
    parser.add_argument('--initial-equity', type=float, default=config.get('initial_equity', 10000.0),
                       help='Initial equity')
    
    args = parser.parse_args()
    
    # If custom config file specified, reload it
    if args.config != config_path:
        config = load_config(args.config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("ASIAN RANGE BREAKOUT STRATEGY BACKTEST")
    logger.info("=" * 80)
    logger.info(f"Start Date: {args.start_date}")
    logger.info(f"End Date: {args.end_date}")
    logger.info(f"Contracts Per Trade: {args.contracts_per_trade}")
    logger.info(f"TP Multiplier: {args.tp_multiplier}x")
    logger.info(f"Stop Loss Buffer: {args.sl_buffer_ticks} ticks")
    logger.info(f"Partial Close Percent: {args.partial_close_percent*100:.0f}% (remaining {100-args.partial_close_percent*100:.0f}% stays open)")
    logger.info(f"Initial Equity: ${args.initial_equity:,.2f}")
    logger.info("")
    
    # Step 1: Fetch Data (1-minute bars required)
    logger.info("Step 1: Fetching 1-minute historical data from TopstepX...")
    fetcher = MGCDataFetcher()
    
    if not fetcher.authenticate():
        logger.error("Failed to authenticate with TopstepX")
        return
    
    # Check for cached data
    cache_file_1m = os.path.join(args.cache_dir, f"mgc_1m_{args.start_date}_{args.end_date}.csv")
    
    if args.use_cached and os.path.exists(cache_file_1m):
        logger.info("Loading cached data...")
        df_1m = pd.read_csv(cache_file_1m, parse_dates=['timestamp'])
        logger.info(f"Loaded {len(df_1m)} 1m bars from cache")
    else:
        logger.info("Fetching data from TopstepX API...")
        df_1m = fetcher.fetch_bars(
            interval="1m",
            start_time=datetime.fromisoformat(args.start_date),
            end_time=datetime.fromisoformat(args.end_date),
            limit=50000
        )
        
        if df_1m.empty:
            logger.error("Failed to fetch data or no data available")
            return
        
        # Cache the data
        df_1m.to_csv(cache_file_1m, index=False)
        logger.info(f"Cached data to {args.cache_dir}")
    
    logger.info(f"Data loaded: {len(df_1m)} 1m bars")
    logger.info("")
    
    # Step 2: Run Backtest
    logger.info("Step 2: Running backtest...")
    engine = AsianRangeBacktestEngine(
        df_1m=df_1m,
        contracts_per_trade=args.contracts_per_trade,
        tp_multiplier=args.tp_multiplier,
        sl_buffer_ticks=args.sl_buffer_ticks,
        partial_close_percent=args.partial_close_percent,
        initial_equity=args.initial_equity
    )
    
    engine.run_backtest()
    logger.info("")
    
    # Step 3: Analyze Results
    logger.info("Step 3: Analyzing results...")
    analyzer = PerformanceAnalyzer(engine.trades, engine.equity_curve)
    
    # Generate report
    report = analyzer.generate_report()
    print("\n" + report)
    
    # Save report
    report_file = os.path.join(args.output_dir, f"asian_range_backtest_{args.start_date}_{args.end_date}.txt")
    with open(report_file, 'w') as f:
        f.write(report)
    logger.info(f"Report saved to {report_file}")
    
    # Export trades
    trades_file = os.path.join(args.output_dir, f"asian_range_trades_{args.start_date}_{args.end_date}.csv")
    analyzer.export_trades_csv(trades_file)
    
    # Generate matplotlib plots
    try:
        import matplotlib
        matplotlib.use('Agg')
        
        equity_curve_file = os.path.join(args.output_dir, f"asian_range_equity_{args.start_date}_{args.end_date}.png")
        analyzer.plot_equity_curve(equity_curve_file)
        
        monthly_returns_file = os.path.join(args.output_dir, f"asian_range_monthly_{args.start_date}_{args.end_date}.png")
        analyzer.plot_monthly_returns(monthly_returns_file)
    except ImportError:
        logger.warning("matplotlib not available, skipping matplotlib plots")
    except Exception as e:
        logger.warning(f"Error generating matplotlib plots: {e}")
    
    # Generate Plotly interactive charts
    try:
        logger.info("Generating Plotly interactive charts...")
        visualizer = AsianRangeVisualizer(
            df_1m=df_1m,
            trades=engine.trades,
            asian_ranges=engine.asian_ranges
        )
        
        # Plot all trades overview
        all_trades_file = os.path.join(args.output_dir, f"asian_range_chart_all_{args.start_date}_{args.end_date}.html")
        visualizer.plot_all_trades(output_file=all_trades_file, max_days=10)
        logger.info(f"All trades chart saved to {all_trades_file}")
        
        # Plot individual trading days
        if engine.trades:
            for i, trade in enumerate(engine.trades[:5]):  # Plot first 5 trades
                trade_date = trade.entry_time.date() if hasattr(trade.entry_time, 'date') else pd.Timestamp(trade.entry_time).date()
                day_file = os.path.join(args.output_dir, f"asian_range_chart_{trade_date}.html")
                from datetime import time
                visualizer.plot_trading_day(
                    date=pd.Timestamp.combine(trade_date, time(0, 0)),
                    output_file=day_file
                )
                logger.info(f"Trading day chart saved to {day_file}")
        
        # Plot equity curve
        equity_html_file = os.path.join(args.output_dir, f"asian_range_equity_curve_{args.start_date}_{args.end_date}.html")
        timestamps = [t.exit_time for t in engine.trades] if engine.trades else None
        visualizer.plot_equity_curve(
            equity_curve=engine.equity_curve,
            timestamps=timestamps,
            output_file=equity_html_file
        )
        logger.info(f"Equity curve chart saved to {equity_html_file}")
        
    except ImportError:
        logger.warning("plotly not available. Install with: pip install plotly")
    except Exception as e:
        logger.warning(f"Error generating Plotly charts: {e}")
        import traceback
        logger.debug(traceback.format_exc())
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("BACKTEST COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

