"""
Main Script to Run MGC Liquidity Sweep Backtest
"""
import os
import sys
import logging
from datetime import datetime
import argparse
import yaml

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mgc_backtest_data import MGCDataFetcher
from mgc_backtest_engine import MGCBacktestEngine
from mgc_performance import PerformanceAnalyzer

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
    
    parser = argparse.ArgumentParser(description='MGC Liquidity Sweep Strategy Backtest')
    parser.add_argument('--config', type=str, default=config_path,
                       help='Path to config file (default: test/backtest_config.yaml)')
    parser.add_argument('--start-date', type=str, default=config.get('start_date', '2025-01-01'),
                       help='Start date (YYYY-MM-DD) - overrides config')
    parser.add_argument('--end-date', type=str, default=config.get('end_date', '2025-11-14'),
                       help='End date (YYYY-MM-DD) - overrides config')
    parser.add_argument('--risk-reward', type=float, default=config.get('risk_reward', 1.5),
                       help='Risk:Reward ratio - overrides config')
    parser.add_argument('--pivot-length', type=int, default=config.get('pivot_length', 5),
                       help='Pivot length for swing detection - overrides config')
    # Handle None/null values from YAML
    contracts_per_trade = config.get('contracts_per_trade')
    if contracts_per_trade is None or contracts_per_trade == 0:
        contracts_per_trade = None
    else:
        contracts_per_trade = int(contracts_per_trade)
    
    risk_per_trade = config.get('risk_per_trade')
    if risk_per_trade is None:
        risk_per_trade = None
    else:
        risk_per_trade = float(risk_per_trade)
    
    parser.add_argument('--contracts-per-trade', type=int, default=contracts_per_trade,
                       help='Fixed number of contracts per trade - overrides config')
    parser.add_argument('--risk-per-trade', type=float, default=risk_per_trade,
                       help='Risk per trade in dollars (only used if contracts_per_trade not set) - overrides config')
    parser.add_argument('--atr-period', type=int, default=config.get('atr_period', 5),
                       help='ATR period - overrides config')
    parser.add_argument('--atr-multiplier', type=float, default=config.get('atr_multiplier', 5.0),
                       help='ATR multiplier for stop sizing - overrides config')
    parser.add_argument('--ema-period', type=int, default=config.get('ema_period', 50),
                       help='EMA period for trend filter - overrides config')
    parser.add_argument('--confirmation-body-ratio', type=float, default=config.get('confirmation_body_ratio', 0.5),
                       help='Confirmation candle body ratio - overrides config')
    parser.add_argument('--use-cached', action='store_true', default=config.get('use_cached_data', False),
                       help='Use cached data if available - overrides config')
    parser.add_argument('--cache-dir', type=str, default=config.get('cache_dir', 'test/cache'),
                       help='Directory for cached data - overrides config')
    parser.add_argument('--output-dir', type=str, default=config.get('output_dir', 'test/results'),
                       help='Directory for output files - overrides config')
    parser.add_argument('--initial-equity', type=float, default=config.get('initial_equity', 10000.0),
                       help='Initial equity - overrides config')
    
    args = parser.parse_args()
    
    # If custom config file specified, reload it
    if args.config != config_path:
        config = load_config(args.config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("MGC LIQUIDITY SWEEP STRATEGY BACKTEST")
    logger.info("=" * 80)
    logger.info(f"Start Date: {args.start_date}")
    logger.info(f"End Date: {args.end_date}")
    logger.info(f"Risk:Reward: {args.risk_reward}")
    logger.info(f"Pivot Length: {args.pivot_length}")
    if args.contracts_per_trade:
        logger.info(f"Contracts Per Trade: {args.contracts_per_trade} (fixed)")
    else:
        logger.info(f"Risk Per Trade: ${args.risk_per_trade} (risk-based sizing)")
    logger.info(f"ATR Period: {args.atr_period}, ATR Multiplier: {args.atr_multiplier}")
    logger.info(f"EMA Period: {args.ema_period}")
    logger.info(f"Confirmation Body Ratio: {args.confirmation_body_ratio}")
    logger.info(f"Initial Equity: ${args.initial_equity:,.2f}")
    logger.info("")
    
    # Step 1: Fetch Data
    logger.info("Step 1: Fetching historical data from TopstepX...")
    fetcher = MGCDataFetcher()
    
    if not fetcher.authenticate():
        logger.error("Failed to authenticate with TopstepX")
        return
    
    # Check for cached data
    cache_file_5m = os.path.join(args.cache_dir, f"mgc_5m_{args.start_date}_{args.end_date}.csv")
    cache_file_15m = os.path.join(args.cache_dir, f"mgc_15m_{args.start_date}_{args.end_date}.csv")
    
    import pandas as pd
    
    if args.use_cached and os.path.exists(cache_file_5m) and os.path.exists(cache_file_15m):
        logger.info("Loading cached data...")
        df_5m = pd.read_csv(cache_file_5m, parse_dates=['timestamp'])
        df_15m = pd.read_csv(cache_file_15m, parse_dates=['timestamp'])
        logger.info(f"Loaded {len(df_5m)} 5m bars and {len(df_15m)} 15m bars from cache")
    else:
        logger.info("Fetching data from TopstepX API...")
        df_5m, df_15m = fetcher.fetch_for_backtest(
            start_date=args.start_date,
            end_date=args.end_date
        )
        
        if df_5m.empty or df_15m.empty:
            logger.error("Failed to fetch data or no data available")
            return
        
        # Cache the data
        df_5m.to_csv(cache_file_5m, index=False)
        df_15m.to_csv(cache_file_15m, index=False)
        logger.info(f"Cached data to {args.cache_dir}")
    
    logger.info(f"Data loaded: {len(df_5m)} 5m bars, {len(df_15m)} 15m bars")
    logger.info("")
    
    # Step 2: Run Backtest
    logger.info("Step 2: Running backtest...")
    engine = MGCBacktestEngine(
        df_5m=df_5m,
        df_15m=df_15m,
        risk_reward=args.risk_reward,
        pivot_length=args.pivot_length,
        contracts_per_trade=args.contracts_per_trade,
        risk_per_trade=args.risk_per_trade,
        atr_period=args.atr_period,
        atr_multiplier=args.atr_multiplier,
        ema_period=args.ema_period,
        confirmation_body_ratio=args.confirmation_body_ratio,
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
    report_file = os.path.join(args.output_dir, f"backtest_report_{args.start_date}_{args.end_date}.txt")
    with open(report_file, 'w') as f:
        f.write(report)
    logger.info(f"Report saved to {report_file}")
    
    # Export trades
    trades_file = os.path.join(args.output_dir, f"trades_{args.start_date}_{args.end_date}.csv")
    analyzer.export_trades_csv(trades_file)
    
    # Generate plots
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        equity_curve_file = os.path.join(args.output_dir, f"equity_curve_{args.start_date}_{args.end_date}.png")
        analyzer.plot_equity_curve(equity_curve_file)
        
        monthly_returns_file = os.path.join(args.output_dir, f"monthly_returns_{args.start_date}_{args.end_date}.png")
        analyzer.plot_monthly_returns(monthly_returns_file)
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
    except Exception as e:
        logger.warning(f"Error generating plots: {e}")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("BACKTEST COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

