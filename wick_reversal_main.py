"""
Main Entry Point for Wick Reversal Strategy
Trades reversals after "fat" hourly wicks with pullback confirmation
"""
import os
import sys
import logging
from threading import Thread
from dotenv import load_dotenv
import uvicorn

# Add project root to path first
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
# Add src to path
sys.path.insert(0, os.path.join(project_root, 'src'))

from strategy.wick_reversal_strategy import WickReversalStrategy
from api.ops import app, set_strategy_instance

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('wick_reversal_trading.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point for Wick Reversal strategy"""
    # Start API server in background (on different port)
    def run_api():
        uvicorn.run(app, host="0.0.0.0", port=8003, log_level="info")
    
    api_thread = Thread(target=run_api, daemon=True)
    api_thread.start()
    
    # Create and run strategy
    strategy = WickReversalStrategy(config_path="wick_reversal_config.yaml")
    set_strategy_instance(strategy)
    
    try:
        strategy.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

