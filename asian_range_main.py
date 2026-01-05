"""
Main Entry Point for Asian Range Breakout Strategy
Separate from grid strategy - runs independently
"""
import os
import sys
import logging
from threading import Thread
from dotenv import load_dotenv
import uvicorn

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from strategy.asian_range_strategy import AsianRangeStrategy
from api.ops import app, set_strategy_instance

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('asian_range_trading.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point for Asian Range strategy"""
    # Start API server in background (on different port)
    def run_api():
        uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
    
    api_thread = Thread(target=run_api, daemon=True)
    api_thread.start()
    
    # Create and run strategy
    strategy = AsianRangeStrategy(config_path="asian_range_config.yaml")
    set_strategy_instance(strategy)
    
    try:
        strategy.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

