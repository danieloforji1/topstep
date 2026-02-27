"""
Main Entry Point for MES-M2K Bounded Spread Grid Strategy
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
sys.path.insert(0, os.path.join(project_root, "src"))

from strategy.mes_m2k_grid_strategy import MESM2KSpreadGridStrategy
from api.ops import app, set_strategy_instance

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("grid_new_trading.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def main():
    def run_api():
        uvicorn.run(app, host="0.0.0.0", port=8003, log_level="info")

    api_thread = Thread(target=run_api, daemon=True)
    api_thread.start()

    strategy = MESM2KSpreadGridStrategy(config_path="grid_new_config.yaml")
    set_strategy_instance(strategy)

    try:
        strategy.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

