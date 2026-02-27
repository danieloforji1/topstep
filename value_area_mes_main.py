"""
Main Entry Point for Value Area MES Strategy
"""
import os
import sys
import logging
from threading import Thread

from dotenv import load_dotenv
import uvicorn

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from strategy.value_area_mes_strategy import ValueAreaMESStrategy
from api.ops import app, set_strategy_instance

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("value_area_mes_trading.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


def main():
    def run_api():
        uvicorn.run(app, host="0.0.0.0", port=8004, log_level="info")

    api_thread = Thread(target=run_api, daemon=True)
    api_thread.start()

    strategy = ValueAreaMESStrategy(config_path="value_area_mes_config.yaml")
    set_strategy_instance(strategy)

    try:
        strategy.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
