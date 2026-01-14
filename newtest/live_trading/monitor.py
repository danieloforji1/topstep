"""
Live Trading Monitor
Simple monitoring script to check strategy status and performance
"""
import os
import sys
import time
import logging
from datetime import datetime
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from src.connectors.topstepx_client import TopstepXClient

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Monitor live trading status"""
    # Initialize API client
    client = TopstepXClient(
        username=os.getenv('TOPSTEPX_USERNAME'),
        api_key=os.getenv('TOPSTEPX_API_KEY'),
        dry_run=False  # We're just reading, not trading
    )
    
    if not client.authenticate():
        logger.error("Failed to authenticate")
        return
    
    # Get account
    accounts = client.get_accounts()
    if not accounts:
        logger.error("No accounts found")
        return
    
    account_id = accounts[0].get('id')
    client.account_id = account_id
    
    logger.info("="*80)
    logger.info("LIVE TRADING MONITOR")
    logger.info("="*80)
    logger.info(f"Account ID: {account_id}")
    logger.info(f"Time: {datetime.now()}")
    logger.info("="*80)
    
    # Get account balance
    balance = client.get_account_balance(account_id)
    if balance:
        logger.info(f"Account Balance: ${balance.get('balance', 0):,.2f}")
        logger.info(f"Equity: ${balance.get('equity', 0):,.2f}")
        logger.info(f"Available Margin: ${balance.get('availableMargin', 0):,.2f}")
    
    # Get open positions
    positions = client.get_positions(account_id)
    logger.info(f"\nOpen Positions: {len(positions)}")
    for pos in positions:
        logger.info(f"  {pos.get('symbol', 'Unknown')}: {pos.get('quantity', 0)} contracts @ ${pos.get('averagePrice', 0):.2f}")
        logger.info(f"    Unrealized P&L: ${pos.get('unrealizedPnL', 0):.2f}")
    
    # Get open orders
    orders = client.get_open_orders(account_id)
    logger.info(f"\nOpen Orders: {len(orders)}")
    for order in orders:
        side = "Buy" if order.get('side') == 0 else "Sell"
        order_type = order.get('type', 'Unknown')
        logger.info(f"  {side} {order.get('size', 0)} @ ${order.get('limitPrice', order.get('stopPrice', 0)):.2f} ({order_type})")
    
    logger.info("="*80)


if __name__ == "__main__":
    main()


