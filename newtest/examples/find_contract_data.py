"""
Utility script to find available contracts and date ranges
"""
import sys
import os
import logging
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

from framework.contract_finder import ContractFinder
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Find available contracts and date ranges"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Find available contracts and date ranges')
    parser.add_argument('--symbol', default='MES', help='Trading symbol')
    parser.add_argument('--interval', default='15m', help='Bar interval')
    parser.add_argument('--start-date', type=str, help='Preferred start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='Preferred end date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    finder = ContractFinder()
    
    preferred_start = None
    preferred_end = None
    if args.start_date:
        preferred_start = datetime.fromisoformat(args.start_date)
    if args.end_date:
        preferred_end = datetime.fromisoformat(args.end_date)
    
    # Find all contracts with data
    print(f"\n{'='*80}")
    print(f"Finding contracts with data for {args.symbol}")
    print(f"{'='*80}\n")
    
    contracts = finder.find_contracts_with_data(
        symbol=args.symbol,
        interval=args.interval,
        target_start=preferred_start,
        target_end=preferred_end
    )
    
    if not contracts:
        print(f"No contracts found with data for {args.symbol}")
        return
    
    print(f"\nFound {len(contracts)} contracts with data:\n")
    for i, contract in enumerate(contracts, 1):
        print(f"{i}. {contract['name']} ({contract['contract_id']})")
        if contract['start_date'] and contract['end_date']:
            print(f"   Available: {contract['start_date'].date()} to {contract['end_date'].date()}")
        if contract['has_target_data']:
            print(f"   ✓ Has data for target range")
        else:
            print(f"   ✗ No data for target range")
        print()
    
    # Get recommendation
    print(f"\n{'='*80}")
    print("RECOMMENDATION")
    print(f"{'='*80}\n")
    
    recommendation = finder.recommend_contract_and_dates(
        symbol=args.symbol,
        interval=args.interval,
        preferred_start=preferred_start,
        preferred_end=preferred_end
    )
    
    if recommendation:
        print(f"Contract: {recommendation['name']} ({recommendation['contract_id']})")
        print(f"Recommended Date Range: {recommendation['start_date'].date()} to {recommendation['end_date'].date()}")
        if recommendation['available_start'] and recommendation['available_end']:
            print(f"Available Range: {recommendation['available_start'].date()} to {recommendation['available_end'].date()}")
        
        print(f"\nUse this in your backtest:")
        print(f"  --start-date {recommendation['start_date'].date()}")
        print(f"  --end-date {recommendation['end_date'].date()}")
    else:
        print("Could not find suitable contract and date range")


if __name__ == "__main__":
    main()

