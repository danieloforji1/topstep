"""
Contract Finder Utility
Helps find the right contract and date range for historical data
"""
import logging
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
from src.connectors.topstepx_client import TopstepXClient

logger = logging.getLogger(__name__)


class ContractFinder:
    """Utility to find contracts and their available date ranges"""
    
    def __init__(self):
        self.client = TopstepXClient()
        if not self.client.authenticate():
            raise Exception("Failed to authenticate with TopstepX")
    
    def find_contracts_with_data(
        self,
        symbol: str,
        interval: str = "15m",
        target_start: Optional[datetime] = None,
        target_end: Optional[datetime] = None
    ) -> List[Dict]:
        """
        Find contracts that have data for the target date range
        
        Returns:
            List of contracts with their available date ranges
        """
        logger.info(f"Finding contracts with data for {symbol}")
        
        # Search for all contracts
        contracts = []
        search_terms = [symbol, symbol.upper()]
        if symbol == "MES":
            search_terms.extend(["MICRO S&P", "MICRO ES"])
        
        all_contracts = []
        for term in search_terms:
            for live in [False, True]:
                found = self.client.search_contracts(symbol=term, live=live)
                for contract in found:
                    contract_id = (
                        contract.get('contractId') or 
                        contract.get('id') or 
                        contract.get('contract_id') or
                        contract.get('contractIdStr')
                    )
                    if contract_id and contract_id not in [c.get('id') for c in all_contracts]:
                        all_contracts.append({
                            'id': str(contract_id),
                            'name': contract.get('name', ''),
                            'description': contract.get('description', ''),
                            'contract': contract
                        })
        
        logger.info(f"Found {len(all_contracts)} unique contracts")
        
        # Test each contract to find available date range
        results = []
        for contract_info in all_contracts:
            contract_id = contract_info['id']
            logger.info(f"Testing contract {contract_info['name']} ({contract_id})...")
            
            # Try to find available date range by testing different dates
            available_range = self._find_available_date_range(
                contract_id, interval, target_start, target_end
            )
            
            if available_range:
                # Ensure timezone consistency
                from datetime import timezone as tz
                start_date = available_range[0]
                end_date = available_range[1]
                if start_date.tzinfo is None:
                    start_date = start_date.replace(tzinfo=tz.utc)
                if end_date.tzinfo is None:
                    end_date = end_date.replace(tzinfo=tz.utc)
                
                results.append({
                    'contract_id': contract_id,
                    'name': contract_info['name'],
                    'description': contract_info['description'],
                    'start_date': start_date,
                    'end_date': end_date,
                    'has_target_data': self._has_data_for_range(
                        contract_id, interval, target_start, target_end
                    )
                })
        
        return results
    
    def _find_available_date_range(
        self,
        contract_id: str,
        interval: str,
        target_start: Optional[datetime],
        target_end: Optional[datetime]
    ) -> Optional[Tuple[datetime, datetime]]:
        """Find the available date range for a contract"""
        # Try recent dates first (more likely to have data)
        test_dates = []
        
        # If target dates provided, test around them
        if target_start and target_end:
            # Test target range
            test_dates.append((target_start, target_end))
            # Test a week before/after
            test_dates.append((target_start - timedelta(days=7), target_end))
            test_dates.append((target_start, target_end + timedelta(days=7)))
        
        # Also test recent dates (last 30 days)
        now = datetime.now()
        test_dates.append((now - timedelta(days=30), now))
        test_dates.append((now - timedelta(days=7), now))
        
        # Try each date range
        for start, end in test_dates:
            for live in [False, True]:
                bars = self.client.get_bars(
                    contract_id=contract_id,
                    interval=interval,
                    start_time=start,
                    end_time=end,
                    limit=10,
                    live=live
                )
                if bars:
                    # Found data, try to find the full range
                    logger.info(f"  Found data for {contract_id} around {start.date()} to {end.date()}")
                    return (start, end)
        
        return None
    
    def _has_data_for_range(
        self,
        contract_id: str,
        interval: str,
        start: Optional[datetime],
        end: Optional[datetime]
    ) -> bool:
        """Check if contract has data for specific range"""
        if not start or not end:
            return False
        
        for live in [False, True]:
            bars = self.client.get_bars(
                contract_id=contract_id,
                interval=interval,
                start_time=start,
                end_time=end,
                limit=10,
                live=live
            )
            if bars:
                return True
        return False
    
    def recommend_contract_and_dates(
        self,
        symbol: str,
        interval: str = "15m",
        preferred_start: Optional[datetime] = None,
        preferred_end: Optional[datetime] = None
    ) -> Optional[Dict]:
        """
        Find the best contract and date range for backtesting
        
        Returns:
            Dict with contract_id, start_date, end_date, and available data info
        """
        contracts = self.find_contracts_with_data(symbol, interval, preferred_start, preferred_end)
        
        if not contracts:
            logger.warning(f"No contracts found with data for {symbol}")
            return None
        
        # Find contract that has data for preferred range
        if preferred_start and preferred_end:
            for contract in contracts:
                if contract['has_target_data']:
                    logger.info(f"Found contract with data for preferred range: {contract['contract_id']}")
                    return {
                        'contract_id': contract['contract_id'],
                        'name': contract['name'],
                        'start_date': preferred_start,
                        'end_date': preferred_end,
                        'available_start': contract['start_date'],
                        'available_end': contract['end_date']
                    }
        
        # Use contract with most recent data
        # Ensure timezone consistency for comparison
        def safe_max_key(x):
            if x['end_date']:
                # Ensure timezone-aware
                end_date = x['end_date']
                if end_date.tzinfo is None:
                    from datetime import timezone
                    end_date = end_date.replace(tzinfo=timezone.utc)
                return end_date
            else:
                from datetime import timezone
                return datetime.min.replace(tzinfo=timezone.utc)
        
        best_contract = max(contracts, key=safe_max_key)
        
        # Determine best date range
        # Ensure all datetimes are timezone-aware for comparison
        from datetime import timezone as tz
        
        if preferred_start and preferred_end:
            # Make preferred dates timezone-aware if needed
            if preferred_start.tzinfo is None:
                preferred_start = preferred_start.replace(tzinfo=tz.utc)
            if preferred_end.tzinfo is None:
                preferred_end = preferred_end.replace(tzinfo=tz.utc)
            
            # Make contract dates timezone-aware if needed
            contract_start = best_contract['start_date']
            contract_end = best_contract['end_date']
            if contract_start and contract_start.tzinfo is None:
                contract_start = contract_start.replace(tzinfo=tz.utc)
            if contract_end and contract_end.tzinfo is None:
                contract_end = contract_end.replace(tzinfo=tz.utc)
            
            # Try to use preferred range, but adjust if needed
            start = max(preferred_start, contract_start) if contract_start else preferred_start
            end = min(preferred_end, contract_end) if contract_end else preferred_end
        else:
            # Use available range
            start = best_contract['start_date']
            end = best_contract['end_date']
        
        logger.info(f"Recommended contract: {best_contract['name']} ({best_contract['contract_id']})")
        logger.info(f"Recommended date range: {start.date()} to {end.date()}")
        
        return {
            'contract_id': best_contract['contract_id'],
            'name': best_contract['name'],
            'start_date': start,
            'end_date': end,
            'available_start': best_contract['start_date'],
            'available_end': best_contract['end_date']
        }

