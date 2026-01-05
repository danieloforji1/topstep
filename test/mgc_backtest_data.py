"""
Data Fetcher for MGC Gold Futures Backtest
Fetches historical data from TopstepX API
"""
import os
import sys
import logging
from datetime import datetime, timedelta
from typing import List, Optional
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.connectors.topstepx_client import TopstepXClient
from src.connectors.market_data_adapter import MarketDataAdapter
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class MGCDataFetcher:
    """Fetches MGC historical data from TopstepX"""
    
    def __init__(self):
        self.client = TopstepXClient(
            username=os.getenv('TOPSTEPX_USERNAME'),
            api_key=os.getenv('TOPSTEPX_API_KEY'),
            dry_run=False  # We need real data for backtesting
        )
        self.adapter = MarketDataAdapter()
        self.contract_id: Optional[str] = None
        
    def authenticate(self) -> bool:
        """Authenticate with TopstepX"""
        return self.client.authenticate()
    
    def find_mgc_contract(self) -> Optional[str]:
        """Find MGC contract ID"""
        logger.info("Searching for MGC contract...")
        
        # Try different search terms
        search_terms = ["MGC", "GOLD", "MICRO GOLD"]
        contracts = []
        
        for term in search_terms:
            logger.info(f"Trying search term: {term}")
            contracts = self.client.search_contracts(symbol=term, live=False)
            if contracts:
                logger.info(f"Found {len(contracts)} contracts with search term '{term}'")
                break
        
        if not contracts:
            logger.error("No contracts found with any search term")
            return None
        
        # Debug: log first few contracts to see structure
        logger.info(f"Sample contract structure: {contracts[0] if contracts else 'None'}")
        
        # Find the most recent/active contract
        # Check multiple possible field names for symbol and contract ID
        for contract in contracts:
            # Try different field names for contract ID
            contract_id = (
                contract.get('contractId') or 
                contract.get('id') or 
                contract.get('contract_id') or
                contract.get('contractIdStr') or
                str(contract.get('contractId', ''))
            )
            
            # Try different field names for symbol
            symbol = (
                contract.get('symbol', '') or 
                contract.get('Symbol', '') or
                contract.get('name', '') or
                contract.get('Name', '') or
                contract.get('instrument', '') or
                contract.get('Instrument', '')
            ).upper()
            
            # Also check if contract_id itself contains MGC
            contract_id_str = str(contract_id).upper()
            
            logger.debug(f"Checking contract - ID: {contract_id}, Symbol: {symbol}, ContractID: {contract_id_str}")
            
            # Check if this is an MGC contract
            if 'MGC' in symbol or 'MGC' in contract_id_str:
                logger.info(f"Found MGC contract: {symbol} (ID: {contract_id})")
                self.contract_id = str(contract_id)
                return str(contract_id)
        
        # If no match found, log all available contracts for debugging
        logger.warning("MGC contract not found in results. Available contracts:")
        for i, contract in enumerate(contracts[:10]):  # Show first 10
            logger.warning(f"  Contract {i+1}: {contract}")
        
        return None
    
    def fetch_bars(
        self,
        interval: str = "5m",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 10000
    ) -> pd.DataFrame:
        """
        Fetch historical bars and return as pandas DataFrame
        
        Args:
            interval: "5m" or "15m"
            start_time: Start datetime
            end_time: End datetime
            limit: Max number of bars to fetch
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        if not self.contract_id:
            if not self.find_mgc_contract():
                raise Exception("Could not find MGC contract")
        
        logger.info(f"Fetching {interval} bars from {start_time} to {end_time}")
        
        # Normalize timezones - ensure both are aware or both are naive
        if start_time:
            if start_time.tzinfo is None:
                # Assume UTC if naive
                from pytz import UTC
                start_time = start_time.replace(tzinfo=UTC)
        if end_time:
            if end_time.tzinfo is None:
                # Assume UTC if naive
                from pytz import UTC
                end_time = end_time.replace(tzinfo=UTC)
        
        # Calculate expected number of bars for logging
        if start_time and end_time:
            time_diff = end_time - start_time
            if interval.endswith('m'):
                minutes = int(interval[:-1])
                expected_bars = int(time_diff.total_seconds() / 60 / minutes)
                logger.info(f"Expected approximately {expected_bars} bars for this date range")
        
        # Fetch in chunks if needed (API may have limits)
        all_bars = []
        current_start = start_time
        max_bars_per_request = 5000  # API limit per request
        request_count = 0
        max_requests = 100  # Safety limit to prevent infinite loops
        
        while current_start and (not end_time or current_start < end_time):
            request_count += 1
            if request_count > max_requests:
                logger.warning(f"Reached max requests limit ({max_requests}), stopping fetch")
                break
            
            bars = self.client.get_bars(
                contract_id=self.contract_id,
                interval=interval,
                start_time=current_start,
                end_time=end_time,
                limit=max_bars_per_request  # API limit per request
            )
            
            if not bars:
                logger.info(f"No more bars returned from API (request {request_count}), stopping fetch. Total bars so far: {len(all_bars)}")
                if current_start and end_time:
                    logger.info(f"  Requested range: {current_start} to {end_time}")
                    logger.warning(f"  API may not have data for the full requested range")
                break
            
            all_bars.extend(bars)
            logger.info(f"Fetched {len(bars)} bars (request {request_count}), total: {len(all_bars)}")
            
            # Log timestamp range of fetched bars
            if bars:
                first_bar_time = bars[0].get('t') or bars[0].get('timestamp', 'N/A')
                last_bar_time = bars[-1].get('t') or bars[-1].get('timestamp', 'N/A')
                logger.debug(f"  Bar range: {first_bar_time} to {last_bar_time}")
            
            # If we got fewer bars than requested, we might be done, but check if we've reached end_time
            if len(bars) < max_bars_per_request:
                # Check if we've reached the end time
                last_bar = bars[-1]
                last_timestamp = last_bar.get('t') or last_bar.get('timestamp')
                if isinstance(last_timestamp, str):
                    try:
                        if last_timestamp.endswith('Z'):
                            last_dt = datetime.fromisoformat(last_timestamp.replace('Z', '+00:00'))
                        else:
                            last_dt = datetime.fromisoformat(last_timestamp)
                        
                        # Ensure timezone awareness matches
                        if end_time:
                            if end_time.tzinfo is None and last_dt.tzinfo is not None:
                                last_dt = last_dt.replace(tzinfo=None)
                            elif end_time.tzinfo is not None and last_dt.tzinfo is None:
                                from pytz import UTC
                                last_dt = last_dt.replace(tzinfo=UTC)
                        
                        # If last bar is close to end_time, we're done
                        if end_time and last_dt >= end_time:
                            break
                    except Exception as e:
                        logger.debug(f"Error checking end_time: {e}")
                        pass
                # If we got fewer bars than max, we're likely done
                break
            
            # Update start time to last bar timestamp + 1 interval for next chunk
            last_bar = bars[-1]
            last_timestamp = last_bar.get('t') or last_bar.get('timestamp')
            if isinstance(last_timestamp, str):
                try:
                    if last_timestamp.endswith('Z'):
                        last_dt = datetime.fromisoformat(last_timestamp.replace('Z', '+00:00'))
                    else:
                        last_dt = datetime.fromisoformat(last_timestamp)
                    
                    # Ensure timezone matches current_start
                    if current_start:
                        if current_start.tzinfo is None and last_dt.tzinfo is not None:
                            last_dt = last_dt.replace(tzinfo=None)
                        elif current_start.tzinfo is not None and last_dt.tzinfo is None:
                            from pytz import UTC
                            last_dt = last_dt.replace(tzinfo=UTC)
                    
                    # Move forward by interval for next request
                    if interval.endswith('m'):
                        minutes = int(interval[:-1])
                        current_start = last_dt + timedelta(minutes=minutes)
                    elif interval.endswith('h'):
                        hours = int(interval[:-1])
                        current_start = last_dt + timedelta(hours=hours)
                    elif interval.endswith('s'):
                        seconds = int(interval[:-1])
                        current_start = last_dt + timedelta(seconds=seconds)
                    else:
                        break
                except Exception as e:
                    logger.warning(f"Error parsing timestamp for chunking: {e}")
                    break
            else:
                break
            
            # Safety check - if we've fetched enough bars, stop
            if limit and len(all_bars) >= limit:
                logger.info(f"Reached requested limit of {limit} bars")
                break
        
        if not all_bars:
            logger.warning("No bars fetched")
            return pd.DataFrame()
        
        logger.info(f"Total bars fetched: {len(all_bars)} across {request_count} API request(s)")
        
        # Convert to candles and then to DataFrame
        candles = self.adapter.normalize_bars(all_bars, "MGC", interval)
        
        if not candles:
            logger.warning("No candles normalized")
            return pd.DataFrame()
        
        # Convert to DataFrame
        data = {
            'timestamp': [c.timestamp for c in candles],
            'open': [c.open for c in candles],
            'high': [c.high for c in candles],
            'low': [c.low for c in candles],
            'close': [c.close for c in candles],
            'volume': [c.volume for c in candles]
        }
        
        df = pd.DataFrame(data)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Log actual date range of fetched data
        if len(df) > 0:
            first_ts = df.iloc[0]['timestamp']
            last_ts = df.iloc[-1]['timestamp']
            logger.info(f"Fetched {len(df)} bars covering {first_ts} to {last_ts}")
            
            # Warn if data range doesn't match requested range
            if start_time and end_time:
                if first_ts > start_time or last_ts < end_time:
                    logger.warning(f"⚠️  Data range mismatch!")
                    logger.warning(f"   Requested: {start_time} to {end_time}")
                    logger.warning(f"   Received:  {first_ts} to {last_ts}")
                    logger.warning(f"   The API may not have historical data for the full requested range.")
                    logger.warning(f"   This is common - APIs often only provide recent data (e.g., last 30-90 days).")
        
        return df
    
    def fetch_for_backtest(
        self,
        start_date: str = "2025-01-01",
        end_date: str = "2025-11-14"
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch both 5m and 15m data for backtest
        
        Returns:
            (df_5m, df_15m) tuple
        """
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
        
        logger.info(f"Fetching data from {start_date} to {end_date}")
        
        # Fetch 5m data
        df_5m = self.fetch_bars(
            interval="5m",
            start_time=start_dt,
            end_time=end_dt,
            limit=50000
        )
        
        # Fetch 15m data
        df_15m = self.fetch_bars(
            interval="15m",
            start_time=start_dt,
            end_time=end_dt,
            limit=20000
        )
        
        return df_5m, df_15m

