"""
Data Fetcher for Statistical Arbitrage Backtest
Fetches historical data for both GC and MGC from TopstepX API
"""
import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.connectors.topstepx_client import TopstepXClient
from src.connectors.market_data_adapter import MarketDataAdapter
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class StatArbDataFetcher:
    """Fetches GC and MGC historical data from TopstepX"""
    
    def __init__(self):
        self.client = TopstepXClient(
            username=os.getenv('TOPSTEPX_USERNAME'),
            api_key=os.getenv('TOPSTEPX_API_KEY'),
            dry_run=False
        )
        self.adapter = MarketDataAdapter()
        self.gc_contract_id: Optional[str] = None
        self.mgc_contract_id: Optional[str] = None
    
    def authenticate(self) -> bool:
        """Authenticate with TopstepX"""
        return self.client.authenticate()
    
    def find_contract(self, symbol: str) -> Optional[str]:
        """Find contract ID for given symbol"""
        logger.info(f"Searching for {symbol} contract...")
        
        search_terms = [symbol, symbol.upper(), symbol.replace("GC", "GOLD")]
        contracts = []
        
        for term in search_terms:
            logger.info(f"Trying search term: {term}")
            contracts = self.client.search_contracts(symbol=term, live=False)
            if contracts:
                logger.info(f"Found {len(contracts)} contracts with search term '{term}'")
                break
        
        if not contracts:
            logger.error(f"No contracts found for {symbol}")
            return None
        
        # Find matching contract
        for contract in contracts:
            contract_id = (
                contract.get('contractId') or 
                contract.get('id') or 
                contract.get('contract_id') or
                contract.get('contractIdStr') or
                str(contract.get('contractId', ''))
            )
            
            symbol_field = (
                contract.get('symbol', '') or 
                contract.get('Symbol', '') or
                contract.get('name', '') or
                contract.get('Name', '')
            ).upper()
            
            contract_id_str = str(contract_id).upper()
            
            if symbol.upper() in symbol_field or symbol.upper() in contract_id_str:
                logger.info(f"Found {symbol} contract: {symbol_field} (ID: {contract_id})")
                return str(contract_id)
        
        logger.warning(f"{symbol} contract not found in results")
        return None
    
    def fetch_bars(
        self,
        contract_id: str,
        symbol: str,
        interval: str = "1m",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 10000
    ) -> pd.DataFrame:
        """
        Fetch historical bars and return as pandas DataFrame
        
        Args:
            contract_id: Contract ID
            symbol: Symbol name (for logging)
            interval: Bar interval ("1m", "5m", etc.)
            start_time: Start datetime
            end_time: End datetime
            limit: Max number of bars to fetch
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        logger.info(f"Fetching {symbol} {interval} bars from {start_time} to {end_time}")
        
        # Normalize timezones
        if start_time and start_time.tzinfo is None:
            from pytz import UTC
            start_time = start_time.replace(tzinfo=UTC)
        if end_time and end_time.tzinfo is None:
            from pytz import UTC
            end_time = end_time.replace(tzinfo=UTC)
        
        # Fetch in chunks
        all_bars = []
        current_start = start_time
        max_bars_per_request = 5000
        request_count = 0
        max_requests = 100
        
        while current_start and (not end_time or current_start < end_time):
            request_count += 1
            if request_count > max_requests:
                logger.warning(f"Reached max requests limit ({max_requests})")
                break
            
            bars = self.client.get_bars(
                contract_id=contract_id,
                interval=interval,
                start_time=current_start,
                end_time=end_time,
                limit=max_bars_per_request
            )
            
            if not bars:
                logger.info(f"No more bars returned (request {request_count})")
                break
            
            all_bars.extend(bars)
            logger.info(f"Fetched {len(bars)} bars (request {request_count}), total: {len(all_bars)}")
            
            if len(bars) < max_bars_per_request:
                break
            
            # Update start time for next chunk
            last_bar = bars[-1]
            last_timestamp = last_bar.get('t') or last_bar.get('timestamp')
            if isinstance(last_timestamp, str):
                try:
                    if last_timestamp.endswith('Z'):
                        last_dt = datetime.fromisoformat(last_timestamp.replace('Z', '+00:00'))
                    else:
                        last_dt = datetime.fromisoformat(last_timestamp)
                    
                    if current_start:
                        if current_start.tzinfo is None and last_dt.tzinfo is not None:
                            last_dt = last_dt.replace(tzinfo=None)
                        elif current_start.tzinfo is not None and last_dt.tzinfo is None:
                            from pytz import UTC
                            last_dt = last_dt.replace(tzinfo=UTC)
                    
                    if interval.endswith('m'):
                        minutes = int(interval[:-1])
                        current_start = last_dt + timedelta(minutes=minutes)
                    elif interval.endswith('h'):
                        hours = int(interval[:-1])
                        current_start = last_dt + timedelta(hours=hours)
                    else:
                        break
                except Exception as e:
                    logger.warning(f"Error parsing timestamp: {e}")
                    break
            else:
                break
            
            if limit and len(all_bars) >= limit:
                logger.info(f"Reached requested limit of {limit} bars")
                break
        
        if not all_bars:
            logger.warning(f"No bars fetched for {symbol}")
            return pd.DataFrame()
        
        logger.info(f"Total bars fetched for {symbol}: {len(all_bars)}")
        
        # Convert to candles and then to DataFrame
        candles = self.adapter.normalize_bars(all_bars, symbol, interval)
        
        if not candles:
            logger.warning(f"No candles normalized for {symbol}")
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
        
        if len(df) > 0:
            logger.info(f"{symbol} data: {len(df)} bars from {df.iloc[0]['timestamp']} to {df.iloc[-1]['timestamp']}")
        
        return df
    
    def fetch_for_backtest(
        self,
        start_date: str = "2025-01-01",
        end_date: str = "2025-11-14",
        interval: str = "1m",
        use_cached: bool = False,
        cache_dir: str = "test/cache"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch both GC and MGC data for backtest
        
        Returns:
            (df_gc, df_mgc) tuple
        """
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
        
        logger.info(f"Fetching StatArb data from {start_date} to {end_date}")
        
        # Check for cached data
        cache_file_gc = os.path.join(cache_dir, f"gc_{interval}_{start_date}_{end_date}.csv")
        cache_file_mgc = os.path.join(cache_dir, f"mgc_{interval}_{start_date}_{end_date}.csv")
        
        if use_cached and os.path.exists(cache_file_gc) and os.path.exists(cache_file_mgc):
            logger.info("Loading cached data...")
            df_gc = pd.read_csv(cache_file_gc, parse_dates=['timestamp'])
            df_mgc = pd.read_csv(cache_file_mgc, parse_dates=['timestamp'])
            logger.info(f"Loaded {len(df_gc)} GC bars and {len(df_mgc)} MGC bars from cache")
            return df_gc, df_mgc
        
        # Find contracts
        if not self.gc_contract_id:
            self.gc_contract_id = self.find_contract("GC")
        if not self.mgc_contract_id:
            self.mgc_contract_id = self.find_contract("MGC")
        
        if not self.gc_contract_id:
            raise Exception("Could not find GC contract")
        if not self.mgc_contract_id:
            raise Exception("Could not find MGC contract")
        
        # Fetch GC data
        logger.info("Fetching GC data...")
        df_gc = self.fetch_bars(
            contract_id=self.gc_contract_id,
            symbol="GC",
            interval=interval,
            start_time=start_dt,
            end_time=end_dt,
            limit=50000
        )
        
        # Fetch MGC data
        logger.info("Fetching MGC data...")
        df_mgc = self.fetch_bars(
            contract_id=self.mgc_contract_id,
            symbol="MGC",
            interval=interval,
            start_time=start_dt,
            end_time=end_dt,
            limit=50000
        )
        
        # Cache the data
        os.makedirs(cache_dir, exist_ok=True)
        if not df_gc.empty:
            df_gc.to_csv(cache_file_gc, index=False)
        if not df_mgc.empty:
            df_mgc.to_csv(cache_file_mgc, index=False)
        logger.info(f"Cached data to {cache_dir}")
        
        return df_gc, df_mgc

