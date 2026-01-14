"""
Data Manager
Handles data fetching, caching, and alignment for backtesting
"""
import os
import pandas as pd
import logging
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
import sys

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Try to load .env from multiple possible locations
    env_paths = [
        os.path.join(os.path.dirname(__file__), '../../.env'),  # topstep/.env
        os.path.join(os.path.dirname(__file__), '../../../.env'),  # topstep/../.env
        '.env'  # Current directory
    ]
    for env_path in env_paths:
        if os.path.exists(env_path):
            load_dotenv(env_path)
            break
    else:
        # If no .env found, try default location
        load_dotenv()
except Exception:
    pass  # .env file is optional, continue without it

# Add parent directory to path to import TopstepX client
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
from src.connectors.topstepx_client import TopstepXClient
from src.connectors.market_data_adapter import MarketDataAdapter

# Import ContractFinder for finding available contracts
try:
    from framework.contract_finder import ContractFinder
    CONTRACT_FINDER_AVAILABLE = True
except ImportError:
    CONTRACT_FINDER_AVAILABLE = False

logger = logging.getLogger(__name__)


class DataManager:
    """
    Manages data fetching, caching, and alignment for backtesting
    """
    
    def __init__(
        self,
        cache_dir: str = "newtest/results/cache",
        use_cache: bool = True
    ):
        """
        Initialize data manager
        
        Args:
            cache_dir: Directory for caching data
            use_cache: Whether to use cached data if available
        """
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize TopstepX client (will authenticate when needed)
        self.client: Optional[TopstepXClient] = None
        self.adapter = MarketDataAdapter()
    
    def _get_client(self) -> TopstepXClient:
        """Get or create TopstepX client"""
        if self.client is None:
            self.client = TopstepXClient()
            if not self.client.authenticate():
                raise Exception("Failed to authenticate with TopstepX")
        return self.client
    
    def _get_cache_path(self, symbol: str, interval: str, start_date: str, end_date: str) -> str:
        """Get cache file path"""
        filename = f"{symbol}_{interval}_{start_date}_{end_date}.csv"
        return os.path.join(self.cache_dir, filename)
    
    def _load_from_cache(self, cache_path: str) -> Optional[pd.DataFrame]:
        """Load data from cache"""
        if not self.use_cache or not os.path.exists(cache_path):
            return None
        
        try:
            df = pd.read_csv(cache_path, parse_dates=['timestamp'])
            logger.info(f"Loaded {len(df)} bars from cache: {cache_path}")
            return df
        except Exception as e:
            logger.warning(f"Error loading cache {cache_path}: {e}")
            return None
    
    def _save_to_cache(self, df: pd.DataFrame, cache_path: str):
        """Save data to cache"""
        try:
            df.to_csv(cache_path, index=False)
            logger.info(f"Cached {len(df)} bars to: {cache_path}")
        except Exception as e:
            logger.warning(f"Error saving cache {cache_path}: {e}")
    
    def fetch_bars(
        self,
        symbol: str,
        interval: str = "15m",
        start_date: datetime = None,
        end_date: datetime = None,
        contract_id: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch historical bars for a symbol
        
        Args:
            symbol: Trading symbol (e.g., "MES", "MGC", "GC")
            interval: Bar interval ("1m", "5m", "15m", "1h", etc.)
            start_date: Start date
            end_date: End date
            contract_id: Optional contract ID (if None, will search)
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        # Format dates for cache
        start_str = start_date.strftime("%Y-%m-%d") if start_date else "all"
        end_str = end_date.strftime("%Y-%m-%d") if end_date else "all"
        
        cache_path = self._get_cache_path(symbol, interval, start_str, end_str)
        
        # Try to load from cache first
        df = self._load_from_cache(cache_path)
        if df is not None:
            logger.info(f"Using cached data: {len(df)} bars")
            return df
        
        # Also check test cache directory (where existing backtests store data)
        test_cache_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'test', 'test', 'cache',
            f"{symbol.lower()}_{interval}_{start_str}_{end_str}.csv"
        )
        if os.path.exists(test_cache_path):
            logger.info(f"Found data in test cache: {test_cache_path}")
            df = self._load_from_cache(test_cache_path)
            if df is not None:
                logger.info(f"Using test cache data: {len(df)} bars")
                return df
        
        # Fetch from API
        logger.info(f"Fetching {symbol} {interval} bars from TopstepX API...")
        client = self._get_client()
        
        # Find contract if not provided
        if contract_id is None:
            # Try multiple search approaches (like statarb_data_fetcher and mgc_backtest_data do)
            contracts = []
            
            # Try different search terms
            search_terms = [symbol, symbol.upper()]
            if symbol == "MES":
                search_terms.extend(["MICRO S&P", "MICRO ES"])
            elif symbol == "MNQ":
                search_terms.extend(["MICRO NASDAQ", "MICRO NQ"])
            elif symbol == "MGC":
                search_terms.extend(["MICRO GOLD", "GOLD"])
            
            # Collect ALL contracts from both live and non-live searches
            all_contracts = []
            contracts_seen = set()  # Track by ID to avoid duplicates
            
            for term in search_terms:
                logger.info(f"Trying search term: {term}")
                # Try both live=False and live=True to get all contracts
                for live_flag in [False, True]:
                    found_contracts = client.search_contracts(symbol=term, live=live_flag)
                    if found_contracts:
                        logger.info(f"Found {len(found_contracts)} contracts with search term '{term}' (live={live_flag})")
                        for contract in found_contracts:
                            # Extract contract ID to check for duplicates
                            potential_id = (
                                contract.get('contractId') or 
                                contract.get('id') or 
                                contract.get('contract_id') or
                                contract.get('contractIdStr')
                            )
                            if potential_id and str(potential_id) not in contracts_seen:
                                all_contracts.append(contract)
                                contracts_seen.add(str(potential_id))
            
            contracts = all_contracts
            
            if not contracts:
                raise Exception(f"Could not find contract for {symbol}. Tried search terms: {search_terms}")
            
            logger.info(f"Total unique contracts found: {len(contracts)}")
            
            # Initialize contract_id
            contract_id = None
            
            # If we have a date range, try to find the contract that was active during that period
            # by trying each contract until we find one with data
            if start_date and end_date:
                logger.info(f"Trying to find contract with data for date range {start_date.date()} to {end_date.date()}")
                
                # Extract year from date range
                target_year = start_date.year
                year_str = str(target_year)[-2:]  # Last 2 digits (e.g., "25" for 2025)
                
                # Try to construct historical contract IDs for the target year
                # Futures contract months: F=Jan, G=Feb, H=Mar, J=Apr, K=May, M=Jun, N=Jul, Q=Aug, U=Sep, V=Oct, X=Nov, Z=Dec
                contract_months = ['H', 'M', 'U', 'Z']  # March, June, September, December (quarterly)
                historical_contracts = []
                
                # Get symbol ID pattern from first contract (e.g., "F.US.MES" from "CON.F.US.MES.H26")
                if contracts:
                    first_contract_id = (
                        contracts[0].get('contractId') or 
                        contracts[0].get('id') or 
                        contracts[0].get('contract_id') or
                        contracts[0].get('contractIdStr') or ''
                    )
                    # Extract base pattern (e.g., "F.US.MES" from "CON.F.US.MES.H26")
                    if 'CON.F.US.' in str(first_contract_id):
                        base_pattern = str(first_contract_id).split('CON.')[1]  # Get "F.US.MES.H26"
                        base_parts = base_pattern.rsplit('.', 1)[0]  # Get "F.US.MES"
                        
                        # Construct contract IDs for target year
                        for month in contract_months:
                            constructed_id = f"CON.{base_parts}.{month}{year_str}"
                            historical_contracts.append({
                                'id': constructed_id,
                                'name': f"{symbol}{month}{year_str}",
                                'description': f"Constructed contract for {target_year}",
                                'constructed': True
                            })
                        
                        logger.info(f"Constructed {len(historical_contracts)} historical contract IDs for {target_year}")
                        # Add constructed contracts to the list (at the beginning for priority)
                        contracts = historical_contracts + contracts
                
                logger.info(f"Testing {len(contracts)} contracts (including {len(historical_contracts)} constructed)...")
                
                # Sort contracts: prioritize contracts that might be from target year
                def contract_priority(contract):
                    """Prioritize contracts that might be from target year"""
                    potential_id = (
                        contract.get('contractId') or 
                        contract.get('id') or 
                        contract.get('contract_id') or
                        contract.get('contractIdStr') or ''
                    )
                    contract_name = (
                        contract.get('name', '') or 
                        contract.get('description', '') or ''
                    )
                    search_text = (str(potential_id) + contract_name).upper()
                    # Higher priority if year string appears in contract ID/name, or if constructed
                    if contract.get('constructed', False):
                        return 0  # Highest priority for constructed contracts
                    if year_str in search_text:
                        return 1
                    return 2
                
                # Sort contracts by priority
                contracts_sorted = sorted(contracts, key=contract_priority)
                
                # Try each contract to see which one has data for the date range
                # Try both live=False and live=True for each contract
                for contract in contracts_sorted:
                    potential_id = (
                        contract.get('contractId') or 
                        contract.get('id') or 
                        contract.get('contract_id') or
                        contract.get('contractIdStr')
                    )
                    
                    if not potential_id:
                        continue
                    
                    potential_id_str = str(potential_id)
                    
                    # Get contract name for logging
                    contract_name = (
                        contract.get('name', '') or 
                        contract.get('description', '') or
                        potential_id_str
                    )
                    logger.info(f"Testing contract {contract_name} ({potential_id_str})...")
                    
                    # Try both live=False and live=True
                    for live_flag in [False, True]:
                        try:
                            # Try fetching a small sample to see if this contract has data
                            test_bars = client.get_bars(
                                contract_id=potential_id_str,
                                interval=interval,
                                start_time=start_date,
                                end_time=min(end_date, start_date + timedelta(days=1)),  # Just test first day
                                limit=10,
                                live=live_flag
                            )
                            
                            if test_bars:
                                contract_id = potential_id_str
                                logger.info(f"✓ Found contract with data: {contract_name} ({contract_id}) [live={live_flag}]")
                                break
                        except Exception as e:
                            logger.debug(f"Error testing contract {potential_id_str} with live={live_flag}: {e}")
                            continue
                    
                    if contract_id:
                        break
                
                if not contract_id:
                    logger.warning(f"None of the {len(contracts)} contracts have data for the requested date range")
                    logger.warning("Available contracts:")
                    for i, contract in enumerate(contracts[:10]):
                        contract_name = contract.get('name', '') or contract.get('description', '') or 'Unknown'
                        potential_id = (
                            contract.get('contractId') or 
                            contract.get('id') or 
                            contract.get('contract_id') or
                            contract.get('contractIdStr')
                        )
                        logger.warning(f"  {i+1}. {contract_name} ({potential_id})")
                    
                    # Try using ContractFinder to find what's actually available
                    if CONTRACT_FINDER_AVAILABLE:
                        logger.info("Using ContractFinder to find available contracts and date ranges...")
                        try:
                            finder = ContractFinder()
                            recommendation = finder.recommend_contract_and_dates(
                                symbol=symbol,
                                interval=interval,
                                preferred_start=start_date,
                                preferred_end=end_date
                            )
                            
                            if recommendation:
                                logger.info(f"ContractFinder recommends: {recommendation['name']} ({recommendation['contract_id']})")
                                logger.info(f"Available date range: {recommendation['start_date'].date()} to {recommendation['end_date'].date()}")
                                
                                # Use the recommended contract
                                contract_id = recommendation['contract_id']
                                
                                # If the recommended range overlaps with requested, adjust to intersection
                                if recommendation['start_date'] <= end_date and recommendation['end_date'] >= start_date:
                                    adjusted_start = max(start_date, recommendation['start_date'])
                                    adjusted_end = min(end_date, recommendation['end_date'])
                                    logger.info(f"Using recommended contract with adjusted dates: {adjusted_start.date()} to {adjusted_end.date()}")
                                    start_date = adjusted_start
                                    end_date = adjusted_end
                                else:
                                    # No overlap - use the available range instead
                                    logger.warning(f"Requested date range {start_date.date()} to {end_date.date()} not available")
                                    logger.warning(f"Using available range instead: {recommendation['start_date'].date()} to {recommendation['end_date'].date()}")
                                    start_date = recommendation['start_date']
                                    end_date = recommendation['end_date']
                        except Exception as e:
                            logger.debug(f"ContractFinder error: {e}")
            
            # If still no contract_id, use matching logic
            if not contract_id:
                symbol_upper = symbol.upper()
                
                for contract in contracts:
                    # Try different field names for contract ID
                    potential_id = (
                        contract.get('contractId') or 
                        contract.get('id') or 
                        contract.get('contract_id') or
                        contract.get('contractIdStr')
                    )
                    
                    if not potential_id:
                        continue
                    
                    potential_id_str = str(potential_id).upper()
                    
                    # Try different field names for symbol
                    symbol_field = (
                        contract.get('symbol', '') or 
                        contract.get('Symbol', '') or
                        contract.get('name', '') or
                        contract.get('Name', '') or
                        contract.get('instrument', '')
                    ).upper()
                    
                    # Check if symbol matches
                    if symbol_upper in symbol_field or symbol_upper in potential_id_str:
                        contract_id = str(potential_id)
                        logger.info(f"Found {symbol} contract: {symbol_field} (ID: {contract_id})")
                        logger.debug(f"Contract details: {contract}")
                        break
                
                # If no match found, use first contract (fallback)
                if not contract_id and contracts:
                    potential_id = (
                        contracts[0].get('contractId') or 
                        contracts[0].get('id') or 
                        contracts[0].get('contract_id') or
                        contracts[0].get('contractIdStr')
                    )
                    if potential_id:
                        contract_id = str(potential_id)
                        logger.warning(f"Using first contract as fallback: {contract_id}")
                        logger.warning(f"Contract details: {contracts[0]}")
            
            if not contract_id:
                logger.error(f"Could not extract contract ID from contracts: {contracts[:3]}")
                raise Exception(f"Could not extract contract ID for {symbol}")
        
        # Ensure dates are timezone-aware (UTC)
        from datetime import timezone
        if start_date and start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date and end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)
        
        # Fetch bars - try both live=False and live=True
        bars = []
        for live_flag in [False, True]:
            logger.info(f"Fetching bars with live={live_flag}...")
            bars = client.get_bars(
                contract_id=contract_id,
                interval=interval,
                start_time=start_date,
                end_time=end_date,
                limit=50000,
                live=live_flag
            )
            if bars:
                logger.info(f"Found {len(bars)} bars with live={live_flag}")
                break
        
        # Convert bars to DataFrame
        if bars:
            # Convert to DataFrame
            data = []
            for bar in bars:
                candle = self.adapter.normalize_candle(bar, symbol, interval)
                if candle:
                    data.append({
                        'timestamp': candle.timestamp,
                        'open': candle.open,
                        'high': candle.high,
                        'low': candle.low,
                        'close': candle.close,
                        'volume': candle.volume
                    })
            
            if not data:
                logger.warning(f"Adapter returned no valid candles from {len(bars)} raw bars")
                logger.debug(f"Sample raw bar: {bars[0] if bars else 'No bars'}")
                raise Exception(f"No valid bars returned for {symbol} (adapter returned empty)")
            
            df = pd.DataFrame(data)
            
            if df.empty:
                raise Exception(f"DataFrame is empty after conversion for {symbol}")
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Save to cache
            self._save_to_cache(df, cache_path)
            
            logger.info(f"Fetched {len(df)} bars for {symbol} from {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            return df
        
        if not bars:
            # Log more details for debugging
            logger.error(f"No bars returned for {symbol}")
            logger.error(f"Contract ID: {contract_id}")
            logger.error(f"Date range: {start_date} to {end_date}")
            logger.error(f"Interval: {interval}")
            
            # Try to find what date range this contract actually has data for
            logger.info("Attempting to find available date range for this contract...")
            available_range = self._probe_contract_date_range(client, contract_id, interval, start_date, end_date)
            
            if available_range:
                logger.info(f"Contract has data from {available_range[0].date()} to {available_range[1].date()}")
                logger.warning(f"Requested range {start_date.date()} to {end_date.date()} is outside available range")
                logger.warning(f"Try using: --start-date {available_range[0].date()} --end-date {available_range[1].date()}")
            
            # Suggest using cached data if available
            logger.warning("Attempting to use cached data from test directory if available...")
            test_cache_glob = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'test', 'test', 'cache',
                f"{symbol.lower()}_{interval}_*.csv"
            )
            import glob
            cache_files = glob.glob(test_cache_glob)
            if cache_files:
                logger.info(f"Found {len(cache_files)} potential cache files")
                # Try to find one that overlaps with our date range
                for cache_file in cache_files:
                    try:
                        test_df = pd.read_csv(cache_file, parse_dates=['timestamp'], nrows=1)
                        if not test_df.empty:
                            logger.info(f"Found cache file: {cache_file}")
                            df = pd.read_csv(cache_file, parse_dates=['timestamp'])
                            # Filter to requested date range
                            if start_date:
                                df = df[df['timestamp'] >= pd.Timestamp(start_date)]
                            if end_date:
                                df = df[df['timestamp'] <= pd.Timestamp(end_date)]
                            if not df.empty:
                                logger.info(f"Using cached data: {len(df)} bars from {cache_file}")
                                return df
                    except Exception as e:
                        logger.debug(f"Error reading cache file {cache_file}: {e}")
                        continue
            
            error_msg = f"No bars returned for {symbol} (contract: {contract_id}, dates: {start_date.date()} to {end_date.date()})"
            if available_range:
                error_msg += f". Contract has data from {available_range[0].date()} to {available_range[1].date()}"
            error_msg += ". Try using: python examples/find_contract_data.py --symbol MES to find available dates"
            raise Exception(error_msg)
    
    def _probe_contract_date_range(
        self,
        client: TopstepXClient,
        contract_id: str,
        interval: str,
        requested_start: datetime,
        requested_end: datetime
    ) -> Optional[Tuple[datetime, datetime]]:
        """Probe contract to find what date range it actually has data for"""
        # Try recent dates (more likely to have data)
        now = datetime.now()
        test_ranges = [
            (now - timedelta(days=7), now),
            (now - timedelta(days=30), now),
            (now - timedelta(days=90), now),
            (requested_start, requested_end),  # Try requested range
        ]
        
        found_start = None
        found_end = None
        
        for test_start, test_end in test_ranges:
            for live in [False, True]:
                try:
                    bars = client.get_bars(
                        contract_id=contract_id,
                        interval=interval,
                        start_time=test_start,
                        end_time=test_end,
                        limit=100,
                        live=live
                    )
                    if bars:
                        # Found data, extract date range from bars
                        timestamps = []
                        for bar in bars:
                            t = bar.get('t') or bar.get('timestamp')
                            if t:
                                if isinstance(t, str):
                                    if t.endswith('Z'):
                                        t = datetime.fromisoformat(t.replace('Z', '+00:00'))
                                    else:
                                        t = datetime.fromisoformat(t)
                                timestamps.append(t)
                        
                        if timestamps:
                            bar_start = min(timestamps)
                            bar_end = max(timestamps)
                            if found_start is None or bar_start < found_start:
                                found_start = bar_start
                            if found_end is None or bar_end > found_end:
                                found_end = bar_end
                        break
                except Exception as e:
                    logger.debug(f"Error probing date range: {e}")
                    continue
        
        if found_start and found_end:
            return (found_start, found_end)
        return None
    
    def fetch_multiple_contracts(
        self,
        symbols: List[str],
        interval: str = "15m",
        start_date: datetime = None,
        end_date: datetime = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch bars for multiple symbols
        
        Args:
            symbols: List of trading symbols
            interval: Bar interval
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        result = {}
        for symbol in symbols:
            try:
                df = self.fetch_bars(symbol, interval, start_date, end_date)
                result[symbol] = df
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
        
        return result
    
    def align_dataframes(
        self,
        dataframes: Dict[str, pd.DataFrame],
        method: str = "inner"
    ) -> Dict[str, pd.DataFrame]:
        """
        Align multiple DataFrames by timestamp
        
        Args:
            dataframes: Dictionary of symbol -> DataFrame
            method: "inner" (intersection) or "outer" (union)
            
        Returns:
            Dictionary of aligned DataFrames
        """
        if len(dataframes) < 2:
            return dataframes
        
        # Get all timestamps
        all_timestamps = set()
        for df in dataframes.values():
            all_timestamps.update(df['timestamp'].tolist())
        
        # Create aligned timestamp index
        aligned_timestamps = sorted(all_timestamps)
        
        if method == "inner":
            # Only keep timestamps present in all DataFrames
            for df in dataframes.values():
                aligned_timestamps = [t for t in aligned_timestamps if t in df['timestamp'].values]
        
        # Align each DataFrame
        aligned = {}
        for symbol, df in dataframes.items():
            # Reindex to aligned timestamps, forward fill missing values
            df_indexed = df.set_index('timestamp')
            df_aligned = df_indexed.reindex(aligned_timestamps, method='ffill')
            df_aligned = df_aligned.reset_index()
            aligned[symbol] = df_aligned
        
        logger.info(f"Aligned {len(dataframes)} DataFrames to {len(aligned_timestamps)} timestamps")
        
        return aligned

