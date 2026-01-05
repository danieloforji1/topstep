"""
Market Data Adapter
Normalizes incoming market data from TopstepX to internal format
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Tick:
    """Normalized tick data"""
    symbol: str
    price: float
    volume: float
    timestamp: datetime
    bid: Optional[float] = None
    ask: Optional[float] = None


@dataclass
class Candle:
    """Normalized candle/bar data"""
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: datetime
    interval: str = "15m"


class MarketDataAdapter:
    """Adapter to normalize TopstepX market data to internal format"""
    
    @staticmethod
    def normalize_tick(data: Dict[str, Any], symbol: str) -> Optional[Tick]:
        """Convert TopstepX quote/trade data to Tick"""
        try:
            # Handle GatewayQuote format
            if "lastPrice" in data:
                return Tick(
                    symbol=symbol,
                    price=data.get("lastPrice", 0.0),
                    volume=data.get("volume", 0.0),
                    timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat()).replace('Z', '+00:00')),
                    bid=data.get("bestBid"),
                    ask=data.get("bestAsk")
                )
            # Handle GatewayTrade format
            elif "price" in data:
                return Tick(
                    symbol=symbol,
                    price=data.get("price", 0.0),
                    volume=data.get("volume", 0.0),
                    timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat()).replace('Z', '+00:00'))
                )
        except Exception as e:
            logger.error(f"Error normalizing tick: {e}")
            return None
    
    @staticmethod
    def normalize_candle(data: Dict[str, Any], symbol: str, interval: str = "15m") -> Optional[Candle]:
        """Convert TopstepX bar data to Candle"""
        try:
            # TopstepX API returns bars with: t (timestamp), o (open), h (high), l (low), c (close), v (volume)
            # Also support standard format: timestamp/time/date, open, high, low, close, volume
            timestamp_str = data.get("t") or data.get("timestamp") or data.get("time") or data.get("date")
            if isinstance(timestamp_str, str):
                # Handle ISO format with or without timezone
                if timestamp_str.endswith('Z'):
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                elif '+' in timestamp_str or timestamp_str.count('-') > 2:
                    timestamp = datetime.fromisoformat(timestamp_str)
                else:
                    timestamp = datetime.fromisoformat(timestamp_str + '+00:00')
            elif isinstance(timestamp_str, (int, float)):
                timestamp = datetime.fromtimestamp(timestamp_str)
            else:
                timestamp = datetime.now()
            
            return Candle(
                symbol=symbol,
                open=float(data.get("o") or data.get("open", 0.0)),
                high=float(data.get("h") or data.get("high", 0.0)),
                low=float(data.get("l") or data.get("low", 0.0)),
                close=float(data.get("c") or data.get("close", 0.0)),
                volume=float(data.get("v") or data.get("volume", 0.0)),
                timestamp=timestamp,
                interval=interval
            )
        except Exception as e:
            logger.error(f"Error normalizing candle: {e}")
            return None
    
    @staticmethod
    def normalize_bars(bars: List[Dict[str, Any]], symbol: str, interval: str = "15m") -> List[Candle]:
        """Convert list of bar data to Candles"""
        candles = []
        for bar in bars:
            candle = MarketDataAdapter.normalize_candle(bar, symbol, interval)
            if candle:
                candles.append(candle)
        return candles

