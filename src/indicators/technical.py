"""
Technical Indicators
ATR, volatility, correlation calculations
"""
import numpy as np
import pandas as pd
from typing import List, Optional
import logging

from connectors.market_data_adapter import Candle

logger = logging.getLogger(__name__)


def calculate_atr(candles: List[Candle], window: int = 14) -> Optional[float]:
    """
    Calculate Average True Range (ATR)
    
    ATR measures volatility by averaging the true range over a period.
    True Range = max(high - low, abs(high - prev_close), abs(low - prev_close))
    """
    if len(candles) < window + 1:
        return None
    
    true_ranges = []
    
    for i in range(1, len(candles)):
        high = candles[i].high
        low = candles[i].low
        prev_close = candles[i-1].close
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = max(tr1, tr2, tr3)
        true_ranges.append(true_range)
    
    if len(true_ranges) < window:
        return None
    
    # Calculate ATR as simple moving average of true ranges
    atr_values = true_ranges[-window:]
    atr = sum(atr_values) / len(atr_values)
    
    return atr


def calculate_volatility(candles: List[Candle], window: int = 100) -> Optional[float]:
    """
    Calculate rolling volatility (standard deviation of returns)
    """
    if len(candles) < window + 1:
        return None
    
    # Calculate returns
    closes = [c.close for c in candles[-window-1:]]
    returns = []
    
    for i in range(1, len(closes)):
        ret = (closes[i] - closes[i-1]) / closes[i-1]
        returns.append(ret)
    
    if len(returns) < window:
        return None
    
    # Calculate standard deviation of returns
    volatility = np.std(returns)
    
    return volatility


def calculate_correlation(
    primary_candles: List[Candle],
    hedge_candles: List[Candle],
    window: int = 100
) -> Optional[float]:
    """
    Calculate rolling correlation between two instruments
    """
    if len(primary_candles) < window or len(hedge_candles) < window:
        return None
    
    # Get closing prices
    primary_closes = [c.close for c in primary_candles[-window:]]
    hedge_closes = [c.close for c in hedge_candles[-window:]]
    
    if len(primary_closes) != len(hedge_closes):
        return None
    
    # Calculate returns
    primary_returns = []
    hedge_returns = []
    
    for i in range(1, len(primary_closes)):
        p_ret = (primary_closes[i] - primary_closes[i-1]) / primary_closes[i-1]
        h_ret = (hedge_closes[i] - hedge_closes[i-1]) / hedge_closes[i-1]
        primary_returns.append(p_ret)
        hedge_returns.append(h_ret)
    
    if len(primary_returns) < 2:
        return None
    
    # Calculate correlation
    correlation = np.corrcoef(primary_returns, hedge_returns)[0, 1]
    
    return correlation if not np.isnan(correlation) else None


def calculate_trend_strength(
    candles: List[Candle],
    window: int = 20,
    atr_window: int = 14
) -> Optional[float]:
    """
    Calculate trend strength in ATR units over a window.

    Positive values indicate uptrend, negative values indicate downtrend.
    """
    if len(candles) < max(window, atr_window) + 1:
        return None
    
    recent = candles[-window:]
    if len(recent) < window:
        return None
    
    atr = calculate_atr(candles[-(atr_window + 1):], atr_window)
    if atr is None or atr == 0:
        return None
    
    start_price = recent[0].close
    end_price = recent[-1].close
    return (end_price - start_price) / atr


def calculate_atr_from_dataframe(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Calculate ATR from pandas DataFrame (more efficient for large datasets)
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate ATR as rolling mean
    atr = tr.rolling(window=window).mean()
    
    return atr


def round_to_tick(price: float, tick_size: float) -> float:
    """Round price to nearest tick"""
    return round(price / tick_size) * tick_size

