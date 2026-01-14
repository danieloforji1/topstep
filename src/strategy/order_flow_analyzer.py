"""
Order Flow Analyzer
Analyzes bid/ask imbalance, volume profile, and order flow metrics
"""
import logging
from typing import Dict, Optional, List, Tuple
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class OrderFlowSnapshot:
    """Snapshot of order flow metrics at a point in time"""
    timestamp: datetime
    price: float
    bid_size: float
    ask_size: float
    bid_ask_imbalance: float  # -1 to +1 (negative = more asks, positive = more bids)
    volume: float
    buy_volume: float
    sell_volume: float
    volume_imbalance: float  # -1 to +1 (negative = more sells, positive = more buys)
    spread: float
    mid_price: float


class OrderFlowAnalyzer:
    """Analyzes order flow to inform grid placement"""
    
    def __init__(
        self,
        lookback_period: int = 20,  # Number of snapshots to keep
        imbalance_threshold: float = 0.3  # Threshold for significant imbalance
    ):
        self.lookback_period = lookback_period
        self.imbalance_threshold = imbalance_threshold
        self.snapshots: deque = deque(maxlen=lookback_period)
        self.volume_profile: Dict[float, float] = {}  # price -> volume
        
    def add_snapshot(
        self,
        price: float,
        bid_size: Optional[float] = None,
        ask_size: Optional[float] = None,
        volume: Optional[float] = None,
        buy_volume: Optional[float] = None,
        sell_volume: Optional[float] = None,
        spread: Optional[float] = None
    ):
        """Add a new order flow snapshot"""
        # Calculate bid/ask imbalance
        if bid_size is not None and ask_size is not None:
            total_size = bid_size + ask_size
            if total_size > 0:
                bid_ask_imbalance = (bid_size - ask_size) / total_size
            else:
                bid_ask_imbalance = 0.0
        else:
            bid_ask_imbalance = 0.0
        
        # Calculate volume imbalance
        if buy_volume is not None and sell_volume is not None:
            total_volume = buy_volume + sell_volume
            if total_volume > 0:
                volume_imbalance = (buy_volume - sell_volume) / total_volume
            else:
                volume_imbalance = 0.0
        else:
            volume_imbalance = 0.0
        
        # Calculate mid price
        if bid_size is not None and ask_size is not None and spread is not None:
            mid_price = price - (spread / 2) if spread else price
        else:
            mid_price = price
        
        snapshot = OrderFlowSnapshot(
            timestamp=datetime.now(),
            price=price,
            bid_size=bid_size or 0.0,
            ask_size=ask_size or 0.0,
            bid_ask_imbalance=bid_ask_imbalance,
            volume=volume or 0.0,
            buy_volume=buy_volume or 0.0,
            sell_volume=sell_volume or 0.0,
            volume_imbalance=volume_imbalance,
            spread=spread or 0.0,
            mid_price=mid_price
        )
        
        self.snapshots.append(snapshot)
        
        # Update volume profile (round price to nearest 0.25 for MES)
        rounded_price = round(price / 0.25) * 0.25
        if rounded_price not in self.volume_profile:
            self.volume_profile[rounded_price] = 0.0
        self.volume_profile[rounded_price] += (volume or 0.0)
    
    def get_bid_ask_imbalance(self) -> float:
        """Get current bid/ask imbalance (-1 to +1)"""
        if not self.snapshots:
            return 0.0
        
        # Average recent imbalances
        recent_imbalances = [s.bid_ask_imbalance for s in list(self.snapshots)[-5:]]
        return sum(recent_imbalances) / len(recent_imbalances) if recent_imbalances else 0.0
    
    def get_volume_imbalance(self) -> float:
        """Get current volume imbalance (-1 to +1)"""
        if not self.snapshots:
            return 0.0
        
        # Average recent imbalances
        recent_imbalances = [s.volume_imbalance for s in list(self.snapshots)[-5:]]
        return sum(recent_imbalances) / len(recent_imbalances) if recent_imbalances else 0.0
    
    def get_combined_imbalance(self) -> float:
        """Get combined order flow imbalance (weighted average)"""
        bid_ask_imb = self.get_bid_ask_imbalance()
        volume_imb = self.get_volume_imbalance()
        
        # Weight: 60% bid/ask, 40% volume (bid/ask is more immediate)
        combined = (0.6 * bid_ask_imb) + (0.4 * volume_imb)
        return max(-1.0, min(1.0, combined))
    
    def get_grid_mid_adjustment(self, current_price: float) -> float:
        """
        Calculate adjustment to grid midpoint based on order flow
        
        Returns:
            Adjustment in price points (positive = shift up, negative = shift down)
        """
        imbalance = self.get_combined_imbalance()
        
        # If strong buying pressure (positive imbalance), shift grid up slightly
        # If strong selling pressure (negative imbalance), shift grid down slightly
        # Adjustment is proportional to imbalance strength and current price
        
        # Use 0.1% of price as max adjustment (very conservative)
        max_adjustment = current_price * 0.001
        
        adjustment = imbalance * max_adjustment
        
        logger.debug(f"Order flow adjustment: {adjustment:.2f} (imbalance: {imbalance:.3f})")
        return adjustment
    
    def get_volume_profile_levels(self, price_range: Tuple[float, float], num_levels: int = 5) -> List[float]:
        """
        Get price levels with highest volume (support/resistance from volume profile)
        
        Args:
            price_range: (min_price, max_price) to search
            num_levels: Number of levels to return
        
        Returns:
            List of price levels sorted by volume (highest first)
        """
        min_price, max_price = price_range
        
        # Filter volume profile to price range
        relevant_levels = {
            price: volume 
            for price, volume in self.volume_profile.items()
            if min_price <= price <= max_price
        }
        
        if not relevant_levels:
            return []
        
        # Sort by volume (descending)
        sorted_levels = sorted(relevant_levels.items(), key=lambda x: x[1], reverse=True)
        
        # Return top N levels
        return [price for price, _ in sorted_levels[:num_levels]]
    
    def should_avoid_price_level(self, price: float, threshold: float = 0.1) -> bool:
        """
        Check if a price level should be avoided (low volume zone)
        
        Args:
            price: Price to check
            threshold: Minimum volume threshold (as fraction of max volume)
        
        Returns:
            True if level should be avoided (low volume)
        """
        rounded_price = round(price / 0.25) * 0.25
        
        if not self.volume_profile:
            return False  # No data, don't avoid
        
        max_volume = max(self.volume_profile.values()) if self.volume_profile else 0.0
        
        if max_volume == 0:
            return False
        
        level_volume = self.volume_profile.get(rounded_price, 0.0)
        volume_ratio = level_volume / max_volume
        
        # Avoid if volume is below threshold
        return volume_ratio < threshold
    
    def get_order_flow_direction(self) -> str:
        """Get current order flow direction"""
        imbalance = self.get_combined_imbalance()
        
        if imbalance > self.imbalance_threshold:
            return "STRONG_BUY"
        elif imbalance < -self.imbalance_threshold:
            return "STRONG_SELL"
        elif imbalance > 0.1:
            return "WEAK_BUY"
        elif imbalance < -0.1:
            return "WEAK_SELL"
        else:
            return "NEUTRAL"
    
    def get_snapshot_summary(self) -> Dict:
        """Get summary of current order flow state"""
        if not self.snapshots:
            return {
                "bid_ask_imbalance": 0.0,
                "volume_imbalance": 0.0,
                "combined_imbalance": 0.0,
                "direction": "NEUTRAL",
                "snapshots_count": 0
            }
        
        return {
            "bid_ask_imbalance": self.get_bid_ask_imbalance(),
            "volume_imbalance": self.get_volume_imbalance(),
            "combined_imbalance": self.get_combined_imbalance(),
            "direction": self.get_order_flow_direction(),
            "snapshots_count": len(self.snapshots),
            "volume_profile_levels": len(self.volume_profile)
        }

