"""
Multi-Timeframe Analyzer
Detects support/resistance levels from higher timeframes and aligns grid
"""
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class SupportResistanceLevel:
    """Represents a support or resistance level"""
    price: float
    strength: float  # 0.0 to 1.0 (how strong the level is)
    level_type: str  # 'support' or 'resistance'
    timeframe: str  # '1h', '4h', etc.
    touches: int  # Number of times price touched this level


class MultiTimeframeAnalyzer:
    """Analyzes multiple timeframes to find support/resistance levels"""
    
    def __init__(
        self,
        primary_timeframe: str = "15m",
        higher_timeframes: List[str] = None,
        lookback_periods: Dict[str, int] = None
    ):
        self.primary_timeframe = primary_timeframe
        self.higher_timeframes = higher_timeframes or ["1h", "4h"]
        self.lookback_periods = lookback_periods or {
            "1h": 50,  # 50 hours = ~2 days
            "4h": 30   # 30 * 4h = 5 days
        }
        
        self.support_levels: List[SupportResistanceLevel] = []
        self.resistance_levels: List[SupportResistanceLevel] = []
    
    def analyze_timeframe(
        self,
        candles: List[Dict],
        timeframe: str
    ) -> Tuple[List[SupportResistanceLevel], List[SupportResistanceLevel]]:
        """
        Analyze a timeframe to find support and resistance levels
        
        Args:
            candles: List of candle dicts with 'high', 'low', 'close', 'open'
            timeframe: Timeframe name (e.g., '1h', '4h')
        
        Returns:
            Tuple of (support_levels, resistance_levels)
        """
        if not candles or len(candles) < 20:
            return [], []
        
        # Extract price data
        highs = [c.get('high') or c.get('close', 0) for c in candles]
        lows = [c.get('low') or c.get('close', 0) for c in candles]
        closes = [c.get('close', 0) for c in candles]
        
        if not highs or not lows:
            return [], []
        
        # Find pivot points (local highs and lows)
        pivot_highs = self._find_pivot_highs(highs, window=5)
        pivot_lows = self._find_pivot_lows(lows, window=5)
        
        # Cluster pivot points to find significant levels
        support_levels = self._cluster_levels(pivot_lows, lows, "support", timeframe)
        resistance_levels = self._cluster_levels(pivot_highs, highs, "resistance", timeframe)
        
        return support_levels, resistance_levels
    
    def _find_pivot_highs(self, highs: List[float], window: int = 5) -> List[Tuple[int, float]]:
        """Find local pivot highs"""
        pivots = []
        for i in range(window, len(highs) - window):
            is_pivot = True
            # Check if current high is higher than surrounding highs
            for j in range(i - window, i + window + 1):
                if j != i and highs[j] >= highs[i]:
                    is_pivot = False
                    break
            if is_pivot:
                pivots.append((i, highs[i]))
        return pivots
    
    def _find_pivot_lows(self, lows: List[float], window: int = 5) -> List[Tuple[int, float]]:
        """Find local pivot lows"""
        pivots = []
        for i in range(window, len(lows) - window):
            is_pivot = True
            # Check if current low is lower than surrounding lows
            for j in range(i - window, i + window + 1):
                if j != i and lows[j] <= lows[i]:
                    is_pivot = False
                    break
            if is_pivot:
                pivots.append((i, lows[i]))
        return pivots
    
    def _cluster_levels(
        self,
        pivots: List[Tuple[int, float]],
        prices: List[float],
        level_type: str,
        timeframe: str
    ) -> List[SupportResistanceLevel]:
        """Cluster pivot points into significant support/resistance levels"""
        if not pivots:
            return []
        
        # Group pivots that are close together (within 0.5% of price)
        clusters = []
        pivot_prices = [p[1] for p in pivots]
        
        for pivot_idx, pivot_price in pivots:
            # Find existing cluster within threshold
            threshold = pivot_price * 0.005  # 0.5%
            cluster_found = False
            
            for cluster in clusters:
                cluster_center = sum(cluster['prices']) / len(cluster['prices'])
                if abs(pivot_price - cluster_center) < threshold:
                    cluster['prices'].append(pivot_price)
                    cluster['indices'].append(pivot_idx)
                    cluster_found = True
                    break
            
            if not cluster_found:
                clusters.append({
                    'prices': [pivot_price],
                    'indices': [pivot_idx]
                })
        
        # Convert clusters to SupportResistanceLevel objects
        levels = []
        for cluster in clusters:
            cluster_prices = cluster['prices']
            cluster_center = sum(cluster_prices) / len(cluster_prices)
            
            # Calculate strength based on:
            # 1. Number of touches (more = stronger)
            # 2. How recent the touches are
            touches = len(cluster_prices)
            
            # Count how many times price came close to this level
            touches_count = 0
            for price in prices:
                if abs(price - cluster_center) < (cluster_center * 0.01):  # Within 1%
                    touches_count += 1
            
            # Strength: 0.3 base + 0.4 for touches + 0.3 for recency
            base_strength = 0.3
            touch_strength = min(0.4, touches / 10.0)  # Max 0.4 for 10+ touches
            recency_strength = 0.3 if cluster['indices'][-1] >= len(prices) - 20 else 0.1
            
            strength = base_strength + touch_strength + recency_strength
            strength = min(1.0, strength)
            
            level = SupportResistanceLevel(
                price=cluster_center,
                strength=strength,
                level_type=level_type,
                timeframe=timeframe,
                touches=touches_count
            )
            levels.append(level)
        
        # Sort by strength (descending)
        levels.sort(key=lambda x: x.strength, reverse=True)
        
        return levels
    
    def update_levels(
        self,
        timeframe_data: Dict[str, List[Dict]]
    ):
        """Update support/resistance levels from multiple timeframes"""
        all_support = []
        all_resistance = []
        
        for timeframe in self.higher_timeframes:
            if timeframe in timeframe_data:
                candles = timeframe_data[timeframe]
                support, resistance = self.analyze_timeframe(candles, timeframe)
                all_support.extend(support)
                all_resistance.extend(resistance)
        
        # Merge and deduplicate levels
        self.support_levels = self._merge_levels(all_support)
        self.resistance_levels = self._merge_levels(all_resistance)
        
        logger.info(
            f"Updated MTF levels: {len(self.support_levels)} support, "
            f"{len(self.resistance_levels)} resistance"
        )
    
    def _merge_levels(self, levels: List[SupportResistanceLevel]) -> List[SupportResistanceLevel]:
        """Merge nearby levels and keep strongest"""
        if not levels:
            return []
        
        # Sort by price
        levels.sort(key=lambda x: x.price)
        
        merged = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            # Check if within 0.5% of current cluster
            cluster_center = sum(l.price for l in current_cluster) / len(current_cluster)
            threshold = cluster_center * 0.005
            
            if abs(level.price - cluster_center) < threshold:
                current_cluster.append(level)
            else:
                # Finalize current cluster (keep strongest)
                strongest = max(current_cluster, key=lambda x: x.strength)
                merged.append(strongest)
                current_cluster = [level]
        
        # Add final cluster
        if current_cluster:
            strongest = max(current_cluster, key=lambda x: x.strength)
            merged.append(strongest)
        
        # Sort by strength and return top 10
        merged.sort(key=lambda x: x.strength, reverse=True)
        return merged[:10]
    
    def get_nearest_levels(
        self,
        price: float,
        max_distance: float = None
    ) -> Tuple[Optional[SupportResistanceLevel], Optional[SupportResistanceLevel]]:
        """
        Get nearest support and resistance levels to current price
        
        Args:
            price: Current price
            max_distance: Maximum distance to consider (as fraction of price, e.g., 0.02 = 2%)
        
        Returns:
            Tuple of (nearest_support, nearest_resistance)
        """
        if max_distance is None:
            max_distance = price * 0.05  # 5% default
        
        nearest_support = None
        nearest_resistance = None
        min_support_dist = float('inf')
        min_resistance_dist = float('inf')
        
        # Find nearest support (below price)
        for level in self.support_levels:
            if level.price < price:
                dist = price - level.price
                if dist < min_support_dist and dist <= max_distance:
                    min_support_dist = dist
                    nearest_support = level
        
        # Find nearest resistance (above price)
        for level in self.resistance_levels:
            if level.price > price:
                dist = level.price - price
                if dist < min_resistance_dist and dist <= max_distance:
                    min_resistance_dist = dist
                    nearest_resistance = level
        
        return nearest_support, nearest_resistance
    
    def should_align_grid_to_level(
        self,
        grid_mid: float,
        threshold: float = 0.002  # 0.2% of price
    ) -> Optional[float]:
        """
        Check if grid should be aligned to a nearby support/resistance level
        
        Args:
            grid_mid: Current grid midpoint
            threshold: Maximum distance to consider alignment (as fraction of price)
        
        Returns:
            Adjusted grid midpoint if alignment recommended, None otherwise
        """
        support, resistance = self.get_nearest_levels(grid_mid, grid_mid * threshold)
        
        # Prefer stronger levels
        candidates = []
        if support and support.strength > 0.5:
            candidates.append((support.price, support.strength, "support"))
        if resistance and resistance.strength > 0.5:
            candidates.append((resistance.price, resistance.strength, "resistance"))
        
        if not candidates:
            return None
        
        # Use strongest level
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_level_price, best_strength, level_type = candidates[0]
        
        # Only align if level is significantly stronger and close enough
        if best_strength > 0.6 and abs(best_level_price - grid_mid) < (grid_mid * threshold):
            logger.info(
                f"Aligning grid to {level_type} level: {best_level_price:.2f} "
                f"(strength: {best_strength:.2f}, current mid: {grid_mid:.2f})"
            )
            return best_level_price
        
        return None
    
    def get_levels_summary(self) -> Dict:
        """Get summary of current support/resistance levels"""
        return {
            "support_levels": [
                {
                    "price": l.price,
                    "strength": l.strength,
                    "timeframe": l.timeframe,
                    "touches": l.touches
                }
                for l in self.support_levels[:5]
            ],
            "resistance_levels": [
                {
                    "price": l.price,
                    "strength": l.strength,
                    "timeframe": l.timeframe,
                    "touches": l.touches
                }
                for l in self.resistance_levels[:5]
            ]
        }

