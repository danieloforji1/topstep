"""
Lot Sizer
Volatility-based dynamic lot sizing with risk parity
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class Sizer:
    """Volatility-adjusted lot sizer"""
    
    def __init__(
        self,
        R_per_trade: float = 150.0,  # Risk per base lot in dollars
        tick_value: float = 5.0,  # Dollar value per tick for MES
        min_lot: int = 1,
        max_lot: int = 10
    ):
        self.R_per_trade = R_per_trade
        self.tick_value = tick_value
        self.min_lot = min_lot
        self.max_lot = max_lot
        self.current_volatility: Optional[float] = None
    
    def update_volatility(self, atr_points: Optional[float]):
        """Update current volatility estimate"""
        self.current_volatility = atr_points
    
    def compute_base_lot(self, atr_points: Optional[float] = None) -> int:
        """
        Calculate base lot size based on volatility
        
        Formula: base_lot = R / (ATR_points * tick_value)
        """
        atr = atr_points or self.current_volatility
        
        if atr is None or atr <= 0:
            logger.warning("ATR not available, using minimum lot size")
            return self.min_lot
        
        # Calculate dollar volatility per contract
        dollar_vol_per_contract = atr * self.tick_value
        
        if dollar_vol_per_contract <= 0:
            return self.min_lot
        
        # Calculate lot size
        lot = self.R_per_trade / dollar_vol_per_contract
        
        # Round to integer and clamp
        lot_int = max(self.min_lot, min(self.max_lot, int(round(lot))))
        
        logger.debug(f"Computed base lot: {lot_int} (ATR={atr:.2f}, R=${self.R_per_trade})")
        return lot_int
    
    def compute_pyramiding_lot(self, base_lot: int, layer: int, alpha: float = 0.25) -> int:
        """
        Calculate lot size for pyramiding layers
        
        Formula: lot_n = base_lot * (1 + α * n)
        """
        if layer <= 0:
            return base_lot
        
        lot = base_lot * (1 + alpha * layer)
        lot_int = max(self.min_lot, min(self.max_lot, int(round(lot))))
        
        return lot_int

