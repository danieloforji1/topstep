"""
MGC Gold Futures Contract Specifications
Defines contract specs for realistic futures trading calculations
"""
from dataclasses import dataclass


@dataclass
class MGCContractSpecs:
    """MGC Micro Gold Futures Contract Specifications"""
    
    # Contract identification
    symbol: str = "MGC"
    name: str = "Micro Gold Futures"
    exchange: str = "COMEX"
    
    # Price specifications
    tick_size: float = 0.10  # Minimum price movement (0.10)
    tick_value: float = 1.00  # Dollar value per tick ($1.00)
    
    # Contract multiplier
    # For MGC: 1 full dollar move = 10 ticks = $10 per contract
    # Example: Price moves from 2000.00 to 2001.00 = 10 ticks = $10 per contract
    dollar_move_value: float = 10.00  # $10 per $1.00 price move per contract
    
    def price_to_ticks(self, price_move: float) -> float:
        """
        Convert price movement to number of ticks
        
        Args:
            price_move: Price difference (e.g., 1.50)
            
        Returns:
            Number of ticks (e.g., 15.0 for 1.50 move)
        """
        return price_move / self.tick_size
    
    def ticks_to_price(self, ticks: float) -> float:
        """
        Convert number of ticks to price movement
        
        Args:
            ticks: Number of ticks (e.g., 15.0)
            
        Returns:
            Price movement (e.g., 1.50)
        """
        return ticks * self.tick_size
    
    def calculate_pnl(self, entry_price: float, exit_price: float, contracts: int, is_long: bool) -> float:
        """
        Calculate PnL for a futures position
        
        Args:
            entry_price: Entry price
            exit_price: Exit price
            contracts: Number of contracts
            is_long: True for long position, False for short
            
        Returns:
            PnL in dollars
        """
        if is_long:
            price_move = exit_price - entry_price
        else:
            price_move = entry_price - exit_price
        
        # Convert price move to ticks
        ticks = self.price_to_ticks(price_move)
        
        # Calculate PnL: ticks × tick_value × contracts
        pnl = ticks * self.tick_value * contracts
        
        return pnl
    
    def calculate_risk_amount(self, entry_price: float, stop_price: float, contracts: int, is_long: bool) -> float:
        """
        Calculate dollar risk amount for a position
        
        Args:
            entry_price: Entry price
            stop_price: Stop loss price
            contracts: Number of contracts
            is_long: True for long position, False for short
            
        Returns:
            Risk amount in dollars
        """
        return abs(self.calculate_pnl(entry_price, stop_price, contracts, is_long))
    
    def round_to_tick(self, price: float) -> float:
        """
        Round price to nearest tick
        
        Args:
            price: Price to round
            
        Returns:
            Price rounded to nearest tick
        """
        return round(price / self.tick_size) * self.tick_size


# Global instance for easy access
MGC_SPECS = MGCContractSpecs()

