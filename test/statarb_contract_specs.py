"""
Contract Specifications for Statistical Arbitrage Strategy
Supports both MGC (Micro Gold) and GC (Full Gold) futures
"""
from dataclasses import dataclass


@dataclass
class ContractSpecs:
    """Base contract specifications for futures trading"""
    
    symbol: str
    name: str
    exchange: str
    tick_size: float
    tick_value: float
    dollar_move_value: float
    
    def price_to_ticks(self, price_move: float) -> float:
        """Convert price movement to number of ticks"""
        return price_move / self.tick_size
    
    def ticks_to_price(self, ticks: float) -> float:
        """Convert number of ticks to price movement"""
        return ticks * self.tick_size
    
    def calculate_pnl(self, entry_price: float, exit_price: float, contracts: int, is_long: bool) -> float:
        """Calculate PnL for a futures position"""
        if is_long:
            price_move = exit_price - entry_price
        else:
            price_move = entry_price - exit_price
        
        ticks = self.price_to_ticks(price_move)
        pnl = ticks * self.tick_value * contracts
        return pnl
    
    def calculate_risk_amount(self, entry_price: float, stop_price: float, contracts: int, is_long: bool) -> float:
        """Calculate dollar risk amount for a position"""
        return abs(self.calculate_pnl(entry_price, stop_price, contracts, is_long))
    
    def round_to_tick(self, price: float) -> float:
        """Round price to nearest tick"""
        return round(price / self.tick_size) * self.tick_size


# MGC (Micro Gold) Contract Specs
MGC_SPECS = ContractSpecs(
    symbol="MGC",
    name="Micro Gold Futures",
    exchange="COMEX",
    tick_size=0.10,
    tick_value=1.00,  # $1.00 per tick
    dollar_move_value=10.00  # $10 per $1.00 price move
)

# GC (Full Gold) Contract Specs
GC_SPECS = ContractSpecs(
    symbol="GC",
    name="Gold Futures",
    exchange="COMEX",
    tick_size=0.10,
    tick_value=10.00,  # $10.00 per tick
    dollar_move_value=100.00  # $100 per $1.00 price move
)

