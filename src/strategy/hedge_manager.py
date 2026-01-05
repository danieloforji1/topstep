"""
Hedge Manager
Computes hedge ratios, places hedge orders, maintains hedge lifecycle
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class HedgeManager:
    """Manages cross-asset hedging"""
    
    def __init__(
        self,
        primary_symbol: str,
        hedge_symbol: str,
        min_hedge_ratio: float = 0.5,
        max_hedge_ratio: float = 1.25,
        correlation_threshold: float = 0.6,
        max_hedge_contracts_multiplier: float = 1.5
    ):
        self.primary_symbol = primary_symbol
        self.hedge_symbol = hedge_symbol
        self.min_hedge_ratio = min_hedge_ratio
        self.max_hedge_ratio = max_hedge_ratio
        self.correlation_threshold = correlation_threshold
        self.max_hedge_contracts_multiplier = max_hedge_contracts_multiplier
        
        self.current_correlation: Optional[float] = None
        self.primary_volatility: Optional[float] = None
        self.hedge_volatility: Optional[float] = None
        self.hedge_ratio: Optional[float] = None
    
    def update_correlation(self, correlation: Optional[float]):
        """Update current correlation"""
        self.current_correlation = correlation
        if correlation is not None:
            logger.debug(f"Updated correlation: {correlation:.3f}")
    
    def update_volatilities(
        self,
        primary_vol: Optional[float],
        hedge_vol: Optional[float]
    ):
        """Update volatility estimates"""
        self.primary_volatility = primary_vol
        self.hedge_volatility = hedge_vol
        self._recalculate_hedge_ratio()
    
    def _recalculate_hedge_ratio(self):
        """Recalculate hedge ratio based on current correlation and volatilities"""
        if (self.current_correlation is None or
            self.primary_volatility is None or
            self.hedge_volatility is None or
            self.hedge_volatility <= 0):
            self.hedge_ratio = None
            return
        
        # Hedge ratio formula: h = ρ * (σ_p / σ_h)
        h = self.current_correlation * (self.primary_volatility / self.hedge_volatility)
        
        # Clamp to min/max
        self.hedge_ratio = max(self.min_hedge_ratio, min(self.max_hedge_ratio, h))
        
        logger.debug(f"Hedge ratio: {self.hedge_ratio:.3f} (corr={self.current_correlation:.3f}, "
                    f"vol_p={self.primary_volatility:.4f}, vol_h={self.hedge_volatility:.4f})")
    
    def compute_hedge_ratio(
        self,
        correlation: Optional[float] = None,
        primary_vol: Optional[float] = None,
        hedge_vol: Optional[float] = None
    ) -> Optional[float]:
        """
        Compute hedge ratio
        
        Args:
            correlation: Rolling correlation (if None, uses current)
            primary_vol: Primary instrument volatility (if None, uses current)
            hedge_vol: Hedge instrument volatility (if None, uses current)
        
        Returns:
            Hedge ratio (None if insufficient data)
        """
        if correlation is not None:
            self.update_correlation(correlation)
        if primary_vol is not None:
            self.primary_volatility = primary_vol
        if hedge_vol is not None:
            self.hedge_volatility = hedge_vol
        
        self._recalculate_hedge_ratio()
        return self.hedge_ratio
    
    def compute_hedge_size(
        self,
        primary_position: int,
        hedge_ratio: Optional[float] = None,
        primary_price: Optional[float] = None,
        hedge_price: Optional[float] = None,
        primary_tick_value: float = 5.0,
        hedge_tick_value: float = 2.0,
        primary_volatility: Optional[float] = None,
        hedge_volatility: Optional[float] = None,
        correlation: Optional[float] = None
    ) -> int:
        """
        Compute hedge position size based on RISK exposure, accounting for volatility differences.
        
        This is critical because MES and MNQ have different:
        - Prices (MES ~$6,800, MNQ ~$25,000)
        - Tick values (MES $5/tick, MNQ $2/tick)
        - Volatilities (MNQ typically 2-3x more volatile than MES)
        
        The calculation matches RISK, not dollar exposure, to properly hedge drawdowns.
        
        Args:
            primary_position: Net position in primary instrument (positive=long, negative=short)
            hedge_ratio: Hedge ratio (if None, uses current) - DEPRECATED, kept for compatibility
            primary_price: Current price of primary instrument
            hedge_price: Current price of hedge instrument
            primary_tick_value: Dollar value per tick for primary (MES = $5)
            hedge_tick_value: Dollar value per tick for hedge (MNQ = $2)
            primary_volatility: Primary instrument volatility (if None, uses stored value)
            hedge_volatility: Hedge instrument volatility (if None, uses stored value)
            correlation: Correlation between instruments (if None, uses stored value)
        
        Returns:
            Hedge position size (positive=long, negative=short, opposite of primary)
        """
        # Use stored values if not provided
        if correlation is None:
            correlation = self.current_correlation
        if primary_volatility is None:
            primary_volatility = self.primary_volatility
        if hedge_volatility is None:
            hedge_volatility = self.hedge_volatility
        
        # Validate required data
        if primary_position == 0:
            return 0
        
        if primary_price is None or hedge_price is None:
            logger.error("Cannot compute hedge size: missing price data")
            return 0
        
        if primary_volatility is None or hedge_volatility is None or primary_volatility <= 0 or hedge_volatility <= 0:
            logger.warning("Cannot compute hedge size: missing or invalid volatility data. Falling back to dollar-based calculation.")
            # Fallback to old method if volatility not available
            return self._compute_hedge_size_dollar_based(
                primary_position, primary_price, hedge_price, 
                primary_tick_value, hedge_tick_value, hedge_ratio
            )
        
        if correlation is None or correlation <= 0:
            logger.warning("Cannot compute hedge size: missing correlation. Using hedge ratio fallback.")
            if hedge_ratio is None:
                hedge_ratio = self.hedge_ratio
            if hedge_ratio is None:
                logger.warning("Cannot compute hedge size: no hedge ratio or correlation available")
                return 0
            # Fallback to old method
            return self._compute_hedge_size_dollar_based(
                primary_position, primary_price, hedge_price,
                primary_tick_value, hedge_tick_value, hedge_ratio
            )
        
        # RISK-BASED CALCULATION
        
        # Step 1: Calculate risk per contract for each instrument
        # Risk per contract = price × tick_value × volatility
        primary_risk_per_contract = primary_price * primary_tick_value * primary_volatility
        hedge_risk_per_contract = hedge_price * hedge_tick_value * hedge_volatility
        
        # Step 2: Calculate total primary risk exposure
        primary_risk_exposure = abs(primary_position) * primary_risk_per_contract
        
        # Step 3: Calculate target hedge risk (with correlation)
        # We match risk exposure, adjusted by correlation
        target_hedge_risk = primary_risk_exposure * correlation
        
        # Step 4: Convert to hedge contracts
        # hedge_contracts = target_hedge_risk / hedge_risk_per_contract
        hedge_contracts = target_hedge_risk / hedge_risk_per_contract
        
        # Round to integer
        hedge_contracts_int = int(round(hedge_contracts))
        
        # Hedge should be opposite direction of primary
        # If primary is long (positive), hedge should be short (negative)
        hedge_size = -hedge_contracts_int if primary_position > 0 else hedge_contracts_int
        
        logger.info(
            f"Computed hedge (RISK-BASED, BEFORE caps): primary={primary_position} contracts @ ${primary_price:.2f} "
            f"(risk_per_contract=${primary_risk_per_contract:,.2f}, total_risk=${primary_risk_exposure:,.2f}), "
            f"correlation={correlation:.3f}, hedge_vol={hedge_volatility:.4f}, "
            f"hedge_risk_per_contract=${hedge_risk_per_contract:,.2f}, target_hedge_risk=${target_hedge_risk:,.2f}, "
            f"calculated_hedge_contracts={hedge_contracts:.2f}, hedge={hedge_size} contracts @ ${hedge_price:.2f}"
        )
        
        # CRITICAL SAFETY CHECKS (applied in order of strictness):
        
        # 1. NEVER exceed configurable multiplier (most restrictive)
        max_hedge_contracts = abs(primary_position) * self.max_hedge_contracts_multiplier
        if abs(hedge_size) > max_hedge_contracts:
            logger.error(
                f"🚨 CRITICAL: Calculated hedge size ({hedge_size}) exceeds config limit ({max_hedge_contracts}). "
                f"CAPPING to {max_hedge_contracts} contracts (primary={primary_position}, multiplier={self.max_hedge_contracts_multiplier})."
            )
            hedge_size = max_hedge_contracts if hedge_size > 0 else -max_hedge_contracts
        
        # 2. NEVER hedge more contracts than primary position (hard limit)
        if abs(hedge_size) > abs(primary_position):
            logger.error(
                f"🚨 CRITICAL: Hedge size ({hedge_size}) exceeds primary position ({primary_position}). "
                f"CAPPING to primary position size for safety!"
            )
            hedge_size = -abs(primary_position) if primary_position > 0 else abs(primary_position)
        
        # 3. Warn if hedge size is more than 0.6x primary (should typically be 0.4-0.5x for MES->MNQ)
        if abs(hedge_size) > abs(primary_position) * 0.6:
            logger.warning(
                f"⚠️ Hedge size ({hedge_size}) is more than 60% of primary position ({primary_position}). "
                f"Expected ratio: ~0.4-0.5x for MES->MNQ. Current ratio: {abs(hedge_size)/abs(primary_position):.2f}x"
            )
        
        logger.info(
            f"Final hedge size (AFTER caps): {hedge_size} contracts "
            f"(primary={primary_position}, ratio={abs(hedge_size)/abs(primary_position) if primary_position != 0 else 0:.2f}x)"
        )
        
        return hedge_size
    
    def _compute_hedge_size_dollar_based(
        self,
        primary_position: int,
        primary_price: float,
        hedge_price: float,
        primary_tick_value: float,
        hedge_tick_value: float,
        hedge_ratio: Optional[float]
    ) -> int:
        """
        Fallback method: Compute hedge size based on dollar exposure (old method).
        Used when volatility data is not available.
        """
        if hedge_ratio is None:
            hedge_ratio = self.hedge_ratio
        
        if hedge_ratio is None:
            logger.warning("Cannot compute hedge size: hedge ratio not available")
            return 0
        
        # Calculate primary dollar exposure
        primary_contract_multiplier = primary_tick_value  # MES = 5
        hedge_contract_multiplier = hedge_tick_value      # MNQ = 2
        
        primary_exposure = abs(primary_position) * primary_price * primary_contract_multiplier
        
        # Calculate target hedge dollar exposure
        target_hedge_exposure = primary_exposure * hedge_ratio
        
        # Convert hedge dollar exposure back to contracts
        hedge_contracts = target_hedge_exposure / (hedge_price * hedge_contract_multiplier)
        
        # Round to integer
        hedge_contracts_int = int(round(hedge_contracts))
        
        # Hedge should be opposite direction of primary
        hedge_size = -hedge_contracts_int if primary_position > 0 else hedge_contracts_int
        
        logger.info(
            f"Computed hedge (DOLLAR-BASED fallback): primary={primary_position} contracts @ ${primary_price:.2f} "
            f"(exposure=${primary_exposure:,.2f}), ratio={hedge_ratio:.3f}, "
            f"target_hedge_exposure=${target_hedge_exposure:,.2f}, "
            f"calculated_hedge_contracts={hedge_contracts:.2f}, hedge={hedge_size} contracts @ ${hedge_price:.2f}"
        )
        
        return hedge_size
    
    def should_activate_hedge(
        self,
        net_exposure_dollars: float,
        spacing: float,
        current_price: float,
        grid_mid: float,
        activation_multiplier: float = 1.5
    ) -> bool:
        """
        Determine if hedging should be activated
        
        Args:
            net_exposure_dollars: Current net exposure in dollars
            spacing: Grid spacing
            current_price: Current market price
            grid_mid: Grid midpoint
            activation_multiplier: Multiplier of spacing to trigger hedge
        
        Returns:
            True if hedge should be activated
        """
        # Check if price has moved beyond threshold
        price_move = abs(current_price - grid_mid)
        threshold = spacing * activation_multiplier
        
        # Also check if correlation is sufficient
        correlation_ok = (self.current_correlation is not None and
                         self.current_correlation >= self.correlation_threshold)
        
        should_activate = price_move > threshold and correlation_ok
        
        if should_activate:
            logger.info(f"Hedge activation: price_move={price_move:.2f} > threshold={threshold:.2f}, "
                        f"corr={self.current_correlation:.3f}")
        
        return should_activate
    
    def is_correlation_healthy(self) -> bool:
        """Check if correlation is above threshold"""
        return (self.current_correlation is not None and
                self.current_correlation >= self.correlation_threshold)
    
    def get_hedge_reduction_factor(self) -> float:
        """Get factor to reduce hedge by if correlation drops"""
        if not self.is_correlation_healthy():
            return 0.5  # Reduce hedge by 50%
        return 1.0

