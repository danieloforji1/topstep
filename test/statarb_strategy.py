"""
Statistical Arbitrage Strategy Logic
Implements spread calculation, z-score normalization, and signal generation
"""
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SpreadSignal:
    """Represents a spread trading signal"""
    timestamp: pd.Timestamp
    spread: float
    zscore: float
    price_a: float  # GC price
    price_b: float  # MGC price
    beta: float
    spread_mean: float
    spread_std: float
    signal: str  # "LONG_SPREAD", "SHORT_SPREAD", "NONE", "EXIT"
    is_entry: bool
    is_exit: bool


class StatArbCalculator:
    """Calculates spread, z-scores, and generates trading signals"""
    
    def __init__(
        self,
        z_entry: float = 2.0,
        z_exit: float = 0.6,
        lookback_periods: int = 1440,  # 1 day of 1-minute bars (default)
        min_lookback: int = 100,  # Minimum periods needed to calculate stats
        beta_lookback: int = 500  # Periods for beta calculation
    ):
        """
        Initialize StatArb calculator
        
        Args:
            z_entry: Z-score threshold for entry (default 2.0)
            z_exit: Z-score threshold for exit (default 0.6)
            lookback_periods: Number of periods for rolling mean/std (default 1440 = 1 day of 1-min bars)
            min_lookback: Minimum periods needed before generating signals
            beta_lookback: Periods to use for beta calculation
        """
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.lookback_periods = lookback_periods
        self.min_lookback = min_lookback
        self.beta_lookback = beta_lookback
        
        # Cache for beta calculation
        self._beta_cache: Optional[float] = None
        self._beta_calculated = False
    
    def calculate_beta(
        self,
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
        current_index: int
    ) -> float:
        """
        Calculate hedge ratio (beta) using OLS linear regression
        
        Regression: X_t = alpha + beta * Y_t + epsilon_t
        Where X = GC, Y = MGC
        
        IMPORTANT: GC and MGC prices are both quoted PER OUNCE
        So prices are very similar (~4100), and beta should be ~1.0
        
        Contract sizes are separate:
        - 1 GC contract = 100 oz
        - 1 MGC contract = 10 oz
        - So 1 GC contract = 10 MGC contracts (for hedging)
        
        But for spread calculation: spread = GC_price - beta * MGC_price
        where beta ≈ 1.0 because prices are per-ounce and nearly identical
        
        Args:
            df_a: DataFrame for instrument A (GC)
            df_b: DataFrame for instrument B (MGC)
            current_index: Current bar index
            
        Returns:
            Beta (hedge ratio) from OLS regression, should be ~1.0
        """
        # Use lookback window for beta calculation
        start_idx = max(0, current_index - self.beta_lookback)
        end_idx = current_index + 1
        
        if end_idx - start_idx < 50:  # Need minimum data
            # Default: for GC/MGC, beta ≈ 1.0 (prices are per-ounce and similar)
            return 1.0
        
        # Get aligned prices (use close prices - these are per-ounce)
        prices_a = df_a.iloc[start_idx:end_idx]['close'].values
        prices_b = df_b.iloc[start_idx:end_idx]['close'].values
        
        if len(prices_a) != len(prices_b) or len(prices_a) < 50:
            return 1.0
        
        # Calculate beta using OLS regression: X = alpha + beta * Y
        # Beta = Cov(X, Y) / Var(Y)
        try:
            mean_a = np.mean(prices_a)
            mean_b = np.mean(prices_b)
            
            # Calculate covariance and variance
            cov = np.mean((prices_a - mean_a) * (prices_b - mean_b))
            var_b = np.var(prices_b)
            
            if var_b == 0 or np.isnan(var_b) or np.isnan(cov):
                # Fallback: use price ratio
                if mean_b != 0:
                    return mean_a / mean_b
                return 1.0
            
            beta = cov / var_b
            
            # For GC/MGC pair, beta should be around 1.0 (prices are per-ounce and similar)
            # Allow reasonable range: 0.5 to 2.0
            if beta < 0.5 or beta > 2.0 or np.isnan(beta):
                # Use simple price ratio as fallback
                if mean_b != 0:
                    beta = mean_a / mean_b
                    # Sanity check: should be reasonable for GC/MGC (prices are similar)
                    if beta < 0.5 or beta > 2.0:
                        return 1.0
                else:
                    return 1.0
            
            return beta
        except Exception as e:
            logger.debug(f"Error calculating beta: {e}, using default 1.0")
            return 1.0
    
    def calculate_spread(
        self,
        price_a: float,
        price_b: float,
        beta: float
    ) -> float:
        """
        Calculate spread: spread = price_A - beta * price_B
        
        Args:
            price_a: Price of instrument A (GC)
            price_b: Price of instrument B (MGC)
            beta: Hedge ratio
            
        Returns:
            Spread value
        """
        return price_a - beta * price_b
    
    def calculate_zscore(
        self,
        current_spread: float,
        spread_history: pd.Series
    ) -> Tuple[float, float, float]:
        """
        Calculate z-score of current spread
        
        Args:
            current_spread: Current spread value
            spread_history: Historical spread values (rolling window)
            
        Returns:
            (zscore, mean, std)
        """
        if len(spread_history) < self.min_lookback:
            return 0.0, 0.0, 1.0
        
        mean = spread_history.mean()
        std = spread_history.std()
        
        if std == 0 or np.isnan(std):
            return 0.0, mean, 1.0
        
        zscore = (current_spread - mean) / std
        
        return zscore, mean, std
    
    def generate_signal(
        self,
        zscore: float,
        current_position: Optional[str] = None
    ) -> Tuple[str, bool, bool]:
        """
        Generate trading signal based on z-score
        
        Args:
            zscore: Current z-score
            current_position: Current position ("LONG_SPREAD" or "SHORT_SPREAD" or None)
            
        Returns:
            (signal, is_entry, is_exit)
            signal: "LONG_SPREAD", "SHORT_SPREAD", "EXIT", or "NONE"
            is_entry: True if this is an entry signal
            is_exit: True if this is an exit signal
        """
        # Exit logic: if in position and z-score crosses exit threshold
        if current_position:
            if abs(zscore) < self.z_exit:
                return "EXIT", False, True
            # Also exit if z-score crosses zero (mean reversion complete)
            if current_position == "LONG_SPREAD" and zscore > 0:
                return "EXIT", False, True
            if current_position == "SHORT_SPREAD" and zscore < 0:
                return "EXIT", False, True
        
        # Entry logic: only if not in position
        if current_position is None:
            if zscore > self.z_entry:
                return "SHORT_SPREAD", True, False  # Short spread (short A, long B)
            elif zscore < -self.z_entry:
                return "LONG_SPREAD", True, False  # Long spread (long A, short B)
        
        return "NONE", False, False
    
    def process_bar(
        self,
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
        index: int,
        current_position: Optional[str] = None,
        spread_history: Optional[pd.Series] = None,
        fixed_beta: Optional[float] = None
    ) -> Optional[SpreadSignal]:
        """
        Process a single bar and generate signal
        
        Args:
            df_a: DataFrame for instrument A (GC)
            df_b: DataFrame for instrument B (MGC)
            index: Current bar index
            current_position: Current position ("LONG_SPREAD" or "SHORT_SPREAD" or None)
            spread_history: Historical spread values for rolling calculation
            fixed_beta: Optional fixed beta to use (ensures consistency with spread_history)
            
        Returns:
            SpreadSignal or None if insufficient data
        """
        if index >= len(df_a) or index >= len(df_b):
            return None
        
        # Get current prices (use mid-price: (high + low) / 2 or close)
        price_a = df_a.iloc[index]['close']
        price_b = df_b.iloc[index]['close']
        timestamp = df_a.iloc[index]['timestamp']
        
        # Use fixed_beta if provided, otherwise calculate dynamically
        # CRITICAL: If spread_history was calculated with a specific beta,
        # we MUST use the same beta for current spread calculation
        if fixed_beta is not None:
            beta = fixed_beta
        else:
            beta = self.calculate_beta(df_a, df_b, index)
        
        # Calculate spread
        spread = self.calculate_spread(price_a, price_b, beta)
        
        # Calculate z-score
        if spread_history is None or len(spread_history) < self.min_lookback:
            return None
        
        zscore, spread_mean, spread_std = self.calculate_zscore(spread, spread_history)
        
        # Generate signal
        signal, is_entry, is_exit = self.generate_signal(zscore, current_position)
        
        return SpreadSignal(
            timestamp=timestamp,
            spread=spread,
            zscore=zscore,
            price_a=price_a,
            price_b=price_b,
            beta=beta,
            spread_mean=spread_mean,
            spread_std=spread_std,
            signal=signal,
            is_entry=is_entry,
            is_exit=is_exit
        )

