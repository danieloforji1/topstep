"""
Risk Manager
Enforces hard stops: daily loss guard, trailing drawdown guard, emergency flatten
"""
import logging
from typing import Optional
from datetime import datetime, date
import pytz

logger = logging.getLogger(__name__)

# Chicago timezone for trading day calculations
CHICAGO_TZ = pytz.timezone('America/Chicago')


class RiskManager:
    """Manages risk limits and safety controls"""
    
    def __init__(
        self,
        max_daily_loss: float = 900.0,
        trailing_drawdown_limit: float = 1800.0,
        max_net_notional: float = 1200.0
    ):
        self.max_daily_loss = max_daily_loss
        self.trailing_drawdown_limit = trailing_drawdown_limit
        self.max_net_notional = max_net_notional
        
        # Use equity instead of balance for accurate intraday tracking
        self.peak_equity: float = 0.0
        self.current_equity: float = 0.0
        self.daily_start_equity: float = 0.0
        self.last_reset_date: Optional[date] = None
        
        # Keep balance for backward compatibility if needed
        self.current_balance: float = 0.0
        
        self.hard_stop_triggered: bool = False
        self.hard_stop_reason: Optional[str] = None
    
    def reset_daily(self):
        """Reset daily tracking (call at start of trading day) - uses Chicago timezone"""
        # Get current date in Chicago timezone (trading day)
        now_chicago = datetime.now(CHICAGO_TZ)
        today = now_chicago.date()
        
        if self.last_reset_date != today:
            self.daily_start_equity = self.current_equity
            self.last_reset_date = today
            logger.info(f"Daily reset (Chicago timezone): starting equity = ${self.daily_start_equity:.2f}")
    
    def update_balance(self, balance: float):
        """Update current balance (kept for backward compatibility)"""
        self.current_balance = balance
        # Also update equity if not set separately
        if self.current_equity == 0.0:
            self.current_equity = balance
            self.peak_equity = balance
            self.daily_start_equity = balance
        
        # Reset daily if needed
        self.reset_daily()
    
    def update_equity(self, equity: float):
        """Update current equity (preferred method - includes unrealized P&L)"""
        self.current_equity = equity
        
        # Update peak equity
        if equity > self.peak_equity:
            self.peak_equity = equity
        
        # Reset daily if needed
        self.reset_daily()
    
    def get_daily_pnl(self) -> float:
        """Get today's P&L (based on equity, not balance)"""
        return self.current_equity - self.daily_start_equity
    
    def get_trailing_drawdown(self) -> float:
        """Get trailing drawdown from peak equity"""
        return self.peak_equity - self.current_equity
    
    def daily_loss_exceeded(self) -> bool:
        """Check if daily loss limit exceeded"""
        daily_pnl = self.get_daily_pnl()
        exceeded = daily_pnl <= -self.max_daily_loss
        
        if exceeded and not self.hard_stop_triggered:
            self.hard_stop_triggered = True
            self.hard_stop_reason = f"Daily loss limit exceeded: ${daily_pnl:.2f} <= ${-self.max_daily_loss:.2f}"
            logger.critical(self.hard_stop_reason)
        
        return exceeded
    
    def trailing_drawdown_exceeded(self) -> bool:
        """Check if trailing drawdown limit exceeded"""
        drawdown = self.get_trailing_drawdown()
        exceeded = drawdown >= self.trailing_drawdown_limit
        
        if exceeded and not self.hard_stop_triggered:
            self.hard_stop_triggered = True
            self.hard_stop_reason = f"Trailing drawdown exceeded: ${drawdown:.2f} >= ${self.trailing_drawdown_limit:.2f}"
            logger.critical(self.hard_stop_reason)
        
        return exceeded
    
    def exposure_cap_exceeded(self, current_exposure: float) -> bool:
        """Check if exposure cap exceeded"""
        return current_exposure >= self.max_net_notional
    
    def check_all_limits(
        self,
        daily_pnl: float,
        trailing_drawdown: float,
        net_exposure: float
    ) -> tuple:
        """
        Check all risk limits
        
        Returns: (should_stop, reason)
        """
        if self.hard_stop_triggered:
            return True, self.hard_stop_reason
        
        # Check daily loss
        if daily_pnl <= -self.max_daily_loss:
            reason = f"Daily loss limit: ${daily_pnl:.2f}"
            self.hard_stop_triggered = True
            self.hard_stop_reason = reason
            return True, reason
        
        # Check trailing drawdown
        if trailing_drawdown >= self.trailing_drawdown_limit:
            reason = f"Trailing drawdown: ${trailing_drawdown:.2f}"
            self.hard_stop_triggered = True
            self.hard_stop_reason = reason
            return True, reason
        
        # Check exposure cap (HARD STOP - was previously only a warning)
        if net_exposure >= self.max_net_notional:
            reason = f"Exposure cap exceeded: ${net_exposure:.2f} >= ${self.max_net_notional:.2f}"
            self.hard_stop_triggered = True
            self.hard_stop_reason = reason
            logger.critical(reason)
            return True, reason
        
        return False, None
    
    def reset_hard_stop(self):
        """Reset hard stop (manual intervention)"""
        self.hard_stop_triggered = False
        self.hard_stop_reason = None
        logger.info("Hard stop reset by manual intervention")

