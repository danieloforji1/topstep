"""
Risk Manager
Enforces hard stops: daily loss guard, trailing drawdown guard, emergency flatten
"""
import logging
from typing import Optional
from datetime import datetime, date

logger = logging.getLogger(__name__)


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
        
        self.peak_balance: float = 0.0
        self.current_balance: float = 0.0
        self.daily_start_balance: float = 0.0
        self.last_reset_date: Optional[date] = None
        
        self.hard_stop_triggered: bool = False
        self.hard_stop_reason: Optional[str] = None
    
    def reset_daily(self):
        """Reset daily tracking (call at start of trading day)"""
        today = date.today()
        if self.last_reset_date != today:
            self.daily_start_balance = self.current_balance
            self.last_reset_date = today
            logger.info(f"Daily reset: starting balance = ${self.daily_start_balance:.2f}")
    
    def update_balance(self, balance: float):
        """Update current balance"""
        self.current_balance = balance
        
        # Update peak balance
        if balance > self.peak_balance:
            self.peak_balance = balance
        
        # Reset daily if needed
        self.reset_daily()
    
    def get_daily_pnl(self) -> float:
        """Get today's P&L"""
        return self.current_balance - self.daily_start_balance
    
    def get_trailing_drawdown(self) -> float:
        """Get trailing drawdown from peak"""
        return self.peak_balance - self.current_balance
    
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
        
        # Check exposure cap (warning, not a hard stop)
        if net_exposure >= self.max_net_notional:
            logger.warning(f"Exposure cap reached: ${net_exposure:.2f}")
        
        return False, None
    
    def reset_hard_stop(self):
        """Reset hard stop (manual intervention)"""
        self.hard_stop_triggered = False
        self.hard_stop_reason = None
        logger.info("Hard stop reset by manual intervention")

