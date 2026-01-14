"""
Performance Analyzer
Standardized performance metrics for strategy comparison
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

from .backtest_engine import Trade

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """
    Calculate standardized performance metrics for strategy comparison
    """
    
    def __init__(self, trades: List[Trade], equity_curve: List[float], initial_equity: float):
        """
        Initialize performance analyzer
        
        Args:
            trades: List of completed trades
            equity_curve: List of equity values over time
            initial_equity: Starting equity
        """
        self.trades = trades
        self.equity_curve = equity_curve
        self.initial_equity = initial_equity
        self.final_equity = equity_curve[-1] if equity_curve else initial_equity
    
    def calculate_returns(self) -> pd.Series:
        """Calculate period returns from equity curve"""
        if len(self.equity_curve) < 2:
            return pd.Series([0.0])
        
        equity_series = pd.Series(self.equity_curve)
        returns = equity_series.pct_change().dropna()
        return returns
    
    def calculate_daily_returns(self) -> pd.Series:
        """Calculate daily returns"""
        if not self.trades:
            return pd.Series([0.0])
        
        # Group trades by day
        trades_df = pd.DataFrame([
            {
                'date': trade.exit_time.date(),
                'pnl': trade.pnl
            }
            for trade in self.trades
        ])
        
        daily_pnl = trades_df.groupby('date')['pnl'].sum()
        daily_returns = daily_pnl / self.initial_equity
        
        return daily_returns
    
    def sharpe_ratio(self, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
        """
        Calculate Sharpe Ratio
        
        Args:
            risk_free_rate: Risk-free rate (annual)
            periods_per_year: Number of trading periods per year
            
        Returns:
            Sharpe ratio
        """
        returns = self.calculate_returns()
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / periods_per_year)
        sharpe = np.sqrt(periods_per_year) * excess_returns.mean() / returns.std()
        
        return sharpe
    
    def sortino_ratio(self, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
        """
        Calculate Sortino Ratio (only penalizes downside volatility)
        
        Args:
            risk_free_rate: Risk-free rate (annual)
            periods_per_year: Number of trading periods per year
            
        Returns:
            Sortino ratio
        """
        returns = self.calculate_returns()
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / periods_per_year)
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return float('inf') if excess_returns.mean() > 0 else 0.0
        
        sortino = np.sqrt(periods_per_year) * excess_returns.mean() / downside_returns.std()
        
        return sortino
    
    def max_drawdown(self) -> float:
        """
        Calculate maximum drawdown
        
        Returns:
            Maximum drawdown as percentage
        """
        if len(self.equity_curve) < 2:
            return 0.0
        
        equity_series = pd.Series(self.equity_curve)
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max
        max_dd = drawdown.min()
        
        return abs(max_dd) * 100  # Return as percentage
    
    def max_drawdown_dollar(self) -> float:
        """Calculate maximum drawdown in dollars"""
        if len(self.equity_curve) < 2:
            return 0.0
        
        equity_series = pd.Series(self.equity_curve)
        running_max = equity_series.expanding().max()
        drawdown = equity_series - running_max
        max_dd = drawdown.min()
        
        return abs(max_dd)
    
    def win_rate(self) -> float:
        """Calculate win rate percentage"""
        if not self.trades:
            return 0.0
        
        winning_trades = [t for t in self.trades if t.pnl > 0]
        return len(winning_trades) / len(self.trades) * 100
    
    def profit_factor(self) -> float:
        """
        Calculate profit factor (gross profit / gross loss)
        
        Returns:
            Profit factor (infinity if no losses)
        """
        if not self.trades:
            return 0.0
        
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def average_win(self) -> float:
        """Average winning trade P&L"""
        winning_trades = [t.pnl for t in self.trades if t.pnl > 0]
        return np.mean(winning_trades) if winning_trades else 0.0
    
    def average_loss(self) -> float:
        """Average losing trade P&L"""
        losing_trades = [t.pnl for t in self.trades if t.pnl < 0]
        return np.mean(losing_trades) if losing_trades else 0.0
    
    def largest_win(self) -> float:
        """Largest winning trade"""
        if not self.trades:
            return 0.0
        return max((t.pnl for t in self.trades), default=0.0)
    
    def largest_loss(self) -> float:
        """Largest losing trade"""
        if not self.trades:
            return 0.0
        return min((t.pnl for t in self.trades), default=0.0)
    
    def expectancy(self) -> float:
        """Average P&L per trade"""
        if not self.trades:
            return 0.0
        return np.mean([t.pnl for t in self.trades])
    
    def total_return(self) -> float:
        """Total return percentage"""
        return ((self.final_equity - self.initial_equity) / self.initial_equity) * 100
    
    def annualized_return(self, days: int = 365) -> float:
        """
        Calculate annualized return
        
        Args:
            days: Number of days in backtest period
        """
        if days == 0:
            return 0.0
        
        total_return = self.total_return() / 100
        years = days / 365.0
        if years == 0:
            return 0.0
        
        annualized = (1 + total_return) ** (1 / years) - 1
        return annualized * 100
    
    def calmar_ratio(self, days: int = 365) -> float:
        """
        Calculate Calmar Ratio (annualized return / max drawdown)
        
        Args:
            days: Number of days in backtest period
        """
        max_dd = self.max_drawdown() / 100
        if max_dd == 0:
            return float('inf') if self.total_return() > 0 else 0.0
        
        annual_return = self.annualized_return(days) / 100
        return annual_return / max_dd
    
    def consistency_score(self) -> float:
        """
        Calculate consistency score (inverse of std dev of daily returns)
        Lower std dev = higher consistency
        """
        daily_returns = self.calculate_daily_returns()
        if len(daily_returns) == 0:
            return 0.0
        
        std_dev = daily_returns.std()
        if std_dev == 0:
            return 100.0  # Perfect consistency
        
        # Convert to score (lower std dev = higher score)
        # Score = 100 / (1 + std_dev * 100)
        score = 100 / (1 + std_dev * 100)
        return score
    
    def average_daily_return(self) -> float:
        """Average daily return in dollars"""
        daily_returns = self.calculate_daily_returns()
        if len(daily_returns) == 0:
            return 0.0
        return daily_returns.mean() * self.initial_equity
    
    def generate_report(self) -> str:
        """Generate formatted performance report"""
        if not self.trades:
            return "No trades executed.\n"
        
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl < 0]
        
        # Calculate days
        if self.trades:
            first_trade = min(t.entry_time for t in self.trades)
            last_trade = max(t.exit_time for t in self.trades)
            days = (last_trade - first_trade).days + 1
        else:
            days = 0
        
        report = f"""
================================================================================
PERFORMANCE REPORT
================================================================================

EQUITY METRICS:
  Initial Equity: ${self.initial_equity:,.2f}
  Final Equity: ${self.final_equity:,.2f}
  Total Return: {self.total_return():.2f}%
  Annualized Return: {self.annualized_return(days):.2f}%
  Total P&L: ${self.final_equity - self.initial_equity:,.2f}

RISK METRICS:
  Maximum Drawdown: {self.max_drawdown():.2f}% (${self.max_drawdown_dollar():,.2f})
  Sharpe Ratio: {self.sharpe_ratio():.2f}
  Sortino Ratio: {self.sortino_ratio():.2f}
  Calmar Ratio: {self.calmar_ratio(days):.2f}

TRADE STATISTICS:
  Total Trades: {len(self.trades)}
  Winning Trades: {len(winning_trades)} ({self.win_rate():.2f}%)
  Losing Trades: {len(losing_trades)} ({100 - self.win_rate():.2f}%)
  
  Profit Factor: {self.profit_factor():.2f}
  Expectancy: ${self.expectancy():.2f} per trade
  
  Average Win: ${self.average_win():.2f}
  Average Loss: ${self.average_loss():.2f}
  Largest Win: ${self.largest_win():.2f}
  Largest Loss: ${self.largest_loss():.2f}

CONSISTENCY:
  Average Daily Return: ${self.average_daily_return():.2f}
  Consistency Score: {self.consistency_score():.2f}

================================================================================
"""
        return report
    
    def get_metrics_dict(self) -> Dict[str, Any]:
        """Get all metrics as dictionary"""
        if self.trades:
            first_trade = min(t.entry_time for t in self.trades)
            last_trade = max(t.exit_time for t in self.trades)
            days = (last_trade - first_trade).days + 1
        else:
            days = 0
        
        return {
            'initial_equity': self.initial_equity,
            'final_equity': self.final_equity,
            'total_return': self.total_return(),
            'annualized_return': self.annualized_return(days),
            'total_pnl': self.final_equity - self.initial_equity,
            'max_drawdown': self.max_drawdown(),
            'max_drawdown_dollar': self.max_drawdown_dollar(),
            'sharpe_ratio': self.sharpe_ratio(),
            'sortino_ratio': self.sortino_ratio(),
            'calmar_ratio': self.calmar_ratio(days),
            'win_rate': self.win_rate(),
            'profit_factor': self.profit_factor(),
            'expectancy': self.expectancy(),
            'num_trades': len(self.trades),
            'winning_trades': len([t for t in self.trades if t.pnl > 0]),
            'losing_trades': len([t for t in self.trades if t.pnl < 0]),
            'average_win': self.average_win(),
            'average_loss': self.average_loss(),
            'largest_win': self.largest_win(),
            'largest_loss': self.largest_loss(),
            'consistency_score': self.consistency_score(),
            'average_daily_return': self.average_daily_return(),
            'days': days
        }

