"""
Performance Metrics and Reporting for MGC Backtest
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

class PerformanceAnalyzer:
    """Analyzes and reports backtest performance"""
    
    def __init__(self, trades: List[Any], equity_curve: List[float]):
        self.trades = trades
        self.equity_curve = equity_curve
        self.df_trades = self._trades_to_dataframe()
    
    def _trades_to_dataframe(self) -> pd.DataFrame:
        """Convert trades to DataFrame"""
        if not self.trades:
            return pd.DataFrame()
        
        data = {
            'entry_time': [t.entry_time for t in self.trades],
            'exit_time': [t.exit_time for t in self.trades],
            'entry_price': [t.entry_price for t in self.trades],
            'exit_price': [t.exit_price for t in self.trades],
            'stop_loss': [getattr(t, 'stop_loss', None) for t in self.trades],
            'take_profit': [getattr(t, 'take_profit', None) for t in self.trades],
            'contracts': [t.contracts for t in self.trades],
            'is_long': [t.is_long for t in self.trades],
            'pnl': [t.pnl for t in self.trades],
            'pnl_pct': [t.pnl_pct for t in self.trades],
            'risk_reward': [t.risk_reward for t in self.trades],
            'duration_minutes': [t.duration_minutes for t in self.trades],
            'exit_reason': [t.exit_reason for t in self.trades]
        }
        
        # Add Asian range size if available (for Asian Range Breakout strategy)
        if hasattr(self.trades[0], 'asian_range_size'):
            data['asian_range_size'] = [getattr(t, 'asian_range_size', None) for t in self.trades]
        
        return pd.DataFrame(data)
    
    def generate_report(self) -> str:
        """Generate comprehensive performance report"""
        if not self.trades:
            return "No trades executed during backtest period."
        
        report = []
        report.append("=" * 80)
        report.append("MGC TRADING STRATEGY - BACKTEST REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary Statistics
        report.append("SUMMARY STATISTICS")
        report.append("-" * 80)
        
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl < 0]
        
        report.append(f"Total Trades: {total_trades}")
        report.append(f"Winning Trades: {len(winning_trades)} ({len(winning_trades)/total_trades*100:.1f}%)")
        report.append(f"Losing Trades: {len(losing_trades)} ({len(losing_trades)/total_trades*100:.1f}%)")
        report.append("")
        
        # PnL Statistics
        total_pnl = sum(t.pnl for t in self.trades)
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        
        report.append("P&L STATISTICS")
        report.append("-" * 80)
        report.append(f"Total P&L: ${total_pnl:,.2f}")
        report.append(f"Gross Profit: ${gross_profit:,.2f}")
        report.append(f"Gross Loss: ${gross_loss:,.2f}")
        report.append(f"Profit Factor: {gross_profit/gross_loss:.2f}" if gross_loss > 0 else "Profit Factor: N/A")
        report.append("")
        
        # Win Rate
        win_rate = len(winning_trades) / total_trades * 100
        report.append(f"Win Rate: {win_rate:.1f}%")
        report.append("")
        
        # Average Statistics
        if winning_trades:
            avg_win = np.mean([t.pnl for t in winning_trades])
            max_win = max(t.pnl for t in winning_trades)
            report.append(f"Average Win: ${avg_win:,.2f}")
            report.append(f"Largest Win: ${max_win:,.2f}")
        
        if losing_trades:
            avg_loss = np.mean([t.pnl for t in losing_trades])
            max_loss = min(t.pnl for t in losing_trades)
            report.append(f"Average Loss: ${avg_loss:,.2f}")
            report.append(f"Largest Loss: ${max_loss:,.2f}")
        
        report.append("")
        
        # Risk Metrics
        avg_rr = np.mean([t.risk_reward for t in self.trades])
        report.append("RISK METRICS")
        report.append("-" * 80)
        report.append(f"Average Risk:Reward: {avg_rr:.2f}")
        
        # Expectancy
        expectancy = (win_rate / 100 * avg_win) - ((100 - win_rate) / 100 * abs(avg_loss)) if winning_trades and losing_trades else 0
        report.append(f"Expectancy: ${expectancy:,.2f} per trade")
        report.append("")
        
        # Drawdown
        if self.equity_curve:
            peak = self.equity_curve[0]
            max_dd = 0
            max_dd_value = 0
            for equity in self.equity_curve:
                if equity > peak:
                    peak = equity
                dd = peak - equity
                dd_pct = (dd / peak * 100) if peak > 0 else 0
                if dd_pct > max_dd:
                    max_dd = dd_pct
                    max_dd_value = dd
            
            report.append("DRAWDOWN")
            report.append("-" * 80)
            report.append(f"Maximum Drawdown: {max_dd:.2f}% (${max_dd_value:,.2f})")
            report.append("")
        
        # Equity Curve
        if self.equity_curve:
            initial_equity = self.equity_curve[0]
            final_equity = self.equity_curve[-1]
            total_return = ((final_equity - initial_equity) / initial_equity * 100) if initial_equity > 0 else 0
            
            report.append("EQUITY CURVE")
            report.append("-" * 80)
            report.append(f"Initial Equity: ${initial_equity:,.2f}")
            report.append(f"Final Equity: ${final_equity:,.2f}")
            report.append(f"Total Return: {total_return:.2f}%")
            report.append("")
        
        # Trade Duration
        avg_duration = np.mean([t.duration_minutes for t in self.trades])
        report.append("TRADE DURATION")
        report.append("-" * 80)
        report.append(f"Average Duration: {avg_duration:.1f} minutes")
        report.append(f"Average Duration: {avg_duration/60:.1f} hours")
        report.append("")
        
        # Exit Reasons
        exit_reasons = {}
        for t in self.trades:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1
        
        report.append("EXIT REASONS")
        report.append("-" * 80)
        for reason, count in exit_reasons.items():
            report.append(f"{reason}: {count} ({count/total_trades*100:.1f}%)")
        report.append("")
        
        # Long vs Short
        long_trades = [t for t in self.trades if t.is_long]
        short_trades = [t for t in self.trades if not t.is_long]
        
        if long_trades:
            long_pnl = sum(t.pnl for t in long_trades)
            long_win_rate = len([t for t in long_trades if t.pnl > 0]) / len(long_trades) * 100
            report.append("LONG TRADES")
            report.append("-" * 80)
            report.append(f"Count: {len(long_trades)}")
            report.append(f"Total P&L: ${long_pnl:,.2f}")
            report.append(f"Win Rate: {long_win_rate:.1f}%")
            report.append("")
        
        if short_trades:
            short_pnl = sum(t.pnl for t in short_trades)
            short_win_rate = len([t for t in short_trades if t.pnl > 0]) / len(short_trades) * 100
            report.append("SHORT TRADES")
            report.append("-" * 80)
            report.append(f"Count: {len(short_trades)}")
            report.append(f"Total P&L: ${short_pnl:,.2f}")
            report.append(f"Win Rate: {short_win_rate:.1f}%")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def plot_equity_curve(self, save_path: Optional[str] = None):
        """Plot equity curve"""
        if not MATPLOTLIB_AVAILABLE:
            print("matplotlib not available, skipping plot")
            return
        
        if not self.equity_curve:
            return
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.equity_curve, linewidth=2)
        plt.title('Equity Curve', fontsize=14, fontweight='bold')
        plt.xlabel('Trade Number', fontsize=12)
        plt.ylabel('Equity ($)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Equity curve saved to {save_path}")
        else:
            plt.show()
    
    def plot_monthly_returns(self, save_path: Optional[str] = None):
        """Plot monthly returns"""
        if not MATPLOTLIB_AVAILABLE:
            print("matplotlib not available, skipping plot")
            return
        
        if not self.df_trades.empty:
            self.df_trades['month'] = pd.to_datetime(self.df_trades['exit_time']).dt.to_period('M')
            monthly_pnl = self.df_trades.groupby('month')['pnl'].sum()
            
            plt.figure(figsize=(12, 6))
            monthly_pnl.plot(kind='bar', color=['green' if x > 0 else 'red' for x in monthly_pnl])
            plt.title('Monthly Returns', fontsize=14, fontweight='bold')
            plt.xlabel('Month', fontsize=12)
            plt.ylabel('P&L ($)', fontsize=12)
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Monthly returns saved to {save_path}")
            else:
                plt.show()
    
    def export_trades_csv(self, filepath: str):
        """Export trades to CSV"""
        if not self.df_trades.empty:
            self.df_trades.to_csv(filepath, index=False)
            print(f"Trades exported to {filepath}")
        else:
            print("No trades to export")

