"""
Plotly visualization for Asian Range Breakout Strategy backtest results
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional, Dict, Any
from datetime import time
import logging

logger = logging.getLogger(__name__)


class AsianRangeVisualizer:
    """Creates interactive Plotly charts for backtest results"""
    
    def __init__(self, df_1m: pd.DataFrame, trades: List[Any], asian_ranges: List[Any]):
        """
        Initialize visualizer
        
        Args:
            df_1m: 1-minute OHLCV DataFrame
            trades: List of Trade objects
            asian_ranges: List of AsianRange objects
        """
        self.df_1m = df_1m.copy()
        self.trades = trades
        self.asian_ranges = asian_ranges
        
        # Create a mapping of date to asian range for quick lookup
        self.range_by_date = {}
        for ar in asian_ranges:
            if isinstance(ar.date, pd.Timestamp):
                date_key = ar.date.date()
            else:
                date_key = ar.date
            self.range_by_date[date_key] = ar
    
    def plot_trading_day(self, date: pd.Timestamp, output_file: Optional[str] = None) -> go.Figure:
        """
        Plot a single trading day with Asian range, entry/exit, SL/TP
        
        Args:
            date: Trading date to plot
            output_file: Optional file path to save HTML
            
        Returns:
            Plotly Figure object
        """
        if isinstance(date, pd.Timestamp):
            trading_date = date.date()
        else:
            trading_date = date
            date = pd.Timestamp.combine(trading_date, time(0, 0))
        
        # Get data for the day (extend to show Asian session)
        prev_day = trading_date - pd.Timedelta(days=1)
        start_time = pd.Timestamp.combine(prev_day, time(20, 0))  # 8 PM prev day
        end_time = pd.Timestamp.combine(trading_date, time(12, 0))  # 12 PM current day
        
        # Handle timezones
        if self.df_1m['timestamp'].dt.tz is not None:
            start_time = start_time.tz_localize(self.df_1m['timestamp'].dt.tz)
            end_time = end_time.tz_localize(self.df_1m['timestamp'].dt.tz)
        
        mask = (self.df_1m['timestamp'] >= start_time) & (self.df_1m['timestamp'] <= end_time)
        day_data = self.df_1m[mask].copy()
        
        if len(day_data) == 0:
            logger.warning(f"No data available for {trading_date}")
            return None
        
        # Get Asian range for this date
        asian_range = self.range_by_date.get(trading_date)
        
        # Get trade for this date
        trade = None
        for t in self.trades:
            trade_date = t.entry_time.date() if hasattr(t.entry_time, 'date') else pd.Timestamp(t.entry_time).date()
            if trade_date == trading_date:
                trade = t
                break
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=(f"MGC Gold - {trading_date}", "Volume")
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=day_data['timestamp'],
                open=day_data['open'],
                high=day_data['high'],
                low=day_data['low'],
                close=day_data['close'],
                name="Price"
            ),
            row=1, col=1
        )
        
        # Asian Range - draw a box around it
        if asian_range:
            # Calculate Asian session time range (8 PM prev day to 2 AM current day)
            prev_day = trading_date - pd.Timedelta(days=1)
            asian_start_time = pd.Timestamp.combine(prev_day, time(20, 0))
            asian_end_time = pd.Timestamp.combine(trading_date, time(2, 0))
            
            # Handle timezones
            if self.df_1m['timestamp'].dt.tz is not None:
                asian_start_time = asian_start_time.tz_localize(self.df_1m['timestamp'].dt.tz)
                asian_end_time = asian_end_time.tz_localize(self.df_1m['timestamp'].dt.tz)
            
            # Draw a box (rectangle) around the Asian range
            fig.add_shape(
                type="rect",
                x0=asian_start_time,
                x1=asian_end_time,
                y0=asian_range.asian_low,
                y1=asian_range.asian_high,
                fillcolor="rgba(255, 165, 0, 0.15)",  # Orange with transparency
                line=dict(
                    color="orange",
                    width=2,
                    dash="dash"
                ),
                layer="below",
                row=1, col=1
            )
            
            # Add text annotation for Asian Range
            mid_time = asian_start_time + (asian_end_time - asian_start_time) / 2
            fig.add_annotation(
                x=mid_time,
                y=asian_range.asian_high,
                text=f"Asian Range<br>High: {asian_range.asian_high:.2f}<br>Low: {asian_range.asian_low:.2f}",
                showarrow=True,
                arrowhead=2,
                arrowcolor="orange",
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="orange",
                borderwidth=1,
                row=1, col=1
            )
        
        # Trade markers
        if trade:
            # Entry point
            entry_color = "green" if trade.is_long else "red"
            fig.add_trace(
                go.Scatter(
                    x=[trade.entry_time],
                    y=[trade.entry_price],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up' if trade.is_long else 'triangle-down',
                        size=15,
                        color=entry_color,
                        line=dict(width=2, color='black')
                    ),
                    name=f"Entry ({'LONG' if trade.is_long else 'SHORT'})",
                    text=[f"Entry: {trade.entry_price:.2f}<br>Time: {trade.entry_time}"],
                    hovertemplate="%{text}<extra></extra>"
                ),
                row=1, col=1
            )
            
            # Exit point
            exit_colors = {
                "TP": "green",
                "SL": "red",
                "BE": "yellow",
                "TimeExit": "blue"
            }
            exit_color = exit_colors.get(trade.exit_reason, "gray")
            fig.add_trace(
                go.Scatter(
                    x=[trade.exit_time],
                    y=[trade.exit_price],
                    mode='markers',
                    marker=dict(
                        symbol='x',
                        size=15,
                        color=exit_color,
                        line=dict(width=2, color='black')
                    ),
                    name=f"Exit ({trade.exit_reason})",
                    text=[f"Exit: {trade.exit_price:.2f}<br>Time: {trade.exit_time}<br>PnL: ${trade.pnl:.2f}"],
                    hovertemplate="%{text}<extra></extra>"
                ),
                row=1, col=1
            )
            
            # Stop Loss line (make it more visible with thicker line)
            fig.add_hline(
                y=trade.stop_loss,
                line_dash="dashdot",
                line_color="red",
                line_width=2,
                annotation_text=f"Stop Loss: {trade.stop_loss:.2f}",
                annotation_position="right",
                row=1, col=1
            )
            
            # Take Profit line
            fig.add_hline(
                y=trade.take_profit,
                line_dash="dot",
                line_color="green",
                annotation_text=f"Take Profit: {trade.take_profit:.2f}",
                annotation_position="right",
                row=1, col=1
            )
        
        # Volume
        if 'volume' in day_data.columns:
            colors = ['red' if day_data.iloc[i]['close'] < day_data.iloc[i]['open'] else 'green' 
                     for i in range(len(day_data))]
            fig.add_trace(
                go.Bar(
                    x=day_data['timestamp'],
                    y=day_data['volume'],
                    name="Volume",
                    marker_color=colors
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=f"Asian Range Breakout Strategy - {trading_date}",
            xaxis_rangeslider_visible=False,
            height=800,
            hovermode='x unified',
            showlegend=True
        )
        
        # Update x-axes
        fig.update_xaxes(title_text="Time", row=2, col=1)
        
        # Update y-axes
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        # Save if requested
        if output_file:
            fig.write_html(output_file)
            logger.info(f"Chart saved to {output_file}")
        
        return fig
    
    def plot_all_trades(self, output_file: Optional[str] = None, max_days: int = 10) -> go.Figure:
        """
        Plot all trading days with trades (limited to avoid overcrowding)
        
        Args:
            output_file: Optional file path to save HTML
            max_days: Maximum number of days to plot (default 10)
            
        Returns:
            Plotly Figure object
        """
        if not self.trades:
            logger.warning("No trades to plot")
            return None
        
        # Get unique trading dates
        trade_dates = []
        for trade in self.trades:
            date = trade.entry_time.date() if hasattr(trade.entry_time, 'date') else pd.Timestamp(trade.entry_time).date()
            if date not in trade_dates:
                trade_dates.append(date)
        
        # Limit to max_days
        if len(trade_dates) > max_days:
            trade_dates = trade_dates[:max_days]
            logger.info(f"Plotting first {max_days} trading days (out of {len(trade_dates)} total)")
        
        # Create figure
        fig = go.Figure()
        
        # Plot candlesticks for all days
        for date in trade_dates:
            if isinstance(date, pd.Timestamp):
                trading_date = date.date()
            else:
                trading_date = date
            
            prev_day = trading_date - pd.Timedelta(days=1)
            start_time = pd.Timestamp.combine(prev_day, time(20, 0))
            end_time = pd.Timestamp.combine(trading_date, time(12, 0))
            
            if self.df_1m['timestamp'].dt.tz is not None:
                start_time = start_time.tz_localize(self.df_1m['timestamp'].dt.tz)
                end_time = end_time.tz_localize(self.df_1m['timestamp'].dt.tz)
            
            mask = (self.df_1m['timestamp'] >= start_time) & (self.df_1m['timestamp'] <= end_time)
            day_data = self.df_1m[mask].copy()
            
            if len(day_data) > 0:
                fig.add_trace(
                    go.Candlestick(
                        x=day_data['timestamp'],
                        open=day_data['open'],
                        high=day_data['high'],
                        low=day_data['low'],
                        close=day_data['close'],
                        name=str(trading_date),
                        showlegend=False
                    )
                )
        
        # Add Asian ranges
        for ar in self.asian_ranges:
            if isinstance(ar.date, pd.Timestamp):
                date_key = ar.date.date()
            else:
                date_key = ar.date
            
            if date_key in trade_dates:
                # Get time range for this day
                prev_day = date_key - pd.Timedelta(days=1)
                start_time = pd.Timestamp.combine(prev_day, time(20, 0))
                end_time = pd.Timestamp.combine(date_key, time(2, 0))
                
                if self.df_1m['timestamp'].dt.tz is not None:
                    start_time = start_time.tz_localize(self.df_1m['timestamp'].dt.tz)
                    end_time = end_time.tz_localize(self.df_1m['timestamp'].dt.tz)
                
                # Add horizontal lines for Asian range
                fig.add_trace(
                    go.Scatter(
                        x=[start_time, end_time],
                        y=[ar.asian_high, ar.asian_high],
                        mode='lines',
                        line=dict(color='orange', dash='dash', width=2),
                        name=f"Asian High {date_key}",
                        showlegend=False
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=[start_time, end_time],
                        y=[ar.asian_low, ar.asian_low],
                        mode='lines',
                        line=dict(color='orange', dash='dash', width=2),
                        name=f"Asian Low {date_key}",
                        showlegend=False
                    )
                )
        
        # Add trade markers
        for trade in self.trades:
            trade_date = trade.entry_time.date() if hasattr(trade.entry_time, 'date') else pd.Timestamp(trade.entry_time).date()
            if trade_date in trade_dates:
                # Entry
                entry_color = "green" if trade.is_long else "red"
                fig.add_trace(
                    go.Scatter(
                        x=[trade.entry_time],
                        y=[trade.entry_price],
                        mode='markers',
                        marker=dict(
                            symbol='triangle-up' if trade.is_long else 'triangle-down',
                            size=12,
                            color=entry_color,
                            line=dict(width=2, color='black')
                        ),
                        name=f"Entry {'LONG' if trade.is_long else 'SHORT'}",
                        text=[f"Entry: {trade.entry_price:.2f}<br>PnL: ${trade.pnl:.2f}"],
                        hovertemplate="%{text}<extra></extra>",
                        showlegend=(trade == self.trades[0])  # Only show legend for first trade
                    )
                )
                
                # Exit
                exit_colors = {"TP": "green", "SL": "red", "BE": "yellow", "TimeExit": "blue"}
                exit_color = exit_colors.get(trade.exit_reason, "gray")
                fig.add_trace(
                    go.Scatter(
                        x=[trade.exit_time],
                        y=[trade.exit_price],
                        mode='markers',
                        marker=dict(
                            symbol='x',
                            size=12,
                            color=exit_color,
                            line=dict(width=2, color='black')
                        ),
                        name=f"Exit {trade.exit_reason}",
                        text=[f"Exit: {trade.exit_price:.2f}<br>Reason: {trade.exit_reason}<br>PnL: ${trade.pnl:.2f}"],
                        hovertemplate="%{text}<extra></extra>",
                        showlegend=(trade == self.trades[0])
                    )
                )
        
        # Update layout
        fig.update_layout(
            title="Asian Range Breakout Strategy - All Trades",
            xaxis_title="Time",
            yaxis_title="Price ($)",
            height=600,
            hovermode='x unified',
            showlegend=True
        )
        
        if output_file:
            fig.write_html(output_file)
            logger.info(f"Chart saved to {output_file}")
        
        return fig
    
    def plot_equity_curve(self, equity_curve: List[float], timestamps: Optional[List[pd.Timestamp]] = None, 
                         output_file: Optional[str] = None) -> go.Figure:
        """
        Plot equity curve
        
        Args:
            equity_curve: List of equity values
            timestamps: Optional list of timestamps (defaults to trade exit times)
            output_file: Optional file path to save HTML
            
        Returns:
            Plotly Figure object
        """
        if not equity_curve:
            logger.warning("No equity curve data")
            return None
        
        if timestamps is None:
            # Use trade exit times
            timestamps = [t.exit_time for t in self.trades]
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=equity_curve,
                mode='lines+markers',
                name="Equity",
                line=dict(color='blue', width=2),
                marker=dict(size=6)
            )
        )
        
        # Add horizontal line at initial equity
        if equity_curve:
            initial_equity = equity_curve[0]
            fig.add_hline(
                y=initial_equity,
                line_dash="dash",
                line_color="gray",
                annotation_text=f"Initial: ${initial_equity:,.2f}"
            )
        
        fig.update_layout(
            title="Equity Curve",
            xaxis_title="Date",
            yaxis_title="Equity ($)",
            height=400,
            hovermode='x unified'
        )
        
        if output_file:
            fig.write_html(output_file)
            logger.info(f"Equity curve saved to {output_file}")
        
        return fig

