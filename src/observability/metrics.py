"""
Metrics Exporter
Prometheus metrics for P&L, exposure, orders, etc.
"""
import logging
from prometheus_client import Counter, Gauge, Histogram
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Metrics
pnl_gauge = Gauge('strategy_pnl_total', 'Total P&L', ['type'])  # type: realized, unrealized, total
pnl_daily = Gauge('strategy_pnl_daily', 'Daily P&L')
drawdown_gauge = Gauge('strategy_drawdown', 'Current drawdown')
exposure_gauge = Gauge('strategy_exposure', 'Net exposure in dollars')
orders_gauge = Gauge('strategy_orders_open', 'Number of open orders', ['symbol'])
positions_gauge = Gauge('strategy_positions', 'Position size', ['symbol', 'side'])
atr_gauge = Gauge('strategy_atr', 'Current ATR', ['symbol'])
correlation_gauge = Gauge('strategy_correlation', 'Hedge correlation')
hedge_ratio_gauge = Gauge('strategy_hedge_ratio', 'Current hedge ratio')

trade_counter = Counter('strategy_trades_total', 'Total trades executed', ['symbol', 'side'])
order_counter = Counter('strategy_orders_total', 'Total orders placed', ['symbol', 'side', 'status'])


class MetricsExporter:
    """Exports metrics to Prometheus"""
    
    @staticmethod
    def update_pnl(realized: float, unrealized: float, total: float, daily: float):
        """Update P&L metrics"""
        pnl_gauge.labels(type='realized').set(realized)
        pnl_gauge.labels(type='unrealized').set(unrealized)
        pnl_gauge.labels(type='total').set(total)
        pnl_daily.set(daily)
    
    @staticmethod
    def update_drawdown(drawdown: float):
        """Update drawdown metric"""
        drawdown_gauge.set(drawdown)
    
    @staticmethod
    def update_exposure(exposure: float):
        """Update exposure metric"""
        exposure_gauge.set(exposure)
    
    @staticmethod
    def update_orders(symbol: str, count: int):
        """Update open orders count"""
        orders_gauge.labels(symbol=symbol).set(count)
    
    @staticmethod
    def update_position(symbol: str, side: str, quantity: int):
        """Update position metric"""
        positions_gauge.labels(symbol=symbol, side=side).set(quantity)
    
    @staticmethod
    def update_atr(symbol: str, atr: float):
        """Update ATR metric"""
        atr_gauge.labels(symbol=symbol).set(atr)
    
    @staticmethod
    def update_correlation(correlation: float):
        """Update correlation metric"""
        correlation_gauge.set(correlation)
    
    @staticmethod
    def update_hedge_ratio(ratio: float):
        """Update hedge ratio metric"""
        hedge_ratio_gauge.set(ratio)
    
    @staticmethod
    def record_trade(symbol: str, side: str):
        """Record a trade"""
        trade_counter.labels(symbol=symbol, side=side).inc()
    
    @staticmethod
    def record_order(symbol: str, side: str, status: str):
        """Record an order"""
        order_counter.labels(symbol=symbol, side=side, status=status).inc()

