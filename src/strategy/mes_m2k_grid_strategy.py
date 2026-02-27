"""
MES-M2K Bounded Spread Grid Strategy
Implements PRD in grid-new.txt for Topstep $50K Combine.
"""
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List, Tuple
from collections import deque
from zoneinfo import ZoneInfo
import yaml
import numpy as np

# Add project root and src to path
current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(os.path.dirname(current_file))
project_root = os.path.dirname(src_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, src_dir)

from connectors.topstepx_client import TopstepXClient
from connectors.market_data_adapter import MarketDataAdapter, Tick, Candle
from execution.order_client import OrderClient
from strategy.position_manager import PositionManager
from indicators.technical import calculate_correlation, calculate_trend_strength, calculate_volatility

logger = logging.getLogger(__name__)

CT_TZ = ZoneInfo("America/Chicago")


@dataclass
class Layer:
    level_index: int
    is_long_spread: bool
    entry_time: datetime
    entry_z: float
    entry_spread: float
    mes_qty: int
    m2k_qty: int
    entry_order_ids: Tuple[Optional[str], Optional[str]]
    open: bool = True
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None
    entry_pending: bool = True


class BarBuilder:
    """Builds fixed-interval bars from ticks."""

    def __init__(self, interval_seconds: int, max_bars: int):
        self.interval_seconds = interval_seconds
        self.bars = deque(maxlen=max_bars)
        self.current_bar: Optional[Candle] = None
        self.current_start: Optional[datetime] = None

    def _bar_start(self, ts: datetime) -> datetime:
        epoch = int(ts.timestamp())
        start_epoch = epoch - (epoch % self.interval_seconds)
        return datetime.fromtimestamp(start_epoch, tz=ts.tzinfo or timezone.utc)

    def update(self, tick: Tick) -> Optional[Candle]:
        ts = tick.timestamp
        bar_start = self._bar_start(ts)
        finished = None

        if self.current_start is None or bar_start != self.current_start:
            if self.current_bar:
                finished = self.current_bar
                self.bars.append(self.current_bar)
            self.current_start = bar_start
            self.current_bar = Candle(
                symbol=tick.symbol,
                open=tick.price,
                high=tick.price,
                low=tick.price,
                close=tick.price,
                volume=tick.volume or 0.0,
                timestamp=bar_start,
                interval=f"{self.interval_seconds}s"
            )
            return finished

        if self.current_bar:
            self.current_bar.high = max(self.current_bar.high, tick.price)
            self.current_bar.low = min(self.current_bar.low, tick.price)
            self.current_bar.close = tick.price
            self.current_bar.volume += tick.volume or 0.0
        return None


class MESM2KSpreadGridStrategy:
    """Bounded spread grid strategy with dynamic hedge ratio."""

    def __init__(self, config_path: str = "grid_new_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        env_dry_run = os.getenv("DRY_RUN", "").lower()
        if env_dry_run:
            self.dry_run = env_dry_run == "true"
        else:
            self.dry_run = self.config.get("dry_run", True)

        self.api_client = TopstepXClient(
            username=os.getenv("TOPSTEPX_USERNAME"),
            api_key=os.getenv("TOPSTEPX_API_KEY"),
            base_url=self.config.get("api_base_url", "https://api.topstepx.com"),
            user_hub_url=self.config.get("user_hub_url", "https://rtc.topstepx.com/hubs/user"),
            market_hub_url=self.config.get("market_hub_url", "https://rtc.topstepx.com/hubs/market"),
            dry_run=self.dry_run
        )

        # Symbols
        self.mes_symbol = self.config.get("mes_symbol", "MES")
        self.m2k_symbol = self.config.get("m2k_symbol", "M2K")

        # Risk and limits
        dll_amount = self.config.get("dll_amount")
        if dll_amount is None or float(dll_amount) <= 0:
            raise ValueError("dll_amount must be set explicitly from Topstep dashboard.")
        self.dll_amount = float(dll_amount)
        self.dll_buffer = float(self.config.get("dll_buffer", 200.0))
        self.mll_buffer = float(self.config.get("mll_buffer", 350.0))
        self.daily_profit_target = float(self.config.get("daily_profit_target", 700.0))
        self.max_daily_trades = int(self.config.get("max_daily_trades", 50))
        self.max_daily_entries = int(self.config.get("max_daily_entries", 20))
        self.max_gross_contracts_mes = int(self.config.get("max_gross_contracts_mes", 5))
        self.max_gross_contracts_m2k = int(self.config.get("max_gross_contracts_m2k", 10))
        self.max_layers = int(self.config.get("max_layers", 2))
        self.max_abs_z = float(self.config.get("max_abs_z", 2.2))
        self.reentry_cooldown_sec = int(self.config.get("reentry_cooldown_sec", 60))
        self.order_reject_limit = int(self.config.get("order_reject_limit", 3))
        self.entry_fill_timeout_sec = int(self.config.get("entry_fill_timeout_sec", 5))
        self.entry_order_check_seconds = int(self.config.get("entry_order_check_seconds", 2))
        self.flatten_order_type = self.config.get("flatten_order_type", "close_position")
        self.exit_order_type = self.config.get("exit_order_type", "aggressive_limit")
        self.aggressive_limit_ticks = int(self.config.get("aggressive_limit_ticks", 2))
        self.mes_tick_size = float(self.config.get("mes_tick_size", 0.25))
        self.m2k_tick_size = float(self.config.get("m2k_tick_size", 0.25))

        # Strategy parameters
        self.entry_levels = [float(x) for x in self.config.get("entry_levels", [1.3, 1.8])]
        self.z_takeprofit = float(self.config.get("z_takeprofit", 0.7))
        self.z_mid = float(self.config.get("z_mid", 0.0))
        self.z_stop = float(self.config.get("z_stop", 2.2))
        self.max_hold_seconds = int(self.config.get("max_hold_seconds", 720))
        self.mes_units_per_layer = int(self.config.get("mes_units_per_layer", 1))
        self.size_scale = float(self.config.get("size_scale", 1.0))

        # Beta and spread windows
        self.beta_window_minutes = int(self.config.get("beta_window_minutes", 60))
        self.spread_window_minutes = int(self.config.get("spread_window_minutes", 120))
        self.beta_min = float(self.config.get("beta_min", 0.5))
        self.beta_max = float(self.config.get("beta_max", 1.5))
        self.min_spread_std = float(self.config.get("min_spread_std", 0.01))
        self.warmup_prefill_from_1m = bool(self.config.get("warmup_prefill_from_1m", True))
        self.warmup_prefill_repeat = int(self.config.get("warmup_prefill_repeat", 60))

        # Regime filters
        self.corr_window = int(self.config.get("corr_window", 60))
        self.corr_min = float(self.config.get("corr_min", 0.5))
        self.trend_window = int(self.config.get("trend_window", 20))
        self.trend_threshold = float(self.config.get("trend_threshold", 2.0))
        self.vol_window = int(self.config.get("vol_window", 60))
        self.vol_threshold = float(self.config.get("vol_threshold", 0.015))

        # Time windows
        self.session_start = self.config.get("session_start", "08:30")
        self.session_end = self.config.get("session_end", "15:00")
        self.disable_after_open_minutes = int(self.config.get("disable_after_open_minutes", 5))
        self.disable_before_close_minutes = int(self.config.get("disable_before_close_minutes", 5))
        self.macro_no_trade_windows = self.config.get("macro_no_trade_windows", [])

        # Data and reconciliation
        self.data_stall_seconds = int(self.config.get("data_stall_seconds", 10))
        self.reconcile_interval_seconds = int(self.config.get("reconcile_interval_seconds", 30))
        self.account_poll_seconds = int(self.config.get("account_poll_seconds", 15))

        # Fee estimate
        self.fees_per_contract = float(self.config.get("fees_per_contract", 0.0))

        # Tick values (fallback if API doesn't provide P&L)
        self.mes_tick_value = float(self.config.get("mes_tick_value", 1.25))
        self.m2k_tick_value = float(self.config.get("m2k_tick_value", 0.5))
        self.mll_mode = self.config.get("mll_mode", "end_of_day")
        self.eod_cutoff_time_ct = self.config.get("eod_cutoff_time_ct", "15:00")

        self.position_manager = PositionManager(
            max_net_notional=self.config.get("max_net_notional", 10000.0),
            tick_values={self.mes_symbol: self.mes_tick_value, self.m2k_symbol: self.m2k_tick_value}
        )

        self.order_client = OrderClient(self.api_client, dry_run=self.dry_run)

        # State
        self.running = False
        self.paused = False
        self.state = "WARMUP"
        self.contract_id_mes: Optional[str] = None
        self.contract_id_m2k: Optional[str] = None
        self.symbol_by_contract: Dict[str, str] = {}

        self.last_tick_time: Optional[datetime] = None
        self.last_reconcile_time = datetime.now(timezone.utc)
        self.last_account_sync = datetime.now(timezone.utc)
        self.last_open_orders_check = datetime.now(timezone.utc)
        self.last_beta_update: Optional[datetime] = None
        self.last_entry_time: Optional[datetime] = None
        self.last_state: Optional[str] = None
        self.last_state_reason: Optional[str] = None
        self.last_entry_block_reason: Optional[str] = None
        self.last_entry_block_log_time: Optional[datetime] = None
        self.last_warmup_log_time: Optional[datetime] = None
        self.last_market_data_log_time: Optional[datetime] = None
        self.warmup_complete_logged: bool = False  # Track if warmup completion has been logged

        self.daily_entry_count = 0
        self.daily_trade_count = 0
        self.daily_contracts_traded = 0
        self.daily_start_equity: Optional[float] = None
        self.equity_high_watermark: Optional[float] = None
        self.daily_equity_high: Optional[float] = None
        self.locked_until: Optional[datetime] = None
        self.order_reject_count = 0

        self.current_price_mes: Optional[float] = None
        self.current_price_m2k: Optional[float] = None
        self.current_beta: Optional[float] = None
        self.current_spread: Optional[float] = None
        self.current_z: Optional[float] = None
        self.current_spread_mean: Optional[float] = None
        self.current_spread_std: Optional[float] = None

        self.spread_history = deque(maxlen=self.spread_window_minutes * 60)
        self.layers: List[Layer] = []

        # Bar builders
        self.mes_1s = BarBuilder(interval_seconds=1, max_bars=7200)
        self.m2k_1s = BarBuilder(interval_seconds=1, max_bars=7200)
        self.mes_1m = BarBuilder(interval_seconds=60, max_bars=3000)
        self.m2k_1m = BarBuilder(interval_seconds=60, max_bars=3000)

        logger.info(f"MES-M2K Spread Grid initialized (dry_run={self.dry_run})")

    def _log_state_change(self, state: str, reason: str):
        if state != self.last_state or reason != self.last_state_reason:
            logger.info(f"STATE={state} reason={reason}")
            self.last_state = state
            self.last_state_reason = reason

    def _log_entry_block(self, reason: str):
        now = datetime.now(timezone.utc)
        if self.last_entry_block_log_time is None:
            should_log = True
        else:
            elapsed = (now - self.last_entry_block_log_time).total_seconds()
            should_log = elapsed >= 15 or reason != self.last_entry_block_reason
        if should_log:
            logger.info(f"ENTRY_BLOCKED reason={reason}")
            self.last_entry_block_reason = reason
            self.last_entry_block_log_time = now

    def _log_warmup_status(self):
        now = datetime.now(timezone.utc)
        if self.last_warmup_log_time is not None:
            if (now - self.last_warmup_log_time).total_seconds() < 15:
                return
        needed = self.spread_window_minutes * 60
        have = len(self.spread_history)
        std = self.current_spread_std
        beta = self.current_beta
        msg = f"WARMUP_PROGRESS spread_points={have}/{needed}"
        if beta is not None:
            msg += f" beta={beta:.4f}"
        if std is not None:
            msg += f" std={std:.5f} min_std={self.min_spread_std:.5f}"
        logger.info(msg)
        self.last_warmup_log_time = now

    def _log_market_data_issue(self, symbol: str, reason: str, data: Dict[str, Any]):
        now = datetime.now(timezone.utc)
        if self.last_market_data_log_time is not None:
            if (now - self.last_market_data_log_time).total_seconds() < 15:
                return
        keys = ",".join(sorted(data.keys())) if isinstance(data, dict) else str(type(data))
        logger.info(f"MARKET_DATA_IGNORED symbol={symbol} reason={reason} keys={keys}")
        self.last_market_data_log_time = now

    def initialize(self) -> bool:
        logger.info("Initializing MES-M2K Spread Grid strategy...")

        if not self.api_client.authenticate():
            logger.error("Failed to authenticate with TopstepX")
            return False

        accounts = self.api_client.get_accounts()
        if not accounts:
            logger.error("No accounts found")
            return False

        account = self._select_account(accounts)
        if not account:
            return False

        account_id = account.get("id") or account.get("accountId")
        if not account_id:
            logger.error("Selected account has no ID")
            return False
        self.api_client.set_account(account_id)
        logger.info(f"Using account: {account.get('name', 'Unknown')} (ID: {account_id})")

        contracts_mes = self.api_client.search_contracts(self.mes_symbol)
        contracts_m2k = self.api_client.search_contracts(self.m2k_symbol)
        if not contracts_mes or not contracts_m2k:
            logger.error("Failed to find MES/M2K contracts")
            return False

        self.contract_id_mes = contracts_mes[0].get("contractId") or contracts_mes[0].get("id")
        self.contract_id_m2k = contracts_m2k[0].get("contractId") or contracts_m2k[0].get("id")
        self.symbol_by_contract[str(self.contract_id_mes)] = self.mes_symbol
        self.symbol_by_contract[str(self.contract_id_m2k)] = self.m2k_symbol

        self._load_historical_bars()
        self._setup_realtime_callbacks()

        self.api_client.connect_realtime(
            account_id=self.api_client.account_id,
            contract_ids=[self.contract_id_mes, self.contract_id_m2k]
        )
        return True

    def _select_account(self, accounts: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        prefer_practice = self.config.get("prefer_practice_account", True)
        specified_account_id = self.config.get("account_id")

        if specified_account_id:
            for acc in accounts:
                acc_id = acc.get("id") or acc.get("accountId")
                if acc_id == specified_account_id:
                    return acc

        if prefer_practice:
            practice = [a for a in accounts if a.get("simulated") or "PRAC" in (a.get("name", "").upper())]
            if practice:
                return practice[0]

        return accounts[0] if accounts else None

    def _load_historical_bars(self):
        logger.info("Loading historical 1m bars for beta/spread warmup...")
        bars_mes = self.api_client.get_bars(self.contract_id_mes, interval="1m", limit=2000)
        bars_m2k = self.api_client.get_bars(self.contract_id_m2k, interval="1m", limit=2000)
        if not bars_mes or not bars_m2k:
            logger.warning("No historical bars loaded")
            return

        adapter = MarketDataAdapter()
        mes_candles = adapter.normalize_bars(bars_mes, self.mes_symbol, "1m")
        m2k_candles = adapter.normalize_bars(bars_m2k, self.m2k_symbol, "1m")

        # Align by timestamp
        ts_to_mes = {c.timestamp: c for c in mes_candles}
        ts_to_m2k = {c.timestamp: c for c in m2k_candles}
        for ts in sorted(set(ts_to_mes.keys()) & set(ts_to_m2k.keys())):
            self.mes_1m.bars.append(ts_to_mes[ts])
            self.m2k_1m.bars.append(ts_to_m2k[ts])

        if self.mes_1m.bars:
            self.current_price_mes = self.mes_1m.bars[-1].close
            self.position_manager.update_price(self.mes_symbol, self.current_price_mes)
        if self.m2k_1m.bars:
            self.current_price_m2k = self.m2k_1m.bars[-1].close
            self.position_manager.update_price(self.m2k_symbol, self.current_price_m2k)

        if self.warmup_prefill_from_1m:
            self._prefill_spread_history_from_1m()

    def _prefill_spread_history_from_1m(self):
        if not self.mes_1m.bars or not self.m2k_1m.bars:
            return
        if self.warmup_prefill_repeat <= 0:
            return

        if self.current_beta is None:
            self.current_beta = self._calculate_beta()
        if self.current_beta is None:
            return

        mes_bars = list(self.mes_1m.bars)[-self.spread_window_minutes:]
        m2k_bars = list(self.m2k_1m.bars)[-self.spread_window_minutes:]
        if len(mes_bars) != len(m2k_bars) or not mes_bars:
            return

        for mes_bar, m2k_bar in zip(mes_bars, m2k_bars):
            spread = m2k_bar.close - self.current_beta * mes_bar.close
            for _ in range(self.warmup_prefill_repeat):
                self.spread_history.append(spread)

        history = np.array(self.spread_history)
        if history.size == 0:
            return
        mean = float(np.mean(history))
        std = float(np.std(history))
        self.current_spread = history[-1]
        self.current_spread_mean = mean
        self.current_spread_std = std
        if std > self.min_spread_std:
            self.current_z = (self.current_spread - mean) / std
            if not self.warmup_complete_logged:
                logger.info(f"✅ WARMUP_PREFILL COMPLETE: spread_points={len(self.spread_history)} beta={self.current_beta:.4f} std={std:.5f} > {self.min_spread_std:.5f}, z={self.current_z:.2f}")
                self.warmup_complete_logged = True
        else:
            if not self.warmup_complete_logged:
                logger.warning(f"⚠️  WARMUP_PREFILL BLOCKED: spread_points={len(self.spread_history)} std={std:.5f} <= {self.min_spread_std:.5f} (spread volatility too low - need more market movement)")
                self.warmup_complete_logged = True  # Log once even if blocked

    def _setup_realtime_callbacks(self):
        def on_market_quote_update(data):
            self._handle_market_data(data)

        def on_market_trade_update(data):
            self._handle_market_data(data)

        def on_trade_update(data):
            self._handle_trade_update(data)

        def on_order_update(data):
            self._handle_order_update(data)

        # Market hub GatewayQuote triggers "on_quote_update" in TopstepXClient
        self.api_client.register_realtime_callback("on_quote_update", on_market_quote_update)
        self.api_client.register_realtime_callback("on_market_trade_update", on_market_trade_update)
        self.api_client.register_realtime_callback("on_trade_update", on_trade_update)
        self.api_client.register_realtime_callback("on_order_update", on_order_update)

    def _handle_market_data(self, data: Dict[str, Any]):
        try:
            contract_id = str(data.get("contractId") or data.get("contract_id") or "")
            symbol = self.symbol_by_contract.get(contract_id) or data.get("symbol")
            if isinstance(symbol, str):
                if symbol.endswith(f".{self.mes_symbol}"):
                    symbol = self.mes_symbol
                elif symbol.endswith(f".{self.m2k_symbol}"):
                    symbol = self.m2k_symbol
            if not symbol:
                symbol_id = data.get("symbolId")
                if isinstance(symbol_id, str):
                    if symbol_id.endswith(f".{self.mes_symbol}"):
                        symbol = self.mes_symbol
                    elif symbol_id.endswith(f".{self.m2k_symbol}"):
                        symbol = self.m2k_symbol
            if symbol not in [self.mes_symbol, self.m2k_symbol]:
                self._log_market_data_issue(str(symbol), "SYMBOL_MISMATCH", data)
                return

            tick = MarketDataAdapter.normalize_tick(data, symbol)
            if not tick:
                self._log_market_data_issue(symbol, "NO_TICK", data)
                return
            if tick.price <= 0:
                self._log_market_data_issue(symbol, "BAD_PRICE", data)
                return

            self.last_tick_time = datetime.now(timezone.utc)

            if symbol == self.mes_symbol:
                self.current_price_mes = tick.price
                self.position_manager.update_price(self.mes_symbol, tick.price)
                finished_1s = self.mes_1s.update(tick)
                finished_1m = self.mes_1m.update(tick)
            else:
                self.current_price_m2k = tick.price
                self.position_manager.update_price(self.m2k_symbol, tick.price)
                finished_1s = self.m2k_1s.update(tick)
                finished_1m = self.m2k_1m.update(tick)

            if finished_1m is not None:
                self._maybe_update_beta()

            if finished_1s is not None:
                self._update_spread_stats()

        except Exception as e:
            logger.error(f"Error handling market data: {e}")

    def _handle_trade_update(self, data: Dict[str, Any]):
        try:
            contract_id = str(data.get("contractId") or data.get("contract_id") or "")
            symbol = self.symbol_by_contract.get(contract_id) or data.get("symbol")
            if symbol not in [self.mes_symbol, self.m2k_symbol]:
                return

            side_raw = data.get("side")
            side = "BUY" if str(side_raw).upper() in ["BUY", "0"] or side_raw == 0 else "SELL"
            qty = int(data.get("quantity") or data.get("size") or 0)
            price = float(data.get("price") or data.get("fillPrice") or 0.0)

            if qty > 0 and price > 0:
                self.position_manager.on_fill(symbol, side, qty, price)
                self.daily_contracts_traded += qty
                self.daily_trade_count += 1
        except Exception as e:
            logger.debug(f"Trade update parse error: {e}")

    def _handle_order_update(self, data: Dict[str, Any]):
        try:
            status = data.get("status") or data.get("orderStatus")
            if isinstance(status, str) and status.upper() in ["REJECTED", "CANCELLED", "FAILED"]:
                self.order_reject_count += 1
            elif isinstance(status, int) and status in [3, 4]:
                self.order_reject_count += 1
        except Exception:
            return

    def _maybe_update_beta(self):
        now = datetime.now(timezone.utc)
        if self.last_beta_update and (now - self.last_beta_update) < timedelta(minutes=1):
            return
        self.last_beta_update = now
        beta = self._calculate_beta()
        if beta is not None:
            self.current_beta = beta

    def _calculate_beta(self) -> Optional[float]:
        if len(self.mes_1m.bars) < self.beta_window_minutes + 1:
            return None
        if len(self.m2k_1m.bars) < self.beta_window_minutes + 1:
            return None

        mes = list(self.mes_1m.bars)[-self.beta_window_minutes - 1:]
        m2k = list(self.m2k_1m.bars)[-self.beta_window_minutes - 1:]
        if len(mes) != len(m2k):
            return None

        mes_returns = []
        m2k_returns = []
        for i in range(1, len(mes)):
            mes_returns.append((mes[i].close - mes[i - 1].close) / mes[i - 1].close)
            m2k_returns.append((m2k[i].close - m2k[i - 1].close) / m2k[i - 1].close)

        if len(mes_returns) < 2:
            return None

        x = np.array(mes_returns)
        y = np.array(m2k_returns)
        var_x = np.var(x)
        if var_x == 0:
            return None
        cov = np.mean((x - np.mean(x)) * (y - np.mean(y)))
        beta = cov / var_x
        beta = max(self.beta_min, min(self.beta_max, beta))
        return beta

    def _update_spread_stats(self):
        if self.current_beta is None:
            return
        if self.current_price_mes is None or self.current_price_m2k is None:
            return

        spread = self.current_price_m2k - self.current_beta * self.current_price_mes
        self.spread_history.append(spread)
        if len(self.spread_history) < self.spread_window_minutes * 60:
            return

        history = np.array(self.spread_history)
        mean = float(np.mean(history))
        std = float(np.std(history))
        if std <= self.min_spread_std:
            self.current_z = None
            self.current_spread = spread
            self.current_spread_mean = mean
            self.current_spread_std = std
            # Only log warmup blocked once
            if self.state == "WARMUP" and len(self.spread_history) >= self.spread_window_minutes * 60 and not self.warmup_complete_logged:
                logger.warning(f"⚠️  WARMUP BLOCKED: std={std:.5f} <= {self.min_spread_std:.5f} (spread volatility too low - need more market movement)")
                self.warmup_complete_logged = True  # Log once even if blocked
            return

        self.current_spread = spread
        self.current_spread_mean = mean
        self.current_spread_std = std
        self.current_z = (spread - mean) / std
        # Only log warmup completion once
        if self.state == "WARMUP" and len(self.spread_history) >= self.spread_window_minutes * 60 and not self.warmup_complete_logged:
            logger.info(f"✅ WARMUP COMPLETE: std={std:.5f} > {self.min_spread_std:.5f}, beta={self.current_beta:.4f}, z={self.current_z:.2f}")
            self.warmup_complete_logged = True

    def _filters_pass_with_reason(self) -> Tuple[bool, str]:
        mes_bars = list(self.mes_1m.bars)
        m2k_bars = list(self.m2k_1m.bars)
        corr = calculate_correlation(mes_bars, m2k_bars, window=self.corr_window)
        if corr is None:
            return False, "FILTER_CORR_UNAVAILABLE"
        if corr < self.corr_min:
            return False, f"FILTER_CORR_LOW corr={corr:.2f} min={self.corr_min:.2f}"

        trend = calculate_trend_strength(mes_bars, window=self.trend_window)
        if trend is not None and abs(trend) > self.trend_threshold:
            return False, f"FILTER_TREND_STRONG trend={trend:.2f} thresh={self.trend_threshold:.2f}"

        vol = calculate_volatility(mes_bars, window=self.vol_window)
        if vol is not None and vol > self.vol_threshold:
            return False, f"FILTER_VOL_HIGH vol={vol:.5f} thresh={self.vol_threshold:.5f}"

        return True, "FILTERS_OK"

    def _in_trade_window(self, now_ct: datetime) -> bool:
        session_start = datetime.combine(now_ct.date(), datetime.strptime(self.session_start, "%H:%M").time(), tzinfo=CT_TZ)
        session_end = datetime.combine(now_ct.date(), datetime.strptime(self.session_end, "%H:%M").time(), tzinfo=CT_TZ)
        if now_ct < session_start or now_ct > session_end:
            return False

        if (now_ct - session_start) < timedelta(minutes=self.disable_after_open_minutes):
            return False
        if (session_end - now_ct) < timedelta(minutes=self.disable_before_close_minutes):
            return False

        for window in self.macro_no_trade_windows:
            start_str = window.get("start")
            end_str = window.get("end")
            try:
                start_dt = datetime.fromisoformat(start_str).replace(tzinfo=CT_TZ)
                end_dt = datetime.fromisoformat(end_str).replace(tzinfo=CT_TZ)
                if start_dt <= now_ct <= end_dt:
                    return False
            except Exception:
                continue

        return True

    def _should_reset_daily(self, now_ct: datetime) -> bool:
        reset_time = datetime.combine(now_ct.date(), datetime.strptime("17:00", "%H:%M").time(), tzinfo=CT_TZ)
        if now_ct >= reset_time and (self.locked_until is None or self.locked_until.date() < now_ct.date()):
            return True
        return False

    def _reset_daily(self, now_ct: datetime):
        self.daily_entry_count = 0
        self.daily_trade_count = 0
        self.daily_contracts_traded = 0
        self.daily_start_equity = self._get_account_equity()
        self.daily_equity_high = self.daily_start_equity
        self.locked_until = None
        self.state = "WARMUP"
        logger.info(f"Daily reset at {now_ct.isoformat()}")

    def _get_account_equity(self) -> Optional[float]:
        accounts = self.api_client.get_accounts()
        if not accounts:
            return None
        account_id = self.api_client.account_id
        account = next((a for a in accounts if (a.get("id") or a.get("accountId")) == account_id), None)
        if not account:
            return None
        balance = account.get("balance")
        return float(balance) if balance is not None else None

    def _update_mll_watermark(self, now_ct: datetime, equity: float):
        if self.mll_mode == "intraday":
            if self.equity_high_watermark is None or equity > self.equity_high_watermark:
                self.equity_high_watermark = equity
            return

        # end_of_day: only update watermark at cutoff time
        if self.daily_equity_high is None or equity > self.daily_equity_high:
            self.daily_equity_high = equity

        cutoff_time = datetime.strptime(self.eod_cutoff_time_ct, "%H:%M").time()
        cutoff = datetime.combine(now_ct.date(), cutoff_time, tzinfo=CT_TZ)
        if now_ct >= cutoff:
            if self.daily_equity_high is not None:
                if self.equity_high_watermark is None or self.daily_equity_high > self.equity_high_watermark:
                    self.equity_high_watermark = self.daily_equity_high

    def _get_intraday_pnl(self) -> float:
        realized = self.position_manager.get_realized_pnl()
        unrealized = self.position_manager.get_unrealized_pnl()
        fees = self.daily_contracts_traded * self.fees_per_contract
        internal_pnl = realized + unrealized - fees

        api_total = self.position_manager.api_total_pnl
        if api_total:
            api_pnl = sum(api_total.values())
            # Use the more conservative (lower) of API vs internal
            return min(api_pnl, internal_pnl)

        return internal_pnl

    def _risk_governor(self, now_ct: datetime) -> Optional[str]:
        equity = self._get_account_equity()
        if equity is None:
            return "PNL_UNAVAILABLE"

        self._update_mll_watermark(now_ct, equity)

        if self.daily_start_equity is None:
            self.daily_start_equity = equity

        intraday_pnl = self._get_intraday_pnl()

        if self.equity_high_watermark is None:
            self.equity_high_watermark = equity
        mll_floor = self.equity_high_watermark - 2000.0
        if equity <= mll_floor + self.mll_buffer:
            return "MLL_NEAR_BREACH"

        if intraday_pnl <= -(self.dll_amount - self.dll_buffer):
            return "DLL_NEAR_BREACH"

        if intraday_pnl >= self.daily_profit_target:
            return "DAILY_PROFIT_TARGET"

        if self.daily_trade_count >= self.max_daily_trades:
            return "MAX_DAILY_TRADES"

        if self.daily_entry_count >= self.max_daily_entries:
            return "MAX_DAILY_ENTRIES"

        return None

    def _enter_layer(self, level_index: int, is_long: bool):
        if self.current_beta is None or self.current_z is None or self.current_spread is None:
            self._log_entry_block("NO_SPREAD_STATS")
            return
        if self.current_price_mes is None or self.current_price_m2k is None:
            self._log_entry_block("NO_LIVE_PRICES")
            return

        if self.last_entry_time and (datetime.now(timezone.utc) - self.last_entry_time).total_seconds() < self.reentry_cooldown_sec:
            self._log_entry_block("REENTRY_COOLDOWN")
            return

        if len([l for l in self.layers if l.open]) >= self.max_layers:
            self._log_entry_block("MAX_LAYERS")
            return

        mes_qty = self.mes_units_per_layer
        m2k_qty = max(1, int(round(self.current_beta * mes_qty * self.size_scale)))

        # Enforce gross contract caps
        gross_mes = abs(self.position_manager.get_net_position(self.mes_symbol)) + mes_qty
        gross_m2k = abs(self.position_manager.get_net_position(self.m2k_symbol)) + m2k_qty
        if gross_mes > self.max_gross_contracts_mes or gross_m2k > self.max_gross_contracts_m2k:
            self._log_entry_block("GROSS_CONTRACT_CAP")
            return

        if is_long:
            side_m2k = "BUY"
            side_mes = "SELL"
        else:
            side_m2k = "SELL"
            side_mes = "BUY"

        order_id_mes = self.order_client.place_limit_order(
            contract_id=self.contract_id_mes,
            side=side_mes,
            quantity=mes_qty,
            price=self.current_price_mes
        )
        order_id_m2k = self.order_client.place_limit_order(
            contract_id=self.contract_id_m2k,
            side=side_m2k,
            quantity=m2k_qty,
            price=self.current_price_m2k
        )

        if not order_id_mes or not order_id_m2k:
            self.order_reject_count += 1
            self._log_entry_block("ENTRY_ORDER_REJECTED")
            return

        layer = Layer(
            level_index=level_index,
            is_long_spread=is_long,
            entry_time=datetime.now(timezone.utc),
            entry_z=float(self.current_z),
            entry_spread=float(self.current_spread),
            mes_qty=mes_qty,
            m2k_qty=m2k_qty,
            entry_order_ids=(order_id_mes, order_id_m2k)
        )
        self.layers.append(layer)
        self.daily_entry_count += 1
        self.last_entry_time = datetime.now(timezone.utc)
        logger.info(f"ENTER layer {level_index} {'LONG' if is_long else 'SHORT'} z={self.current_z:.2f}")

    def _place_exit_order(
        self,
        contract_id: str,
        side: str,
        quantity: int,
        price: float,
        tick_size: float
    ) -> Optional[str]:
        if self.exit_order_type == "market":
            return self.order_client.place_market_order(
                contract_id=contract_id,
                side=side,
                quantity=quantity
            )
        if self.exit_order_type == "aggressive_limit":
            limit_price = price
            if side.upper() == "BUY":
                limit_price = price + (self.aggressive_limit_ticks * tick_size)
            else:
                limit_price = price - (self.aggressive_limit_ticks * tick_size)
            return self.order_client.place_limit_order(
                contract_id=contract_id,
                side=side,
                quantity=quantity,
                price=limit_price
            )
        return self.order_client.place_limit_order(
            contract_id=contract_id,
            side=side,
            quantity=quantity,
            price=price
        )

    def _close_layer(self, layer: Layer, reason: str):
        if not layer.open:
            return
        if layer.is_long_spread:
            side_m2k = "SELL"
            side_mes = "BUY"
        else:
            side_m2k = "BUY"
            side_mes = "SELL"

        order_id_mes = self._place_exit_order(
            contract_id=self.contract_id_mes,
            side=side_mes,
            quantity=layer.mes_qty,
            price=self.current_price_mes or 0.0,
            tick_size=self.mes_tick_size
        )
        order_id_m2k = self._place_exit_order(
            contract_id=self.contract_id_m2k,
            side=side_m2k,
            quantity=layer.m2k_qty,
            price=self.current_price_m2k or 0.0,
            tick_size=self.m2k_tick_size
        )
        if order_id_mes and order_id_m2k:
            layer.open = False
            layer.exit_time = datetime.now(timezone.utc)
            layer.exit_reason = reason
            self.daily_trade_count += 1
            logger.info(f"EXIT layer {layer.level_index} reason={reason}")
        else:
            self.order_reject_count += 1

    def _process_layers(self):
        if self.current_z is None:
            return
        z = self.current_z
        now = datetime.now(timezone.utc)

        # LIFO exit
        for layer in sorted([l for l in self.layers if l.open], key=lambda l: l.entry_time, reverse=True):
            age = (now - layer.entry_time).total_seconds()
            if abs(z) <= self.z_takeprofit:
                self._close_layer(layer, "TAKE_PROFIT")
                continue
            if layer.is_long_spread and z >= self.z_mid:
                self._close_layer(layer, "Z_MID")
                continue
            if not layer.is_long_spread and z <= -self.z_mid:
                self._close_layer(layer, "Z_MID")
                continue
            if abs(z) >= self.z_stop:
                self._close_layer(layer, "Z_STOP")
                continue
            if age >= self.max_hold_seconds:
                self._close_layer(layer, "TIME_STOP")

    def _process_entries(self):
        if self.current_z is None:
            self._log_entry_block("Z_UNAVAILABLE")
            return
        if any(l.entry_pending for l in self.layers if l.open):
            self._log_entry_block("ENTRY_PENDING")
            return
        z = self.current_z
        if abs(z) > self.max_abs_z:
            self._log_entry_block(f"Z_TOO_FAR z={z:.2f} max_abs_z={self.max_abs_z:.2f}")
            return

        for idx, level in enumerate(self.entry_levels, start=1):
            if abs(z) >= level:
                exists = any(l.open and l.level_index == idx and l.is_long_spread == (z < 0) for l in self.layers)
                if exists:
                    continue
                self._enter_layer(idx, is_long=(z < 0))
                return
        if self.entry_levels:
            self._log_entry_block(f"Z_BELOW_ENTRY z={z:.2f} min_level={min(self.entry_levels):.2f}")

    def _flatten_all(self, reason: str):
        logger.warning(f"FLATTEN: {reason}")
        # Cancel all open orders first
        open_orders = self.api_client.get_open_orders(self.api_client.account_id)
        open_ids = [str(order.get("orderId") or order.get("id")) for order in open_orders if order]
        if open_ids:
            self.order_client.cancel_all_orders(open_ids)

        # Close positions at the broker
        if self.flatten_order_type == "close_position":
            positions = self.api_client.get_positions(self.api_client.account_id)
            for pos in positions:
                contract_id = pos.get("contractId")
                if contract_id:
                    self.api_client.close_position(contract_id)
        else:
            positions = self.api_client.get_positions(self.api_client.account_id)
            for pos in positions:
                contract_id = pos.get("contractId")
                size = pos.get("size", 0)
                pos_type = pos.get("type", 0)  # 1=Long, 2=Short
                if not contract_id or size <= 0:
                    continue
                side = "SELL" if pos_type == 1 else "BUY"
                if self.flatten_order_type == "market":
                    self.order_client.place_market_order(
                        contract_id=contract_id,
                        side=side,
                        quantity=size
                    )
                else:
                    # aggressive limit
                    tick_size = self.mes_tick_size if self.symbol_by_contract.get(str(contract_id)) == self.mes_symbol else self.m2k_tick_size
                    price = self.current_price_mes if self.symbol_by_contract.get(str(contract_id)) == self.mes_symbol else self.current_price_m2k
                    if price is None:
                        self.order_client.place_market_order(
                            contract_id=contract_id,
                            side=side,
                            quantity=size
                        )
                    else:
                        limit_price = price + (self.aggressive_limit_ticks * tick_size) if side == "BUY" else price - (self.aggressive_limit_ticks * tick_size)
                        self.order_client.place_limit_order(
                            contract_id=contract_id,
                            side=side,
                            quantity=size,
                            price=limit_price
                        )

        self.position_manager.flatten_all()

    def _update_state(self):
        now_utc = datetime.now(timezone.utc)
        now_ct = now_utc.astimezone(CT_TZ)

        if self._should_reset_daily(now_ct):
            self._reset_daily(now_ct)

        # Data stall
        if self.last_tick_time and (now_utc - self.last_tick_time).total_seconds() > self.data_stall_seconds:
            self.state = "LOCKED"
            self.locked_until = self._next_reset_time(now_ct)
            self._flatten_all("DATA_STALL")
            self._log_state_change(self.state, "DATA_STALL")
            return

        # Order reject guard
        if self.order_reject_count >= self.order_reject_limit:
            self.state = "ERROR"
            self._flatten_all("ORDER_REJECT")
            self._log_state_change(self.state, "ORDER_REJECT")
            return

        # Warmup
        if self.current_beta is None or self.current_z is None or self.current_spread_std is None:
            self.state = "WARMUP"
            missing = []
            if self.current_beta is None:
                missing.append("beta")
            if self.current_z is None:
                missing.append("z-score")
            if self.current_spread_std is None:
                missing.append("spread_std")
            self._log_state_change(self.state, f"WARMUP_MISSING: {', '.join(missing)}")
            self._log_warmup_status()
            return
        if self.current_spread_std <= self.min_spread_std:
            self.state = "WARMUP"
            self._log_state_change(self.state, f"WARMUP_STD std={self.current_spread_std:.5f} min={self.min_spread_std:.5f} (volatility too low)")
            self._log_warmup_status()
            return

        # Locked state
        if self.locked_until and now_ct < self.locked_until:
            self.state = "LOCKED"
            self._log_state_change(self.state, f"LOCKED_UNTIL {self.locked_until.isoformat()}")
            return

        risk_reason = self._risk_governor(now_ct)
        if risk_reason == "PNL_UNAVAILABLE":
            self.state = "ERROR"
            self._flatten_all("PNL_UNAVAILABLE")
            self._log_state_change(self.state, "PNL_UNAVAILABLE")
            return
        if risk_reason in ["MLL_NEAR_BREACH", "DLL_NEAR_BREACH"]:
            self.state = "LOCKED"
            self.locked_until = self._next_reset_time(now_ct)
            self._flatten_all(risk_reason)
            self._log_state_change(self.state, risk_reason)
            return
        if risk_reason in ["DAILY_PROFIT_TARGET", "MAX_DAILY_TRADES", "MAX_DAILY_ENTRIES"]:
            self.state = "LOCKED"
            self.locked_until = self._next_reset_time(now_ct)
            self._flatten_all(risk_reason)
            self._log_state_change(self.state, risk_reason)
            return

        filters_ok, filters_reason = self._filters_pass_with_reason()
        if not filters_ok:
            self.state = "RISK_OFF"
            self._log_state_change(self.state, filters_reason)
            return

        if self.current_z is not None and abs(self.current_z) >= self.max_abs_z:
            self.state = "MANAGE_ONLY"
            self._log_state_change(self.state, f"Z_TOO_FAR z={self.current_z:.2f} max_abs_z={self.max_abs_z:.2f}")
            return

        if not self._in_trade_window(now_ct):
            self.state = "RISK_OFF"
            self._log_state_change(self.state, "OUTSIDE_TRADE_WINDOW")
            return

        self.state = "RISK_ON"
        self._log_state_change(self.state, "OK")

    def _next_reset_time(self, now_ct: datetime) -> datetime:
        reset = datetime.combine(now_ct.date(), datetime.strptime("17:00", "%H:%M").time(), tzinfo=CT_TZ)
        if now_ct >= reset:
            reset = reset + timedelta(days=1)
        return reset

    def _reconcile_positions(self):
        positions = self.api_client.get_positions(self.api_client.account_id)
        if positions:
            self.position_manager.reconcile_with_api_positions(positions)

    def _check_entry_orders(self):
        now = datetime.now(timezone.utc)
        open_orders = self.api_client.get_open_orders(self.api_client.account_id)
        open_ids = {str(order.get("orderId") or order.get("id")) for order in open_orders if order}

        for layer in [l for l in self.layers if l.open and l.entry_pending]:
            mes_id, m2k_id = layer.entry_order_ids
            mes_open = mes_id and str(mes_id) in open_ids
            m2k_open = m2k_id and str(m2k_id) in open_ids

            if not mes_open and not m2k_open:
                layer.entry_pending = False
                continue

            elapsed = (now - layer.entry_time).total_seconds()
            if elapsed < self.entry_fill_timeout_sec:
                continue

            # If both legs are still open after timeout, cancel and drop layer
            if mes_open and m2k_open:
                if mes_id:
                    self.order_client.cancel_order(str(mes_id))
                if m2k_id:
                    self.order_client.cancel_order(str(m2k_id))
                layer.open = False
                layer.entry_pending = False
                layer.exit_reason = "ENTRY_TIMEOUT"
                logger.warning("Entry timeout: canceled both legs")
                continue

            # One leg filled, one still open -> hedge/flatten
            if mes_open != m2k_open:
                if mes_open and mes_id:
                    self.order_client.cancel_order(str(mes_id))
                if m2k_open and m2k_id:
                    self.order_client.cancel_order(str(m2k_id))
                logger.warning("Entry hedge timeout: one leg filled, flattening")
                self._flatten_all("ENTRY_HEDGE_TIMEOUT")
                self.state = "MANAGE_ONLY"
                return

    def run(self):
        if not self.initialize():
            logger.error("Failed to initialize strategy")
            return

        self.running = True
        logger.info("Starting MES-M2K Spread Grid loop...")

        try:
            while self.running:
                now = datetime.now(timezone.utc)
                if (now - self.last_reconcile_time).total_seconds() >= self.reconcile_interval_seconds:
                    self._reconcile_positions()
                    self.last_reconcile_time = now

                if (now - self.last_account_sync).total_seconds() >= self.account_poll_seconds:
                    self._get_account_equity()
                    self.last_account_sync = now

                if (now - self.last_open_orders_check).total_seconds() >= self.entry_order_check_seconds:
                    self._check_entry_orders()
                    self.last_open_orders_check = now

                self._update_state()

                if self.state in ["LOCKED", "ERROR", "FLATTEN"]:
                    time.sleep(1)
                    continue

                if self.state in ["RISK_OFF", "MANAGE_ONLY"]:
                    self._process_layers()
                elif self.state == "RISK_ON":
                    self._process_layers()
                    self._process_entries()

                time.sleep(1)

        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        except Exception as e:
            logger.error(f"Error in strategy loop: {e}", exc_info=True)
        finally:
            self._flatten_all("SHUTDOWN")
            self.api_client.disconnect()
            self.running = False

