"""
Value Area Strategy for MES
Session-aware strategy using prior-session VAH/VAL/POC for rejection and acceptance setups.
"""
import os
import sys
import time
import logging
import yaml
from dataclasses import dataclass
from datetime import datetime, timedelta, time as dtime
from collections import defaultdict
from typing import Optional, Dict, Any, List, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

# Get absolute paths
current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(os.path.dirname(current_file))
project_root = os.path.dirname(src_dir)

sys.path.insert(0, project_root)
sys.path.insert(0, src_dir)

from connectors.topstepx_client import TopstepXClient
from connectors.market_data_adapter import MarketDataAdapter
from strategy.position_manager import PositionManager
from strategy.risk_manager import RiskManager
from execution.order_client import OrderClient

logger = logging.getLogger(__name__)


@dataclass
class SessionLevels:
    session_date: datetime
    poc: float
    vah: float
    val: float


@dataclass
class ValueAreaPosition:
    side: str  # LONG or SHORT
    entry_price: float
    stop_price: float
    target_price: float
    quantity: int
    entry_time: datetime
    setup_type: str  # rejection or acceptance


class ValueAreaMESStrategy:
    """MES Value Area strategy with funded-account-safe controls."""

    def __init__(self, config_path: str = "value_area_mes_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        env_dry_run = os.getenv("DRY_RUN", "").lower()
        self.dry_run = env_dry_run == "true" if env_dry_run else self.config.get("dry_run", True)

        self.symbol = self.config.get("instrument", "MES")
        self.interval_fast = self.config.get("fast_interval", "1m")
        self.interval_slow = self.config.get("slow_interval", "5m")
        self.tick_size = float(self.config.get("tick_size", 0.25))
        self.tick_value = float(self.config.get("tick_value", 1.25))
        self.contracts = int(self.config.get("contracts", 1))
        self.poll_seconds = int(self.config.get("poll_interval_seconds", 5))

        self.value_area_pct = float(self.config.get("value_area_percent", 0.70))
        self.rejection_excursion_ticks = int(self.config.get("rejection_excursion_ticks", 3))
        self.rejection_confirm_ticks = int(self.config.get("rejection_confirm_ticks", 1))
        self.rejection_state_ttl_minutes = int(self.config.get("rejection_state_ttl_minutes", 20))
        self.acceptance_closes_required = int(self.config.get("acceptance_closes_required", 2))
        self.acceptance_break_ticks = int(self.config.get("acceptance_break_ticks", 1))
        self.acceptance_volume_multiplier = float(self.config.get("acceptance_volume_multiplier", 1.1))
        self.acceptance_volume_lookback = int(self.config.get("acceptance_volume_lookback", 20))
        self.acceptance_retest_required = bool(self.config.get("acceptance_retest_required", True))
        self.acceptance_retest_tolerance_ticks = int(self.config.get("acceptance_retest_tolerance_ticks", 2))
        self.acceptance_retest_lookback_bars = int(self.config.get("acceptance_retest_lookback_bars", 15))

        self.stop_ticks = int(self.config.get("stop_ticks", 12))
        self.target_rr = float(self.config.get("target_rr", 1.2))
        self.min_target_ticks = int(self.config.get("min_target_ticks", 8))

        self.max_trades_per_day = int(self.config.get("max_trades_per_day", 4))
        self.max_daily_loss = float(self.config.get("max_daily_loss", 900.0))
        self.max_consecutive_losses = int(self.config.get("max_consecutive_losses", 3))
        self.cooldown_after_loss_minutes = int(self.config.get("cooldown_after_loss_minutes", 20))
        self.flatten_on_start = bool(self.config.get("flatten_on_start", True))
        self.post_exit_cooldown_bars = int(self.config.get("post_exit_cooldown_bars", 2))
        self.max_acceptance_entries_per_hour_per_side = int(self.config.get("max_acceptance_entries_per_hour_per_side", 1))
        self.max_acceptance_entries_per_session_per_side = int(self.config.get("max_acceptance_entries_per_session_per_side", 3))
        self.heartbeat_log_interval_seconds = int(self.config.get("heartbeat_log_interval_seconds", 45))

        self.session_timezone = ZoneInfo(self.config.get("session_timezone", "America/New_York"))
        self.rth_start = dtime.fromisoformat(self.config.get("rth_start", "09:30"))
        self.rth_end = dtime.fromisoformat(self.config.get("rth_end", "16:00"))
        self.trade_start = dtime.fromisoformat(self.config.get("trade_start", "09:35"))
        self.trade_end = dtime.fromisoformat(self.config.get("trade_end", "15:30"))
        self.force_flatten_time = dtime.fromisoformat(self.config.get("force_flatten_time", "15:55"))

        self.api_client = TopstepXClient(
            username=os.getenv("TOPSTEPX_USERNAME"),
            api_key=os.getenv("TOPSTEPX_API_KEY"),
            base_url=self.config.get("api_base_url", "https://api.topstepx.com"),
            user_hub_url=self.config.get("user_hub_url", "https://rtc.topstepx.com/hubs/user"),
            market_hub_url=self.config.get("market_hub_url", "https://rtc.topstepx.com/hubs/market"),
            dry_run=self.dry_run,
        )
        self.position_manager = PositionManager(tick_values={self.symbol: self.tick_value})
        self.risk_manager = RiskManager(
            max_daily_loss=self.max_daily_loss,
            trailing_drawdown_limit=self.config.get("trailing_drawdown_limit", 1800.0),
            max_net_notional=self.config.get("max_net_notional", 500000.0),
        )
        self.order_client = OrderClient(self.api_client, dry_run=self.dry_run)
        self.adapter = MarketDataAdapter()

        self.running = False
        self.paused = False
        self.contract_id: Optional[str] = None
        self.current_price: Optional[float] = None
        self.current_price_ts: Optional[datetime] = None

        self.session_levels: Optional[SessionLevels] = None
        self.session_levels_for_date: Optional[datetime] = None
        self.position: Optional[ValueAreaPosition] = None

        self.trades_today = 0
        self.daily_realized_pnl = 0.0
        self.consecutive_losses = 0
        self.last_trade_day: Optional[datetime] = None
        self.cooldown_until: Optional[datetime] = None
        self._last_slow_refresh: Optional[datetime] = None
        self._bars_fast = pd.DataFrame()
        self._bars_slow = pd.DataFrame()
        self._rejection_above: Optional[datetime] = None
        self._rejection_below: Optional[datetime] = None
        self._last_exit_fast_bar_ts: Optional[datetime] = None
        self._acceptance_entries_hourly: Dict[str, Dict[Tuple[str, int], int]] = {
            "LONG": defaultdict(int),
            "SHORT": defaultdict(int),
        }
        self._acceptance_entries_session: Dict[Tuple[str, str], int] = defaultdict(int)
        self._last_heartbeat: Optional[datetime] = None
        self._last_no_trade_reason: str = "initializing"

        logger.info(f"Value Area MES Strategy initialized (dry_run={self.dry_run})")

    def initialize(self) -> bool:
        """Authenticate, select account, resolve contract."""
        if not self.api_client.authenticate():
            logger.error("Failed to authenticate with TopstepX")
            return False

        accounts = self.api_client.get_accounts()
        if not accounts:
            logger.error("No accounts found")
            return False

        account = self._select_account(accounts)
        if not account:
            logger.error("Could not select account")
            return False

        account_id = account.get("id") or account.get("accountId")
        self.api_client.set_account(account_id)
        logger.info(f"Using account: {account.get('name', 'Unknown')} ({account_id})")

        contracts = self.api_client.search_contracts(self.symbol)
        if not contracts:
            logger.error(f"No contracts found for {self.symbol}")
            return False
        self.contract_id = contracts[0].get("contractId") or contracts[0].get("id")
        logger.info(f"{self.symbol} contract: {self.contract_id}")

        self._refresh_market_data(force_slow=True)
        self._refresh_session_levels()

        if self.flatten_on_start and not self.dry_run:
            positions = self.api_client.get_positions(self.api_client.account_id)
            for p in positions:
                if str(p.get("contractId", "")) == str(self.contract_id):
                    logger.warning("Flattening existing MES position on startup")
                    self.api_client.close_position(self.contract_id, self.api_client.account_id)
                    break

        return True

    def _select_account(self, accounts: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        prefer_practice = self.config.get("prefer_practice_account", True)
        specified_account_id = self.config.get("account_id")
        if specified_account_id:
            for acc in accounts:
                if acc.get("id") == specified_account_id or acc.get("accountId") == specified_account_id:
                    return acc
        if prefer_practice:
            for acc in accounts:
                name = str(acc.get("name", "")).upper()
                if acc.get("simulated", False) or "PRAC" in name or "PRACTICE" in name:
                    return acc
        return accounts[0] if accounts else None

    def _to_et(self, ts: datetime) -> datetime:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=ZoneInfo("UTC"))
        return ts.astimezone(self.session_timezone)

    def _round_tick(self, price: float) -> float:
        return round(price / self.tick_size) * self.tick_size

    def _refresh_market_data(self, force_slow: bool = False):
        now = datetime.utcnow()
        start_fast = now - timedelta(hours=3)
        bars_fast_raw = self.api_client.get_bars(
            contract_id=self.contract_id,
            interval=self.interval_fast,
            start_time=start_fast,
            end_time=now,
            limit=180,
        )
        if bars_fast_raw:
            fast = self.adapter.normalize_bars(bars_fast_raw, self.symbol, self.interval_fast)
            self._bars_fast = pd.DataFrame([{
                "timestamp": c.timestamp,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
            } for c in fast]).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
            if not self._bars_fast.empty:
                self.current_price = float(self._bars_fast.iloc[-1]["close"])
                self.current_price_ts = datetime.utcnow()
                self.position_manager.update_price(self.symbol, self.current_price)

        need_slow = force_slow or self._last_slow_refresh is None or (datetime.utcnow() - self._last_slow_refresh).total_seconds() >= 60
        if need_slow:
            start_slow = now - timedelta(days=7)
            bars_slow_raw = self.api_client.get_bars(
                contract_id=self.contract_id,
                interval=self.interval_slow,
                start_time=start_slow,
                end_time=now,
                limit=2500,
            )
            if bars_slow_raw:
                slow = self.adapter.normalize_bars(bars_slow_raw, self.symbol, self.interval_slow)
                self._bars_slow = pd.DataFrame([{
                    "timestamp": c.timestamp,
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume,
                } for c in slow]).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
                self._last_slow_refresh = datetime.utcnow()

    def _build_prior_session_profile(self) -> Optional[SessionLevels]:
        if self._bars_slow.empty:
            return None
        bars = self._bars_slow.copy()
        bars["ts_et"] = bars["timestamp"].apply(self._to_et)
        bars["date_et"] = bars["ts_et"].dt.date
        bars["time_et"] = bars["ts_et"].dt.time
        bars = bars[(bars["time_et"] >= self.rth_start) & (bars["time_et"] <= self.rth_end)]
        if bars.empty:
            return None

        today_et = datetime.now(self.session_timezone).date()
        prior_dates = sorted([d for d in bars["date_et"].unique() if d < today_et])
        if not prior_dates:
            return None
        prior_date = prior_dates[-1]
        session_bars = bars[bars["date_et"] == prior_date]
        if session_bars.empty:
            return None

        volume_by_price = defaultdict(float)
        for _, row in session_bars.iterrows():
            tp = self._round_tick((row["high"] + row["low"] + row["close"]) / 3.0)
            volume_by_price[tp] += float(row["volume"])

        if not volume_by_price:
            return None

        prices = sorted(volume_by_price.keys())
        volumes = [volume_by_price[p] for p in prices]
        total_vol = sum(volumes)
        target_vol = total_vol * self.value_area_pct
        poc_idx = max(range(len(prices)), key=lambda i: volumes[i])
        cum = volumes[poc_idx]
        left = poc_idx
        right = poc_idx

        while cum < target_vol and (left > 0 or right < len(prices) - 1):
            next_left_vol = volumes[left - 1] if left > 0 else -1
            next_right_vol = volumes[right + 1] if right < len(prices) - 1 else -1
            if next_left_vol >= next_right_vol and left > 0:
                left -= 1
                cum += volumes[left]
            elif right < len(prices) - 1:
                right += 1
                cum += volumes[right]
            else:
                break

        return SessionLevels(
            session_date=datetime.combine(prior_date, dtime(0, 0)),
            poc=prices[poc_idx],
            vah=prices[right],
            val=prices[left],
        )

    def _refresh_session_levels(self):
        levels = self._build_prior_session_profile()
        if not levels:
            return
        if not self.session_levels_for_date or self.session_levels_for_date.date() != levels.session_date.date():
            self.session_levels = levels
            self.session_levels_for_date = levels.session_date
            logger.info(
                f"Session levels loaded ({levels.session_date.date()}): "
                f"VAL={levels.val:.2f}, POC={levels.poc:.2f}, VAH={levels.vah:.2f}"
            )

    def _in_trade_window(self) -> bool:
        now_et = datetime.now(self.session_timezone).time()
        return self.trade_start <= now_et <= self.trade_end

    def _should_force_flatten(self) -> bool:
        return datetime.now(self.session_timezone).time() >= self.force_flatten_time

    def _reset_daily_counters_if_needed(self):
        today = datetime.now(self.session_timezone).date()
        if self.last_trade_day != today:
            self.trades_today = 0
            self.daily_realized_pnl = 0.0
            self.consecutive_losses = 0
            self.last_trade_day = today
            self._acceptance_entries_hourly = {
                "LONG": defaultdict(int),
                "SHORT": defaultdict(int),
            }
            self._acceptance_entries_session = defaultdict(int)

    def _trade_block_reason(self) -> Optional[str]:
        self._reset_daily_counters_if_needed()
        if self.cooldown_until and datetime.utcnow() < self.cooldown_until:
            remaining = int((self.cooldown_until - datetime.utcnow()).total_seconds())
            return f"loss-cooldown-active ({remaining}s remaining)"
        if self.trades_today >= self.max_trades_per_day:
            return f"daily-trade-limit-reached ({self.trades_today}/{self.max_trades_per_day})"
        if self.daily_realized_pnl <= -abs(self.max_daily_loss):
            return f"daily-loss-limit-reached (${self.daily_realized_pnl:.2f})"
        if self.consecutive_losses >= self.max_consecutive_losses:
            return f"max-consecutive-losses-reached ({self.consecutive_losses})"
        if not self._in_trade_window():
            return "outside-trade-window"
        if self.post_exit_cooldown_bars > 0 and self._last_exit_fast_bar_ts is not None and not self._bars_fast.empty:
            bars_since_exit = len(self._bars_fast[self._bars_fast["timestamp"] > self._last_exit_fast_bar_ts])
            if bars_since_exit < self.post_exit_cooldown_bars:
                return f"post-exit-cooldown ({bars_since_exit}/{self.post_exit_cooldown_bars} bars)"
        return None
    
    def _can_trade(self) -> bool:
        return self._trade_block_reason() is None
    
    def _acceptance_cap_reached(self, side: str) -> Tuple[bool, str]:
        now_et = datetime.now(self.session_timezone)
        hour_key = (now_et.date().isoformat(), now_et.hour)
        session_key = (now_et.date().isoformat(), side)
        
        hour_count = self._acceptance_entries_hourly[side][hour_key]
        session_count = self._acceptance_entries_session[session_key]
        
        if self.max_acceptance_entries_per_hour_per_side > 0 and hour_count >= self.max_acceptance_entries_per_hour_per_side:
            return True, f"acceptance-{side.lower()}-hour-cap ({hour_count}/{self.max_acceptance_entries_per_hour_per_side})"
        if self.max_acceptance_entries_per_session_per_side > 0 and session_count >= self.max_acceptance_entries_per_session_per_side:
            return True, f"acceptance-{side.lower()}-session-cap ({session_count}/{self.max_acceptance_entries_per_session_per_side})"
        return False, ""
    
    def _record_acceptance_entry(self, side: str):
        now_et = datetime.now(self.session_timezone)
        hour_key = (now_et.date().isoformat(), now_et.hour)
        session_key = (now_et.date().isoformat(), side)
        self._acceptance_entries_hourly[side][hour_key] += 1
        self._acceptance_entries_session[session_key] += 1
    
    def _emit_heartbeat(self):
        if self.heartbeat_log_interval_seconds <= 0:
            return
        now = datetime.utcnow()
        if self._last_heartbeat and (now - self._last_heartbeat).total_seconds() < self.heartbeat_log_interval_seconds:
            return
        can_trade = self._can_trade()
        reason = "ready" if can_trade else (self._trade_block_reason() or self._last_no_trade_reason)
        logger.info(
            f"Heartbeat: price={self.current_price}, has_position={self.position is not None}, "
            f"trades_today={self.trades_today}, can_trade={can_trade}, reason={reason}"
        )
        self._last_heartbeat = now

    def _build_entry(self, side: str, setup_type: str) -> Tuple[float, float]:
        stop_distance = self.stop_ticks * self.tick_size
        if side == "LONG":
            stop = self.current_price - stop_distance
            rr_target = self.current_price + max(stop_distance * self.target_rr, self.min_target_ticks * self.tick_size)
            target = max(rr_target, self.session_levels.poc)
        else:
            stop = self.current_price + stop_distance
            rr_target = self.current_price - max(stop_distance * self.target_rr, self.min_target_ticks * self.tick_size)
            target = min(rr_target, self.session_levels.poc)
        return stop, target

    def _try_open_position(self, side: str, setup_type: str) -> bool:
        if self.position is not None:
            return False
        stop, target = self._build_entry(side, setup_type)
        order_side = "BUY" if side == "LONG" else "SELL"
        order_id = self.order_client.place_market_order(self.contract_id, order_side, self.contracts)
        if not order_id:
            logger.warning(f"Failed to place entry order ({setup_type}/{side})")
            return False

        self.position = ValueAreaPosition(
            side=side,
            entry_price=self.current_price,
            stop_price=stop,
            target_price=target,
            quantity=self.contracts,
            entry_time=datetime.utcnow(),
            setup_type=setup_type,
        )
        self.position_manager.on_fill(self.symbol, order_side, self.contracts, self.current_price)
        self.trades_today += 1
        logger.info(
            f"Opened {side} ({setup_type}) {self.contracts} {self.symbol} @ {self.current_price:.2f} "
            f"stop={stop:.2f} target={target:.2f}"
        )
        return True

    def _exit_position(self, reason: str):
        if self.position is None:
            return
        exit_side = "SELL" if self.position.side == "LONG" else "BUY"
        self.order_client.place_market_order(self.contract_id, exit_side, self.position.quantity)
        self.position_manager.on_fill(self.symbol, exit_side, self.position.quantity, self.current_price)

        price_diff = (self.current_price - self.position.entry_price) if self.position.side == "LONG" else (self.position.entry_price - self.current_price)
        pnl = (price_diff / self.tick_size) * self.tick_value * self.position.quantity
        self.daily_realized_pnl += pnl
        self.consecutive_losses = self.consecutive_losses + 1 if pnl < 0 else 0
        if pnl < 0 and self.cooldown_after_loss_minutes > 0:
            self.cooldown_until = datetime.utcnow() + timedelta(minutes=self.cooldown_after_loss_minutes)
        if not self._bars_fast.empty:
            self._last_exit_fast_bar_ts = self._bars_fast.iloc[-1]["timestamp"]
        logger.info(f"Closed {self.position.side} position @ {self.current_price:.2f} | pnl=${pnl:.2f} | reason={reason}")
        self.position = None

    def _manage_position(self):
        if not self.position:
            return
        if self._should_force_flatten():
            self._exit_position("Session force flatten")
            return
        if self.position.side == "LONG":
            if self.current_price <= self.position.stop_price:
                self._exit_position("Stop hit")
            elif self.current_price >= self.position.target_price:
                self._exit_position("Target hit")
        else:
            if self.current_price >= self.position.stop_price:
                self._exit_position("Stop hit")
            elif self.current_price <= self.position.target_price:
                self._exit_position("Target hit")

    def _run_signal_logic(self):
        if not self.session_levels or self._bars_fast.empty or self.current_price is None:
            self._last_no_trade_reason = "waiting-for-levels-or-bars"
            return
        trade_block = self._trade_block_reason()
        if trade_block:
            self._last_no_trade_reason = trade_block
            return

        now = datetime.utcnow()
        ttl = timedelta(minutes=self.rejection_state_ttl_minutes)
        if self._rejection_above and now - self._rejection_above > ttl:
            self._rejection_above = None
        if self._rejection_below and now - self._rejection_below > ttl:
            self._rejection_below = None

        vah = self.session_levels.vah
        val = self.session_levels.val
        excursion = self.rejection_excursion_ticks * self.tick_size
        confirm = self.rejection_confirm_ticks * self.tick_size
        break_buf = self.acceptance_break_ticks * self.tick_size

        if self.current_price >= vah + excursion:
            self._rejection_above = now
        if self.current_price <= val - excursion:
            self._rejection_below = now

        # Rejection fades
        if self._rejection_above and self.current_price <= vah - confirm:
            if self._try_open_position("SHORT", "rejection"):
                self._rejection_above = None
                self._rejection_below = None
                self._last_no_trade_reason = "entered-short-rejection"
                return
        if self._rejection_below and self.current_price >= val + confirm:
            if self._try_open_position("LONG", "rejection"):
                self._rejection_above = None
                self._rejection_below = None
                self._last_no_trade_reason = "entered-long-rejection"
                return

        # Acceptance breakouts
        if len(self._bars_fast) >= max(self.acceptance_closes_required, self.acceptance_volume_lookback):
            recent = self._bars_fast.tail(self.acceptance_closes_required)
            vol_ref = self._bars_fast.tail(self.acceptance_volume_lookback)["volume"].mean()
            vol_recent = recent["volume"].mean()
            high_accept = (recent["close"] > (vah + break_buf)).all() and vol_recent >= (vol_ref * self.acceptance_volume_multiplier)
            low_accept = (recent["close"] < (val - break_buf)).all() and vol_recent >= (vol_ref * self.acceptance_volume_multiplier)
            
            if self.acceptance_retest_required:
                lookback_bars = max(self.acceptance_retest_lookback_bars, self.acceptance_closes_required)
                accept_window = self._bars_fast.tail(lookback_bars)
                high_retest_level = vah + (self.acceptance_retest_tolerance_ticks * self.tick_size)
                low_retest_level = val - (self.acceptance_retest_tolerance_ticks * self.tick_size)
                high_retest_seen = (accept_window["low"] <= high_retest_level).any()
                low_retest_seen = (accept_window["high"] >= low_retest_level).any()
                high_hold = (recent["close"] > vah).all()
                low_hold = (recent["close"] < val).all()
                high_accept = high_accept and high_retest_seen and high_hold
                low_accept = low_accept and low_retest_seen and low_hold
            
            if high_accept:
                cap_reached, cap_reason = self._acceptance_cap_reached("LONG")
                if cap_reached:
                    self._last_no_trade_reason = cap_reason
                elif self._try_open_position("LONG", "acceptance"):
                    self._record_acceptance_entry("LONG")
                    self._last_no_trade_reason = "entered-long-acceptance"
            elif low_accept:
                cap_reached, cap_reason = self._acceptance_cap_reached("SHORT")
                if cap_reached:
                    self._last_no_trade_reason = cap_reason
                elif self._try_open_position("SHORT", "acceptance"):
                    self._record_acceptance_entry("SHORT")
                    self._last_no_trade_reason = "entered-short-acceptance"
            else:
                self._last_no_trade_reason = "no-valid-setup"
        else:
            self._last_no_trade_reason = "insufficient-bars-for-acceptance"

    def set_paused(self, paused: bool):
        self.paused = paused
        logger.warning(f"Strategy {'paused' if paused else 'resumed'}")

    def emergency_flatten(self, reason: str = "Emergency"):
        logger.critical(f"EMERGENCY FLATTEN: {reason}")
        if self.position:
            self._exit_position(reason)
        if self.contract_id and not self.dry_run:
            self.api_client.close_position(self.contract_id, self.api_client.account_id)
        self.running = False

    def get_status(self) -> Dict[str, Any]:
        return {
            "status": "running" if self.running else "stopped",
            "timestamp": datetime.utcnow().isoformat(),
            "trading_enabled": self.running and not self.paused,
            "dry_run": self.dry_run,
            "primary_symbol": self.symbol,
            "current_price": self.current_price,
            "session_levels": {
                "poc": self.session_levels.poc if self.session_levels else None,
                "vah": self.session_levels.vah if self.session_levels else None,
                "val": self.session_levels.val if self.session_levels else None,
            },
            "position": {
                "side": self.position.side,
                "entry_price": self.position.entry_price,
                "stop_price": self.position.stop_price,
                "target_price": self.position.target_price,
                "quantity": self.position.quantity,
                "setup_type": self.position.setup_type,
            } if self.position else None,
            "daily_pnl": self.daily_realized_pnl,
            "total_pnl": self.position_manager.get_total_pnl(),
            "drawdown": self.risk_manager.get_trailing_drawdown(),
            "trades_today": self.trades_today,
            "cooldown_until": self.cooldown_until.isoformat() if self.cooldown_until else None,
        }

    def get_metrics(self) -> Dict[str, Any]:
        return self.get_status()

    def run(self):
        if not self.initialize():
            logger.error("Failed to initialize Value Area MES strategy")
            return
        self.running = True
        logger.info("Starting Value Area MES trading loop...")

        try:
            while self.running:
                if self.paused:
                    time.sleep(1)
                    continue

                self._refresh_market_data(force_slow=False)
                self._refresh_session_levels()

                if self.current_price is None:
                    self._last_no_trade_reason = "no-current-price"
                    time.sleep(self.poll_seconds)
                    continue

                self._manage_position()
                if not self.position:
                    self._run_signal_logic()
                self._emit_heartbeat()

                time.sleep(self.poll_seconds)
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        except Exception as e:
            logger.error(f"Error in Value Area MES loop: {e}", exc_info=True)
        finally:
            logger.info("Shutting down Value Area MES strategy...")
            if self.position:
                self._exit_position("Shutdown")
            self.api_client.disconnect()
