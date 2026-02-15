"""Risk management for Polymarket trading.

Enforces position limits, daily/weekly loss limits, drawdown protection,
category exposure, cooldown, black swan protection, and Kelly-based sizing.

Based on: Ernest Chan Ch.6-7 (Kelly, stop-loss for MR vs momentum,
fat tails, trailing Kelly), FinAgent (progressive drawdown cuts).

Usage:
    rm = RiskManager(config)
    allowed, reason = rm.can_open(equity, cash, n_positions, token_id, ...)
    size = rm.compute_size(equity, cash, p_model, p_market)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Literal

import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)


class Regime(Enum):
    MEAN_REVERTING = "mr"
    MOMENTUM = "momentum"
    UNKNOWN = "unknown"


@dataclass
class RiskConfig:
    """Risk management parameters."""
    # Position limits
    max_position_pct: float = 0.10       # Max % of equity per position
    max_position_usd: float = 500.0      # Absolute $ limit per market
    max_positions: int = 10              # Max concurrent positions
    max_correlated: int = 3              # Max positions in same event

    # Category exposure (sports != politics != crypto -- Chan Ch.6 contagion risk)
    max_category_pct: float = 0.40       # Max % of equity in one category
    max_category_positions: int = 5      # Max positions in one category

    # Total exposure
    cash_reserve_pct: float = 0.20       # Keep 20% cash reserve
    max_total_exposure_pct: float = 0.80 # Max 80% of equity in positions

    # Loss limits
    daily_loss_limit_pct: float = 0.05   # Stop trading after 5% daily loss
    weekly_loss_limit_pct: float = 0.08  # Reduce size after 8% weekly loss
    max_drawdown_pct: float = 0.15       # Halt after 15% drawdown from peak

    # Progressive drawdown cuts (FinAgent: 10%→reduce 30%, 15%→reduce 50%, 20%→halt)
    drawdown_cut_10_pct: float = 0.10    # At 10% DD, reduce size by 30%
    drawdown_cut_15_pct: float = 0.15    # At 15% DD, reduce size by 50%
    drawdown_halt_pct: float = 0.20      # At 20% DD, halt trading

    # Cooldown (pause after a large loss)
    cooldown_loss_threshold: float = 0.03   # Single-trade loss > 3% equity → cooldown
    cooldown_minutes: int = 60              # Cooldown duration

    # Edge & fees
    min_edge: float = 0.01              # Minimum EV to open position
    kelly_fraction: float = 0.5          # Half-Kelly (Chan Ch.6)

    # Market quality
    min_liquidity: float = 1000.0        # Min market liquidity ($)
    max_spread: float = 0.05             # Max bid-ask spread

    # Black swan (Chan Ch.6: max_leverage = min(half_Kelly, max_DD / max_hist_loss))
    max_high_prob_pct: float = 0.05      # Max 5% on $0.90+ bonds
    max_loss_per_resolution: float = 0.10  # Max 10% equity loss per single resolution

    # Stop-loss strategy (Chan Ch.6-7: stop loss HARMFUL for MR, ok for momentum)
    time_exit_hours: float = 72.0        # Default max hold time for active trades
    edge_exit_threshold: float = 0.005   # Exit if edge drops below this


@dataclass
class DailyStats:
    """Track daily trading statistics."""
    date: str = ""
    starting_equity: float = 0.0
    realized_pnl: float = 0.0
    n_trades: int = 0
    n_wins: int = 0
    peak_equity: float = 0.0

    @property
    def daily_return(self) -> float:
        if self.starting_equity <= 0:
            return 0.0
        return self.realized_pnl / self.starting_equity

    @property
    def win_rate(self) -> float:
        if self.n_trades == 0:
            return 0.0
        return self.n_wins / self.n_trades


class RiskManager:
    """Centralized risk management.

    Implements:
    - Half-Kelly sizing (Chan Ch.6)
    - Progressive drawdown cuts (FinAgent)
    - Stop-loss only for momentum (Chan Ch.6-7)
    - Category diversification (Chan Ch.6 contagion risk)
    - Black swan protection
    - Cooldown after big losses
    """

    def __init__(self, config: RiskConfig | None = None):
        self.config = config or RiskConfig()
        self.daily = DailyStats()
        self.peak_equity = 0.0
        self._halted = False
        self._halt_reason = ""
        self._event_positions: dict[str, int] = {}  # event_id → count
        self._category_exposure: dict[str, float] = {}  # category → $ invested
        self._category_positions: dict[str, int] = {}   # category → count
        self._weekly_pnl: float = 0.0
        self._week_start: str = ""
        self._cooldown_until: datetime | None = None
        self._trailing_returns: list[float] = []  # for trailing Kelly update

    # --- Lifecycle ---

    def reset_daily(self, equity: float):
        """Reset daily stats (call at start of each trading day)."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.daily.date != today:
            # Reset daily halt — new day, fresh start
            if self._halted:
                log.info(f"Daily reset clears halt: {self._halt_reason}")
                self._halted = False
                self._halt_reason = ""
            self.daily = DailyStats(
                date=today,
                starting_equity=equity,
                peak_equity=equity,
            )
            self.peak_equity = max(self.peak_equity, equity)
            # Weekly tracking
            week = datetime.now(timezone.utc).strftime("%Y-W%W")
            if self._week_start != week:
                self._weekly_pnl = 0.0
                self._week_start = week
            log.info(f"Daily reset: equity=${equity:.2f}")

    def update_equity(self, equity: float):
        """Update peak equity for drawdown tracking."""
        self.peak_equity = max(self.peak_equity, equity)
        self.daily.peak_equity = max(self.daily.peak_equity, equity)

    def record_trade(self, pnl: float, equity: float = 0.0):
        """Record a completed trade."""
        self.daily.realized_pnl += pnl
        self.daily.n_trades += 1
        self._weekly_pnl += pnl
        if pnl > 0:
            self.daily.n_wins += 1
        # Track for trailing Kelly
        if equity > 0:
            self._trailing_returns.append(pnl / equity)
            # Keep last ~126 trades (~6 months of daily trades, Chan Ch.6)
            if len(self._trailing_returns) > 126:
                self._trailing_returns = self._trailing_returns[-126:]
        # Cooldown check: big single-trade loss
        if equity > 0 and pnl < 0 and abs(pnl) / equity >= self.config.cooldown_loss_threshold:
            self._cooldown_until = datetime.now(timezone.utc) + timedelta(
                minutes=self.config.cooldown_minutes
            )
            log.warning(
                f"Cooldown activated: loss {pnl/equity:.1%} > "
                f"{self.config.cooldown_loss_threshold:.1%}, "
                f"pausing {self.config.cooldown_minutes}min"
            )

    # --- Properties ---

    @property
    def is_halted(self) -> bool:
        return self._halted

    @property
    def halt_reason(self) -> str:
        return self._halt_reason

    @property
    def is_cooling_down(self) -> bool:
        if self._cooldown_until is None:
            return False
        if datetime.now(timezone.utc) >= self._cooldown_until:
            self._cooldown_until = None
            return False
        return True

    @property
    def drawdown_pct(self) -> float:
        """Current drawdown from peak."""
        if self.peak_equity <= 0:
            return 0.0
        current = self.daily.starting_equity + self.daily.realized_pnl
        return max(0.0, (self.peak_equity - current) / self.peak_equity)

    @property
    def drawdown_size_multiplier(self) -> float:
        """Progressive size reduction based on drawdown (FinAgent).

        10% DD → 0.7x, 15% DD → 0.5x, 20% DD → 0.0x (halt)
        """
        dd = self.drawdown_pct
        if dd >= self.config.drawdown_halt_pct:
            return 0.0
        if dd >= self.config.drawdown_cut_15_pct:
            return 0.5
        if dd >= self.config.drawdown_cut_10_pct:
            return 0.7
        return 1.0

    @property
    def trailing_kelly_fraction(self) -> float:
        """Adaptive Kelly fraction from trailing returns (Chan Ch.6).

        As mean return → 0, Kelly → 0 (gradual shutdown).
        """
        if len(self._trailing_returns) < 10:
            return self.config.kelly_fraction
        rets = np.array(self._trailing_returns)
        m = rets.mean()
        s2 = rets.var()
        if s2 <= 0 or m <= 0:
            return 0.0
        raw_kelly = m / s2
        # Half-Kelly, capped
        return min(raw_kelly * 0.5, self.config.kelly_fraction)

    # --- Checks ---

    def check_limits(self, equity: float, n_positions: int) -> tuple[bool, str]:
        """Check if trading should continue.

        Returns:
            (ok, reason) — ok=True if trading allowed
        """
        if self._halted:
            return False, f"HALTED: {self._halt_reason}"

        # Daily loss limit (checked before cooldown — halt is more serious)
        if self.daily.starting_equity > 0:
            daily_loss = -self.daily.realized_pnl / self.daily.starting_equity
            if daily_loss >= self.config.daily_loss_limit_pct:
                self._halted = True
                self._halt_reason = (
                    f"Daily loss limit: {daily_loss:.1%} >= "
                    f"{self.config.daily_loss_limit_pct:.1%}"
                )
                log.warning(f"HALT: {self._halt_reason}")
                return False, self._halt_reason

        # Cooldown (after loss limits, before position limits)
        if self.is_cooling_down:
            remaining = (self._cooldown_until - datetime.now(timezone.utc)).total_seconds() / 60
            return False, f"Cooldown: {remaining:.0f}min remaining"

        # Max drawdown (hard halt)
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - equity) / self.peak_equity
            if drawdown >= self.config.drawdown_halt_pct:
                self._halted = True
                self._halt_reason = (
                    f"Max drawdown halt: {drawdown:.1%} >= "
                    f"{self.config.drawdown_halt_pct:.1%}"
                )
                log.warning(f"HALT: {self._halt_reason}")
                return False, self._halt_reason

        # Max positions
        if n_positions >= self.config.max_positions:
            return False, f"Max positions reached: {n_positions}"

        return True, "OK"

    def can_open(
        self,
        equity: float,
        cash: float,
        n_positions: int,
        token_id: str,
        event_id: str = "",
        category: str = "",
        market_liquidity: float = 0.0,
        market_spread: float = 0.0,
        market_price: float = 0.0,
    ) -> tuple[bool, str]:
        """Check if a new position can be opened.

        Returns:
            (allowed, reason)
        """
        # Basic limits
        ok, reason = self.check_limits(equity, n_positions)
        if not ok:
            return False, reason

        # Cash reserve
        available = cash - (equity * self.config.cash_reserve_pct)
        if available <= 0:
            return False, f"Cash reserve: ${cash:.0f} < {self.config.cash_reserve_pct:.0%} of ${equity:.0f}"

        # Total exposure limit
        total_invested = equity - cash
        if equity > 0 and total_invested / equity >= self.config.max_total_exposure_pct:
            return False, (
                f"Total exposure: {total_invested/equity:.0%} >= "
                f"{self.config.max_total_exposure_pct:.0%}"
            )

        # Correlation limit (same event)
        if event_id and self._event_positions.get(event_id, 0) >= self.config.max_correlated:
            return False, f"Event concentration: {self._event_positions[event_id]} positions in event"

        # Category exposure limit
        if category:
            cat_exposure = self._category_exposure.get(category, 0.0)
            if equity > 0 and cat_exposure / equity >= self.config.max_category_pct:
                return False, (
                    f"Category exposure: {category} at "
                    f"{cat_exposure/equity:.0%} >= {self.config.max_category_pct:.0%}"
                )
            cat_count = self._category_positions.get(category, 0)
            if cat_count >= self.config.max_category_positions:
                return False, f"Category positions: {category} has {cat_count} positions"

        # Black swan: limit high-probability bond exposure
        if market_price >= 0.90 or market_price <= 0.10:
            # Already investing in high-prob market — check total exposure
            # (a $0.95 bond losing = 95¢ loss per share, catastrophic)
            pass  # checked in compute_size via max_high_prob_pct

        # Liquidity check
        if market_liquidity > 0 and market_liquidity < self.config.min_liquidity:
            return False, f"Low liquidity: ${market_liquidity:.0f} < ${self.config.min_liquidity:.0f}"

        # Spread check
        if market_spread > 0 and market_spread > self.config.max_spread:
            return False, f"Wide spread: {market_spread:.3f} > {self.config.max_spread:.3f}"

        # Weekly loss — reduce sizing (checked in compute_size), but still allow opening
        if self.daily.starting_equity > 0 and self._weekly_pnl < 0:
            weekly_loss = abs(self._weekly_pnl) / self.daily.starting_equity
            if weekly_loss >= self.config.weekly_loss_limit_pct:
                return False, f"Weekly loss limit: {weekly_loss:.1%} >= {self.config.weekly_loss_limit_pct:.1%}"

        return True, "OK"

    # --- Position tracking ---

    def position_opened(self, event_id: str = "", category: str = "",
                        invested: float = 0.0):
        """Track that a position was opened."""
        if event_id:
            self._event_positions[event_id] = self._event_positions.get(event_id, 0) + 1
        if category:
            self._category_exposure[category] = self._category_exposure.get(category, 0.0) + invested
            self._category_positions[category] = self._category_positions.get(category, 0) + 1

    def position_closed(self, event_id: str = "", category: str = "",
                        pnl: float = 0.0, invested: float = 0.0,
                        equity: float = 0.0):
        """Track that a position was closed."""
        if event_id and event_id in self._event_positions:
            self._event_positions[event_id] = max(0, self._event_positions[event_id] - 1)
            if self._event_positions[event_id] == 0:
                del self._event_positions[event_id]
        if category and category in self._category_exposure:
            self._category_exposure[category] = max(0.0, self._category_exposure[category] - invested)
            if self._category_exposure[category] <= 0:
                del self._category_exposure[category]
            self._category_positions[category] = max(0, self._category_positions.get(category, 1) - 1)
            if self._category_positions.get(category, 0) <= 0:
                self._category_positions.pop(category, None)
        self.record_trade(pnl, equity)

    # --- Exit rules (Chan Ch.6-7) ---

    def should_exit(
        self,
        regime: Regime,
        entry_time: datetime,
        entry_price: float,
        current_price: float,
        p_model_current: float = 0.0,
        p_market_current: float = 0.0,
        side: str = "YES",
    ) -> tuple[bool, str]:
        """Determine if a position should be exited.

        Chan Ch.6: stop loss HARMFUL for MR, beneficial for momentum.
        Chan Ch.7: exit MR at target (μ) or half-life; exit momentum on signal reversal.
        """
        now = datetime.now(timezone.utc)
        hold_hours = (now - entry_time).total_seconds() / 3600

        # 1. Time-based exit (applies to all)
        if hold_hours >= self.config.time_exit_hours:
            return True, f"Time exit: held {hold_hours:.0f}h >= {self.config.time_exit_hours:.0f}h"

        # 2. Edge-based exit: if model edge disappeared
        if p_model_current > 0 and p_market_current > 0:
            if side == "YES":
                current_edge = p_model_current - p_market_current
            else:
                current_edge = (1 - p_model_current) - (1 - p_market_current)
            if current_edge < self.config.edge_exit_threshold:
                return True, f"Edge gone: {current_edge:.4f} < {self.config.edge_exit_threshold}"

        # 3. Regime-specific exit
        # current_price is always in our token's terms (NO price for NO, YES price for YES)
        # so profit = current - entry for both sides
        move = current_price - entry_price

        if regime == Regime.MOMENTUM:
            # Stop loss ok for momentum (Chan Ch.6)
            if move <= -0.05:
                return True, f"Momentum SL: move {move:+.3f}"
            # Take profit on strong move
            if move >= 0.10:
                return True, f"Momentum TP: move {move:+.3f}"

        elif regime == Regime.MEAN_REVERTING:
            # NO stop loss for MR (Chan Ch.6-7: price will revert!)
            # Only exit on target (mean) or time
            if move >= 0.05:
                return True, f"MR target hit: move {move:+.3f}"
            # Don't exit on losses — MR will revert

        else:  # UNKNOWN
            # Conservative: use wide stops
            if move <= -0.10:
                return True, f"Wide SL: move {move:+.3f}"
            if move >= 0.08:
                return True, f"TP: move {move:+.3f}"

        # 4. Portfolio-level daily stop
        if self.daily.starting_equity > 0:
            daily_loss_pct = -self.daily.realized_pnl / self.daily.starting_equity
            if daily_loss_pct >= self.config.daily_loss_limit_pct:
                return True, f"Portfolio daily stop: {daily_loss_pct:.1%} loss"

        return False, "Hold"

    # --- Sizing ---

    def compute_size(
        self,
        equity: float,
        cash: float,
        p_model: float,
        p_market: float,
        fee_rate: float = 0.0175,
        category: str = "",
    ) -> float:
        """Compute position size using adaptive Half-Kelly.

        Applies:
        - Half-Kelly (or trailing adaptive Kelly)
        - Progressive drawdown cuts
        - Per-market caps (% and absolute $)
        - Black swan protection for high-prob bonds
        - Fat tails: min(half_Kelly, max_DD_tolerance / max_hist_loss) — Chan Ch.6

        Args:
            equity: current portfolio equity
            cash: available cash
            p_model: model's probability estimate
            p_market: market price (= implied probability)
            fee_rate: transaction fee rate
            category: market category for exposure checks

        Returns:
            bet size in dollars (0 if should not trade)
        """
        # Kelly fraction: f* = (p*b - q) / b
        fee = p_market * (1 - p_market) * fee_rate

        if p_model > p_market:  # BUY YES
            win_payoff = (1.0 - p_market) - fee
            loss_payoff = p_market + fee
        else:  # BUY NO
            no_price = 1.0 - p_market
            fee_no = no_price * (1 - no_price) * fee_rate
            win_payoff = p_market - fee_no
            loss_payoff = no_price + fee_no
            p_model = 1.0 - p_model  # flip for NO side

        if win_payoff <= 0 or loss_payoff <= 0:
            return 0.0

        b = win_payoff / loss_payoff  # odds
        kelly_f = (p_model * b - (1 - p_model)) / b
        kelly_f = max(0.0, kelly_f)

        # Use trailing adaptive Kelly if available, otherwise config fraction
        fraction = self.trailing_kelly_fraction
        f = kelly_f * fraction

        # Cap at max_position_pct
        f = min(f, self.config.max_position_pct)

        # Progressive drawdown reduction
        f *= self.drawdown_size_multiplier
        if f <= 0:
            return 0.0

        # Black swan cap: high-prob bonds limited
        if p_market >= 0.90 or p_market <= 0.10:
            f = min(f, self.config.max_high_prob_pct)

        # Max loss per resolution
        potential_loss_pct = f  # worst case: lose entire position
        if potential_loss_pct > self.config.max_loss_per_resolution:
            f = self.config.max_loss_per_resolution

        # Available cash (minus reserve)
        available = cash - (equity * self.config.cash_reserve_pct)
        if available <= 0:
            return 0.0

        # Category cap
        if category and equity > 0:
            cat_current = self._category_exposure.get(category, 0.0)
            cat_max = equity * self.config.max_category_pct
            cat_available = max(0.0, cat_max - cat_current)
            available = min(available, cat_available)

        size = min(f * equity, available, self.config.max_position_usd)
        return max(0.0, size)

    # --- Control ---

    def resume(self):
        """Resume trading after halt (manual override)."""
        if self._halted:
            log.info(f"Resuming trading (was halted: {self._halt_reason})")
            self._halted = False
            self._halt_reason = ""

    def clear_cooldown(self):
        """Manually clear cooldown."""
        self._cooldown_until = None

    def status(self) -> dict:
        """Return current risk status."""
        return {
            "halted": self._halted,
            "halt_reason": self._halt_reason,
            "daily_pnl": self.daily.realized_pnl,
            "daily_return": self.daily.daily_return,
            "daily_trades": self.daily.n_trades,
            "daily_wr": self.daily.win_rate,
            "weekly_pnl": self._weekly_pnl,
            "peak_equity": self.peak_equity,
            "drawdown_pct": self.drawdown_pct,
            "drawdown_multiplier": self.drawdown_size_multiplier,
            "trailing_kelly": self.trailing_kelly_fraction,
            "cooling_down": self.is_cooling_down,
            "event_positions": dict(self._event_positions),
            "category_exposure": dict(self._category_exposure),
            "category_positions": dict(self._category_positions),
        }
