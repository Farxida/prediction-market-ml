"""Trading strategies for Polymarket.

Implements rule-based strategies from research (Chan Ch.7, EDA findings):
- MeanReversionStrategy: z-score entry, target exit (77% of markets are MR)
- MomentumStrategy: breakout entry, stop loss ok (Chan Ch.6)
- ResolutionConvergenceStrategy: buy near-certain at discount
- ContrarianStrategy: bet against favourite-longshot bias (YES bias +0.217)
- NegRiskArbStrategy: sum(outcomes) != $1 in multi-outcome markets
- MarketMakingStrategy: two-sided quotes, earn spread minus fees
- EventDrivenNLPStrategy: sentiment contrarian — negative sentiment → BUY YES

All strategies produce Signal objects compatible with SignalEngine.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.risk.manager import Regime
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class StrategySignal:
    """Output from a rule-based strategy."""
    token_id: str
    condition_id: str
    side: Literal["YES", "NO"]
    p_model: float       # Strategy's estimated probability
    p_market: float       # Current market price
    edge: float           # Estimated edge
    strategy: str         # Strategy name
    regime: Regime        # Market regime (for exit rules)
    confidence: float     # 0-1 confidence
    meta: dict = None     # Strategy-specific metadata

    def __post_init__(self):
        if self.meta is None:
            self.meta = {}


from src.utils.fees import polymarket_fee as fee  # noqa: E402


class MeanReversionStrategy:
    """Mean reversion strategy using z-score (Chan Ch.7).

    77% of Polymarket markets are mean-reverting (VR<1, ADF analysis).
    Entry: z-score deviates beyond threshold from mean.
    Exit: revert to mean (target μ) or half-life timeout.
    NO stop loss (Chan Ch.6).

    Uses Ornstein-Uhlenbeck half-life for optimal holding period.
    """

    def __init__(
        self,
        z_entry: float = 2.0,      # Enter when |z-score| > 2
        z_exit: float = 0.5,       # Exit when |z-score| < 0.5
        min_bars: int = 20,        # Need enough history for z-score
        fee_rate: float = 0.0175,
        min_edge: float = 0.02,
    ):
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.min_bars = min_bars
        self.fee_rate = fee_rate
        self.min_edge = min_edge

    def generate(self, market_data: dict) -> StrategySignal | None:
        """Generate MR signal from market data.

        Expected market_data keys:
            token_id, condition_id, midpoint,
            z_score (pre-computed), price_mean, price_std, n_bars
        """
        z = market_data.get("z_score")
        p_market = market_data.get("midpoint")
        p_mean = market_data.get("price_mean")
        n_bars = market_data.get("n_bars", 0)

        if z is None or p_market is None or p_mean is None:
            return None
        if n_bars < self.min_bars:
            return None

        # Entry: price deviated from mean
        if z > self.z_entry:
            # Price too high → expect reversion down → BUY NO
            side = "NO"
            p_model = 1.0 - p_mean  # expect reversion to mean
            edge = p_model - (1.0 - p_market)
        elif z < -self.z_entry:
            # Price too low → expect reversion up → BUY YES
            side = "YES"
            p_model = p_mean
            edge = p_model - p_market
        else:
            return None

        # Check edge after fees
        entry_fee = fee(p_market, self.fee_rate)
        net_edge = abs(edge) - entry_fee
        if net_edge < self.min_edge:
            return None

        return StrategySignal(
            token_id=market_data.get("token_id", ""),
            condition_id=market_data.get("condition_id", ""),
            side=side,
            p_model=p_model,
            p_market=p_market,
            edge=edge,
            strategy="mean_reversion",
            regime=Regime.MEAN_REVERTING,
            confidence=min(abs(z) / 4.0, 1.0),
            meta={"z_score": z, "price_mean": p_mean},
        )


class MomentumStrategy:
    """Momentum strategy (Chan Ch.7).

    Entry: strong price move in one direction (breakout).
    Exit: signal reversal or stop loss (ok for momentum — Chan Ch.6).

    Momentum causes (Chan Ch.7): slow info diffusion, herd behavior.
    """

    def __init__(
        self,
        lookback_hours: int = 24,
        min_return: float = 0.05,   # Min return to trigger signal
        fee_rate: float = 0.0175,
        min_edge: float = 0.02,
    ):
        self.lookback_hours = lookback_hours
        self.min_return = min_return
        self.fee_rate = fee_rate
        self.min_edge = min_edge

    def generate(self, market_data: dict) -> StrategySignal | None:
        """Generate momentum signal.

        Expected market_data keys:
            token_id, condition_id, midpoint,
            ret_24h (return over lookback), volume_ratio (current/avg)
        """
        p_market = market_data.get("midpoint")
        ret = market_data.get("ret_24h", 0.0)
        vol_ratio = market_data.get("volume_ratio", 1.0)

        if p_market is None or ret is None:
            return None

        # Momentum: strong move + high volume confirmation
        if abs(ret) < self.min_return:
            return None
        if vol_ratio < 1.2:  # Need above-average volume
            return None

        if ret > 0:
            side = "YES"
            # Project continuation: price + expected momentum
            p_model = min(p_market + ret * 0.5, 0.95)
            edge = p_model - p_market
        else:
            side = "NO"
            p_model = max(p_market + ret * 0.5, 0.05)
            edge = (1 - p_model) - (1 - p_market)

        entry_fee = fee(p_market, self.fee_rate)
        net_edge = abs(edge) - entry_fee
        if net_edge < self.min_edge:
            return None

        return StrategySignal(
            token_id=market_data.get("token_id", ""),
            condition_id=market_data.get("condition_id", ""),
            side=side,
            p_model=p_model,
            p_market=p_market,
            edge=edge,
            strategy="momentum",
            regime=Regime.MOMENTUM,
            confidence=min(abs(ret) / 0.15, 1.0),
            meta={"ret_24h": ret, "volume_ratio": vol_ratio},
        )


class ResolutionConvergenceStrategy:
    """Resolution convergence strategy (Chan Ch.7 time decay).

    Buy markets where outcome is near-certain but price hasn't fully
    converged to 0 or 1. Prediction markets: binary payoff → hold to
    resolution for guaranteed profit if estimate is correct.

    Best for: markets close to resolution with clear outcomes.
    """

    def __init__(
        self,
        high_prob_threshold: float = 0.90,  # Consider "near certain"
        discount_threshold: float = 0.03,   # Min discount from fair value
        fee_rate: float = 0.0175,
    ):
        self.high_prob_threshold = high_prob_threshold
        self.discount_threshold = discount_threshold
        self.fee_rate = fee_rate

    def generate(self, market_data: dict) -> StrategySignal | None:
        """Generate convergence signal.

        Expected market_data keys:
            token_id, condition_id, midpoint,
            p_model (model's probability), time_to_resolution_hours
        """
        p_market = market_data.get("midpoint")
        p_model = market_data.get("p_model")
        hours_left = market_data.get("time_to_resolution_hours")

        if p_market is None or p_model is None:
            return None

        # Look for near-certain YES outcomes at discount
        if p_model >= self.high_prob_threshold and p_market < p_model - self.discount_threshold:
            side = "YES"
            edge = p_model - p_market
            entry_fee = fee(p_market, self.fee_rate)
            if edge - entry_fee < 0.01:
                return None

            return StrategySignal(
                token_id=market_data.get("token_id", ""),
                condition_id=market_data.get("condition_id", ""),
                side=side,
                p_model=p_model,
                p_market=p_market,
                edge=edge,
                strategy="convergence",
                regime=Regime.MEAN_REVERTING,  # converging to final value
                confidence=min(p_model, 1.0),
                meta={"hours_left": hours_left, "discount": edge},
            )

        # Near-certain NO outcomes (price should be near 0)
        if p_model <= (1 - self.high_prob_threshold) and p_market > p_model + self.discount_threshold:
            side = "NO"
            edge = p_market - p_model
            entry_fee = fee(1 - p_market, self.fee_rate)
            if edge - entry_fee < 0.01:
                return None

            return StrategySignal(
                token_id=market_data.get("token_id", ""),
                condition_id=market_data.get("condition_id", ""),
                side=side,
                p_model=p_model,
                p_market=p_market,
                edge=edge,
                strategy="convergence",
                regime=Regime.MEAN_REVERTING,
                confidence=min(1 - p_model, 1.0),
                meta={"hours_left": hours_left, "discount": edge},
            )

        return None


class ContrarianStrategy:
    """Contrarian strategy exploiting favourite-longshot bias.

    EDA findings: YES bias +0.217 on Polymarket (traders overtrade YES).
    Longshots overperform on Polymarket (unlike Kalshi).
    Strategy: systematically bet against crowd bias.
    """

    def __init__(
        self,
        yes_bias: float = 0.02,     # Min YES overpricing to exploit
        longshot_threshold: float = 0.20,  # What counts as longshot
        fee_rate: float = 0.0175,
        min_edge: float = 0.02,
    ):
        self.yes_bias = yes_bias
        self.longshot_threshold = longshot_threshold
        self.fee_rate = fee_rate
        self.min_edge = min_edge

    def generate(self, market_data: dict) -> StrategySignal | None:
        """Generate contrarian signal.

        Expected market_data keys:
            token_id, condition_id, midpoint,
            calibration_error (market price - true probability),
            volume (for liquidity check)
        """
        p_market = market_data.get("midpoint")
        cal_error = market_data.get("calibration_error")

        if p_market is None:
            return None

        # Strategy 1: YES overpricing — bet NO
        if cal_error is not None and cal_error > self.yes_bias:
            side = "NO"
            p_model = p_market - cal_error  # true probability
            edge = cal_error
            entry_fee = fee(1 - p_market, self.fee_rate)
            if edge - entry_fee < self.min_edge:
                return None

            return StrategySignal(
                token_id=market_data.get("token_id", ""),
                condition_id=market_data.get("condition_id", ""),
                side=side,
                p_model=p_model,
                p_market=p_market,
                edge=edge,
                strategy="contrarian",
                regime=Regime.MEAN_REVERTING,
                confidence=min(cal_error / 0.10, 1.0),
                meta={"calibration_error": cal_error, "type": "yes_bias"},
            )

        return None


class NegRiskArbStrategy:
    """NegRisk arbitrage: sum(outcomes) != $1 in multi-outcome markets.

    From research: $39.6M arb extracted from Polymarket,
    ~100 arb opportunities per NegRisk market.
    NO buying dominates arb profits ($17.3M of $39.6M).
    """

    def __init__(
        self,
        min_deviation: float = 0.02,  # Min sum deviation from $1
        fee_rate: float = 0.0175,
    ):
        self.min_deviation = min_deviation
        self.fee_rate = fee_rate

    def generate(self, market_data: dict) -> list[StrategySignal]:
        """Generate NegRisk arb signals.

        Expected market_data keys:
            outcomes: list of dicts with token_id, condition_id, midpoint
        """
        outcomes = market_data.get("outcomes", [])
        if len(outcomes) < 2:
            return []

        total = sum(o.get("midpoint", 0) for o in outcomes)
        deviation = total - 1.0

        if abs(deviation) < self.min_deviation:
            return []

        signals = []
        if deviation > 0:
            # Sum > $1 → sell (buy NO on) overpriced outcomes
            for o in outcomes:
                p = o.get("midpoint", 0)
                if p > 0.1:  # Don't short already cheap
                    edge = deviation / len(outcomes)  # Simplified
                    entry_fee = fee(1 - p, self.fee_rate)
                    if edge - entry_fee > 0.005:
                        signals.append(StrategySignal(
                            token_id=o.get("token_id", ""),
                            condition_id=o.get("condition_id", ""),
                            side="NO",
                            p_model=p - edge,
                            p_market=p,
                            edge=edge,
                            strategy="negrisk_arb",
                            regime=Regime.MEAN_REVERTING,
                            confidence=min(abs(deviation) / 0.05, 1.0),
                            meta={"sum": total, "deviation": deviation},
                        ))
        else:
            # Sum < $1 → buy (YES on) underpriced outcomes
            for o in outcomes:
                p = o.get("midpoint", 0)
                if p < 0.9:
                    edge = abs(deviation) / len(outcomes)
                    entry_fee = fee(p, self.fee_rate)
                    if edge - entry_fee > 0.005:
                        signals.append(StrategySignal(
                            token_id=o.get("token_id", ""),
                            condition_id=o.get("condition_id", ""),
                            side="YES",
                            p_model=p + edge,
                            p_market=p,
                            edge=edge,
                            strategy="negrisk_arb",
                            regime=Regime.MEAN_REVERTING,
                            confidence=min(abs(deviation) / 0.05, 1.0),
                            meta={"sum": total, "deviation": deviation},
                        ))

        return signals


class MarketMakingStrategy:
    """Market making: two-sided limit orders to earn the spread.

    From research: @defiance_cr $10K→$200-800/day, 80-200% annualized.
    Maker fees lower than taker (+110% PnL from backtest).
    Profitable when: half_spread > fee(mid) + inventory_risk.

    Fee is parabolic: max at p=0.50, minimal at extremes.
    Best MM opportunities: price < 0.20 or > 0.80 (low fees).

    Reference: Polymarket/poly-market-maker (official), bands strategy.
    """

    def __init__(
        self,
        min_spread: float = 0.02,       # Min market spread to participate
        quote_offset: float = 0.01,     # Offset from mid for our quotes
        max_inventory: float = 500.0,   # Max $ inventory per side
        fee_rate: float = 0.0175,
        min_edge: float = 0.005,        # Min profit per round-trip
        skew_factor: float = 0.5,       # Inventory skew strength
    ):
        self.min_spread = min_spread
        self.quote_offset = quote_offset
        self.max_inventory = max_inventory
        self.fee_rate = fee_rate
        self.min_edge = min_edge
        self.skew_factor = skew_factor

    def generate(self, market_data: dict) -> list[StrategySignal]:
        """Generate market making signals (bid + ask).

        Expected market_data keys:
            token_id, condition_id, best_bid, best_ask,
            midpoint, inventory_yes (current YES position $),
            inventory_no (current NO position $), volatility
        """
        bid = market_data.get("best_bid")
        ask = market_data.get("best_ask")
        mid = market_data.get("midpoint")

        if bid is None or ask is None or mid is None:
            return []

        spread = ask - bid
        if spread < self.min_spread:
            return []  # Spread too tight, can't profit

        # Round-trip cost: entry fee + exit fee
        entry_fee = fee(mid, self.fee_rate)
        exit_fee = fee(mid, self.fee_rate)
        round_trip_cost = entry_fee + exit_fee

        # Expected profit per round trip = spread_captured - fees
        half_spread = spread / 2
        expected_profit = half_spread - round_trip_cost
        if expected_profit < self.min_edge:
            return []

        # Inventory skew: shift quotes away from concentrated side
        inv_yes = market_data.get("inventory_yes", 0.0)
        inv_no = market_data.get("inventory_no", 0.0)
        net_inventory = inv_yes - inv_no  # positive = long YES
        skew = self.skew_factor * net_inventory / max(self.max_inventory, 1.0)
        skew = np.clip(skew, -0.03, 0.03)

        # Volatility adjustment: widen quotes in volatile markets
        vol = market_data.get("volatility", 0.0)
        vol_adj = min(vol * 0.5, 0.02)  # up to 2 cents wider

        # Compute quote prices
        our_bid = mid - self.quote_offset - vol_adj - skew
        our_ask = mid + self.quote_offset + vol_adj - skew

        our_bid = np.clip(our_bid, 0.01, 0.99)
        our_ask = np.clip(our_ask, 0.01, 0.99)

        if our_bid >= our_ask:
            return []

        signals = []
        confidence = min(expected_profit / 0.02, 1.0)

        # BUY YES at our_bid (we want to buy low)
        if inv_yes < self.max_inventory:
            signals.append(StrategySignal(
                token_id=market_data.get("token_id", ""),
                condition_id=market_data.get("condition_id", ""),
                side="YES",
                p_model=mid,
                p_market=our_bid,
                edge=expected_profit,
                strategy="market_making",
                regime=Regime.MEAN_REVERTING,
                confidence=confidence,
                meta={
                    "quote_type": "bid",
                    "spread": spread,
                    "our_bid": our_bid,
                    "our_ask": our_ask,
                    "round_trip_cost": round_trip_cost,
                    "skew": skew,
                },
            ))

        # BUY NO at (1 - our_ask) = SELL YES at our_ask
        if inv_no < self.max_inventory:
            signals.append(StrategySignal(
                token_id=market_data.get("token_id", ""),
                condition_id=market_data.get("condition_id", ""),
                side="NO",
                p_model=1 - mid,
                p_market=1 - our_ask,
                edge=expected_profit,
                strategy="market_making",
                regime=Regime.MEAN_REVERTING,
                confidence=confidence,
                meta={
                    "quote_type": "ask",
                    "spread": spread,
                    "our_bid": our_bid,
                    "our_ask": our_ask,
                    "round_trip_cost": round_trip_cost,
                    "skew": skew,
                },
            ))

        return signals


class EventDrivenNLPStrategy:
    """Event-driven strategy using NLP sentiment as contrarian signal.

    Validated on 174 resolved markets (2026-03-06):
    - Comment sentiment inversely correlated with outcome (r=-0.212, p=0.005)
    - Negative sentiment → market resolves YES more often
    - Confirms YES bias (+0.217) from EDA
    - AI-Trader paper: use NLP as feature, NOT standalone buy/sell

    Strategy logic:
    - Collect comments → FinBERT sentiment → contrarian signal
    - Strong negative sentiment + low price → BUY YES (crowd is wrong)
    - Strong positive sentiment + high price → BUY NO (crowd is overconfident)
    - Comment velocity amplifies signal (more comments = stronger conviction)
    - Best combined with other signals (metamodel ensemble)
    """

    def __init__(
        self,
        sentiment_threshold: float = -0.15,  # Min negative sentiment to trigger
        min_comments: int = 5,                # Need enough comments for signal
        velocity_boost: float = 0.3,          # Extra confidence from high velocity
        fee_rate: float = 0.0175,
        min_edge: float = 0.02,
    ):
        self.sentiment_threshold = sentiment_threshold
        self.min_comments = min_comments
        self.velocity_boost = velocity_boost
        self.fee_rate = fee_rate
        self.min_edge = min_edge

    def generate(self, market_data: dict) -> StrategySignal | None:
        """Generate event-driven NLP signal.

        Expected market_data keys:
            token_id, condition_id, midpoint,
            nlp_comment_sentiment (-1 to +1),
            nlp_comment_count,
            nlp_comment_velocity (comments/hour),
            nlp_bullish_keyword_ratio (0 to 1),
            nlp_news_sentiment (optional, for confirmation)
        """
        p_market = market_data.get("midpoint")
        sentiment = market_data.get("nlp_comment_sentiment")
        n_comments = market_data.get("nlp_comment_count", 0)
        velocity = market_data.get("nlp_comment_velocity", 0.0)
        bullish_ratio = market_data.get("nlp_bullish_keyword_ratio", 0.5)

        if p_market is None or sentiment is None:
            return None
        if n_comments < self.min_comments:
            return None

        # Contrarian logic: negative sentiment → BUY YES
        if sentiment < self.sentiment_threshold:
            side = "YES"
            # Edge estimate: stronger negative sentiment = larger expected mispricing
            # r=-0.212 means ~21% of price variance explained by inverse sentiment
            raw_edge = abs(sentiment) * 0.15  # conservative: 15% of sentiment magnitude
            # Boost if keyword ratio confirms (bearish keywords dominant)
            if bullish_ratio < 0.4:
                raw_edge *= 1.2

        # Positive sentiment + high price → BUY NO (overconfident crowd)
        elif sentiment > abs(self.sentiment_threshold) and p_market > 0.60:
            side = "NO"
            raw_edge = sentiment * 0.10  # weaker signal for positive side
        else:
            return None

        # Fee check
        entry_price = p_market if side == "YES" else (1 - p_market)
        entry_fee = fee(entry_price, self.fee_rate)
        net_edge = raw_edge - entry_fee
        if net_edge < self.min_edge:
            return None

        # Confidence: base from sentiment + velocity boost
        base_confidence = min(abs(sentiment) / 0.5, 0.8)
        vel_boost = min(velocity / 10.0, 1.0) * self.velocity_boost
        confidence = min(base_confidence + vel_boost, 1.0)

        if side == "YES":
            p_model = p_market + raw_edge
        else:
            p_model = p_market - raw_edge

        p_model = np.clip(p_model, 0.01, 0.99)

        return StrategySignal(
            token_id=market_data.get("token_id", ""),
            condition_id=market_data.get("condition_id", ""),
            side=side,
            p_model=p_model,
            p_market=p_market,
            edge=raw_edge,
            strategy="event_driven_nlp",
            regime=Regime.MEAN_REVERTING,  # contrarian = mean-reverting behavior
            confidence=confidence,
            meta={
                "sentiment": sentiment,
                "n_comments": n_comments,
                "velocity": velocity,
                "bullish_ratio": bullish_ratio,
                "signal_type": "contrarian_sentiment",
            },
        )
