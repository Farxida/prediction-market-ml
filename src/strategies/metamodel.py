"""Strategy Router (Metamodel) — routes markets to the best strategy.

Based on backtest results (688 HTR trades, 2925 maker trades, 90 baseline trades):
- Price is the #1 routing feature: low prices (0-0.20) → HTR dominates ($19/trade)
- Category matters at low prices: crypto/sports best, culture worst at high prices
- Liquidity: $1K-100K optimal, stable across levels
- Maker orders: +110% vs taker at all price levels

Strategy selection:
  1. NegRisk arb (if sum deviation exists) — highest priority
  2. Convergence (near resolution + high confidence)
  3. Mean reversion at low price (0-0.20, best PnL zone)
  4. Contrarian (if calibration error detected)
  5. Event-driven NLP (contrarian sentiment, r=-0.212)
  6. Mean reversion at any price
  7. Momentum (only if confirmed regime + volume)
  8. Market making (if spread wide enough to profit)

Order type: always maker (limit) unless urgency requires taker.
"""

from dataclasses import dataclass, field
from typing import Literal

from src.strategies.strategies import (
    ContrarianStrategy,
    EventDrivenNLPStrategy,
    MarketMakingStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    NegRiskArbStrategy,
    ResolutionConvergenceStrategy,
    StrategySignal,
)
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class RouteDecision:
    """Output of strategy router."""
    strategy_name: str
    order_type: Literal["maker", "taker"]
    priority: int                    # lower = higher priority
    reason: str
    skip: bool = False               # True = don't trade this market
    skip_reason: str = ""


@dataclass
class StrategyRouter:
    """Routes markets to the best strategy based on backtest findings.

    Backtest evidence:
    - HTR at p<0.20: avg $19/trade, WR 81%, 474 maker trades
    - HTR at p 0.20-0.50: avg $1.8/trade, WR 68% (marginal)
    - HTR at p>0.80: avg $0.97/trade, WR 70% (barely profitable)
    - Culture at high prices: LOSES money (avg -$5.28, WR 29%)
    - Crypto at all prices: consistently profitable (WR 89%)
    """

    # Price thresholds from backtest analysis
    price_sweet_spot: float = 0.20       # Best performance below this
    price_marginal_upper: float = 0.50   # Marginal above this
    price_danger_zone: float = 0.70      # Poor performance above this

    # Minimum liquidity
    min_liquidity: float = 500.0         # $500 minimum

    # Strategy instances (lazy init)
    _strategies: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._strategies = {
            "mean_reversion": MeanReversionStrategy(),
            "momentum": MomentumStrategy(),
            "convergence": ResolutionConvergenceStrategy(),
            "contrarian": ContrarianStrategy(),
            "negrisk_arb": NegRiskArbStrategy(),
            "market_making": MarketMakingStrategy(),
            "event_driven_nlp": EventDrivenNLPStrategy(),
        }

    def route(self, market_data: dict) -> RouteDecision:
        """Decide which strategy to use for a market.

        Args:
            market_data: dict with keys:
                midpoint, liquidity, cat_is_sports, cat_is_politics,
                cat_is_crypto, cat_is_culture, cat_is_games,
                is_negrisk, outcomes (for arb), z_score, ret_24h,
                volume_ratio, p_model, time_to_resolution_hours,
                calibration_error
        """
        p = market_data.get("midpoint")
        liq = market_data.get("liquidity", 0)

        if p is None:
            return RouteDecision("none", "maker", 99, "", skip=True,
                                 skip_reason="No price data")

        if liq > 0 and liq < self.min_liquidity:
            return RouteDecision("none", "maker", 99, "", skip=True,
                                 skip_reason=f"Low liquidity: ${liq:.0f}")

        # Culture at high prices loses money
        if market_data.get("cat_is_culture", 0) and p > self.price_danger_zone:
            return RouteDecision("none", "maker", 99, "", skip=True,
                                 skip_reason="Culture + high price = negative edge")

        # Priority 1: NegRisk arbitrage
        if market_data.get("is_negrisk") and market_data.get("outcomes"):
            signals = self._strategies["negrisk_arb"].generate(market_data)
            if signals:
                return RouteDecision("negrisk_arb", "maker", 1,
                                     f"NegRisk arb: {len(signals)} signals")

        # Priority 2: Resolution convergence (near-certain outcomes)
        if market_data.get("p_model") is not None:
            sig = self._strategies["convergence"].generate(market_data)
            if sig:
                return RouteDecision("convergence", "maker", 2,
                                     f"Convergence: edge={sig.edge:.3f}")

        # Priority 3: Mean reversion at low prices (best PnL zone)
        if p <= self.price_sweet_spot and market_data.get("z_score") is not None:
            sig = self._strategies["mean_reversion"].generate(market_data)
            if sig:
                return RouteDecision("mean_reversion", "maker", 3,
                                     f"MR at low price: z={market_data['z_score']:.2f}")

        # Priority 4: Contrarian (YES bias exploit)
        if market_data.get("calibration_error") is not None:
            sig = self._strategies["contrarian"].generate(market_data)
            if sig:
                return RouteDecision("contrarian", "maker", 4,
                                     f"Contrarian: cal_err={market_data['calibration_error']:.3f}")

        # Priority 5: Event-driven NLP (contrarian sentiment, r=-0.212)
        if market_data.get("nlp_comment_sentiment") is not None:
            sig = self._strategies["event_driven_nlp"].generate(market_data)
            if sig:
                return RouteDecision("event_driven_nlp", "maker", 5,
                                     f"NLP contrarian: sent={market_data['nlp_comment_sentiment']:.2f}")

        # Priority 6: Mean reversion at any price
        if market_data.get("z_score") is not None:
            sig = self._strategies["mean_reversion"].generate(market_data)
            if sig:
                return RouteDecision("mean_reversion", "maker", 6,
                                     f"MR: z={market_data['z_score']:.2f}")

        # Priority 7: Momentum (only with volume confirmation)
        if market_data.get("ret_24h") is not None:
            sig = self._strategies["momentum"].generate(market_data)
            if sig:
                return RouteDecision("momentum", "maker", 7,
                                     f"Momentum: ret={market_data['ret_24h']:.3f}")

        # Priority 8: Market making (if spread wide enough)
        if market_data.get("best_bid") is not None and market_data.get("best_ask") is not None:
            signals = self._strategies["market_making"].generate(market_data)
            if signals:
                return RouteDecision("market_making", "maker", 8,
                                     f"MM: spread={market_data.get('best_ask', 0) - market_data.get('best_bid', 0):.3f}")

        # No strategy triggered
        zone = "sweet" if p <= self.price_sweet_spot else \
               "marginal" if p <= self.price_marginal_upper else \
               "danger" if p <= self.price_danger_zone else "extreme"
        return RouteDecision("none", "maker", 99, "", skip=True,
                             skip_reason=f"No signal (price zone: {zone})")

    def generate_signals(self, market_data: dict) -> list[StrategySignal]:
        """Generate all applicable signals for a market, ranked by priority."""
        signals = []

        for name, strategy in self._strategies.items():
            if name == "negrisk_arb":
                if market_data.get("is_negrisk") and market_data.get("outcomes"):
                    arb_signals = strategy.generate(market_data)
                    signals.extend(arb_signals)
            elif name == "market_making":
                if market_data.get("best_bid") is not None:
                    mm_signals = strategy.generate(market_data)
                    signals.extend(mm_signals)
            else:
                sig = strategy.generate(market_data)
                if sig:
                    signals.append(sig)

        # Sort by edge (descending)
        signals.sort(key=lambda s: abs(s.edge), reverse=True)
        return signals

    def should_use_maker(self, market_data: dict) -> bool:
        """Always prefer maker orders (110% better PnL from backtest)."""
        # Only use taker if market is about to resolve and edge is large
        hours_left = market_data.get("time_to_resolution_hours")
        edge = market_data.get("edge", 0)
        if hours_left is not None and hours_left < 1 and edge > 0.10:
            return False  # Taker for urgent high-edge trades
        return True

    def get_price_zone(self, price: float) -> str:
        """Classify market by price zone for logging."""
        if price <= self.price_sweet_spot:
            return "sweet_spot"
        elif price <= self.price_marginal_upper:
            return "marginal"
        elif price <= self.price_danger_zone:
            return "caution"
        else:
            return "high_price"
