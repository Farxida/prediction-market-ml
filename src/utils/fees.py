"""Polymarket fee calculation utility.

Shared across strategies, signal engine, and paper trader to avoid duplication.
Fee model: fee(p) = p * (1 - p) * rate (parabolic, max at p=0.50).
"""

import numpy as np


def polymarket_fee(price: float, rate: float = 0.0175) -> float:
    """Compute Polymarket fee for a trade.

    Args:
        price: trade price (0-1)
        rate: fee rate (default 0.0175 = maker, taker is ~0.10)

    Returns:
        fee amount per share
    """
    p = np.clip(price, 0.01, 0.99)
    return p * (1 - p) * rate
