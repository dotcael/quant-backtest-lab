"""Single source of truth for trading costs.

Ground rule from onboarding: NO backtest runs fee-free or slippage-free.
Every backtest imports its cost model from here — a backtest that doesn't
import this module is wrong by construction.

Base-tier SPOT fees (0.001 = 0.1%):
  Binance: 0.10% maker / 0.10% taker (base tier)
  Kraken:  0.16% maker / 0.26% taker (base tier)
Re-check exchange docs before going deeper; these are the base tiers
before volume discounts.

Withdrawal/network fees are separate on-chain costs — factored in only
when a strategy actually moves funds between venues.
"""
from dataclasses import dataclass

MAKER_TAKER = {
    "binance": (0.001, 0.001),
    "kraken": (0.0016, 0.0026),
}

# Slippage assumption for high-liquidity pairs (BTC, ETH) on top exchanges.
# Small-cap coins need a bigger number — per-symbol modeling comes later.
DEFAULT_SLIPPAGE = 0.0005  # 5 basis points


@dataclass(frozen=True)
class FeeModel:
    maker: float
    taker: float
    slippage: float

    @classmethod
    def for_exchange(cls, exchange_id: str, stress: float = 1.0) -> "FeeModel":
        """stress > 1 is the torture-test knob: rerun a backtest with all
        costs multiplied and see if the edge survives. If a strategy only
        works at stress=1, it was priced right at the edge of noise."""
        maker, taker = MAKER_TAKER[exchange_id]
        return cls(maker=maker * stress, taker=taker * stress, slippage=DEFAULT_SLIPPAGE * stress)

    def round_trip_cost(self) -> float:
        """Friction for one entry + exit (taker fees + slippage on each side).

        Example (Binance): 2 * (0.001 + 0.0005) = 0.003 — a strategy must
        clear +0.3% per trade before it earns anything. This number is why
        fee-free backtests lie, especially for small accounts.
        """
        return 2 * (self.taker + self.slippage)


def book_slippage(levels: list, notional: float) -> float:
    """Real market-order cost by walking an order book.

    `levels` is one side of the book, best price first (ccxt format:
    [[price, amount], ...], amount in base units). `notional` is quote
    currency to spend (e.g. 1000 = $1000). Returns VWAP-fill vs best
    price as a fraction — positive for asks (buying pushes price up),
    negative for bids.

    This measures what the flat DEFAULT_SLIPPAGE assumption can only
    estimate: the actual depth behind the best quote.
    """
    best = levels[0][0]
    remaining = notional
    units = 0.0
    for price, amount in levels:
        spend = min(price * amount, remaining)
        units += spend / price
        remaining -= spend
        if remaining <= 1e-9:
            break
    if remaining > 1e-9:
        raise ValueError(f"book too thin for {notional} notional")
    vwap = notional / units
    return vwap / best - 1.0
