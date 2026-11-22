"""Capital allocation: the rules that outlive any strategy.

From the onboarding doc: "Position sizing and risk management matters more
than the strategy itself. A mediocre strategy with strict risk management
can survive; a brilliant strategy with none will eventually blow up an
account." This module is those rules, made executable:

  - diversification: weights normalized, per-coin capped; capped excess is
    left as CASH, never force-fed elsewhere
  - the 1-2% rule: a position is sized so that hitting its stop loses
    exactly the agreed risk fraction of total equity
  - the drawdown watchdog: the max acceptable drawdown is decided UP FRONT
    and monitored, not rationalized after the fact
"""
MAX_DD = 0.20  # agreed in advance: -20% from peak stops the experiment
DEFAULT_PER_COIN_CAP = 0.25  # max 25% of equity in any one coin
START_BANKROLL = 5_000.0  # notional portfolio the advisor sizes against


def allocate(
    equity: float,
    weights: dict[str, float],
    per_coin_cap: float = DEFAULT_PER_COIN_CAP,
) -> dict[str, float]:
    """Dollar amounts per symbol.

    Weights are TARGET FRACTIONS OF EQUITY. If they sum above 100% they are
    scaled DOWN to fit — never scaled up (a modest request must not get
    inflated by a greedy neighbor). The per-coin cap applies after, and
    whatever the cap squeezes out stays in cash.
    """
    if not weights:
        raise ValueError("no weights to allocate")
    if equity <= 0:
        raise ValueError(f"equity must be positive, got {equity}")
    if per_coin_cap <= 0:
        raise ValueError(f"per_coin_cap must be positive, got {per_coin_cap}")

    total = sum(weights.values())
    scale = min(1.0, 1.0 / total)
    return {
        symbol: equity * min(weight * scale, per_coin_cap) for symbol, weight in weights.items()
    }


def position_size(equity: float, risk_pct: float, entry: float, stop: float) -> float:
    """Units of an asset such that a fill at entry and a stop-out at `stop`
    loses exactly risk_pct of total equity. The 1-2% rule, as a formula."""
    if stop >= entry:
        raise ValueError(f"stop {stop} must be below entry {entry}")
    risk_dollars = equity * risk_pct
    return risk_dollars / (entry - stop)


def drawdown_watch(equity_curve: list[float], max_dd: float = MAX_DD) -> tuple[bool, float]:
    """(currently_breached, current_drawdown) — peak-to-trough from the
    running maximum. A past breach that has recovered is a lesson, not an
    alarm; the flag reflects where you ARE."""
    if not equity_curve:
        return False, 0.0
    peak = max(equity_curve)
    current_dd = equity_curve[-1] / peak - 1
    return current_dd < -max_dd, current_dd
