"""Out-of-sample evaluation: tune on old data, prove on new data.

Why this module exists: any parameter choice made by looking at a data set
and then scored on that SAME data set is graded with the answer key in hand.
With 12 SMA settings, one of them will look brilliant by pure luck. The only
honest score is on data the selection process never saw.

Protocol (stated plainly):
  1. split history chronologically: older `fraction` = design, rest = proof
  2. try every config in the grid on DESIGN only; rank by return per unit
     of drawdown ("return per unit of pain" — avoids crowning lottery tickets)
  3. take the single design winner, run it ONCE on PROOF
  4. verdict: LIVED if it beats buy&hold on proof, else DIED
  5. disclose how many configs were tried, and what pure hindsight would
     have scored on proof — that gap is the price of selection bias

Note: the proof run starts cold — its SMA warmup eats the first bars, just
like a real deployment starting mid-history would. No peeking backwards.
"""
from dataclasses import dataclass, field

import pandas as pd

from quant.backtest import BacktestResult, run
from quant.strategies import buy_and_hold, sma_cross

DEFAULT_GRID: list[tuple[int, int]] = [
    (fast, slow)
    for fast in (5, 10, 20)
    for slow in (30, 50, 100, 200)
    if fast < slow
]


def split(df: pd.DataFrame, fraction: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split -> (design, proof). Never shuffled."""
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")
    cut = int(len(df) * fraction)
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)


def rank_key(total_return: float, max_drawdown: float) -> float:
    """Return per unit of pain. Zero-drawdown edge case: denominator floored
    so a monotonic dream-run ranks huge but never crashes."""
    pain = abs(max_drawdown) if max_drawdown < 0 else 1e-9
    return total_return / pain


@dataclass
class SweepResult:
    fast: int
    slow: int
    rank: float
    result: BacktestResult


def sweep(
    df: pd.DataFrame,
    grid: list[tuple[int, int]],
    exchange_id: str,
    timeframe: str,
    fee_model=None,
) -> list[SweepResult]:
    """Score every config on the given frame ONLY (leakage-free by
    construction — the caller passes design data, never proof)."""
    out = []
    for fast, slow in grid:
        res = run(df, sma_cross(df, fast, slow), exchange_id, timeframe, fee_model=fee_model)
        out.append(SweepResult(fast=fast, slow=slow, rank=rank_key(res.total_return, res.max_drawdown), result=res))
    return sorted(out, key=lambda sr: sr.rank, reverse=True)


@dataclass
class OosReport:
    design_start: int
    design_end: int
    proof_start: int
    proof_end: int
    design_results: list[SweepResult]
    winner_fast: int
    winner_slow: int
    proof_return: float
    proof_buy_hold: float
    proof_max_drawdown: float
    proof_trades: int
    hindsight_best: float
    verdict: str
    extras: dict = field(default_factory=dict)

    def report(self) -> str:
        win = self.design_results[0]
        lines = [
            f"design {pd.to_datetime(self.design_start, unit='ms'):%Y-%m-%d} -> "
            f"{pd.to_datetime(self.design_end, unit='ms'):%Y-%m-%d} | "
            f"proof {pd.to_datetime(self.proof_start, unit='ms'):%Y-%m-%d} -> "
            f"{pd.to_datetime(self.proof_end, unit='ms'):%Y-%m-%d}",
            f"tried {len(self.design_results)} configs on design; "
            f"winner SMA {self.winner_fast}/{self.winner_slow} "
            f"(design: return {win.result.total_return:+.1%}, maxdd {win.result.max_drawdown:.1%})",
            f"PROOF (unseen): return {self.proof_return:+.1%} vs buy&hold {self.proof_buy_hold:+.1%} "
            f"(maxdd {self.proof_max_drawdown:.1%}, trades {self.proof_trades})",
            f"hindsight best on proof was {self.hindsight_best:+.1%} — that gap is luck we did NOT capture",
            f"verdict: {self.verdict}",
        ]
        return "\n".join(lines)


def run_oos(
    df: pd.DataFrame,
    grid: list[tuple[int, int]],
    exchange_id: str,
    timeframe: str,
    fraction: float = 0.7,
) -> OosReport:
    """Full protocol on one data set. Selection touches design only."""
    design, proof = split(df, fraction)

    design_results = sweep(design, grid, exchange_id, timeframe)
    winner = design_results[0]

    proof_res = run(proof, sma_cross(proof, winner.fast, winner.slow), exchange_id, timeframe)
    proof_bh = run(proof, buy_and_hold(proof), exchange_id, timeframe)

    hindsight_best = max(
        run(proof, sma_cross(proof, fast, slow), exchange_id, timeframe).total_return
        for fast, slow in grid
    )

    return OosReport(
        design_start=int(design["timestamp"].iloc[0]),
        design_end=int(design["timestamp"].iloc[-1]),
        proof_start=int(proof["timestamp"].iloc[0]),
        proof_end=int(proof["timestamp"].iloc[-1]),
        design_results=design_results,
        winner_fast=winner.fast,
        winner_slow=winner.slow,
        proof_return=proof_res.total_return,
        proof_buy_hold=proof_bh.total_return,
        proof_max_drawdown=proof_res.max_drawdown,
        proof_trades=proof_res.n_trades,
        hindsight_best=hindsight_best,
        verdict="LIVED" if proof_res.total_return > proof_bh.total_return else "DIED",
    )


def main() -> None:
    """CLI: honest out-of-sample check for every stored daily series."""
    from quant import config, store

    for symbol in config.SYMBOLS:
        path = store.parquet_path(config.EXCHANGES[0], symbol, "1d")
        if not path.exists():
            print(f"[oos] no daily data for {symbol}, skipping")
            continue
        df = pd.read_parquet(path)
        print(f"\n=== {symbol} 1d ({len(df)} bars) ===")
        print(run_oos(df, DEFAULT_GRID, config.EXCHANGES[0], "1d").report())


if __name__ == "__main__":
    main()
