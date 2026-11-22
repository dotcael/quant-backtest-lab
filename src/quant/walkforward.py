"""Walk-forward evaluation: the tune-then-test exam, repeated across history.

One 70/30 pass (oos.py) answers "did it work once?". One pass is a coin flip.
Walk-forward repeats the exam: split the post-design era into consecutive
proof windows; before each, re-tune on everything earlier; test the fresh
winner on that window exactly once. The deliverable is the CHAINED return —
what you would have actually experienced, window after window, re-tuning as
you go, after fees — against buy&hold over the same span.

Selection still never sees the future: each design slice is anchored at day
zero and ends the bar before its proof window (pinned by tests).
"""
from dataclasses import dataclass

import pandas as pd

from quant.oos import DEFAULT_GRID, OosReport, sweep
from quant.strategies import buy_and_hold
from quant.backtest import run

MIN_WINDOW_BARS = 60  # a proof window shorter than ~2 months tells us nothing


def windows(
    df: pd.DataFrame,
    n_windows: int = 5,
    min_design_frac: float = 0.5,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Anchored walk-forward windows -> [(design, proof), ...].

    Proofs are consecutive, non-overlapping, and together cover everything
    after the initial design slice. Designs grow: each includes all history
    before its proof.
    """
    if not 0.0 < min_design_frac < 1.0:
        raise ValueError(f"min_design_frac must be in (0, 1), got {min_design_frac}")
    min_design = int(len(df) * min_design_frac)
    step = (len(df) - min_design) // n_windows
    if step < MIN_WINDOW_BARS:
        raise ValueError(
            f"proof windows would be {step} bars (< {MIN_WINDOW_BARS}) — "
            "use fewer windows or more history"
        )
    out = []
    for i in range(n_windows):
        start = min_design + i * step
        end = start + step if i < n_windows - 1 else len(df)
        out.append((df.iloc[:start].reset_index(drop=True), df.iloc[start:end].reset_index(drop=True)))
    return out


@dataclass
class WindowResult:
    proof_start: int
    proof_end: int
    label: str  # e.g. "SMA 10/30" (tuned) or "breakout 20/10" (fixed)
    proof_return: float
    proof_buy_hold: float
    verdict: str


@dataclass
class WalkForwardReport:
    window_results: list[WindowResult]
    lived: int
    total_windows: int
    chained_return: float
    buy_hold_return: float
    param_changes: int
    buy_hold_start: int
    buy_hold_end: int

    def report(self) -> str:
        lines = []
        prev_params = None
        changes = 0
        for wr in self.window_results:
            mark = "LIVED" if wr.verdict == "LIVED" else "DIED"
            lines.append(
                f"  proof {pd.to_datetime(wr.proof_start, unit='ms'):%Y-%m-%d} -> "
                f"{pd.to_datetime(wr.proof_end, unit='ms'):%Y-%m-%d} | {wr.label} "
                f"| {wr.proof_return:+7.1%} vs B&H {wr.proof_buy_hold:+7.1%} | {mark}"
            )
            if prev_params is not None and wr.label != prev_params:
                changes += 1
            prev_params = wr.label
        span = (
            f"{pd.to_datetime(self.buy_hold_start, unit='ms'):%Y-%m-%d} -> "
            f"{pd.to_datetime(self.buy_hold_end, unit='ms'):%Y-%m-%d}"
        )
        lines.append(
            f"summary: LIVED {self.lived}/{self.total_windows} windows | "
            f"chained {self.chained_return:+.1%} vs buy&hold {self.buy_hold_return:+.1%} "
            f"over {span} | winner params changed {changes}/{self.total_windows} times"
        )
        return "\n".join(lines)


def run_walkforward(
    df: pd.DataFrame,
    grid: list[tuple[int, int]] = None,
    exchange_id: str = "binance",
    timeframe: str = "1d",
    n_windows: int = 5,
    min_design_frac: float = 0.5,
    fee_model=None,
) -> WalkForwardReport:
    """Repeated tune-on-past / test-on-next. Grid defaults to oos.DEFAULT_GRID."""
    grid = grid or DEFAULT_GRID
    from quant.strategies import sma_cross  # local import keeps module deps honest

    results: list[WindowResult] = []
    first_proof_start = None
    for design, proof in windows(df, n_windows, min_design_frac):
        if first_proof_start is None:
            first_proof_start = proof["timestamp"].iloc[0]
        winner = sweep(design, grid, exchange_id, timeframe, fee_model=fee_model)[0]
        res = run(proof, sma_cross(proof, winner.fast, winner.slow), exchange_id, timeframe, fee_model=fee_model)
        bh = run(proof, buy_and_hold(proof), exchange_id, timeframe, fee_model=fee_model)
        results.append(
            WindowResult(
                proof_start=int(proof["timestamp"].iloc[0]),
                proof_end=int(proof["timestamp"].iloc[-1]),
                label=f"SMA {winner.fast}/{winner.slow}",
                proof_return=res.total_return,
                proof_buy_hold=bh.total_return,
                verdict="LIVED" if res.total_return > bh.total_return else "DIED",
            )
        )

    return _finalize(results, df, first_proof_start, exchange_id, timeframe, fee_model)


def fixed_walkforward(
    df: pd.DataFrame,
    signal_fn,
    label: str,
    exchange_id: str = "binance",
    timeframe: str = "1d",
    n_windows: int = 5,
    min_design_frac: float = 0.5,
    fee_model=None,
) -> WalkForwardReport:
    """Walk-forward a FIXED-parameter strategy: signal_fn(proof_df) -> signal.

    No design window is consumed because nothing is tuned — textbook
    parameters only. That removes selection bias entirely; whatever this
    reports is what the plain recipe would have actually done.
    """
    results: list[WindowResult] = []
    first_proof_start = None
    for _, proof in windows(df, n_windows, min_design_frac):
        if first_proof_start is None:
            first_proof_start = proof["timestamp"].iloc[0]
        res = run(proof, signal_fn(proof), exchange_id, timeframe, fee_model=fee_model)
        bh = run(proof, buy_and_hold(proof), exchange_id, timeframe, fee_model=fee_model)
        results.append(
            WindowResult(
                proof_start=int(proof["timestamp"].iloc[0]),
                proof_end=int(proof["timestamp"].iloc[-1]),
                label=label,
                proof_return=res.total_return,
                proof_buy_hold=bh.total_return,
                verdict="LIVED" if res.total_return > bh.total_return else "DIED",
            )
        )

    return _finalize(results, df, first_proof_start, exchange_id, timeframe, fee_model)


def _finalize(results, df, first_proof_start, exchange_id, timeframe, fee_model) -> WalkForwardReport:
    chained = 1.0
    for wr in results:
        chained *= 1 + wr.proof_return

    tail = df[df["timestamp"] >= first_proof_start]
    bh_tail = run(tail, buy_and_hold(tail), exchange_id, timeframe, fee_model=fee_model)

    labels = [wr.label for wr in results]
    param_changes = sum(1 for a, b in zip(labels, labels[1:]) if a != b)

    return WalkForwardReport(
        window_results=results,
        lived=sum(1 for wr in results if wr.verdict == "LIVED"),
        total_windows=len(results),
        chained_return=chained - 1,
        buy_hold_return=bh_tail.total_return,
        param_changes=param_changes,
        buy_hold_start=int(tail["timestamp"].iloc[0]),
        buy_hold_end=int(tail["timestamp"].iloc[-1]),
    )


def main() -> None:
    """CLI: walk-forward every stored daily series. --stress X reruns with
    all costs multiplied by X (torture test)."""
    import argparse

    from quant import config, fees, store
    from quant.strategies import breakout, crash_shield, mean_reversion

    parser = argparse.ArgumentParser()
    parser.add_argument("--stress", type=float, default=1.0, help="multiply all costs by this")
    args = parser.parse_args()

    fee_model = fees.FeeModel.for_exchange(config.EXCHANGES[0], stress=args.stress) if args.stress != 1.0 else None
    banner = f" (STRESS x{args.stress:g} costs — round-trip {fee_model.round_trip_cost():.2%})" if fee_model else ""

    for symbol in config.SYMBOLS:
        path = store.parquet_path(config.EXCHANGES[0], symbol, "1d")
        if not path.exists():
            print(f"[walkforward] no daily data for {symbol}, skipping")
            continue
        df = pd.read_parquet(path)
        print(f"\n=== {symbol} 1d walk-forward ({len(df)} bars){banner} ===")
        print("[tuned SMA family]")
        print(run_walkforward(df, exchange_id=config.EXCHANGES[0], fee_model=fee_model).report())
        print("[fixed breakout 20/10 — untuned]")
        print(
            fixed_walkforward(
                df, breakout, "breakout 20/10", exchange_id=config.EXCHANGES[0], fee_model=fee_model
            ).report()
        )
        print("[fixed mean-reversion RSI 14/30/50 — untuned]")
        print(
            fixed_walkforward(
                df, mean_reversion, "RSI 14/30/50", exchange_id=config.EXCHANGES[0], fee_model=fee_model
            ).report()
        )
        print("[fixed crash-shield 200 — untuned airbag]")
        print(
            fixed_walkforward(
                df, crash_shield, "crash-shield 200", exchange_id=config.EXCHANGES[0], fee_model=fee_model
            ).report()
        )


if __name__ == "__main__":
    main()
