import pandas as pd
import pytest

from quant.backtest import run
from quant.oos import DEFAULT_GRID, rank_key, run_oos, split, sweep
from quant.strategies import sma_cross

D = 86_400_000  # one day in ms


def daily_df(closes):
    closes = list(closes)
    return pd.DataFrame(
        {
            "timestamp": [i * D for i in range(len(closes))],
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1.0] * len(closes),
        }
    )


def rising(n=100, start=100.0, drift=1.0):
    return daily_df([start + drift * i for i in range(n)])


# --- split ---------------------------------------------------------------


def test_split_chronological_no_overlap_no_gap():
    df = rising(10)
    design, proof = split(df, fraction=0.7)
    assert len(design) == 7
    assert len(proof) == 3
    # time only moves forward: everything in design precedes everything in proof
    assert design["timestamp"].max() < proof["timestamp"].min()
    # no gap: proof starts on the very next bar
    assert proof["timestamp"].iloc[0] - design["timestamp"].iloc[-1] == D
    # nothing lost, nothing duplicated
    assert list(design["timestamp"]) + list(proof["timestamp"]) == list(df["timestamp"])


def test_split_rejects_silly_fraction():
    with pytest.raises(ValueError):
        split(rising(10), fraction=0.0)
    with pytest.raises(ValueError):
        split(rising(10), fraction=1.0)


# --- sweep ---------------------------------------------------------------


def test_sweep_matches_manual_runs_on_same_data():
    """Sweep must be a thin loop over run() on the EXACT frame given —
    the leakage guarantee is by construction, this pins it."""
    df = rising(80, drift=0.7)
    grid = [(5, 30), (10, 50)]
    results = sweep(df, grid, exchange_id="binance", timeframe="1d")
    assert len(results) == 2
    for sr, (fast, slow) in zip(results, grid):
        manual = run(df, sma_cross(df, fast, slow), "binance", "1d")
        assert (sr.fast, sr.slow) == (fast, slow)
        assert sr.result.total_return == pytest.approx(manual.total_return)
        assert sr.result.max_drawdown == pytest.approx(manual.max_drawdown)


def test_sweep_sorted_best_rank_first():
    results = sweep(rising(120, drift=0.9), DEFAULT_GRID, exchange_id="binance", timeframe="1d")
    ranks = [sr.rank for sr in results]
    assert ranks == sorted(ranks, reverse=True)


def test_rank_key_return_per_unit_of_pain():
    assert rank_key(total_return=1.0, max_drawdown=-0.5) == pytest.approx(2.0)
    assert rank_key(total_return=0.5, max_drawdown=-0.25) == pytest.approx(2.0)


# --- full out-of-sample pipeline -----------------------------------------


def test_run_oos_consistent_verdict_and_no_leakage_into_selection():
    """The verdict must follow from the numbers reported: LIVED iff the
    tuned strategy beats buy&hold on the proof window."""
    df = rising(200, drift=0.8)
    report = run_oos(df, DEFAULT_GRID, exchange_id="binance", timeframe="1d")
    assert len(report.design_results) == len(DEFAULT_GRID)
    winner = report.design_results[0]
    assert (report.winner_fast, report.winner_slow) == (winner.fast, winner.slow)
    expected_verdict = "LIVED" if report.proof_return > report.proof_buy_hold else "DIED"
    assert report.verdict == expected_verdict
    # hindsight disclosure must exist and be at least as good as our pick
    assert report.hindsight_best >= report.proof_return


def test_default_grid_is_small_and_sane():
    """12 honest configs, fast always below slow."""
    assert len(DEFAULT_GRID) == 12
    assert all(f < s for f, s in DEFAULT_GRID)
