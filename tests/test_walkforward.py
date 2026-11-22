import pandas as pd
import pytest

from quant.walkforward import fixed_walkforward, run_walkforward, windows
from quant.fees import FeeModel

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


def rising(n=1000, start=100.0, drift=0.5):
    return daily_df([start + drift * i for i in range(n)])


SMALL_GRID = [(5, 30), (10, 50), (20, 100)]


# --- windows -------------------------------------------------------------


def test_windows_back_to_back_no_overlap_complete_tail():
    df = rising(1000)
    wins = windows(df, n_windows=5, min_design_frac=0.5)
    assert len(wins) == 5
    prev_end = None
    for design, proof in wins:
        assert len(proof) > 0
        if prev_end is not None:
            assert proof["timestamp"].iloc[0] == prev_end + D  # back-to-back
        prev_end = proof["timestamp"].iloc[-1]
    # together the proofs cover everything after the initial design point
    first_proof_start = wins[0][1]["timestamp"].iloc[0]
    tail = df[df["timestamp"] >= first_proof_start]
    covered = [t for _, proof in wins for t in proof["timestamp"]]
    assert covered == list(tail["timestamp"])


def test_windows_anchored_design_never_sees_proof():
    """Anchored = each design starts at day zero and ends the bar before
    its proof. This is the no-lookahead guarantee for the selection step."""
    df = rising(1000)
    wins = windows(df, n_windows=4, min_design_frac=0.5)
    for design, proof in wins:
        assert design["timestamp"].iloc[0] == 0
        assert design["timestamp"].iloc[-1] < proof["timestamp"].iloc[0]
        assert proof["timestamp"].iloc[0] - design["timestamp"].iloc[-1] == D


def test_windows_rejects_degenerate_setup():
    with pytest.raises(ValueError):
        windows(rising(100), n_windows=50, min_design_frac=0.9)  # slivers


# --- full walk-forward ---------------------------------------------------


def test_verdicts_follow_numbers_window_by_window():
    report = run_walkforward(rising(600), SMALL_GRID, exchange_id="binance", timeframe="1d", n_windows=4)
    assert len(report.window_results) == 4
    for wr in report.window_results:
        expected = "LIVED" if wr.proof_return > wr.proof_buy_hold else "DIED"
        assert wr.verdict == expected


def test_chained_return_is_compounded_window_returns():
    report = run_walkforward(rising(600), SMALL_GRID, exchange_id="binance", timeframe="1d", n_windows=4)
    chained = 1.0
    for wr in report.window_results:
        chained *= 1 + wr.proof_return
    assert report.chained_return == pytest.approx(chained - 1)


def test_summary_lived_count_matches_windows():
    report = run_walkforward(rising(600), SMALL_GRID, exchange_id="binance", timeframe="1d", n_windows=4)
    assert report.lived == sum(1 for wr in report.window_results if wr.verdict == "LIVED")
    assert report.total_windows == 4


def test_walkforward_honors_fee_model():
    """Zero fees must strictly beat default fees on a trading series —
    proves the override reaches every run() call inside the loop."""
    df = rising(600)
    free = FeeModel(maker=0.0, taker=0.0, slippage=0.0)
    base = run_walkforward(df, SMALL_GRID, exchange_id="binance", timeframe="1d", n_windows=3)
    free_rep = run_walkforward(df, SMALL_GRID, exchange_id="binance", timeframe="1d", n_windows=3, fee_model=free)
    assert free_rep.chained_return > base.chained_return


# --- fixed-parameter walk-forward (no tuning = no selection bias) ---------


def always_long(df):
    return pd.Series(1.0, index=df.index)


def test_fixed_walkforward_verdicts_chained_and_labels():
    df = rising(600)
    rep = fixed_walkforward(df, always_long, "always", exchange_id="binance", timeframe="1d", n_windows=3)
    assert len(rep.window_results) == 3
    chained = 1.0
    for wr in rep.window_results:
        assert wr.label == "always"
        expected = "LIVED" if wr.proof_return > wr.proof_buy_hold else "DIED"
        assert wr.verdict == expected
        chained *= 1 + wr.proof_return
    assert rep.chained_return == pytest.approx(chained - 1)


def test_fixed_always_long_never_beats_buyhold_with_fees():
    """Same trades as buy&hold minus one entry cost per window: must lose
    every window. Costs always bite — this pins it."""
    df = rising(600)
    rep = fixed_walkforward(df, always_long, "always", exchange_id="binance", timeframe="1d", n_windows=3)
    assert all(wr.verdict == "DIED" for wr in rep.window_results)
    assert rep.chained_return < rep.buy_hold_return
