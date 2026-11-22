import pandas as pd
import pytest

from quant.backtest import run
from quant.fees import FeeModel


def price_df(closes):
    closes = list(closes)
    return pd.DataFrame(
        {
            "timestamp": range(len(closes)),
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1.0] * len(closes),
        }
    )


BINANCE_SIDE = 0.001 + 0.0005  # taker + slippage, per side


def test_flat_prices_round_trip_costs_exact():
    """Constant prices, one entry + exit: result must be exactly the cost.

    This is the ground rule from the README made executable — no backtest
    can ever report a cost-free profit.
    """
    df = price_df([100.0] * 5)
    signal = pd.Series([0, 1, 1, 0, 0], dtype=float)
    res = run(df, signal, exchange_id="binance", timeframe="1d")
    assert res.total_return == pytest.approx((1 - BINANCE_SIDE) ** 2 - 1)
    assert res.cost_drag == pytest.approx(1 - (1 - BINANCE_SIDE) ** 2)
    assert res.n_trades == 2  # entry + exit
    assert res.max_drawdown <= 0


def test_no_lookahead_signal_acts_next_bar():
    """A signal set at bar i must earn bar i+1's return, never bar i's.

    Prices [100, 110, 90], go long after seeing bar 1 close: the strategy
    must MISS the +10% of bar 1 and eat the -9.09% of bar 2.
    """
    df = price_df([100.0, 110.0, 90.0])
    signal = pd.Series([0.0, 1.0, 1.0])
    res = run(df, signal, exchange_id="binance", timeframe="1d")
    bar2 = 90.0 / 110.0 - 1
    assert res.total_return == pytest.approx((1 + bar2 - BINANCE_SIDE) - 1)


def test_buy_and_hold_matches_underlying_minus_entry():
    """All-long signal on rising prices: wealth = price gain minus one entry cost."""
    df = price_df([100.0, 120.0, 150.0])
    res = run(df, pd.Series([1.0, 1.0, 1.0]), exchange_id="binance", timeframe="1d")
    # entry cost paid at bar 1 compounds through bar 2's gain
    expected = (1 + 0.2 - BINANCE_SIDE) * (1 + 0.25) - 1
    assert res.total_return == pytest.approx(expected)
    assert res.n_trades == 1
    # bar 0 must be flat (signal decided at its close) -> 2 of 3 bars long
    assert res.exposure == pytest.approx(2 / 3)


def test_max_drawdown_known_path():
    """Wealth 1 -> 2 -> 1 -> 3: deepest peak-to-trough is -50%."""
    df = price_df([100.0, 200.0, 100.0, 300.0])
    res = run(df, pd.Series([1.0, 1.0, 1.0, 1.0]), exchange_id="binance", timeframe="1d")
    assert res.max_drawdown == pytest.approx(-0.5, abs=1e-9)


def test_too_good_result_is_flagged_not_celebrated():
    """Ground rule 2: absurd returns get flagged as possible overfits."""
    df = price_df([100.0 * 2**i for i in range(12)])  # 100x+ in 11 bars
    res = run(df, pd.Series([1.0] * 12), exchange_id="binance", timeframe="1d")
    assert res.flagged
    assert "overfit" in res.flag_reason.lower()


def test_sane_result_not_flagged():
    df = price_df([100.0, 102.0, 101.0, 103.0, 104.0])
    res = run(df, pd.Series([1.0] * 5), exchange_id="binance", timeframe="1d")
    assert not res.flagged
    assert res.flag_reason == ""


def test_run_accepts_custom_fee_model():
    """Stress testing needs cost overrides: a zero-fee model on flat prices
    with a round trip must produce EXACTLY zero loss."""
    df = price_df([100.0] * 5)
    signal = pd.Series([0, 1, 1, 0, 0], dtype=float)
    free = FeeModel(maker=0.0, taker=0.0, slippage=0.0)
    res = run(df, signal, exchange_id="binance", timeframe="1d", fee_model=free)
    assert res.total_return == 0.0
    assert res.cost_drag == 0.0
