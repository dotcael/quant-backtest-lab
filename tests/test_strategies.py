import pandas as pd
import pytest

from quant.strategies import breakout, buy_and_hold, crash_shield, mean_reversion, rsi, sma_cross


def ohlcv(closes):
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


def test_buy_and_hold_always_long():
    sig = buy_and_hold(ohlcv([100.0, 90.0, 80.0]))
    assert list(sig) == [1.0, 1.0, 1.0]


def test_sma_cross_enters_when_fast_above_slow():
    sig = sma_cross(ohlcv([1.0, 2.0, 3.0, 4.0, 5.0]), fast=2, slow=3)
    # sma2=[_,1.5,2.5,3.5,4.5] sma3=[_,_,2,3,4] -> fast>slow from index 2
    assert list(sig) == [0.0, 0.0, 1.0, 1.0, 1.0]


def test_sma_cross_exits_when_fast_falls_below():
    closes = [5.0, 4.0, 3.0, 2.0, 1.0]  # falling: fast under slow after warmup
    sig = sma_cross(ohlcv(closes), fast=2, slow=3)
    assert list(sig) == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_sma_cross_warmup_is_flat():
    """Not enough data to compute either SMA -> stay out, never guess."""
    sig = sma_cross(ohlcv([1.0, 2.0]), fast=2, slow=3)
    assert list(sig) == [0.0, 0.0]


# --- RSI (the dip detector) ----------------------------------------------


def test_rsi_all_rising_is_max():
    df = ohlcv([float(i) for i in range(1, 31)])
    assert rsi(df).iloc[1:].min() == pytest.approx(100.0)


def test_rsi_all_falling_is_min():
    df = ohlcv([float(31 - i) for i in range(1, 31)])
    assert rsi(df).iloc[1:].max() == pytest.approx(0.0)


def test_rsi_flat_market_is_neutral_50():
    df = ohlcv([100.0] * 20)
    assert (rsi(df) == 50.0).all()


# --- breakout (Donchian: buy new highs, exit on breakdown) ----------------


def test_breakout_enters_on_new_high_after_warmup():
    df = ohlcv([float(i) for i in range(1, 41)])  # monotonic new highs
    sig = breakout(df, entry_bars=20, exit_bars=10)
    assert list(sig[:20]) == [0.0] * 20  # warmup: no history to break out of
    assert list(sig[20:]) == [1.0] * 20


def test_breakout_exits_when_price_falls_below_prior_low():
    closes = [float(i) for i in range(1, 31)] + [5.0] + [4.0] * 5
    sig = breakout(ohlcv(closes), entry_bars=20, exit_bars=10)
    assert sig[29] == 1.0
    assert sig[30] == 0.0


def test_breakout_never_enters_without_a_breakout():
    df = ohlcv([100.0] * 30)  # dead flat: nothing to break out of
    assert breakout(df).sum() == 0.0


# --- mean reversion (buy oversold, exit on recovery) ----------------------


def test_mean_reversion_buys_dip_exits_recovery():
    closes = [100.0 - i for i in range(30)] + [71.0 + i for i in range(30)]
    sig = mean_reversion(ohlcv(closes), rsi_period=14, entry_level=30, exit_level=50)
    assert sig[0] == 0.0  # neutral first bar, never a guess
    assert sig[5] == 1.0  # deep in decline: RSI ~0, oversold
    assert sig[35] == 1.0  # RSI needs ~10 bars of gains to cross 50
    assert sig[45] == 0.0  # recovered: out


def test_mean_reversion_stays_out_of_flat_market():
    df = ohlcv([100.0] * 30)
    assert mean_reversion(df).sum() == 0.0


# --- crash shield (hold with an airbag) -----------------------------------


def test_crash_shield_warmup_flat():
    sig = crash_shield(ohlcv([100.0] * 10), shield=200)
    assert sig.sum() == 0.0


def test_crash_shield_holds_above_its_ma():
    df = ohlcv([float(i) for i in range(1, 301)])  # steady climb
    sig = crash_shield(df, shield=200)
    assert list(sig[:199]) == [0.0] * 199  # MA undefined -> parked
    assert list(sig[199:]) == [1.0] * 101  # price above MA -> riding


def test_crash_shield_steps_out_below_ma_and_reenters():
    closes = [float(i) for i in range(1, 301)]  # climb above MA...
    closes += [1.0] * 5                          # ...crash below it
    closes += [900.0 + i for i in range(210)]    # ...recover above it, rising
    sig = crash_shield(ohlcv(closes), shield=200)
    assert sig[299] == 1.0  # riding before the crash
    assert sig[300] == 0.0  # airbag deploys
    assert sig.iloc[-1] == 1.0  # back in once recovered
