import pytest

from quant.fees import FeeModel, book_slippage


def test_stress_multiplier_scales_all_costs():
    """The torture-test knob: stress=2 doubles fees AND slippage."""
    base = FeeModel.for_exchange("binance")
    stressed = FeeModel.for_exchange("binance", stress=2.0)
    assert stressed.taker == pytest.approx(2 * base.taker)
    assert stressed.maker == pytest.approx(2 * base.maker)
    assert stressed.slippage == pytest.approx(2 * base.slippage)
    assert stressed.round_trip_cost() == pytest.approx(2 * base.round_trip_cost())


def test_default_stress_is_unchanged():
    assert FeeModel.for_exchange("binance").round_trip_cost() == pytest.approx(0.003)


# --- book_slippage: what a market order REALLY costs, from the order book ---


def test_flat_book_zero_slippage():
    levels = [[100.0, 10.0]] * 5  # 5 levels, all 100.0 x 10 units
    assert book_slippage(levels, 1000.0) == pytest.approx(0.0)


def test_book_slippage_walks_levels():
    """Buy $200 into: level1 100.0 x 1.0 ($100), level2 101.0 x 10 ($1010).
    Fill: $100 @ 100.0 + $100 @ 101.0 -> VWAP 200/(1+100/101) = 100.4975."""
    levels = [[100.0, 1.0], [101.0, 10.0]]
    vwap = 200.0 / (1.0 + 100.0 / 101.0)
    assert book_slippage(levels, 200.0) == pytest.approx(vwap / 100.0 - 1)


def test_book_too_thin_raises():
    with pytest.raises(ValueError):
        book_slippage([[100.0, 1.0]], 10_000.0)
