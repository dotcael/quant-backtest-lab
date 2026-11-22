import pytest

from quant.sizing import DEFAULT_PER_COIN_CAP, allocate, drawdown_watch, position_size


# --- allocation (the diversification rule) --------------------------------


def test_equal_weights_five_coins_no_cap_binding():
    out = allocate(1000.0, {"A": 1, "B": 1, "C": 1, "D": 1, "E": 1})
    assert out == {"A": 200.0, "B": 200.0, "C": 200.0, "D": 200.0, "E": 200.0}


def test_weights_are_normalized_first():
    """Weights are relative, not absolute: {2,2} behaves like {1,1}."""
    out = allocate(1000.0, {"A": 2, "B": 2}, per_coin_cap=1.0)
    assert out == {"A": 500.0, "B": 500.0}


def test_per_coin_cap_limits_and_leaves_cash_unallocated():
    """{A: 0.6, B: 0.2} with a 25% cap: A gets capped at 250, B keeps its
    full 200, and the excess is NOT force-fed anywhere — it stays cash."""
    out = allocate(1000.0, {"A": 0.6, "B": 0.2}, per_coin_cap=DEFAULT_PER_COIN_CAP)
    assert out["A"] == pytest.approx(250.0)
    assert out["B"] == pytest.approx(200.0)
    assert sum(out.values()) <= 1000.0


def test_allocate_rejects_bad_inputs():
    with pytest.raises(ValueError):
        allocate(1000.0, {})  # nothing to allocate
    with pytest.raises(ValueError):
        allocate(-5.0, {"A": 1.0})  # negative equity
    with pytest.raises(ValueError):
        allocate(1000.0, {"A": 1.0}, per_coin_cap=0.0)  # cap allows nothing


# --- the 1-2% risk rule ---------------------------------------------------


def test_position_size_risks_exactly_the_fraction():
    """$1000 equity, 1% risk, entry 100 stop 90: $10 of pain allowed,
    $10 of distance -> $100 position -> 1.0 units."""
    assert position_size(equity=1000.0, risk_pct=0.01, entry=100.0, stop=90.0) == pytest.approx(1.0)


def test_position_size_scales_with_risk_and_distance():
    assert position_size(2000.0, 0.02, 50.0, 45.0) == pytest.approx(8.0)


def test_position_size_rejects_nonsense_stops():
    with pytest.raises(ValueError):
        position_size(1000.0, 0.01, entry=90.0, stop=100.0)  # stop above entry
    with pytest.raises(ValueError):
        position_size(1000.0, 0.01, entry=100.0, stop=100.0)  # zero distance


# --- drawdown watchdog (decide the max BEFORE you start) -------------------


def test_drawdown_measured_from_running_peak():
    breached, dd = drawdown_watch([100.0, 120.0, 60.0, 90.0], max_dd=0.40)
    assert dd == pytest.approx(90.0 / 120.0 - 1)  # current, from peak 120
    assert not breached  # -25% now, even though -50% happened once


def test_drawdown_breach_flagged_when_current_exceeds_max():
    breached, dd = drawdown_watch([100.0, 120.0, 60.0], max_dd=0.40)
    assert dd == pytest.approx(-0.5)
    assert breached


def test_drawdown_short_curve_is_flat():
    breached, dd = drawdown_watch([100.0], max_dd=0.10)
    assert dd == 0.0
    assert not breached
