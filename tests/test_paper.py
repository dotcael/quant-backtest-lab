import json
from datetime import datetime, timezone

import pytest

from quant.fees import FeeModel
from quant.paper import (
    Account,
    append_event,
    apply_fill,
    decide_action,
    equity,
    is_stale,
    load_state,
    power_ok_from_output,
    save_state,
)

SIDE = 0.001 + 0.0005  # binance taker + slippage per side


# --- decisions -----------------------------------------------------------


def test_decide_action_enters_exits_and_holds():
    assert decide_action(units=0.0, desired=1.0) == "buy"
    assert decide_action(units=2.5, desired=0.0) == "sell"
    assert decide_action(units=0.0, desired=0.0) is None  # agree: stay flat
    assert decide_action(units=2.5, desired=1.0) is None  # agree: stay long


# --- fills and accounting ------------------------------------------------


def test_buy_fill_costs_and_units():
    acct = Account(cash=1000.0, units=0.0)
    filled = apply_fill(acct, "buy", price=100.0, fee_model=FeeModel.for_exchange("binance"))
    assert filled.cash == pytest.approx(0.0)
    assert filled.units == pytest.approx(1000.0 * (1 - SIDE) / 100.0)


def test_round_trip_costs_exact():
    """Same ground rule as the backtest, made executable: a round trip at
    the SAME price must lose exactly the cost — nothing else."""
    fee_model = FeeModel.for_exchange("binance")
    acct = Account(cash=1000.0, units=0.0)
    acct = apply_fill(acct, "buy", price=100.0, fee_model=fee_model)
    acct = apply_fill(acct, "sell", price=100.0, fee_model=fee_model)
    assert acct.units == pytest.approx(0.0)
    assert acct.cash == pytest.approx(1000.0 * (1 - SIDE) ** 2)


def test_equity_flat_and_long():
    assert equity(Account(cash=500.0, units=0.0), price=123.0) == pytest.approx(500.0)
    assert equity(Account(cash=0.0, units=2.0), price=150.0) == pytest.approx(300.0)


# --- persistence ---------------------------------------------------------


def test_state_round_trip(tmp_path):
    save_state("BTC/USDT", Account(cash=12.5, units=0.01), directory=tmp_path)
    loaded = load_state("BTC/USDT", directory=tmp_path)
    assert loaded == Account(cash=12.5, units=0.01)


def test_load_state_missing_starts_flat(tmp_path):
    assert load_state("NEW/USDT", directory=tmp_path) == Account(cash=1000.0, units=0.0)


def test_events_append_as_jsonl(tmp_path):
    for i in range(3):
        append_event({"i": i}, directory=tmp_path)
    lines = (tmp_path / "log.jsonl").read_text().strip().splitlines()
    assert [json.loads(line)["i"] for line in lines] == [0, 1, 2]


# --- power guard (run when charging, or battery >= 80%) --------------------

PMSET_AC = "Now drawing from 'AC Power' - InternalBattery-0 (id=1) 100%; charged; 0:00 remaining present: true"
PMSET_BATT_85 = "Now drawing from 'Battery Power' - InternalBattery-0 (id=1) 85%; discharging; 3:00 remaining present: true"
PMSET_BATT_76 = "Now drawing from 'Battery Power' - InternalBattery-0 (id=1) 76%; discharging; 4:10 remaining present: true"


def test_power_ok_on_ac():
    assert power_ok_from_output(PMSET_AC)


def test_power_ok_battery_at_or_above_threshold():
    assert power_ok_from_output(PMSET_BATT_85)


def test_power_blocks_low_battery():
    assert not power_ok_from_output(PMSET_BATT_76)


def test_power_fails_open_on_unknown_output():
    """A parse error must never silently stop the scans."""
    assert power_ok_from_output("total garbage, no numbers here... 0")


# --- staleness alarm (bracket closing, no scan today) -----------------------


def test_not_stale_when_scanned_today():
    events = [{"ts": "2026-08-30T08:09:07+00:00"}]
    now = datetime(2026, 8, 30, 20, 0, tzinfo=timezone.utc)
    assert not is_stale(events, now=now)


def test_stale_when_no_scan_today_and_past_deadline():
    events = [{"ts": "2026-08-29T08:09:07+00:00"}]
    now = datetime(2026, 8, 30, 19, 0, tzinfo=timezone.utc)
    assert is_stale(events, now=now)


def test_not_stale_before_deadline_even_with_missed_scans():
    """Morning misses are fine — there is still time; don't cry wolf."""
    events = [{"ts": "2026-08-29T08:09:07+00:00"}]
    now = datetime(2026, 8, 30, 10, 0, tzinfo=timezone.utc)
    assert not is_stale(events, now=now)


def test_stale_on_empty_log_past_deadline():
    now = datetime(2026, 8, 30, 22, 0, tzinfo=timezone.utc)
    assert is_stale([], now=now)
