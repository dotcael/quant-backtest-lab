import json

import pytest

from quant.scorecard import summarize


def ev(symbol, ts="2026-08-27T10:00:00+00:00", action=None, equity=1000.0, price=100.0, **extra):
    base = {
        "ts": ts,
        "mode": "paper",
        "symbol": symbol,
        "strategy": "crash_shield 200",
        "price": price,
        "desired": 1.0,
        "action": action,
        "cash": 0.0 if action else 1000.0,
        "units": 10.0 if action else 0.0,
        "equity": equity,
    }
    base.update(extra)
    return base


def test_summarize_groups_by_symbol_and_counts_fills():
    events = [
        ev("BTC/USDT", ts="2026-08-27T10:00:00+00:00", action="buy", equity=998.5),
        ev("BTC/USDT", ts="2026-08-27T16:00:00+00:00", equity=999.0),
        ev("ETH/USDT", ts="2026-08-27T10:00:00+00:00", action="buy", equity=998.5),
    ]
    out = summarize(events)
    assert set(out) == {"BTC/USDT", "ETH/USDT"}
    assert out["BTC/USDT"]["fills"] == 1
    assert out["BTC/USDT"]["events"] == 2
    assert out["ETH/USDT"]["events"] == 1


def test_summarize_tracks_equity_path_and_drawdown():
    events = [
        ev("BTC/USDT", ts="2026-08-27T10:00:00+00:00", equity=1000.0),
        ev("BTC/USDT", ts="2026-08-27T16:00:00+00:00", equity=1200.0),
        ev("BTC/USDT", ts="2026-08-28T10:00:00+00:00", equity=900.0),
    ]
    out = summarize(events)
    assert out["BTC/USDT"]["equity"] == pytest.approx(900.0)
    assert out["BTC/USDT"]["drawdown"] == pytest.approx(900.0 / 1200.0 - 1)


def test_summarize_tolerates_older_events_missing_fields():
    """The log schema gained fields mid-stream (alloc_usd); the scorecard
    must read the whole history, not just the new format."""
    old = ev("BTC/USDT")  # predates alloc_usd and friends
    out = summarize([old])
    assert out["BTC/USDT"]["events"] == 1


def test_summarize_last_action_reflects_most_recent():
    events = [
        ev("BTC/USDT", ts="2026-08-27T10:00:00+00:00", action="buy"),
        ev("BTC/USDT", ts="2026-08-27T16:00:00+00:00", action="sell"),
        ev("BTC/USDT", ts="2026-08-27T20:00:00+00:00", action=None),
    ]
    out = summarize(events)
    assert out["BTC/USDT"]["last_action"] == "sell"
