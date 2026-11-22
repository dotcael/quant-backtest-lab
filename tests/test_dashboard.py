"""Dashboard exporter: one JSON snapshot of everything the learning
console needs — candles, live signal state, paper-trader log, scorecard.

The console (larpSim) is a static page; it cannot read parquet. This
module is the bridge: local files in, one portable JSON out. Offline and
read-only like the rest of the local pipeline — it never touches the
network and never writes into data/raw.
"""
import json

import numpy as np
import pandas as pd
import pytest

from quant import dashboard


def make_df(closes, start_ms=1_600_000_000_000):
    """Synthetic daily OHLCV where close[i] is the given series."""
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    return pd.DataFrame(
        {
            "timestamp": [start_ms + i * 86_400_000 for i in range(n)],
            "open": closes,
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "volume": [100.0] * n,
        }
    )


def trending_closes(n=320, break_at=280):
    """Above a rising floor for `break_at` bars, then collapsing below it —
    guarantees a shield exit (and an SMA cross) near `break_at`."""
    return np.concatenate(
        [
            np.linspace(100.0, 300.0, break_at),
            np.linspace(300.0, 120.0, n - break_at),
        ]
    )


def test_candles_to_rows_rounds_and_formats():
    df = make_df([4285.083, 4108.371], start_ms=1_502_928_000_000)
    rows = dashboard.candles_to_rows(df)
    assert rows[0][0] == "2017-08-17"  # ms -> ISO date
    assert rows[0][4] == pytest.approx(4285.08)  # close rounded to cents
    assert len(rows[1]) == 6  # [date, o, h, l, c, v]


def test_signal_flips_detect_the_exit():
    df = make_df(trending_closes())
    info = dashboard.signal_state(df, dashboard.STRATEGIES["crash_shield"])
    assert info["position"] == 0.0  # collapse put price below the shield MA
    assert info["flips"], "expected at least one long->flat flip"
    last = info["flips"][-1]
    assert last["to"] == info["position"]
    assert info["since"] == last["date"]  # 'since' is the last flip date


def test_signal_state_warmup_is_flat_never_partial():
    df = make_df(np.linspace(100.0, 200.0, 50))  # shorter than 200-day shield
    info = dashboard.signal_state(df, dashboard.STRATEGIES["crash_shield"])
    assert info["position"] == 0.0  # warmup: incomplete indicator = no signal
    assert info["flips"] == []


def test_build_snapshot_reads_parquet_and_paper(tmp_path):
    data_dir = tmp_path / "raw"
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir()
    df = make_df(trending_closes())
    sym = "TEST/USDT"
    (data_dir / "binance" / "TEST-USDT").mkdir(parents=True)
    df.to_parquet(data_dir / "binance" / "TEST-USDT" / "1d.parquet", index=False)
    (paper_dir / "log.jsonl").write_text(
        json.dumps(
            {
                "ts": "2026-08-27T10:00:00+00:00",
                "mode": "paper",
                "symbol": sym,
                "strategy": "crash_shield 200",
                "price": 300.0,
                "ma200": 250.0,
                "desired": 1.0,
                "action": "buy",
                "cash": 0.0,
                "units": 3.3,
                "equity": 998.5,
            }
        )
        + "\n"
    )
    (paper_dir / "TEST-USDT.json").write_text(json.dumps({"cash": 0.0, "units": 3.3}))

    snap = dashboard.build_snapshot(data_dir=data_dir, paper_dir=paper_dir, symbols=[sym])

    assert snap["meta"]["exchange"] == "binance"
    assert snap["meta"]["fee"]["round_trip"] == pytest.approx(0.003)  # 2*(taker+slip)
    assert snap["meta"]["max_dd"] == pytest.approx(0.20)
    assert snap["coins"] == [sym]
    assert len(snap["candles"][sym]) == len(df)
    assert set(snap["signals"][sym]) == {"crash_shield", "sma_cross"}
    assert len(snap["paper"]["events"]) == 1
    assert snap["paper"]["state"][sym]["units"] == pytest.approx(3.3)
    assert snap["paper"]["scorecard"][sym]["fills"] == 1


def test_build_snapshot_tolerates_missing_paper_entirely(tmp_path):
    data_dir = tmp_path / "raw"
    sym = "TEST/USDT"
    (data_dir / "binance" / "TEST-USDT").mkdir(parents=True)
    make_df(trending_closes(n=260, break_at=200)).to_parquet(
        data_dir / "binance" / "TEST-USDT" / "1d.parquet", index=False
    )
    snap = dashboard.build_snapshot(data_dir=data_dir, paper_dir=tmp_path / "nope", symbols=[sym])
    assert snap["paper"]["events"] == []
    assert snap["paper"]["state"] == {}
    assert snap["paper"]["scorecard"] == {}


def test_build_snapshot_skips_symbol_with_no_data(tmp_path):
    snap = dashboard.build_snapshot(
        data_dir=tmp_path / "raw", paper_dir=tmp_path / "paper", symbols=["GONE/USDT"]
    )
    assert snap["coins"] == []
    assert snap["candles"] == {}


def test_write_snapshot_makes_valid_json(tmp_path):
    snap = dashboard.build_snapshot(data_dir=tmp_path / "raw", paper_dir=tmp_path / "paper", symbols=[])
    out = dashboard.write_snapshot(snap, out_dir=tmp_path)
    assert out.name == "snapshot.json"
    loaded = json.loads(out.read_text())
    assert loaded["meta"] == snap["meta"]
