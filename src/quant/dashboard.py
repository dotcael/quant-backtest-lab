"""Dashboard snapshot exporter — one JSON, everything the console needs.

The learning console (larpSim) is a static HTML page: it cannot read
parquet and it must not reach the network. This module is the bridge.

    python -m quant.dashboard   ->  data/dashboard/snapshot.json

Everything is read-only and local, like the rest of the offline pipeline:
raw data is never touched, the snapshot is a derived artifact that can be
regenerated at any time. It is also called at the end of paper.main() so
the console stays fresh on the same 6h cadence.

Snapshot shape (versioned so the console can detect old files):
  meta     — generated ts, exchange, fee model, max_dd, per-coin cap
  coins    — symbols present, in config order
  candles  — {symbol: [[YYYY-MM-DD, o, h, l, c, v], ...]} daily bars
  signals  — per symbol: current position + flip history for each
             strategy, so the console shows what the pipeline ACTUALLY
             believes, not a browser reimplementation of it
  paper    — full event log, per-symbol state, scorecard summary
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from quant import config, fees, scorecard, sizing
from quant.strategies import crash_shield, sma_cross

DASH_DIR = config.PROJECT_ROOT / "data" / "dashboard"
SNAPSHOT_VERSION = 1

# Same definitions the backtest CLI reports — one source of truth.
STRATEGIES = {
    "crash_shield": {"label": "crash_shield 200", "fn": lambda df: crash_shield(df, 200)},
    "sma_cross": {"label": "sma_cross 10/50", "fn": lambda df: sma_cross(df, 10, 50)},
}


def _parquet_path(exchange_id: str, symbol: str, timeframe: str, data_dir: Path = None) -> Path:
    root = data_dir if data_dir is not None else config.DATA_DIR
    return root / exchange_id / symbol.replace("/", "-") / f"{timeframe}.parquet"


def candles_to_rows(df: pd.DataFrame) -> list[list]:
    """Parquet frame -> compact JSON rows [date, o, h, l, c, v].

    Prices rounded to cents, volume to one decimal: enough precision for
    a chart and a journal, a fraction of the bytes.
    """
    dates = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    return [
        [
            date,
            round(float(row.open), 2),
            round(float(row.high), 2),
            round(float(row.low), 2),
            round(float(row.close), 2),
            round(float(row.volume), 1),
        ]
        for date, row in zip(dates, df.itertuples(index=False))
    ]


def signal_state(df: pd.DataFrame, strategy: dict, max_flips: int = 30) -> dict:
    """Where a strategy stands right now, and how it got here.

    position  — 1.0 long / 0.0 flat at the latest close
    since     — date of the last flip (when the current stance began)
    flips     — most recent position changes with the price that caused
                them, newest last; the console turns these into lessons
                ('here is where the airbag fired, here is what it cost')
    """
    positions = strategy["fn"](df)
    closes = df["close"]
    dates = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")

    flips = []
    previous = float(positions.iloc[0]) if len(positions) else 0.0
    for i in range(1, len(positions)):
        current = float(positions.iloc[i])
        if current != previous:
            flips.append(
                {
                    "date": str(dates.iloc[i]),
                    "price": round(float(closes.iloc[i]), 2),
                    "to": current,
                }
            )
            previous = current

    position = previous
    since = flips[-1]["date"] if flips else (str(dates.iloc[0]) if len(dates) else None)
    return {
        "label": strategy["label"],
        "position": position,
        "since": since,
        "flips": flips[-max_flips:],
    }


def _read_daily(exchange_id: str, symbol: str, data_dir: Path = None) -> pd.DataFrame | None:
    path = _parquet_path(exchange_id, symbol, "1d", data_dir)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return df.sort_values("timestamp").reset_index(drop=True) if len(df) else None


def _load_paper(paper_dir: Path) -> dict:
    """Events + per-symbol state + scorecard; missing dir = empty, not error."""
    paper_dir = paper_dir if paper_dir.exists() else paper_dir
    events = scorecard.load_events(paper_dir) if (paper_dir / "log.jsonl").exists() else []

    state = {}
    if paper_dir.exists():
        for path in sorted(paper_dir.glob("*-USDT.json")):
            symbol = path.stem.replace("-", "/")
            try:
                state[symbol] = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue  # a corrupt state file must not sink the snapshot

    summary = scorecard.summarize(events) if events else {}
    return {"events": events, "state": state, "scorecard": summary}


def build_snapshot(
    data_dir: Path = None,
    paper_dir: Path = None,
    out_exchange: str = None,
    symbols: list[str] = None,
) -> dict:
    """Assemble the whole snapshot. Reads local files only."""
    exchange_id = out_exchange or config.EXCHANGES[0]
    symbols = symbols if symbols is not None else config.SYMBOLS
    fee_model = fees.FeeModel.for_exchange(exchange_id)

    candles: dict[str, list] = {}
    signals: dict[str, dict] = {}
    for symbol in symbols:
        df = _read_daily(exchange_id, symbol, data_dir)
        if df is None:
            continue
        candles[symbol] = candles_to_rows(df)
        signals[symbol] = {
            key: signal_state(df, strategy) for key, strategy in STRATEGIES.items()
        }

    return {
        "version": SNAPSHOT_VERSION,
        "meta": {
            "generated": datetime.now(timezone.utc).isoformat(),
            "exchange": exchange_id,
            "timeframe": "1d",
            "fee": {
                "maker": fee_model.maker,
                "taker": fee_model.taker,
                "slippage": fee_model.slippage,
                "round_trip": fee_model.round_trip_cost(),
            },
            "max_dd": sizing.MAX_DD,
            "per_coin_cap": sizing.DEFAULT_PER_COIN_CAP,
        },
        "coins": [s for s in symbols if s in candles],
        "candles": candles,
        "signals": signals,
        "paper": _load_paper(paper_dir if paper_dir is not None else scorecard.PAPER_DIR),
    }


def write_snapshot(snapshot: dict, out_dir: Path = None) -> Path:
    out_dir = out_dir if out_dir is not None else DASH_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "snapshot.json"
    path.write_text(json.dumps(snapshot))
    return path


def main() -> None:
    path = write_snapshot(build_snapshot())
    snap = json.loads(path.read_text())
    n_bars = sum(len(rows) for rows in snap["candles"].values())
    print(
        f"[dashboard] snapshot v{SNAPSHOT_VERSION} -> {path} | "
        f"{len(snap['coins'])} coins, {n_bars:,} daily bars, "
        f"{len(snap['paper']['events'])} paper events"
    )


if __name__ == "__main__":
    main()
