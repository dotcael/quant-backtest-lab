"""Parquet storage for raw OHLCV data.

Layout: data/raw/{exchange}/{SYMBOL-QUOTE}/{timeframe}.parquet
e.g.     data/raw/binance/BTC-USDT/1h.parquet

Policy: data/raw is APPEND-ONLY. fetch adds new candles here; nothing
else ever edits raw data after the fact (cleaning/resampling later write
to a separate zone). This is what makes a future out-of-sample test
honest — you can always reproduce exactly what a backtest saw.

Why parquet: columnar + compressed (~10x smaller than CSV), fast bulk
reads from pandas, one file per series keeps appends cheap.
"""
from pathlib import Path

import pandas as pd

from quant import config


def parquet_path(exchange_id: str, symbol: str, timeframe: str) -> Path:
    safe_symbol = symbol.replace("/", "-")  # BTC/USDT -> BTC-USDT (no dirs in names)
    return config.DATA_DIR / exchange_id / safe_symbol / f"{timeframe}.parquet"


def last_timestamp(exchange_id: str, symbol: str, timeframe: str) -> int | None:
    """End of what we already have — the resume point for the next fetch."""
    path = parquet_path(exchange_id, symbol, timeframe)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return int(df["timestamp"].max()) if len(df) else None


def append(exchange_id: str, symbol: str, timeframe: str, new_df: pd.DataFrame) -> None:
    """Merge new candles into the parquet file, dedupe, keep sorted order."""
    path = parquet_path(exchange_id, symbol, timeframe)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        df = pd.concat([pd.read_parquet(path), new_df])
    else:
        df = new_df
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    df.to_parquet(path, index=False)
