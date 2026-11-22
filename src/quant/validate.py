"""Sanity checks run on every fetch, before data is stored or trusted.

Why: a backtest fed corrupt data produces garbage with total confidence.
These checks catch the classic corruptions:
  - duplicates   (same candle counted twice -> double-weighted history)
  - gaps         (missing candles; crypto trades 24/7 so a gap is real,
                  e.g. exchange downtime — you must KNOW about it)
  - bad OHLC     (high below open/close/low etc. -> API glitch)
  - nonpositive  (price <= 0 or volume < 0 -> never valid)
"""
from dataclasses import dataclass, field

from quant.timeframes import timeframe_ms


@dataclass
class ValidationReport:
    duplicates: int = 0
    gaps: list[tuple[int, int]] = field(default_factory=list)
    bad_ohlc: int = 0
    nonpositive: int = 0

    @property
    def ok(self) -> bool:
        return not (self.duplicates or self.gaps or self.bad_ohlc or self.nonpositive)

    def summary(self) -> str:
        return (
            f"duplicates={self.duplicates}, gaps={len(self.gaps)}, "
            f"bad_ohlc={self.bad_ohlc}, nonpositive={self.nonpositive}"
        )


def validate_ohlcv(df, timeframe: str) -> ValidationReport:
    report = ValidationReport()

    report.duplicates = int(df["timestamp"].duplicated().sum())

    step = timeframe_ms(timeframe)
    ts = df["timestamp"].sort_values().to_numpy()
    for prev, nxt in zip(ts[:-1], ts[1:]):
        if nxt - prev != step:
            report.gaps.append((int(prev), int(nxt)))

    prices = df[["open", "high", "low", "close"]]
    report.bad_ohlc = int(
        (
            (df["high"] < prices.drop(columns="high").max(axis=1))
            | (df["low"] > prices.drop(columns="low").min(axis=1))
        ).sum()
    )

    report.nonpositive = int(((prices <= 0).any(axis=1) | (df["volume"] < 0)).sum())

    return report
