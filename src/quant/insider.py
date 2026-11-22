"""Insider-trading watchlist map: schema, uniform scoring, seed loaders.

Pre-registered scoring spec (fixed before any real data was scored — do not
tune after seeing results):

- WINDOWS = (7, 30, 90) trading-day forward-return windows vs SPY.
- abnormal_return: ticker forward return minus SPY forward return, same window.
- timing_z: |trade_date - nearest event| in days divided by the person's own
  median inter-trade interval (days, min 1). Low = trade sits unusually close
  to an event *for that person's* trading frequency.
- repetition: fraction of a person's scored trades with 30d abnormal return
  > +5%. Only defined when the person has >= MIN_TRADES scored trades.
- lag_score: STOCK Act disclosure lateness, clamp((lag_days - 45) / 45, 0, 1).
- composite: 0.4 * abnormal + 0.2 * timing + 0.2 * repetition + 0.2 * lag,
  each component rescaled to [0, 1]; composite is None if any component is
  undefined. Tier (legal outcome) is a label, never an input to the score.
- Virality / press attention carries zero weight by design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

WINDOWS = (7, 30, 90)
EVENT_WINDOW_DAYS = 7
REP_WINDOW = 30
REP_THRESHOLD = 0.05
MIN_TRADES = 5
STOCK_ACT_LIMIT = 45
WEIGHTS = {"abnormal": 0.4, "timing": 0.2, "repetition": 0.2, "lag": 0.2}

TRADE_COLUMNS = [
    "person",
    "role",
    "party",
    "chamber",
    "state",
    "ticker",
    "asset_type",
    "transaction_type",
    "amount_min",
    "amount_max",
    "transaction_date",
    "filing_date",
    "disclosure_lag_days",
    "owner",
    "linked_via",
    "source_dataset",
    "source_url",
    "provenance",
]

TIERS = [
    "CONVICTED",
    "INVESTIGATED_CLEARED",
    "INVESTIGATED_NO_CHARGES_YET",
    "DISCLOSURE_VIOLATION",
    "STATISTICAL_OUTLIER",
    "UNVERIFIED_GROK",
]

EVENT_COLUMNS = ["event_id", "date", "event_type", "description", "source"]


@dataclass
class Trade:
    person: str
    ticker: str
    transaction_date: date
    filing_date: Optional[date] = None
    amount_min: float = 0.0
    amount_max: float = 0.0

    @property
    def disclosure_lag_days(self) -> Optional[int]:
        if self.filing_date is None:
            return None
        return (self.filing_date - self.transaction_date).days


@dataclass
class PersonStats:
    trades: list[Trade] = field(default_factory=list)

    def median_interval_days(self) -> float:
        days = sorted(
            (b.transaction_date - a.transaction_date).days
            for a, b in zip(self.trades, self.trades[1:])
            if b.transaction_date > a.transaction_date
        )
        if not days:
            return 0.0
        mid = len(days) // 2
        if len(days) % 2:
            return float(days[mid])
        return (days[mid - 1] + days[mid]) / 2 or 1.0


def forward_return(prices: pd.Series, trade_date: date, window: int) -> Optional[float]:
    pos = prices.index.asof(pd.Timestamp(trade_date))
    if pd.isna(pos):
        return None
    start = float(prices.asof(pos))
    target = pos + pd.Timedelta(days=window * 7 // 5)
    if target > prices.index[-1]:
        return None
    end = float(prices.asof(target))
    if pd.isna(end) or start == 0:
        return None
    return (end - start) / start


def abnormal_return(
    ticker_prices: pd.Series,
    spy_prices: pd.Series,
    trade_date: date,
    window: int = REP_WINDOW,
) -> Optional[float]:
    t = forward_return(ticker_prices, trade_date, window)
    b = forward_return(spy_prices, trade_date, window)
    if t is None or b is None:
        return None
    return t - b


def nearest_event_gap(trade_date: date, events: Iterable[date]) -> int:
    gaps = [abs((trade_date - e).days) for e in events]
    return min(gaps) if gaps else 10_000


def timing_z(trade_date: date, events: Iterable[date], person: PersonStats) -> float:
    interval = max(person.median_interval_days(), 1.0)
    return nearest_event_gap(trade_date, events) / interval


def repetition_ratio(person: PersonStats, abnormals: list[Optional[float]]) -> Optional[float]:
    scored = [a for a in abnormals if a is not None]
    if len(scored) < MIN_TRADES:
        return None
    hits = sum(1 for a in scored if a > REP_THRESHOLD)
    return hits / len(scored)


def lag_score(lag_days: Optional[int]) -> Optional[float]:
    if lag_days is None:
        return None
    return min(max((lag_days - STOCK_ACT_LIMIT) / STOCK_ACT_LIMIT, 0.0), 1.0)


def composite_score(
    abnormal: Optional[float],
    timing: float,
    repetition: Optional[float],
    lag: Optional[float],
) -> Optional[float]:
    if abnormal is None or repetition is None or lag is None:
        return None
    abnormal_s = min(max(abnormal, 0.0) / 0.50, 1.0)
    timing_s = min(max(1.0 - timing / 5.0, 0.0), 1.0)
    return (
        WEIGHTS["abnormal"] * abnormal_s
        + WEIGHTS["timing"] * timing_s
        + WEIGHTS["repetition"] * repetition
        + WEIGHTS["lag"] * lag
    )


def load_trades_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(TRADE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"trades file missing columns: {sorted(missing)}")
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
    return df


def load_events_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(EVENT_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"events file missing columns: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def next_quarter_end(d: date) -> date:
    if d.month <= 3:
        return date(d.year, 3, 31)
    if d.month <= 6:
        return date(d.year, 6, 30)
    if d.month <= 9:
        return date(d.year, 9, 30)
    return date(d.year, 12, 31)


def filing_deadline(transaction_date: date) -> date:
    return next_quarter_end(transaction_date) + timedelta(days=STOCK_ACT_LIMIT)


# --- options-aware scoring (Phase 3) ---------------------------------------
# Linearized delta proxy, NOT Black-Scholes: the norm schema carries no
# implied-vol surface. Undisclosed strike/expiry -> ATM delta with
# assumed_atm=True (pre-registered default). Composite weights are unchanged.
ATM_DELTA = 0.5
DELTA_CLIP = (0.05, 0.95)


def parse_option_terms(asset_name: str) -> Optional[dict]:
    """Extract structured terms from a PTR asset-name string when ALL of
    option type, strike, and expiry are present; otherwise None (which maps
    to the ATM proxy + assumed_atm=True)."""
    import re

    if not asset_name:
        return None
    s = str(asset_name)
    m_type = re.search(r"\b(calls?|puts?)\b", s, re.IGNORECASE)
    m_strike = re.search(
        r"(?:\$(\d+(?:\.\d+)?))|(?:(\d+(?:\.\d+)?)\s*strike)|(?:strike\s*(\d+(?:\.\d+)?))",
        s,
        re.IGNORECASE,
    )
    strike = next((g for g in (m_strike.groups() if m_strike else ()) if g), None)
    m_exp = re.search(r"(\d{2}/\d{2}/\d{4})|(\d{4}-\d{2}-\d{2})", s)
    if not (m_type and m_strike and m_exp):
        return None
    raw = m_exp.group(1) or m_exp.group(2)
    fmt = "%m/%d/%Y" if "/" in raw else "%Y-%m-%d"
    return {
        "opt_type": m_type.group(1).lower().rstrip("s"),
        "strike": float(strike),
        "expiry": pd.to_datetime(raw, format=fmt).date(),
    }


def option_delta_proxy(strike: Optional[float] = None, spot: Optional[float] = None) -> float:
    """Linearized call delta in DELTA_CLIP; ATM (missing terms) -> ATM_DELTA."""
    if not strike or not spot:
        return ATM_DELTA
    moneyness = (spot - strike) / strike
    return min(max(0.5 + 0.5 * moneyness, DELTA_CLIP[0]), DELTA_CLIP[1])


def option_pnl_proxy(
    abnormal_30d: Optional[float],
    transaction_type: str,
    strike: Optional[float] = None,
    spot: Optional[float] = None,
) -> Optional[float]:
    """30d option P/L proxy: delta * abnormal underlying move; a sell (write/
    close-short) flips sign. Direction of the underlying bet follows the
    transaction side (call-shaped); put-ness is unknowable without terms and
    is covered by the assumed_atm flag."""
    if abnormal_30d is None:
        return None
    delta = option_delta_proxy(strike, spot)
    sign = -1.0 if str(transaction_type or "").strip().lower().startswith("sell") else 1.0
    return sign * delta * abnormal_30d
