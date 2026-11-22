"""Daily watcher logic: kadoa refresh diff -> score new trades -> alert >= P99.

Pure functions only; I/O lives in scripts/insider_watch.py. Idempotency comes
from the caller persisting trade keys into data/insider/watch_state.json.
Alerts are STATISTICAL FLAGS, never accusations; tiers stay document-tiered.
"""

from __future__ import annotations

from datetime import date
from typing import Iterable, Optional

import pandas as pd

from quant.insider import (
    PersonStats,
    abnormal_return,
    composite_score,
    lag_score,
    timing_z,
)

WATCH_COLUMNS = [
    "person", "ticker", "asset_type", "transaction_type", "amount_min",
    "amount_max", "transaction_date", "filing_date", "owner", "source_url",
]


def trade_key(person: str, ticker: str, tx_date: str, lo: float, hi: float) -> str:
    """Stable id for dedup: person+ticker+date+amount-range (ingest convention)."""
    return "|".join([
        str(person).strip().lower(),
        str(ticker).strip().upper(),
        str(tx_date)[:10],
        f"{float(lo):.0f}",
        f"{float(hi):.0f}",
    ])


def normalize_kadoa(records: Iterable[dict]) -> pd.DataFrame:
    """kadoa trades.json records -> watch rows (WATCH_COLUMNS + key)."""
    rows = []
    for r in records:
        person = r.get("filer_name") or ""
        ticker = str(r.get("ticker") or "").upper()
        tx = r.get("transaction_date") or ""
        lo = float(r.get("amount_range_low") or 0)
        hi = float(r.get("amount_range_high") or 0)
        rows.append({
            "person": person,
            "ticker": ticker,
            "asset_type": r.get("asset_type") or "",
            "transaction_type": (r.get("transaction_type") or "").strip().lower(),
            "amount_min": lo,
            "amount_max": hi,
            "transaction_date": tx,
            "filing_date": r.get("filing_date") or "",
            "owner": (r.get("owner") or "unknown").strip().lower(),
            "source_url": r.get("doc_url") or "",
            "key": trade_key(person, ticker, tx, lo, hi),
        })
    return pd.DataFrame(rows)


def diff_new(rows: pd.DataFrame, seen_keys: set[str]) -> pd.DataFrame:
    """Rows whose key is not in seen_keys. Empty-key rows are skipped."""
    if rows.empty:
        return rows.copy()
    mask = rows["key"].isin(seen_keys) | (rows["key"].str.strip() == "")
    return rows[~mask].copy()


def p99_threshold(map_csv: pd.DataFrame) -> Optional[float]:
    """P99 of non-null composites in the rescored map. None if unusable."""
    s = map_csv["composite"].dropna() if "composite" in map_csv.columns else pd.Series(dtype=float)
    if len(s) < 100:
        return None
    return float(s.quantile(0.99))


def repetition_from_map(map_csv: pd.DataFrame, person: str) -> Optional[float]:
    """Person's repetition ratio from the rescored map (most recent value)."""
    sub = map_csv[map_csv["person"] == person]["repetition"].dropna()
    return float(sub.iloc[-1]) if len(sub) else None


def score_new(
    new_rows: pd.DataFrame,
    events: Iterable[date],
    person_stats: dict[str, PersonStats],
    map_csv: pd.DataFrame,
    prices: dict[str, pd.Series],
    spy: pd.Series,
) -> pd.DataFrame:
    """Score new trades with the frozen spec. Components come from: prices
    (abnormal), events v2 + person history (timing), the rescored map
    (repetition), filing dates (lag). composite is None when any component is
    undefined — same rule as the pipeline."""
    out = new_rows.copy()
    comps: list[Optional[float]] = []
    abns: list[Optional[float]] = []
    for r in out.itertuples():
        d = pd.Timestamp(r.transaction_date).date() if r.transaction_date else None
        abn = None
        if d is not None and r.ticker in prices:
            abn = abnormal_return(prices[r.ticker], spy, d, 30)
        stats = person_stats.get(r.person) or PersonStats()
        tz = timing_z(d, events, stats) if d else None
        rep = repetition_from_map(map_csv, r.person)
        lag = None
        if r.filing_date and d:
            lag = (pd.Timestamp(r.filing_date) - pd.Timestamp(r.transaction_date)).days
        ls = lag_score(lag)
        comp = (
            composite_score(abn, tz, rep, ls)
            if (abn is not None and tz is not None and rep is not None and ls is not None)
            else None
        )
        abns.append(abn)
        comps.append(comp)
    out["abnormal_30d"] = abns
    out["composite"] = comps
    return out


def alert_rows(scored: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """New trades whose composite meets the P99 threshold."""
    return scored[scored["composite"] >= threshold].copy()
