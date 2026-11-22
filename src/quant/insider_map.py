"""Build the insider map from normalized congressional trades.

Scores every trade with the pre-registered spec in quant.insider, aggregates
a person leaderboard, emits a person/ticker/event GraphML graph, the
null-baseline sanity check, and round3_queue.csv (top unverified rows for a
future verification round; QUEUED_UNVERIFIED is not an accusation). Usage:

    python src/quant/insider_map.py \
        --trades data/insider/congress_trades_norm.csv \
        --events data/insider/events.csv \
        --seed data/insider/grok_seed.csv \
        --out data/insider/map
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd

try:
    from quant.insider import (  # repo layout: src/quant package
        REP_THRESHOLD,
        PersonStats,
        Trade,
        abnormal_return,
        composite_score,
        lag_score,
        load_events_csv,
        option_delta_proxy,
        option_pnl_proxy,
        repetition_ratio,
        timing_z,
    )
except ImportError:  # portable layout: insider.py sits next to this file
    from insider import (  # type: ignore[no-redef]
        REP_THRESHOLD,
        PersonStats,
        Trade,
        abnormal_return,
        composite_score,
        lag_score,
        load_events_csv,
        option_delta_proxy,
        option_pnl_proxy,
        repetition_ratio,
        timing_z,
    )

SPY = "SPY"
PRICE_CHUNK = 100
ROUND3_N = 25
OPTION_TYPE = "stock option"

# Persons already carrying receipts or an active verification queue (round 1:
# v1 map receipts; round 2: the 17 verified/verifying candidates). Matched on
# the LAST token of the person string, case-insensitive. Conservative: generic
# surnames (e.g. "Smith") exclude broadly — the queue under-fills rather than
# re-queues a verified name.
VERIFIED_SURNAMES = {
    # round 1 receipts
    "pelosi", "greene", "bresnahan", "shreve", "trump",
    # round 2 candidates (2026-09 verification)
    "menefee", "scanlon", "britt", "walberg", "beaudreau", "hegseth",
    "burns", "lutnick", "isaacman", "bentz", "zeldin", "gottheimer",
    "jacobs", "smith",
}


def round3_queue(scored: pd.DataFrame, n: int = ROUND3_N) -> pd.DataFrame:
    """Top-N unverified rows by composite, for the (not yet scheduled) round-3
    verification. Labels are QUEUED_UNVERIFIED — a queue is not an accusation."""
    s = scored.dropna(subset=["composite"]).copy()
    last = s["person"].astype(str).str.strip().str.split().str[-1].str.lower()
    s = s[~last.isin(VERIFIED_SURNAMES)]
    q = s.nlargest(n, "composite")[
        ["person", "ticker", "transaction_date", "transaction_type",
         "composite", "abnormal_30d", "timing_z", "asset_type"]
    ].copy()
    q.insert(0, "rank", range(1, len(q) + 1))
    q["status"] = "QUEUED_UNVERIFIED"
    return q


def fetch_prices(tickers: list[str], cache: Path) -> dict[str, pd.Series]:
    import yfinance as yf

    cache.mkdir(parents=True, exist_ok=True)
    out: dict[str, pd.Series] = {}
    missing = [t for t in tickers if not (cache / f"{t}.parquet").exists()]
    for i in range(0, len(missing), PRICE_CHUNK):
        chunk = missing[i : i + PRICE_CHUNK]
        try:
            df = yf.download(chunk, period="10y", auto_adjust=True, progress=False)
        except Exception:
            continue
        if len(chunk) == 1:
            closes = df["Close"]
            if isinstance(closes, pd.DataFrame):
                closes = closes.iloc[:, 0]
            if not closes.empty:
                closes.to_frame().to_parquet(cache / f"{chunk[0]}.parquet")
        else:
            for t in chunk:
                try:
                    closes = df["Close"][t].dropna()
                except KeyError:
                    continue
                if closes.empty:
                    continue
                closes.to_frame().to_parquet(cache / f"{t}.parquet")
    for t in tickers:
        p = cache / f"{t}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            out[t] = df.iloc[:, 0] if isinstance(df, pd.DataFrame) else df
    return out


def build_person_stats(trades: pd.DataFrame) -> dict[str, PersonStats]:
    stats: dict[str, PersonStats] = defaultdict(PersonStats)
    for name, grp in trades.groupby("person"):
        rows = grp.dropna(subset=["transaction_date"]).sort_values("transaction_date")
        for _, r in rows.iterrows():
            stats[name].trades.append(
                Trade(
                    person=name,
                    ticker=str(r.get("ticker") or ""),
                    transaction_date=r["transaction_date"].date(),
                )
            )
    return dict(stats)


def score_trades(
    trades: pd.DataFrame,
    event_dates: list[date],
    prices: dict[str, pd.Series],
    spy: pd.Series,
    person_stats: dict[str, PersonStats],
) -> pd.DataFrame:
    abnormals_by_person: dict[str, list[float | None]] = defaultdict(list)
    records: list[dict] = []
    for _, r in trades.iterrows():
        tdate = r["transaction_date"]
        if pd.isna(tdate):
            continue
        d = tdate.date()
        ticker = str(r.get("ticker") or "")
        person = str(r.get("person") or "")
        abn = None
        if ticker in prices:
            abn = abnormal_return(prices[ticker], spy, d, 30)
        tstats = person_stats.get(person) or PersonStats()
        tz = timing_z(d, event_dates, tstats)
        lag = None
        lag_days = r.get("disclosure_lag_days")
        if pd.notna(lag_days):
            try:
                lag = int(float(lag_days))
            except (TypeError, ValueError):
                lag = None
        ls = lag_score(lag)
        records.append(
            {
                "person": person,
                "ticker": ticker,
                "transaction_date": d.isoformat(),
                "transaction_type": r.get("transaction_type"),
                "asset_type": r.get("asset_type"),
                "amount_min": r.get("amount_min"),
                "amount_max": r.get("amount_max"),
                "owner": r.get("owner"),
                "abnormal_30d": abn,
                "timing_z": tz,
                "lag_score": ls,
            }
        )
        abnormals_by_person[person].append(abn)
    rep: dict[str, float | None] = {}
    for person, vals in abnormals_by_person.items():
        rep[person] = repetition_ratio(person_stats.get(person) or PersonStats(trades=[]), vals)
    scored = pd.DataFrame(records)
    scored["repetition"] = scored["person"].map(rep)
    comps = [
        composite_score(r.abnormal_30d, r.timing_z, r.repetition, r.lag_score)
        for r in scored.itertuples()
    ]
    scored["composite"] = comps

    # Options-aware columns (Phase 3): composite spec is UNCHANGED — these are
    # annotation columns. The norm schema carries no strike/expiry, so every
    # option row uses the ATM proxy with assumed_atm=True (pre-registered
    # default; parse_option_terms exists for a future PDF-parsing lane).
    is_opt = scored["asset_type"].astype(str).str.fullmatch(OPTION_TYPE)
    scored["assumed_atm"] = is_opt
    scored["option_delta_proxy"] = [
        option_delta_proxy() if flag else None for flag in is_opt
    ]
    scored["option_pnl_proxy_30d"] = [
        option_pnl_proxy(r.abnormal_30d, r.transaction_type) if flag else None
        for r, flag in zip(scored.itertuples(), is_opt)
    ]
    return scored


def leaderboard(scored: pd.DataFrame, seed: pd.DataFrame) -> pd.DataFrame:
    tiers = {}
    if not seed.empty and "person" in seed.columns:
        s = seed.dropna(subset=["person"]).copy()
        s["key"] = s["person"].str.strip().str.lower()
        tiers = dict(zip(s["key"], s.get("tier", pd.Series(dtype=str))))
    agg = scored.groupby("person").agg(
        n_trades=("ticker", "size"),
        n_scored=("abnormal_30d", lambda s: int(s.notna().sum())),
        median_abnormal=("abnormal_30d", "median"),
        max_abnormal=("abnormal_30d", "max"),
        hit_rate=("abnormal_30d", lambda s: float((s.dropna() > REP_THRESHOLD).mean()) if s.notna().any() else math.nan),
        median_timing_z=("timing_z", "median"),
        median_composite=("composite", "median"),
    )
    agg["tier"] = [tiers.get(str(i).strip().lower(), "") for i in agg.index]
    return agg.sort_values(["median_composite", "hit_rate"], ascending=[False, False])


def build_graph(scored: pd.DataFrame, events: pd.DataFrame, top_n: int = 400):
    import networkx as nx

    g = nx.Graph()
    top = scored.dropna(subset=["composite"]).nlargest(top_n, "composite")
    for _, r in top.iterrows():
        person = f"person:{r.person}"
        ticker = f"ticker:{r.ticker}"
        g.add_node(person, kind="person", label=r.person)
        g.add_node(ticker, kind="ticker", label=r.ticker)
        edge_attrs = {"kind": "trade", "composite": float(r.composite), "timing_z": float(r.timing_z), "date": str(r.transaction_date)}
        if pd.notna(r.abnormal_30d):
            edge_attrs["abnormal_30d"] = float(r.abnormal_30d)
        g.add_edge(person, ticker, **edge_attrs)
    ev_dates = []
    for _, e in events.iterrows():
        if pd.isna(e["date"]):
            continue
        node = f"event:{e['event_id']}"
        g.add_node(node, kind="event", label=str(e.get("description", "")), date=str(e["date"].date()))
        ev_dates.append((node, e["date"].date()))
    for _, r in top.iterrows():
        d = pd.Timestamp(r.transaction_date).date()
        for node, ed in ev_dates:
            gap = abs((d - ed).days)
            if gap <= 7:
                g.add_edge(f"person:{r.person}", node, kind="near_event", gap_days=gap)
    return g


def baseline_report(scored: pd.DataFrame) -> str:
    s = scored.dropna(subset=["abnormal_30d"])
    if s.empty:
        return "NO SCORED TRADES — price fetch failed; no baseline."
    lines = [
        "Null-baseline check (all filers, all scored trades, 30d abnormal vs SPY):",
        f"  scored trades: {len(s)}",
        f"  median abnormal return: {s['abnormal_30d'].median():+.4f}",
        f"  mean abnormal return:   {s['abnormal_30d'].mean():+.4f}",
        f"  hit rate (> +5%):       {(s['abnormal_30d'] > REP_THRESHOLD).mean():.3f}",
        "Expectation per Belmont/Sacerdote 2022: median ~ 0, hit rate ~ coin flip.",
        "Anyone far above this baseline is an outlier AGAINST the null, not proof of intent.",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trades", required=True, type=Path)
    ap.add_argument("--events", required=True, type=Path)
    ap.add_argument("--seed", type=Path, default=None)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--price-cache", type=Path, default=None)
    ap.add_argument("--top-graph", type=int, default=400)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    trades = pd.read_csv(
        args.trades,
        low_memory=False,
        parse_dates=["transaction_date", "filing_date"],
    )
    events = load_events_csv(args.events)
    event_dates = [d.date() for d in events["date"].dropna()]
    seed = pd.read_csv(args.seed, dtype=str) if args.seed and args.seed.exists() else pd.DataFrame()

    tickers = sorted({str(t).upper() for t in trades["ticker"].dropna() if str(t).strip()} | {SPY})
    prices = fetch_prices(tickers, args.price_cache or args.out / "price_cache")
    spy = prices.get(SPY)
    if spy is None:
        raise SystemExit("SPY price series unavailable — cannot compute abnormal returns")
    prices = {k: v for k, v in prices.items() if k != SPY}

    person_stats = build_person_stats(trades)
    scored = score_trades(trades, event_dates, prices, spy, person_stats)
    scored.to_csv(args.out / "insider_map.csv", index=False)

    board = leaderboard(scored, seed)
    board.to_csv(args.out / "leaderboard.csv")

    q = round3_queue(scored)
    q.to_csv(args.out / "round3_queue.csv", index=False)

    g = build_graph(scored, events, args.top_graph)
    try:
        import networkx as nx

        nx.write_graphml(g, args.out / "insider_graph.graphml")
    except ImportError:
        (args.out / "insider_graph.graphml").write_text("networkx not installed")

    with open(args.out / "baseline.md", "w") as f:
        f.write(baseline_report(scored) + "\n")
    print(f"scored {len(scored)} trades, {scored['person'].nunique()} people -> {args.out}")


if __name__ == "__main__":
    main()
