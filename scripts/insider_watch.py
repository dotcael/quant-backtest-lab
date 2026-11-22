#!/usr/bin/env python
"""Daily insider watcher: kadoa refresh -> diff -> score -> P99 alerts.

    .venv/bin/python scripts/insider_watch.py [--dry-run] [--ntfy]

State: data/insider/watch_state.json (seen trade keys + p99 + last_run).
Alerts: appended to docs/insider-alerts.md; optional ntfy push (--ntfy).
Idempotent: keys persist only after a successful scoring pass; a failed run
leaves state untouched so the next run retries the same window.
Bootstrap: run ONCE with --bootstrap to seed seen_keys from the norm CSV
(58,576 rows through 2026-09-22) so the first daily diff only scores trades
disclosed AFTER the last ingest. Without it, the first run re-scores the whole
kadoa window (slow, wasteful).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from quant.insider import load_events_csv  # noqa: E402
from quant.insider_map import build_person_stats, fetch_prices  # noqa: E402
from quant.watch import (  # noqa: E402
    alert_rows,
    diff_new,
    normalize_kadoa,
    p99_threshold,
    score_new,
)

KADOA_URLS = [
    "https://raw.githubusercontent.com/kadoa-org/congress-trading-monitor/main/public/data/trades.json",
    "https://raw.githubusercontent.com/kadoa-org/congress-trading-monitor/master/public/data/trades.json",
]
STATE = ROOT / "data/insider/watch_state.json"
ALERTS_MD = ROOT / "docs/insider-alerts.md"
MAP_CSV = ROOT / "data/insider/map/insider_map.csv"
NORM_CSV = ROOT / "data/insider/congress_trades_norm.csv"
EVENTS_CSV = ROOT / "data/insider/events.csv"
PRICE_CACHE = ROOT / "data/insider/watch_cache"
NTFY_TOPIC = "https://ntfy.sh/quant-insider"


def fetch_kadoa() -> list[dict]:
    last_err = None
    for url in KADOA_URLS:
        # curl over urllib: system trust store just works here (venv python
        # lacks a CA bundle on this box — verified 2026-09-29)
        p = subprocess.run(["curl", "-sL", "--max-time", "120", "-f", url],
                           capture_output=True, timeout=150)
        if p.returncode == 0:
            try:
                return json.loads(p.stdout)
            except json.JSONDecodeError as e:  # noqa: PERF203
                last_err = e
        else:
            last_err = RuntimeError(f"curl rc={p.returncode} for {url}")
    raise RuntimeError(f"kadoa fetch failed: {last_err}")


def load_state() -> dict:
    if STATE.exists():
        return json.load(open(STATE))
    return {"seen_keys": [], "p99": None, "last_run": None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="no state write, no ntfy")
    ap.add_argument("--ntfy", action="store_true", help="push alerts to ntfy.sh/quant-insider")
    ap.add_argument("--bootstrap", action="store_true",
                    help="seed seen_keys from norm CSV before diffing (first run only)")
    args = ap.parse_args()

    state = load_state()
    if args.bootstrap and state["seen_keys"]:
        print("state already has keys; --bootstrap ignored")
    elif args.bootstrap:
        norm_all = pd.read_csv(NORM_CSV, low_memory=False,
                               usecols=["person", "ticker", "transaction_date",
                                        "amount_min", "amount_max"])
        from quant.watch import trade_key
        keys = {
            trade_key(r.person, r.ticker, r.transaction_date, r.amount_min, r.amount_max)
            for r in norm_all.itertuples()
        }
        state["seen_keys"] = sorted(keys)
        print(f"bootstrapped {len(keys)} keys from norm CSV")

    records = fetch_kadoa()
    rows = normalize_kadoa(records)
    new = diff_new(rows, set(state["seen_keys"]))
    print(f"kadoa records: {len(rows)}, new vs state: {len(new)}")

    if new.empty:
        state["last_run"] = datetime.now(timezone.utc).isoformat()
        if not args.dry_run:
            STATE.write_text(json.dumps(state, indent=1))
        print("nothing new; state last_run updated" if not args.dry_run else "nothing new (dry-run)")
        return 0

    map_csv = pd.read_csv(MAP_CSV, low_memory=False)
    norm = pd.read_csv(NORM_CSV, low_memory=False, parse_dates=["transaction_date"])
    events = [d.date() for d in load_events_csv(EVENTS_CSV)["date"].dropna()]
    stats = build_person_stats(norm)

    if state.get("p99") is None:
        state["p99"] = p99_threshold(map_csv)
    tickers = sorted({t for t in new["ticker"] if t} | {"SPY"})
    prices = fetch_prices(tickers, PRICE_CACHE)
    spy = prices.get("SPY")
    if spy is None:
        print("SPY unavailable; cannot score", file=sys.stderr)
        return 1

    scored = score_new(new, events, stats, map_csv, prices, spy)
    alerts = (
        alert_rows(scored, state["p99"])
        if state.get("p99")
        else scored.iloc[0:0]
    )
    print(f"scored {len(scored)} new trades; p99={state['p99']}; alerts={len(alerts)}")

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    if not alerts.empty:
        lines = ["", f"## {now} — {len(alerts)} new trade(s) ≥ P99", ""]
        lines.append("| person | ticker | date | type | amount | composite | abnormal_30d | source |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in alerts.itertuples():
            lines.append(
                f"| {r.person} | {r.ticker} | {r.transaction_date} | {r.transaction_type} "
                f"| ${r.amount_min:,.0f}-${r.amount_max:,.0f} | {r.composite:.3f} "
                f"| {r.abnormal_30d:+.3f} | {r.source_url} |"
            )
        lines.append("\n_Statistical flag vs P99 of the rescored composite distribution — "
                     "not an accusation. Verify against the primary filing before anything else._\n")
        with open(ALERTS_MD, "a") as f:
            f.write("\n".join(lines))
        if args.ntfy and not args.dry_run:
            for r in alerts.itertuples():
                msg = f"insider-watch: {r.person} {r.transaction_type} {r.ticker} {r.transaction_date} composite {r.composite:.3f} (≥P99)"
                subprocess.run(["curl", "-s", "-d", msg, NTFY_TOPIC], timeout=30)
        for r in alerts.itertuples():
            print(f"  ALERT {r.person} {r.ticker} {r.transaction_date} comp={r.composite:.3f}")

    state["seen_keys"] = sorted(set(state["seen_keys"]) | set(new["key"]))
    state["last_run"] = datetime.now(timezone.utc).isoformat()
    if not args.dry_run:
        STATE.write_text(json.dumps(state, indent=1))
        print(f"state updated: {len(state['seen_keys'])} keys")
    else:
        print("dry-run: state NOT written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
