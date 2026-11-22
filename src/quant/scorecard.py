"""The paper-trading scorecard: what actually happened vs the plan.

Reads data/paper/log.jsonl (append-only events from paper.py) and reports
per coin: events, fills, current equity, drawdown vs the agreed maximum,
and the sizing recommendation from sizing.py.

Today this is short (the log is hours old). Its job is to be BORING and
CORRECT for months, so when the accuracy verdict is due, the data is there
and the arithmetic was watched by tests the whole time.
"""
import json
from collections import defaultdict
from pathlib import Path

from quant import config, sizing

PAPER_DIR = config.PROJECT_ROOT / "data" / "paper"


def summarize(events: list[dict]) -> dict[str, dict]:
    """Pure aggregation over log events — the testable core of the CLI."""
    by_symbol: dict[str, list[dict]] = defaultdict(list)
    for event in events:
        by_symbol[event["symbol"]].append(event)

    out = {}
    for symbol, events_for_symbol in sorted(by_symbol.items()):
        events_for_symbol.sort(key=lambda e: e["ts"])
        equity_path = [e["equity"] for e in events_for_symbol]
        fills = [e for e in events_for_symbol if e["action"]]
        last_action = next((e["action"] for e in reversed(events_for_symbol) if e["action"]), None)
        breached, dd = sizing.drawdown_watch(equity_path)
        out[symbol] = {
            "events": len(events_for_symbol),
            "fills": len(fills),
            "last_action": last_action,
            "first_ts": events_for_symbol[0]["ts"],
            "last_ts": events_for_symbol[-1]["ts"],
            "equity": equity_path[-1],
            "drawdown": dd,
            "breached": breached,
        }
    return out


def load_events(directory: Path = None) -> list[dict]:
    directory = directory or PAPER_DIR
    log = directory / "log.jsonl"
    if not log.exists():
        return []
    with log.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    events = load_events()
    if not events:
        print("[scorecard] no paper events yet — run `python -m quant.paper` first")
        return

    summary = summarize(events)
    print(f"paper scorecard — {len(events)} events, {len(summary)} symbols")
    print(f"agreed max drawdown: {sizing.MAX_DD:.0%}")
    for symbol, stats in summary.items():
        fill_note = f"last fill {stats['last_action']}" if stats["last_action"] else "no fills yet"
        breach = "  [MAX DD BREACHED]" if stats["breached"] else ""
        print(
            f"  {symbol}: {stats['events']} events, {stats['fills']} fills ({fill_note}) | "
            f"equity {stats['equity']:,.2f} | dd {stats['drawdown']:+.1%}{breach}"
        )

    # allocation recommendation for the real phase (option A: advisory only)
    bankroll = sum(stats["equity"] for stats in summary.values())
    weights = {symbol: 1.0 for symbol in summary}
    print("\nrecommended real-phase allocation (advisory, per sizing.py):")
    for symbol, dollars in sizing.allocate(bankroll, weights).items():
        print(f"  {symbol}: ${dollars:,.2f}")


if __name__ == "__main__":
    main()
