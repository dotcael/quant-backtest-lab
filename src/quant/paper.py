"""Paper trading: real prices, fake money, zero risk, full honesty.

Why this exists: backtests prove what WOULD have happened to history;
paper trading measures what our simulation gets WRONG live — real signal
timing, real fills vs the close, real cost gaps. Every event is logged so
the eventual accuracy scorecard (predicted vs delivered) has raw material.

Model (kept identical to the backtest's assumptions so gaps we measure are
execution gaps, not model gaps):
  - all-in / all-flat, one asset at a time, spot only
  - fills at the latest daily close, cost = taker + slippage per side
  - every decision and fill appended to an append-only JSONL log

Run it daily (cron/launchd or by hand): `python -m quant.paper`
Each run: sync candles -> compute signal -> act if changed -> log -> report.
"""
import json
import re
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from quant import config, dashboard, fetch, fees, sizing, store, validate
from quant.strategies import crash_shield

START_CASH = 1_000.0
PAPER_DIR = config.PROJECT_ROOT / "data" / "paper"

POWER_MIN_BATTERY = 0.80  # run on battery only above this charge
STALE_DEADLINE_UTC_HOUR = 18  # warn once the UTC day is 6h from closing
STALE_BAR_HOURS = 48  # newest daily candle older than this = the sync is broken


@dataclass(frozen=True)
class Account:
    cash: float
    units: float


def decide_action(units: float, desired: float) -> str | None:
    """'buy' / 'sell' when position disagrees with signal, else None."""
    long_now = units > 0.0
    if long_now and desired < 1.0:
        return "sell"
    if not long_now and desired > 0.0:
        return "buy"
    return None


def apply_fill(acct: Account, action: str, price: float, fee_model: fees.FeeModel) -> Account:
    """All-in buy / full exit at `price`, paying taker+slippage per side —
    the same per-side cost the backtest charges."""
    side = fee_model.taker + fee_model.slippage
    if action == "buy":
        invested = acct.cash * (1 - side)
        return Account(cash=0.0, units=invested / price)
    if action == "sell":
        proceeds = acct.units * price * (1 - side)
        return Account(cash=proceeds, units=0.0)
    raise ValueError(f"unknown action {action!r}")


def equity(acct: Account, price: float) -> float:
    return acct.cash + acct.units * price


def _state_path(symbol: str, directory: Path) -> Path:
    safe = symbol.replace("/", "-")
    return directory / f"{safe}.json"


def save_state(symbol: str, acct: Account, directory: Path = None) -> None:
    directory = directory or PAPER_DIR
    directory.mkdir(parents=True, exist_ok=True)
    _state_path(symbol, directory).write_text(json.dumps(asdict(acct)))


def load_state(symbol: str, directory: Path = None) -> Account:
    directory = directory or PAPER_DIR
    path = _state_path(symbol, directory)
    if not path.exists():
        return Account(cash=START_CASH, units=0.0)
    return Account(**json.loads(path.read_text()))


def append_event(event: dict, directory: Path = None) -> None:
    directory = directory or PAPER_DIR
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "log.jsonl").open("a") as fh:
        fh.write(json.dumps(event) + "\n")


def power_ok_from_output(pmset_output: str, min_battery: float = POWER_MIN_BATTERY) -> bool:
    """True if the machine may spend battery on a scan: on AC power, or
    charged at/above min_battery. Unknown output fails OPEN — a parse
    error must never silently stop the scans."""
    if "AC Power" in pmset_output:
        return True
    match = re.search(r"(\d+)%", pmset_output)
    if match:
        return int(match.group(1)) >= min_battery * 100
    return True


def power_ok() -> bool:
    try:
        result = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=10)
        return power_ok_from_output(result.stdout)
    except Exception:
        return True  # fail open (non-mac, pmset missing, timeout...)


def is_stale(events: list[dict], now: datetime | None = None, deadline_hour: int = STALE_DEADLINE_UTC_HOUR) -> bool:
    """True when the UTC day is nearly over and no scan was logged today —
    the 'bracket about to close' alarm. Before the deadline we stay quiet:
    there is still time, and crying wolf trains people to ignore alarms."""
    now = now or datetime.now(timezone.utc)
    today = now.strftime("%Y-%m-%d")
    scanned_today = any(event["ts"][:10] == today for event in events)
    return (not scanned_today) and now.hour >= deadline_hour


def notify(title: str, message: str) -> None:
    """macOS notification. Best effort only: a broken notifier must never
    break trading."""
    try:
        subprocess.run(
            ["osascript", "-e", f'display notification "{message}" with title "{title}" sound name "Ping"'],
            capture_output=True,
            timeout=10,
        )
    except Exception:
        pass


def _alarm_if_stale() -> None:
    from quant import scorecard  # local import: scorecard -> sizing only, no cycle

    if is_stale(scorecard.load_events()):
        print("[paper] STALE: no scan logged today and the UTC day is closing — plug in / open the lid")
        notify("quant paper trader", "No scan yet today — the daily bracket is closing. Plug in or open the Mac.")


def sync_candles(ex, exchange_id: str, symbol: str, timeframe: str = "1d") -> pd.DataFrame:
    """Pull whatever candles are missing since the last stored one."""
    resume = store.last_timestamp(exchange_id, symbol, timeframe)
    since = resume + fetch.timeframe_ms(timeframe) if resume is not None else 0
    df = fetch.fetch_ohlcv(ex, symbol, timeframe, since, limit=config.FETCH_LIMITS.get(exchange_id, 500))
    if not df.empty:
        report = validate.validate_ohlcv(df, timeframe)
        if not report.ok:
            print(f"[paper] {symbol} {timeframe} validation: {report.summary()} (storing, noted)")
        store.append(exchange_id, symbol, timeframe, df)
    return store.parquet_path(exchange_id, symbol, timeframe)


def newest_bar_age_hours(exchange_id: str, timeframe: str = "1d") -> float | None:
    """Age of the freshest stored candle. A run that 'succeeds' but leaves the
    parquet days old is the failure mode that froze the console on 2026-11-07 —
    the scan wrote a snapshot, so nothing looked broken, while the newest bar
    was five days stale. Make that impossible to miss."""
    newest = None
    for symbol in config.SYMBOLS:
        ts = store.last_timestamp(exchange_id, symbol, timeframe)
        if ts is not None and (newest is None or ts > newest):
            newest = ts
    if newest is None:
        return None
    now_ms = datetime.now(timezone.utc).timestamp() * 1000
    return (now_ms - newest) / 3_600_000.0


def _alarm_if_bars_stale() -> None:
    """Loud warning when the newest stored candle is older than a full
    trading day — the console renders happily on stale bars otherwise."""
    age = newest_bar_age_hours(config.EXCHANGES[0])
    if age is not None and age > STALE_BAR_HOURS:
        msg = f"newest 1d candle is {age:.0f}h old — the daily sync is not landing"
        print(f"[paper] STALE BARS: {msg}")
        notify("quant paper trader", msg)


def main() -> None:
    try:
        if not power_ok():
            print("[paper] on battery below 80% — skipping this run; next scheduled run retries")
            return
        _trade()
    finally:
        # runs after success, power-skip, AND crash: the alarm must fire
        # exactly when the human said "tell me if a day goes by unscanned"
        _alarm_if_stale()
        _alarm_if_bars_stale()


def _trade() -> None:
    ex = fetch.create_exchange_with_fallback()
    exchange_id = ex.id
    fee_model = fees.FeeModel.for_exchange(exchange_id)

    # option A: paper fills stay all-in (comparable to backtests); sizing is
    # an advisory layer logged alongside, consumed by the real phase
    weights = {symbol: 1.0 for symbol in config.SYMBOLS}
    allocation = sizing.allocate(sizing.START_BANKROLL, weights)

    for symbol in config.SYMBOLS:
        parquet = sync_candles(ex, exchange_id, symbol)
        df = pd.read_parquet(parquet)
        window = df.tail(400)  # enough history for the 200-day shield
        signal = crash_shield(window)
        price = float(window["close"].iloc[-1])
        ma = float(window["close"].rolling(200).mean().iloc[-1])
        desired = float(signal.iloc[-1])

        acct = load_state(symbol)
        action = decide_action(acct.units, desired)
        if action:
            acct = apply_fill(acct, action, price, fee_model)
            save_state(symbol, acct)
        append_event(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "mode": "paper",
                "symbol": symbol,
                "strategy": "crash_shield 200",
                "price": price,
                "ma200": round(ma, 2),
                "desired": desired,
                "action": action,
                "cash": round(acct.cash, 6),
                "units": round(acct.units, 8),
                "equity": round(equity(acct, price), 6),
                "alloc_usd": round(allocation[symbol], 2),
            }
        )
        state = "LONG" if acct.units > 0 else "FLAT"
        moved = f" -> {action.upper()} @ {price:,.2f}" if action else " (no change)"
        print(
            f"[paper] {symbol}: price {price:,.2f} vs MA200 {ma:,.2f} -> want "
            f"{'LONG' if desired else 'FLAT'}{moved} | {state} | equity "
            f"{equity(acct, price):,.2f} of {START_CASH:,.0f} | advise ${allocation[symbol]:,.0f}"
        )


if __name__ == "__main__":
    main()
    dashboard.main()  # refresh the console snapshot on the same cadence
