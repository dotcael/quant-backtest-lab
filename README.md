# quant

Side-project phase. **No real money, no live trading.** Goal: clean market
data flowing and stored locally, so strategies can be backtested honestly
before anything else.

See `docs/crypto_market_analysis_prep.docx` for the required background
(order types, slippage, position sizing, fees, overfitting).

## Ground rules (non-negotiable)

1. No fee-free or slippage-free backtests — costs always come from
   `src/quant/fees.py`.
2. Any "too good" backtest result gets flagged as a possible overfit,
   not celebrated.
3. Requests for live-trading logic get redirected to the
   backtest / paper-trade phase first.

## Layout

```
src/quant/
  config.py    # exchange, symbols, timeframes, history depth — one place
  fetch.py     # ccxt pull with pagination + exchange fallback  (online)
  validate.py  # duplicate/gap/OHLC sanity checks before storing (local)
  store.py     # parquet storage, append-only raw zone            (local)
  fees.py      # maker/taker + slippage cost model (imported by every backtest)
  strategies.py # signal generators: buy&hold, sma_cross (pure functions)
  backtest.py  # no-lookahead engine, mandatory costs, overfit tripwire (local)
  timeframes.py # '1h' -> ms parsing shared everywhere (breaks an import cycle)
tests/         # pytest, runs fully offline
data/raw/      # gitignored parquet: {exchange}/{SYMBOL-QUOTE}/{timeframe}.parquet
```

Only `fetch.py` touches the network. Everything else runs locally —
heavy analysis later happens on the Windows PC by cloning this repo and
re-running the same fetch there.

## Backtesting

```bash
.venv/bin/python -m quant.backtest   # buy&hold + SMA-cross on all stored data
```

Execution model: signal at bar i's close earns bar i+1's return (no
look-ahead); every position change pays taker fee + slippage from
`fees.py`. Results that look too good are flagged as possible overfits,
not celebrated.

## Paper trading

```bash
.venv/bin/python -m quant.paper      # sync, signal, fill-at-close, log
```

Fake money, real prices: crash-shield on every configured symbol, fills
cost the same fees as backtests, every event lands in the append-only
`data/paper/log.jsonl` (that log is the raw material for the accuracy
scorecard).

Automated via launchd (macOS): `deploy/com.quant.paper.plist` runs it at
login and at 4 fixed times daily (02:30 / 08:30 / 14:30 / 20:30 local —
first run lands 30min after the UTC daily candle closes). Missed jobs
while asleep fire on wake. Install:

```bash
cp deploy/com.quant.paper.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.quant.paper.plist
tail -f logs/paper-launchd.log      # watch it work
```

Power + watchdog rules (built in to `paper.py`):
- a run is **skipped** if the Mac is on battery below 80% (charging always
  runs; skips retry at the next scheduled time)
- if no scan has been logged by 18:00 UTC, the next attempt raises a
  **macOS notification** — "the daily bracket is closing, plug in"

(Windows PC clone will need the Task Scheduler equivalent — same
command, different scheduler.)

## Risk & sizing

```bash
.venv/bin/python -m quant.scorecard  # what happened vs the plan
```

The onboarding doc's rules, executable: diversification with a per-coin
cap (`sizing.allocate`), the 1-2% risk rule (`sizing.position_size`),
and a drawdown watchdog with the max decided up front (`sizing.MAX_DD`,
currently 20%). Paper fills stay all-in for backtest comparability;
sizing is logged alongside as an advisory (`alloc_usd`) for the real
phase.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest          # offline sanity check
.venv/bin/python -m quant.fetch   # first data pull
```

## Data policy

`data/raw` is append-only. Fetch adds new candles; nothing edits raw data
after the fact. Any cleaning/resampling writes to a separate zone later.
This keeps the future in-sample / out-of-sample split honest — a backtest
can always be reproduced against exactly the data it saw.
