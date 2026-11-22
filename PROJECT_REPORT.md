# quant — Project State Report (2026-08-27)

**One-liner:** Local-first crypto OHLCV pipeline with mandatory-fee backtesting, live paper-trader, and an honesty-first philosophy. Zero live money, zero hype.

---

## What Exists (Code)

| Module | Purpose | Key Feature |
|---|---|---|
| `fetch.py` | Download OHLCV from Binance/Kraken | Paginated, resumable, fallback, rate-limited |
| `validate.py` | Sanity-check every pull | Duplicates, gaps, bad OHLC, nonpositive |
| `store.py` | Append-only Parquet | `data/raw/{ex}/{SYM}/1d.parquet` |
| `fees.py` | Single cost truth | Maker/taker + 5bp slippage; `stress` multiplier |
| `backtest.py` | No-lookahead engine | `pos[i]=signal[i-1]`, cost per trade, maxdd/CAGR/cost_drag |
| `strategies.py` | Pure signal functions | `buy_and_hold`, `sma_cross`, `breakout`, `mean_reversion`, `crash_shield`, `rsi` |
| `oos.py` | Single split (70/30) | Design sweep → one proof run; overfit flag |
| `walkforward.py` | Repeated exam | Anchored expanding design, 5 windows, chained return |
| `paper.py` | Live paper trader | Sync → signal → fill at close → JSONL log (`data/paper/`) |

**Tests:** 56 passing (TDD discipline — no production code without a failing test first).

---

## Data Status

| Symbol | 1d bars | 1h bars | Span (1d) | Validation |
|---|---|---|---|---|
| BTC/USDT | 3,298 | 26,280 | 2017-08-17 → 2026-08-27 | Clean |
| ETH/USDT | 3,298 | 26,280 | 2017-08-17 → 2026-08-27 | Clean |
| SOL/USDT | 2,208 | 26,280 | 2020-08-20 → 2026-08-27 | Clean |
| BNB/USDT | 3,217 | 26,280 | 2017-11-14 → 2026-08-27 | Clean |
| XRP/USDT | 3,038 | 26,280 | 2017-12-15 → 2026-08-27 | Clean |

All validation: 0 duplicates, 0 gaps, 0 bad OHLC, 0 nonpositive.

---

## Walk-Forward Results (Chained Return, 2022-02-21 → 2026-08-27)

*Families: SMA (tuned 12-config sweep), Breakout 20/10, RSI 14/30/50, Crash-Shield 200 (all untuned except SMA)*

| Coin | Buy&Hold | SMA | Breakout | RSI | **Shield** |
|---|---|---|---|---|---|
| BTC | **+115%** | +61% | +22% | +67% | +50% |
| ETH | −2% | +43% | −44% | −43% | **+100%** |
| SOL | +380% | **+458%** | +317% | −22% | +81% |
| BNB | +63% | +42% | **+108%** | −30% | +43% |
| XRP | **+357%** | −31% | +35% | +73% | +127% |

**Conclusions from the filter phase:**

1. **No family dominates.** Each has one coin where it shines and others where it bleeds.
2. **RSI mean-reversion is the weakest** — negative on 3/5 coins.
3. **Crash-shield is the only family positive on ALL 5 coins.** It never maximizes bull returns, but it never gets destroyed either. That "never loses the plot" quality is the robustness signature we've been hunting.
4. **The lone survivor is suspicious.** Tuned SMA's +43% on ETH stands alone vs two untuned classics that lost — classic lucky-winner-of-12 profile.
5. **On strong bulls, nothing beats holding.** Real choice: max growth (hold, eat crashes) vs. max resilience (shield, lag rips).

---

## Paper Trader (Live)

```bash
python -m quant.paper
```

First run (2026-08-27): BTC & ETH above 200-day MA → paper BUY both → equity 998.50/1000 (honest −0.15% entry cost visible immediately).

**Log:** `data/paper/log.jsonl` (append-only JSONL — raw material for accuracy scorecard).
**State:** `data/paper/{SYM}.json` (cash, units).

Runs daily: sync candles → compute crash_shield signal → fill at close with same fees/slippage as backtest → log → report.

---

## Key Honesty Gates (Already Enforced)

- **Fees mandatory** — every backtest imports `fees.FeeModel`; no fee-free path exists.
- **No look-ahead** — signal at bar i's close earns bar i+1's return.
- **Out-of-sample discipline** — oos (single split) + walkforward (5 sequential exams).
- **Fixed params untuned** — breakout/RSI/shield use textbook numbers; zero selection bias.
- **Cost stress-testing** — `--stress 2` reruns with doubled costs; ETH shield survived (+94.5%).
- **Real microstructure check** — `fetch --microstructure`: BTC/ETH spreads ~0–3bp vs assumed 50bp (10–50× conservative in calm markets).
- **Append-only raw data** — fetch adds, nothing rewrites; future OOS splits stay honest.

---

## Survivorship Caveat (Explicit)

All 5 coins chosen because they *survived* to today's top list. XRP's +357% buy&hold is the smoking gun — it had a monster run we didn't predict. A true test would include dead/delisted coins. Acknowledged, not solved.

---

## Next Gates (In Order)

1. **Accumulate paper-trader accuracy data** — run `python -m quant.paper` daily (cron/launchd). Build the execution-gap scorecard: predicted vs. actual fills, signal agreement, PnL gap.
2. **Position-sizing rules** — implement the sizing math from `docs/crypto_market_analysis_prep.docx` (fixed-fraction, max drawdown cap, Kelly fraction) before any real-money conversation.
3. (Optional) **Basket shield** — run crash_shield across all 5 coins simultaneously, diversify the airbag.

---

## Quick Commands

```bash
# Daily paper run
python -m quant.paper

# Full walk-forward (all families, all 5 coins)
python -m quant.walkforward

# Torture test (2x costs)
python -m quant.walkforward --stress 2

# Live microstructure check
python -m quant.fetch --microstructure

# Full test suite
pytest
```

---

## Git History (Clean, One Commit Per Milestone)

```
4927466 widen coin set to 5 majors; crash-shield joins walkforward CLI
f607547 paper: live paper-trader — crash_shield on real prices
980e859 strategies: crash_shield (200-day airbag, untuned)
95d16e0 strategies: breakout + mean-reversion; walkforward fixed-param
408d123 fees: order-book slippage, stress multiplier, fee_model override
a289dcb walkforward: repeated anchored tune-then-test
8f1fca0 oos: single chronological split, design-only sweep
ba6a670 backtest: no-lookahead engine, mandatory fees, SMA + buy&hold
40afdf4 scaffold: ccxt OHLCV pipeline skeleton
```

---

## Philosophy (From README)

> Side-project phase. **No real money, no live trading.** Goal: clean market data flowing and stored locally, so strategies can be backtested honestly before anything else.

> 1. No fee-free or slippage-free backtests — costs always come from `src/quant/fees.py`.
> 2. Any "too good" backtest result gets flagged as a possible overfit, not celebrated.
> 3. Requests for live-trading logic get redirected to the backtest / paper-trade phase first.

---

*End of report. This project is in the "filter phase" — we've killed the weak strategies, found one robust signal (crash-shield), and the paper-trader is now accumulating the real-world accuracy data that gates the next phase.*