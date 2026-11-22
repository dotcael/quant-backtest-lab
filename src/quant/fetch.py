"""Pull historical OHLCV candles via ccxt, with automatic exchange fallback.

Why pagination: exchanges cap candles per request (Binance: 1000,
Kraken: 720). Three years of 1h candles is ~26,000 rows, so we walk
forward from a start timestamp page by page until reaching now.

Why resume logic: each run starts from the last candle already stored,
so re-runs fetch only what's missing — never re-pulling years of data.

This is the ONLY module in the package that touches the network.
"""
import time

import ccxt
import pandas as pd

from quant import config, fees, store, validate
from quant.timeframes import timeframe_ms


REQUEST_TIMEOUT_MS = 30_000  # ccxt default is 10s — too short right after a wake
RETRY_DELAYS_S = (10, 30, 90)  # backoff after a full pass over all exchanges


def _make_exchange(exchange_id: str) -> ccxt.Exchange:
    """Public-data client. No API keys: OHLCV history is public, so there
    is no reason to identify ourselves to the exchange."""
    exchange_class = getattr(ccxt, exchange_id)
    return exchange_class({"enableRateLimit": True, "timeout": REQUEST_TIMEOUT_MS})  # ccxt self-throttles


def create_exchange_with_fallback(
    exchange_ids: list[str] | None = None, retry_delays: tuple[int, ...] = RETRY_DELAYS_S
) -> ccxt.Exchange:
    """Try exchanges in priority order; return the first that responds.

    Why retries: the scheduled scan fires at 02:30/08:30 — often the moment the
    Mac wakes, before Wi-Fi has re-associated. A single failed pass used to
    abort the whole scan, and the console then showed candles frozen for days
    (2026-11-07: 5 days stale, last bar Nov 2). So a failed pass backs off and
    tries again instead of giving up on the first NetworkError."""
    last_err: Exception | None = None
    for delay in (0, *retry_delays):
        if delay:
            print(f"[fetch] no exchange yet — retrying in {delay}s")
            time.sleep(delay)
        for exchange_id in exchange_ids or config.EXCHANGES:
            try:
                ex = _make_exchange(exchange_id)
                ex.load_markets()  # one round-trip: proves reachability + symbol map
                print(f"[fetch] using {ex.id}")
                return ex
            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.DDoSProtection, ccxt.ExchangeError) as err:
                print(f"[fetch] {exchange_id} unavailable ({type(err).__name__}), trying next...")
                last_err = err
    raise RuntimeError(f"no exchange reachable ({type(last_err).__name__ if last_err else 'unknown'})")


def start_timestamp(timeframe: str, lookback_days: int | None) -> int:
    """Where the first pull begins. since=0 means 'from the very beginning';
    exchanges simply clamp to their earliest available candle."""
    if lookback_days is None:
        return 0
    return int(time.time() * 1000) - lookback_days * 86_400_000


def fetch_ohlcv(ex: ccxt.Exchange, symbol: str, timeframe: str, since_ms: int, limit: int) -> pd.DataFrame:
    """Walk forward page by page until 'now'. Returns deduped, sorted candles."""
    step = timeframe_ms(timeframe)
    now_ms = int(time.time() * 1000)
    rows: list[list] = []
    cursor = since_ms
    while cursor < now_ms:
        batch = ex.fetch_ohlcv(symbol, timeframe, since=cursor, limit=limit)
        if not batch:
            break  # exchange has nothing after cursor
        rows.extend(batch)
        cursor = batch[-1][0] + step  # resume point: just past the last candle
        print(f"\r[fetch] {symbol} {timeframe}: {len(rows)} candles", end="", flush=True)
    print()
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def microstructure_report(ex: ccxt.Exchange, symbol: str, notionals: tuple = (1_000.0, 10_000.0)) -> None:
    """Reality-check the cost assumptions against the LIVE order book.

    Backtests assume a flat slippage (fees.DEFAULT_SLIPPAGE). This prints
    the actual spread plus the real market-order cost of lifting/walking
    the book for a few position sizes — the number that assumption has to
    survive contact with. Snapshot in time, not a historical average: run
    it more than once, including in choppy hours.
    """
    ob = ex.fetch_order_book(symbol, limit=25)
    best_bid, best_ask = ob["bids"][0][0], ob["asks"][0][0]
    mid = (best_bid + best_ask) / 2
    spread = (best_ask - best_bid) / mid
    print(f"\n{symbol}: spread {spread:.4%} (bid {best_bid:.2f} / ask {best_ask:.2f})")
    for usd in notionals:
        buy = abs(fees.book_slippage(ob["asks"], usd))
        sell = abs(fees.book_slippage(ob["bids"], usd))
        verdict = "OK" if max(buy, sell) <= fees.DEFAULT_SLIPPAGE else "WORSE THAN ASSUMED"
        print(
            f"  ${usd:>8,.0f} market order: buy +{buy:.4%} / sell -{sell:.4%} "
            f"vs assumed {fees.DEFAULT_SLIPPAGE:.4%} -> {verdict}"
        )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--microstructure",
        action="store_true",
        help="check live spread + real market-order slippage vs our assumptions, then exit",
    )
    args = parser.parse_args()

    ex = create_exchange_with_fallback()

    if args.microstructure:
        for symbol in config.SYMBOLS:
            microstructure_report(ex, symbol)
        return

    limit = config.FETCH_LIMITS.get(ex.id, 500)
    for symbol in config.SYMBOLS:
        for timeframe in config.TIMEFRAMES:
            resume_from = store.last_timestamp(ex.id, symbol, timeframe)
            if resume_from is not None:
                since = resume_from + timeframe_ms(timeframe)
                print(f"[fetch] resuming {symbol} {timeframe} after {resume_from}")
            else:
                since = start_timestamp(timeframe, config.LOOKBACK.get(timeframe))
            df = fetch_ohlcv(ex, symbol, timeframe, since, limit)
            if df.empty:
                print(f"[fetch] {symbol} {timeframe}: nothing new")
                continue
            report = validate.validate_ohlcv(df, timeframe)
            print(f"[validate] {symbol} {timeframe}: {report.summary()}")
            if not report.ok:
                print("[validate] issues found — storing anyway, inspect before trusting")
            store.append(ex.id, symbol, timeframe, df)
            print(f"[store] {symbol} {timeframe}: {len(df)} candles -> {store.parquet_path(ex.id, symbol, timeframe)}")


if __name__ == "__main__":
    main()
