"""Central configuration — everything you'd ever want to change lives here.

Why a config module instead of constants scattered in scripts: one place to
audit what the pipeline pulls, and switching exchange/symbol/depth never
means touching logic.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Priority order: first entry is primary, the rest are fallbacks if the
# primary is unreachable or geo-blocked.
# Verified 2026-08-27 from Rwanda: Binance spot data reachable (HTTP 200).
EXCHANGES = ["binance", "kraken"]

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]
TIMEFRAMES = ["1h", "1d"]

# History depth in days per timeframe. None = as far back as the exchange has.
LOOKBACK = {
    "1h": 3 * 365,
    "1d": None,
}

# Max candles per single API call (exchange-documented limits).
FETCH_LIMITS = {"binance": 1000, "kraken": 720}
