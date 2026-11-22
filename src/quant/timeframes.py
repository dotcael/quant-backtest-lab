"""Timeframe string parsing shared by fetch, validate, and backtest.

Lives in its own module because everything needs it and it needs nothing —
importing it from fetch (as this project once did) creates a fetch <->
validate circular import the moment any third module imports fetch first.
"""
UNIT_MS = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}


def timeframe_ms(timeframe: str) -> int:
    """'1h' -> 3_600_000. Milliseconds are the unit all exchange APIs speak."""
    return int(timeframe[:-1]) * UNIT_MS[timeframe[-1]]
