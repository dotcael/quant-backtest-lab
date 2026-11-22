import pandas as pd

from quant.validate import validate_ohlcv

H = 3_600_000  # one hour in ms


def make_df(timestamps, open_=100.0, high=110.0, low=90.0, close=105.0, volume=10.0):
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def test_clean_data_passes():
    df = make_df([0, H, 2 * H])
    report = validate_ohlcv(df, "1h")
    assert report.ok
    assert report.summary().startswith("duplicates=0")


def test_duplicate_timestamp_detected():
    df = make_df([0, H, H, 2 * H])
    report = validate_ohlcv(df, "1h")
    assert report.duplicates == 1


def test_gap_detected():
    df = make_df([0, H, 3 * H])  # hour 2 missing
    report = validate_ohlcv(df, "1h")
    assert len(report.gaps) == 1
    assert report.gaps[0] == (H, 3 * H)


def test_bad_ohlc_detected():
    df = make_df([0, H], high=95.0)  # high below open/close -> impossible
    report = validate_ohlcv(df, "1h")
    assert report.bad_ohlc == 2


def test_nonpositive_price_detected():
    df = make_df([0, H], low=[90.0, -1.0])  # bad value in one row only
    report = validate_ohlcv(df, "1h")
    assert report.nonpositive == 1


def test_daily_timeframe_gap_ignored():
    D = 86_400_000
    df = make_df([0, D, 2 * D])
    report = validate_ohlcv(df, "1d")
    assert report.ok  # 24h steps are the expected step for 1d candles
