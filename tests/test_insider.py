from datetime import date

import numpy as np
import pandas as pd
import pytest

from quant.insider import (
    PersonStats,
    Trade,
    abnormal_return,
    composite_score,
    filing_deadline,
    forward_return,
    lag_score,
    load_events_csv,
    load_trades_csv,
    nearest_event_gap,
    repetition_ratio,
    timing_z,
)


def daily_series(start: date, days: int, base: float, drift: float = 0.0) -> pd.Series:
    idx = pd.bdate_range(start, periods=days)
    vals = base * (1 + drift) ** np.arange(days)
    return pd.Series(vals, index=idx)


class TestForwardReturn:
    def test_positive_drift(self):
        prices = daily_series(date(2026, 1, 5), 60, 100.0, drift=0.005)
        r = forward_return(prices, date(2026, 1, 5), 30)
        assert r is not None and r > 0.10

    def test_flat_is_zero(self):
        prices = daily_series(date(2026, 1, 5), 60, 100.0)
        r = forward_return(prices, date(2026, 1, 5), 30)
        assert r == pytest.approx(0.0, abs=1e-9)

    def test_trade_date_after_data_returns_none(self):
        prices = daily_series(date(2026, 1, 5), 10, 100.0)
        assert forward_return(prices, date(2027, 1, 5), 30) is None


class TestAbnormalReturn:
    def test_ticker_beats_spy(self):
        ticker = daily_series(date(2026, 1, 5), 60, 100.0, drift=0.008)
        spy = daily_series(date(2026, 1, 5), 60, 500.0, drift=0.001)
        a = abnormal_return(ticker, spy, date(2026, 1, 5), 30)
        assert a is not None and a > 0.20

    def test_missing_spy_gives_none(self):
        ticker = daily_series(date(2026, 1, 5), 60, 100.0, drift=0.008)
        spy = daily_series(date(2026, 1, 5), 5, 500.0)
        assert abnormal_return(ticker, spy, date(2026, 1, 5), 30) is None


class TestTiming:
    def test_nearest_event_gap(self):
        events = [date(2020, 1, 24), date(2022, 8, 9)]
        assert nearest_event_gap(date(2020, 2, 13), events) == 20
        assert nearest_event_gap(date(2022, 8, 20), events) == 11
        assert nearest_event_gap(date(2030, 1, 1), []) == 10_000

    def test_timing_z_normalizes_by_own_frequency(self):
        events = [date(2026, 1, 26)]
        frequent = PersonStats(
            trades=[
                Trade("A", "X", date(2026, 1, 2)),
                Trade("A", "X", date(2026, 1, 9)),
                Trade("A", "X", date(2026, 1, 16)),
            ]
        )
        infrequent = PersonStats(
            trades=[
                Trade("B", "X", date(2025, 1, 2)),
                Trade("B", "X", date(2025, 1, 3)),
                Trade("B", "X", date(2026, 1, 23)),
            ]
        )
        z_freq = timing_z(date(2026, 2, 2), events, frequent)
        z_rare = timing_z(date(2026, 2, 2), events, infrequent)
        assert z_rare < z_freq
        assert z_freq == pytest.approx(1.0)

    def test_timing_z_single_trader_min_interval(self):
        solo = PersonStats(trades=[Trade("C", "X", date(2026, 1, 20))])
        assert timing_z(date(2026, 1, 26), [date(2026, 1, 26)], solo) == 0.0


class TestRepetition:
    def test_below_min_trades_is_none(self):
        p = PersonStats(trades=[Trade("A", "X", date(2026, 1, i)) for i in range(1, 5)])
        assert repetition_ratio(p, [0.1, 0.2, 0.0, -0.1]) is None

    def test_hit_fraction(self):
        p = PersonStats(trades=[Trade("A", "X", date(2026, 1, i)) for i in range(1, 7)])
        abnormals = [0.10, 0.06, 0.00, -0.02, 0.08, None]
        assert repetition_ratio(p, abnormals) == pytest.approx(3 / 5)


class TestLag:
    def test_within_limit_zero(self):
        assert lag_score(45) == 0.0
        assert lag_score(0) == 0.0

    def test_late_scales_to_one(self):
        assert lag_score(90) == pytest.approx(1.0)
        assert lag_score(67) == pytest.approx((67 - 45) / 45)

    def test_missing_filing_is_none(self):
        assert lag_score(None) is None


class TestComposite:
    def test_none_when_components_missing(self):
        assert composite_score(None, 1.0, 0.5, 0.0) is None
        assert composite_score(0.1, 1.0, None, 0.0) is None

    def test_perfect_components(self):
        score = composite_score(0.50, 0.0, 1.0, 1.0)
        assert score == pytest.approx(0.4 + 0.2 + 0.2 + 0.2)

    def test_negative_abnormal_contributes_zero(self):
        assert composite_score(-0.30, 0.0, 1.0, 0.0) == pytest.approx(0.2 + 0.2)


class TestFilingDeadline:
    def test_quarter_ends_plus_45(self):
        assert filing_deadline(date(2026, 7, 24)) == date(2026, 11, 14)
        assert filing_deadline(date(2026, 1, 2)) == date(2026, 5, 15)


class TestLoaders:
    def test_load_trades_rejects_missing_columns(self, tmp_path):
        p = tmp_path / "bad.csv"
        p.write_text("person,ticker\nA,X\n")
        with pytest.raises(ValueError, match="missing columns"):
            load_trades_csv(p)

    def test_load_trades_parses_dates(self, tmp_path):
        p = tmp_path / "ok.csv"
        p.write_text(
            "person,role,party,chamber,state,ticker,asset_type,transaction_type,"
            "amount_min,amount_max,transaction_date,filing_date,disclosure_lag_days,"
            "owner,linked_via,source_dataset,source_url,provenance\n"
            "Test Person,Senator,R,Senate,NC,XX,stock,sell,1000,15000,"
            "2020-02-13,2020-03-19,,self,,filing,http://x,A\n"
        )
        df = load_trades_csv(p)
        assert df.loc[0, "transaction_date"] == pd.Timestamp("2020-02-13")

    def test_load_events_rejects_missing_columns(self, tmp_path):
        p = tmp_path / "bad.csv"
        p.write_text("date,description\n2020-01-24,x\n")
        with pytest.raises(ValueError, match="missing columns"):
            load_events_csv(p)


class TestOptionTerms:
    def test_parse_call_terms(self):
        from quant.insider import parse_option_terms

        t = parse_option_terms("Call $200 06/21/2024")
        assert t == {"opt_type": "call", "strike": 200.0, "expiry": date(2024, 6, 21)}

    def test_parse_put_terms(self):
        from quant.insider import parse_option_terms

        t = parse_option_terms("Puts 45 strike exp 2025-06-20")
        assert t == {"opt_type": "put", "strike": 45.0, "expiry": date(2025, 6, 20)}

    def test_no_terms_returns_none(self):
        from quant.insider import parse_option_terms

        assert parse_option_terms("some fund name") is None
        assert parse_option_terms("") is None
        assert parse_option_terms("Call without details") is None


class TestOptionDeltaProxy:
    def test_atm_default(self):
        from quant.insider import option_delta_proxy

        assert option_delta_proxy() == 0.5
        assert option_delta_proxy(None, None) == 0.5

    def test_moneyness_monotonic(self):
        from quant.insider import option_delta_proxy

        itm = option_delta_proxy(90.0, 100.0)
        atm = option_delta_proxy(100.0, 100.0)
        otm = option_delta_proxy(110.0, 100.0)
        assert itm > atm > otm
        assert atm == pytest.approx(0.5)
        assert itm == pytest.approx(0.5 + 0.5 * (100 - 90) / 90)

    def test_clipped(self):
        from quant.insider import option_delta_proxy

        assert option_delta_proxy(1.0, 1000.0) == 0.95
        assert option_delta_proxy(1000.0, 1.0) == 0.05


class TestOptionPnlProxy:
    def test_buy_sign(self):
        from quant.insider import option_pnl_proxy

        assert option_pnl_proxy(0.10, "buy") == pytest.approx(0.05)

    def test_sell_sign_flip(self):
        from quant.insider import option_pnl_proxy

        assert option_pnl_proxy(0.10, "sell") == pytest.approx(-0.05)

    def test_none_abnormal(self):
        from quant.insider import option_pnl_proxy

        assert option_pnl_proxy(None, "buy") is None

    def test_disclosed_strike_shifts_delta(self):
        from quant.insider import option_pnl_proxy

        atm = option_pnl_proxy(0.10, "buy")
        itm = option_pnl_proxy(0.10, "buy", strike=90.0, spot=100.0)
        assert itm > atm


class TestScoredOptionsAndQueue:
    def _tiny_inputs(self, persons=("Opt Buyer", "Opt Buyer", "Opt Seller", "Stock Holder"),
                     asset_types=("stock option", "stock option", "stock option", "stock"),
                     txn_types=("buy", "sell", "sell", "buy")):
        import quant.insider_map as im

        idx = pd.bdate_range("2026-01-05", periods=60)
        spy = pd.Series(100.0, index=idx)
        spy.iloc[20:] = 110.0  # +10% SPY drift after day 20
        tick = spy.copy()
        tick.iloc[20:] = 140.0  # ticker +40% -> abnormal_30d > 0 for early trades
        # 6 trades per person so repetition (>=MIN_TRADES) is defined
        rows = []
        for p, atype, txn in zip(persons, asset_types, txn_types):
            for k in range(6):
                rows.append(
                    {
                        "person": p,
                        "ticker": "AAA",
                        "asset_type": atype,
                        "transaction_type": txn,
                        "transaction_date": pd.Timestamp("2026-01-06"),
                        "filing_date": pd.Timestamp("2026-02-06"),
                        "disclosure_lag_days": 31,
                        "amount_min": 1000.0,
                        "amount_max": 5000.0,
                        "owner": "self",
                    }
                )
        trades = pd.DataFrame(rows)
        stats = im.build_person_stats(trades)
        return trades, {"AAA": tick}, spy, stats

    def test_option_columns(self):
        import quant.insider_map as im

        trades, prices, spy, stats = self._tiny_inputs()
        scored = im.score_trades(trades, [], prices, spy, stats)
        assert "assumed_atm" in scored.columns and "option_pnl_proxy_30d" in scored.columns
        opts = scored[scored["asset_type"] == "stock option"]
        assert opts["assumed_atm"].all()
        assert opts["option_delta_proxy"].eq(0.5).all()
        stock = scored[scored["asset_type"] == "stock"].iloc[0]
        assert stock["assumed_atm"] is np.False_ or not stock["assumed_atm"]
        buy_row = opts[opts["transaction_type"] == "buy"].iloc[0]
        sell_row = opts[opts["transaction_type"] == "sell"].iloc[0]
        assert buy_row["option_pnl_proxy_30d"] > 0  # abnormal positive
        assert sell_row["option_pnl_proxy_30d"] < 0  # sign flip
        assert buy_row["option_pnl_proxy_30d"] == pytest.approx(
            -sell_row["option_pnl_proxy_30d"]
        )
        # composite spec unchanged: option and stock rows score on the same formula
        assert scored["composite"].notna().all()

    def test_round3_queue_excludes_verified(self):
        import quant.insider_map as im

        trades, prices, spy, stats = self._tiny_inputs(
            persons=("Nancy Pelosi", "Jane Doe", "Opt Seller", "Stock Holder"),
            asset_types=("stock option",) * 4,
        )
        scored = im.score_trades(trades, [], prices, spy, stats)
        q = im.round3_queue(scored, n=10)
        assert "Pelosi" not in " ".join(q["person"].astype(str))
        assert "Jane Doe" in " ".join(q["person"].astype(str))
        assert set(q["status"]) == {"QUEUED_UNVERIFIED"}
        assert list(q["rank"]) == list(range(1, len(q) + 1))
