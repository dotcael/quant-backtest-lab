from datetime import date

import numpy as np
import pandas as pd
import pytest

from quant.insider import PersonStats
from quant.watch import (
    alert_rows,
    diff_new,
    normalize_kadoa,
    p99_threshold,
    repetition_from_map,
    score_new,
    trade_key,
)

KADOA_SAMPLE = {
    "id": "house_20035499_g0",
    "filer_name": "Pete Sessions",
    "ticker": "AAPL",
    "asset_type": "ST",
    "transaction_type": "Sale (Full)",
    "amount_range_low": 100001,
    "amount_range_high": 250000,
    "transaction_date": "2026-09-22",
    "filing_date": "2026-09-22",
    "owner": "SP",
    "doc_url": "https://example.gov/filing.pdf",
}


class TestKeys:
    def test_key_stable_and_normalized(self):
        a = trade_key("Nancy Pelosi", "be", "2026-07-24", 1000001, 5000000)
        b = trade_key("  nancy pelosi ", "BE", "2026-07-24", 1000001.0, 5000000.0)
        assert a == b

    def test_normalize_kadoa(self):
        rows = normalize_kadoa([KADOA_SAMPLE])
        r = rows.iloc[0]
        assert r["person"] == "Pete Sessions"
        assert r["ticker"] == "AAPL"
        assert r["transaction_type"] == "sale (full)"
        assert r["amount_min"] == 100001.0
        assert r["source_url"].startswith("https://")
        assert r["key"] == trade_key("Pete Sessions", "AAPL", "2026-09-22", 100001, 250000)


class TestDiff:
    def test_diff_new_skips_seen_and_empty(self):
        rows = normalize_kadoa([KADOA_SAMPLE])
        fresh = rows.copy()
        fresh["person"] = "Someone New"
        fresh["key"] = trade_key("Someone New", "AAPL", "2026-09-22", 1, 2)
        blank = rows.copy()
        blank["key"] = ""
        out = diff_new(pd.concat([rows, fresh, blank]), {rows.iloc[0]["key"]})
        assert len(out) == 1
        assert out.iloc[0]["person"] == "Someone New"

    def test_diff_empty_input(self):
        assert diff_new(pd.DataFrame(), set()).empty


class TestThresholdAndScore:
    def _map(self):
        return pd.DataFrame({
            "person": ["A"] * 140 + ["B"] * 40,
            "composite": [0.1] * 137 + [0.9] * 3 + [np.nan] * 40,
            "repetition": [0.5] * 180,
        })

    def test_p99(self):
        # 3/140 outliers = 2.1% tail -> P99 sits in the 0.9 block
        assert p99_threshold(self._map()) == pytest.approx(0.9)

    def test_p99_none_when_sparse(self):
        m = pd.DataFrame({"composite": [0.1, 0.2]})
        assert p99_threshold(m) is None

    def test_repetition_from_map(self):
        assert repetition_from_map(self._map(), "A") == 0.5
        assert repetition_from_map(self._map(), "Zz") is None

    def test_score_new_flags_high_abnormal(self):
        idx = pd.bdate_range("2026-01-05", periods=40)
        spy = pd.Series(100.0, index=idx)
        tick = spy.copy()
        tick.iloc[10:] = 200.0  # +100% vs SPY after day 10
        rows = normalize_kadoa([{**KADOA_SAMPLE, "transaction_date": "2026-01-06",
                                 "filing_date": "2026-02-06"}])
        stats = {"Pete Sessions": PersonStats()}
        map_csv = pd.DataFrame({"person": ["Pete Sessions"],
                                "repetition": [0.5], "composite": [0.5]})
        scored = score_new(rows, [date(2026, 1, 6)], stats, map_csv,
                           {"AAPL": tick}, spy)
        r = scored.iloc[0]
        assert r["abnormal_30d"] is not None and r["abnormal_30d"] > 0.5
        assert r["composite"] is not None and r["composite"] > 0.5
        alerts = alert_rows(scored, threshold=0.5)
        assert len(alerts) == 1

    def test_score_new_none_without_price(self):
        rows = normalize_kadoa([KADOA_SAMPLE])
        scored = score_new(rows, [], {}, pd.DataFrame(columns=["person", "repetition"]),
                           {}, pd.Series(dtype=float))
        assert scored.iloc[0]["composite"] is None
