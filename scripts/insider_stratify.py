"""Stratified insider-map analysis: executive-branch vs House vs Senate.

Post-hoc EXPLORATORY stratification of the already-scored map (spec untouched,
nothing re-scored): per-chamber null baselines + per-person leaderboards.
Labeled exploratory in STATUS.md — unlike the pipeline, this split was chosen
after seeing round-1..3 results (exec filers kept topping the queues).

    .venv/bin/python scripts/insider_stratify.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from quant.insider import REP_THRESHOLD  # noqa: E402

MAP_CSV = ROOT / "data/insider/map/insider_map.csv"
NORM_CSV = ROOT / "data/insider/congress_trades_norm.csv"
OUT_CSV = ROOT / "data/insider/map/leaderboard_stratified.csv"


def chamber_person_map(norm: pd.DataFrame) -> dict[str, str]:
    m = norm.dropna(subset=["person"]).groupby("person")["chamber"].agg(
        lambda s: s.mode().iat[0]
    )
    return m.to_dict()


def per_person(scored: pd.DataFrame) -> pd.DataFrame:
    g = scored.groupby("person").agg(
        n_trades=("ticker", "size"),
        n_scored=("abnormal_30d", lambda s: int(s.notna().sum())),
        median_abnormal=("abnormal_30d", "median"),
        max_abnormal=("abnormal_30d", "max"),
        hit_rate=("abnormal_30d",
                  lambda s: float((s.dropna() > REP_THRESHOLD).mean()) if s.notna().any() else float("nan")),
        median_timing_z=("timing_z", "median"),
        median_composite=("composite", "median"),
    )
    return g


def baseline(scored: pd.DataFrame) -> dict:
    s = scored.dropna(subset=["abnormal_30d"])
    if s.empty:
        return {"scored": 0}
    return {
        "scored": len(s),
        "median_abnormal": float(s["abnormal_30d"].median()),
        "mean_abnormal": float(s["abnormal_30d"].mean()),
        "hit_rate": float((s["abnormal_30d"] > REP_THRESHOLD).mean()),
    }


def main() -> None:
    scored = pd.read_csv(MAP_CSV, low_memory=True)
    norm = pd.read_csv(NORM_CSV, low_memory=False, usecols=["person", "chamber"])
    cmap = chamber_person_map(norm)
    scored["chamber"] = scored["person"].map(cmap).fillna("(unknown)")

    frames = []
    print("=== chamber baselines (null check per cohort) ===")
    print(f"{'cohort':10s} {'filers':>7s} {'trades':>7s} {'scored':>7s} {'med_abn':>9s} {'hit_rate':>9s}")
    for ch in ("executive", "house", "senate"):
        sub = scored[scored["chamber"] == ch]
        b = baseline(sub)
        ppl = sub["person"].nunique()
        print(f"{ch:10s} {ppl:>7d} {len(sub):>7d} {b.get('scored',0):>7d} "
              f"{b.get('median_abnormal',float('nan')):>9.4f} {b.get('hit_rate',float('nan')):>9.3f}")
        pp = per_person(sub)
        pp["chamber"] = ch
        frames.append(pp.reset_index())
    allsub = scored
    b = baseline(allsub)
    print(f"{'ALL':10s} {allsub['person'].nunique():>7d} {len(allsub):>7d} "
          f"{b.get('scored',0):>7d} {b.get('median_abnormal',float('nan')):>9.4f} "
          f"{b.get('hit_rate',float('nan')):>9.3f}")

    out = pd.concat(frames, ignore_index=True)
    order = {"executive": 0, "house": 1, "senate": 2}
    out["_o"] = out["chamber"].map(order)
    out = out.sort_values(["_o", "median_composite"], ascending=[True, False]).drop(columns="_o")
    out.to_csv(OUT_CSV, index=False)

    print("\n=== top 10 per cohort (by median composite) ===")
    for ch in ("executive", "house", "senate"):
        sub = out[out["chamber"] == ch].head(10)
        print(f"\n-- {ch} --")
        print(sub[["person", "n_trades", "n_scored", "median_abnormal",
                   "hit_rate", "median_composite"]].to_string(index=False))
    print(f"\nwrote {OUT_CSV} ({len(out)} filers)")


if __name__ == "__main__":
    main()
