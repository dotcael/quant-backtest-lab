# Grok weekly brief — congressional trading monitor

Week: <WEEK_START> → <WEEK_END> (fill in; run every Monday 09:00 local)

## Task

Find congressional / executive-branch trades DISCLOSED during this window
(new PTRs on clerk.house.gov, new Senate eFD periodic filings, new OGE 278-Ts)
that are NOT already in our dataset. Prioritize: (1) trades dated within 7 days
before or after a major policy event (see data/insider/events.csv for the
calendar: FOMC decisions, tariff actions, legislation); (2) large amounts
(≥ $100k); (3) options; (4) filers with repeat hits in
data/insider/map/leaderboard.csv.

## Output schema (EXACT — one row per trade, append-ready to data/insider/grok_seed.csv)

person,role,party,chamber,state,ticker,asset_type,transaction_type,amount_min,amount_max,transaction_date,filing_date,disclosure_lag_days,owner,linked_via,source_dataset,source_url,provenance,tier,convergence,status,notes

- asset_type: stock | stock option | fund | other (kadoa codes decoded: ST=stock, OP=stock option)
- transaction_type: buy | sell | exchange
- owner: self | spouse | joint | child | unknown
- source_url: the ORIGINAL filing PDF (doc_url) — mandatory per row
- provenance: how you found it, e.g. `grok_weekly; clerk_ptr; fetched=<date>`
- tier: UNVERIFIED_GROK unless you retrieved the primary PDF yourself → then
  note "PRIMARY_RETRIEVED" in notes (a human confirms before tier upgrades)
- status: `new` (always, for weekly output; pipeline dedups vs norm CSV)
- notes: one line, factual (shares, strike/expiry if options, context event)

## Hard rules

1. Tiers are document labels, never accusations. Describe filings, not motives.
2. Every row needs a source_url that resolves to the filing (or the eFD record).
3. If you cannot confirm a trade in a primary or credible secondary source,
   drop the row — noise costs more than gaps.
4. Output the CSV rows in a single fenced code block, nothing else in it.
