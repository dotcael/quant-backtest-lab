"""Backtest engine: signals in, honest statistics out.

Ground rules from the README, made mechanical:
  1. Costs ALWAYS come from fees.FeeModel — there is no code path that
     runs a backtest without them.
  2. Results that look too good get flagged as possible overfits, never
     celebrated.

Execution model (deliberately simple, stated openly):
  - signal at bar i is decided at bar i's close, executed at bar i+1 —
    the position earns bar i+1's close-to-close return (no look-ahead)
  - a position change of any size pays (taker + slippage) as a fraction
    of position value, applied to that bar's return
  - no shorts, no leverage, one asset at a time — spot side-project only
"""
from dataclasses import dataclass

import pandas as pd

from quant import fees
from quant.timeframes import timeframe_ms

MS_PER_YEAR = 365.25 * 86_400_000  # crypto trades 24/7 — no calendar gaps


@dataclass
class BacktestResult:
    total_return: float
    buy_hold_return: float
    cagr: float
    max_drawdown: float
    n_trades: int
    cost_drag: float
    exposure: float
    flagged: bool
    flag_reason: str

    def summary(self) -> str:
        flag = f"  [FLAG: {self.flag_reason}]" if self.flagged else ""
        return (
            f"return={self.total_return:+.1%} (buy&hold {self.buy_hold_return:+.1%}) "
            f"cagr={self.cagr:+.1%} maxdd={self.max_drawdown:.1%} "
            f"trades={self.n_trades} cost_drag={self.cost_drag:.2%} "
            f"exposure={self.exposure:.0%}{flag}"
        )


def _overfit_flag(total_return: float, buy_hold_return: float) -> tuple[bool, str]:
    """Heuristic tripwires, not proof:
      - +300% overall AND 3x buy&hold -> suspicious edge vs the market
      - +10,000% overall -> absurd in absolute terms whatever the benchmark
      Either way the result demands an out-of-sample check before being
      believed."""
    if total_return > 3.0 and total_return > 3 * max(buy_hold_return, 0.0):
        return True, "suspiciously good vs buy&hold — possible overfit, verify out-of-sample"
    if total_return > 100.0:
        return True, "absurd absolute return — possible overfit, verify out-of-sample"
    return False, ""


def run(
    df: pd.DataFrame,
    signal: pd.Series,
    exchange_id: str,
    timeframe: str = "1d",
    fee_model: fees.FeeModel | None = None,
) -> BacktestResult:
    """Execute a signal series against OHLCV data with mandatory costs.

    fee_model overrides the exchange default — used for stress reruns
    (same trades, doubled costs). There is still no fee-free path: the
    override must itself be a FeeModel; pass zeros deliberately and own it.
    """
    if len(df) < 2:
        raise ValueError("need at least 2 bars to backtest")

    fee_model = fee_model or fees.FeeModel.for_exchange(exchange_id)
    side_cost = fee_model.taker + fee_model.slippage

    ret = df["close"].pct_change().fillna(0.0)
    pos = signal.shift(1).fillna(0.0).to_numpy()  # act on next bar — no look-ahead
    trade = pd.Series(pos, index=df.index).diff().abs().fillna(pos[0]).to_numpy()

    bar = pos * ret.to_numpy() - trade * side_cost
    wealth = float((1 + pd.Series(bar)).prod())
    total_return = wealth - 1

    cost_drag = 1 - float((1 - pd.Series(trade * side_cost)).prod())

    buy_hold_return = float(df["close"].iloc[-1] / df["close"].iloc[0] - 1)

    bars_per_year = MS_PER_YEAR / timeframe_ms(timeframe)
    years = len(df) / bars_per_year
    cagr = wealth ** (1 / years) - 1 if wealth > 0 else -1.0

    path = pd.Series(bar).add(1).cumprod()
    max_drawdown = float((path / path.cummax() - 1).min())

    flagged, flag_reason = _overfit_flag(total_return, buy_hold_return)

    return BacktestResult(
        total_return=total_return,
        buy_hold_return=buy_hold_return,
        cagr=cagr,
        max_drawdown=max_drawdown,
        n_trades=int((trade > 0).sum()),
        cost_drag=cost_drag,
        exposure=float(pos.mean()),
        flagged=flagged,
        flag_reason=flag_reason,
    )


def main() -> None:
    """CLI: run buy&hold and SMA-cross on every stored series, print comparison."""
    from quant import config, store
    from quant.strategies import buy_and_hold, sma_cross

    for symbol in config.SYMBOLS:
        for timeframe in config.TIMEFRAMES:
            path = store.parquet_path(config.EXCHANGES[0], symbol, timeframe)
            if not path.exists():
                print(f"[backtest] no data for {symbol} {timeframe} ({path}), skipping")
                continue
            df = pd.read_parquet(path)
            fee_model = fees.FeeModel.for_exchange(config.EXCHANGES[0])
            print(f"\n{symbol} {timeframe} — {len(df)} bars, "
                  f"round-trip cost {fee_model.round_trip_cost():.2%}")
            print(f"  buy&hold : {run(df, buy_and_hold(df), config.EXCHANGES[0], timeframe).summary()}")
            print(f"  sma10/50 : {run(df, sma_cross(df, 10, 50), config.EXCHANGES[0], timeframe).summary()}")


if __name__ == "__main__":
    main()
