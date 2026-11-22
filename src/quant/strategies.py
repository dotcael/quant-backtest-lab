"""Signal generators: price frame in, position series out.

Contract every strategy must honor:
  - output is a pandas Series aligned to the input index
  - values are 0.0 (flat) or 1.0 (long) — spot-only, no shorts, no leverage
  - a signal value at bar i is decided using data up to and including bar i's
    close; the backtest engine executes it from bar i+1 (no look-ahead)

Strategies are pure functions — no state, no network, trivially testable.
"""
import pandas as pd


def buy_and_hold(df: pd.DataFrame) -> pd.Series:
    """Always long. The bar every actual strategy has to beat — an active
    strategy that loses to this was a waste of fees."""
    return pd.Series(1.0, index=df.index)


def sma_cross(df: pd.DataFrame, fast: int, slow: int) -> pd.Series:
    """Long while the fast simple moving average is above the slow one.

    The classic first strategy: trend-following, one parameter pair, easy
    to reason about. Warmup bars (either SMA undefined) are flat — never
    trade on incomplete indicators.
    """
    fast_ma = df["close"].rolling(fast).mean()
    slow_ma = df["close"].rolling(slow).mean()
    signal = (fast_ma > slow_ma).astype(float)
    signal[fast_ma.isna() | slow_ma.isna()] = 0.0
    return signal


def _stateful_signal(entry: pd.Series, exit_cond: pd.Series) -> pd.Series:
    """1 from the first True entry until a True exit, then 0 until entry again.

    Breakout and mean-reversion both have memory: 'am I already in?' decides
    what the current bar means. Vectorless on purpose — a plain loop over a
    few thousand bars is instant and obviously correct.
    """
    out = []
    in_pos = False
    for enter, exit_now in zip(entry, exit_cond):
        if not in_pos and bool(enter):
            in_pos = True
        elif in_pos and bool(exit_now):
            in_pos = False
        out.append(1.0 if in_pos else 0.0)
    return pd.Series(out, index=entry.index)


def rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Relative Strength Index: ~0 = relentless selling, ~100 = relentless
    buying, 50 = nothing happening. Wilder smoothing via ewm(alpha=1/period).

    Flat history (0 gain, 0 loss) is defined as neutral 50 — never a signal.
    """
    delta = df["close"].diff().fillna(0.0)
    avg_gain = delta.clip(lower=0.0).ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = (-delta.clip(upper=0.0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return (100 - 100 / (1 + rs)).fillna(50.0)


def breakout(df: pd.DataFrame, entry_bars: int = 20, exit_bars: int = 10) -> pd.Series:
    """Donchian channel: long when close breaks ABOVE the highest high of
    the prior `entry_bars` bars; out when close falls BELOW the lowest low
    of the prior `exit_bars` bars. Buys strength — 'new highs mean buyers
    are in charge'.

    Defaults 20/10 are the textbook turtle numbers, used UNtuned — no
    parameter selection on our data means no selection bias to account for.
    """
    prior_high = df["high"].rolling(entry_bars).max().shift(1)
    prior_low = df["low"].rolling(exit_bars).min().shift(1)
    return _stateful_signal(df["close"] > prior_high, df["close"] < prior_low)


def mean_reversion(df: pd.DataFrame, rsi_period: int = 14, entry_level: float = 30.0, exit_level: float = 50.0) -> pd.Series:
    """Buy oversold (RSI below entry_level), exit on recovery (RSI above
    exit_level). Buys weakness — the opposite personality to breakout:
    where breakout assumes momentum continues, this assumes it snaps back.

    Textbook 14/30/50, UNtuned on purpose.
    """
    momentum = rsi(df, rsi_period)
    return _stateful_signal(momentum < entry_level, momentum > exit_level)


def crash_shield(df: pd.DataFrame, shield: int = 200) -> pd.Series:
    """The airbag: hold the asset while its price is above the long trend
    line (shield-day SMA), step out while below it.

    Why this exists: every family we tested only ever 'won' by dodging the
    two bear markets — so make that the explicit design instead of a side
    effect. 200 days is the famous institutional rule of thumb, UNtuned.

    Personality: misses the first ~x% of every crash (that is the fee for
    the airbag) and lags re-entries; in exchange, avoids most drawdowns.
    """
    ma = df["close"].rolling(shield).mean()
    signal = (df["close"] > ma).astype(float)
    signal[ma.isna()] = 0.0
    return signal
