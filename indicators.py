"""Technical indicators on daily OHLCV data."""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    return s.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI (0-100)."""
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0).where(dn.notna())


def rolling_vwap(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Rolling N-day VWAP on daily bars using typical price (H+L+C)/3.

    Intraday-anchored VWAP is not possible on daily candles, so this is the
    volume-weighted average typical price over the last ``window`` sessions.
    """
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    return (tp * df["Volume"]).rolling(window).sum() / df["Volume"].rolling(window).sum()


def rvol(volume: pd.Series, window: int = 20) -> pd.Series:
    """Relative volume: today's volume / mean of the previous ``window`` days."""
    return volume / volume.rolling(window).mean().shift(1)


def volume_profile(df: pd.DataFrame, bins: int = 40) -> tuple[pd.DataFrame, float]:
    """Price-bucket volume histogram (typical price). Returns (profile, point_of_control)."""
    tp = ((df["High"] + df["Low"] + df["Close"]) / 3).to_numpy()
    hist, edges = np.histogram(tp, bins=bins, weights=df["Volume"].to_numpy())
    mids = (edges[:-1] + edges[1:]) / 2
    prof = pd.DataFrame({"price": mids, "volume": hist})
    return prof, float(mids[int(np.argmax(hist))])


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with EMA9/21/200, RSI14, VWAP20, RVOL20 columns."""
    out = df.copy()
    for n in (9, 21, 200):
        out[f"EMA{n}"] = ema(out["Close"], n)
    out["RSI14"] = rsi(out["Close"])
    out["VWAP20"] = rolling_vwap(out)
    out["RVOL20"] = rvol(out["Volume"])
    return out
