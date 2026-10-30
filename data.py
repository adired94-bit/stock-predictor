"""Data engine: OHLCV fetching via yfinance with caching and graceful errors."""
from __future__ import annotations

import datetime as dt
from typing import Optional

import numpy as np
import pandas as pd

try:
    import streamlit as st
    _cache = st.cache_data(ttl=3600, show_spinner=False)
except Exception:  # pragma: no cover
    def _cache(f):  # type: ignore
        return f

REQUIRED = ["Open", "High", "Low", "Close", "Volume"]


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance MultiIndex columns, keep OHLCV, drop NaNs."""
    if df is None or df.empty:
        return pd.DataFrame(columns=REQUIRED)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[[c for c in REQUIRED if c in df.columns]].dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None) if getattr(df.index, "tz", None) else pd.to_datetime(df.index)
    return df.sort_index()


@_cache
def fetch_ohlcv(ticker: str, start: dt.date, end: dt.date, interval: str = "1d") -> tuple[pd.DataFrame, Optional[str]]:
    """Fetch OHLCV for ``ticker`` in [start, end]. Returns (df, error_message)."""
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end + dt.timedelta(days=1), interval=interval,
                         auto_adjust=True, progress=False, threads=False)
        df = _clean(df)
        if df.empty:
            return df, f"No data returned for '{ticker}' (bad ticker, range, or network blocked)."
        return df, None
    except Exception as e:  # network / parsing errors
        return pd.DataFrame(columns=REQUIRED), f"Data fetch failed: {e}"


def synthetic_ohlcv(n: int = 1260, seed: int = 0, start_price: float = 100.0) -> pd.DataFrame:
    """Generate a random-walk OHLCV frame (used for demo mode and tests)."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, 0.02, n)
    close = start_price * np.exp(np.cumsum(rets))
    open_ = close * np.exp(rng.normal(0, 0.005, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.008, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.008, n)))
    vol = rng.lognormal(15, 0.4, n)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx)
