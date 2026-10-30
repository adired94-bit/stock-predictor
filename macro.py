"""Macro engine: VIX level/trend + index trend -> market regime and sentiment-weight adjustment."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from data import fetch_ohlcv

REGIME_SENTIMENT_MULT = {"Calm": 1.0, "Normal": 0.9, "Elevated": 0.7, "Stress": 0.5, "Unknown": 0.8}


@dataclass
class MacroState:
    regime: str
    vix: Optional[float]
    vix_trend: float  # 20d pct change
    index_trends: dict[str, float] = field(default_factory=dict)  # close vs 50d EMA, pct
    score: float = 0.0  # macro signal -1..1
    sentiment_mult: float = 1.0
    notes: list[str] = field(default_factory=list)


def classify_regime(vix: Optional[float], vix_trend: float) -> str:
    """Calm <15, Normal 15-20, Elevated 20-28, Stress >=28; a >25% 20d VIX spike bumps one level."""
    if vix is None or np.isnan(vix):
        return "Unknown"
    levels = ["Calm", "Normal", "Elevated", "Stress"]
    i = 0 if vix < 15 else 1 if vix < 20 else 2 if vix < 28 else 3
    if vix_trend > 0.25:
        i = min(i + 1, 3)
    return levels[i]


def macro_from_frames(vix_df: Optional[pd.DataFrame], idx: dict[str, pd.DataFrame]) -> MacroState:
    """Build MacroState from already-fetched frames (pure; testable offline)."""
    vix = trend = None
    if vix_df is not None and len(vix_df) > 21:
        vix = float(vix_df["Close"].iloc[-1]); trend = float(vix_df["Close"].iloc[-1] / vix_df["Close"].iloc[-21] - 1)
    regime = classify_regime(vix, trend or 0.0)
    trends = {}
    for name, d in idx.items():
        if d is not None and len(d) > 50:
            e = d["Close"].ewm(span=50, adjust=False).mean().iloc[-1]
            trends[name] = float(d["Close"].iloc[-1] / e - 1)
    idx_sig = float(np.clip(np.mean(list(trends.values())) / 0.05, -1, 1)) if trends else 0.0
    vix_sig = {"Calm": 0.4, "Normal": 0.1, "Elevated": -0.3, "Stress": -0.7, "Unknown": 0.0}[regime]
    st = MacroState(regime, vix, trend or 0.0, trends, float(np.clip(0.5 * idx_sig + 0.5 * vix_sig, -1, 1)),
                    REGIME_SENTIMENT_MULT[regime])
    if vix is None:
        st.notes.append("VIX unavailable – regime Unknown.")
    return st


def fetch_macro(end: dt.date, lookback_days: int = 200) -> MacroState:
    """Fetch ^VIX, ^GSPC, ^IXIC and compute macro state; degrades gracefully."""
    start = end - dt.timedelta(days=lookback_days)
    vix, _ = fetch_ohlcv("^VIX", start, end)
    idx = {n: fetch_ohlcv(t, start, end)[0] for n, t in (("S&P 500", "^GSPC"), ("Nasdaq", "^IXIC"))}
    return macro_from_frames(vix if len(vix) else None, idx)
