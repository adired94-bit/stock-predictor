"""News & analyst sentiment engine."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

try:
    import streamlit as st
    _cache = st.cache_data(ttl=900, show_spinner=False)
except Exception:  # pragma: no cover
    def _cache(f):  # type: ignore
        return f

FINANCE_LEXICON = {
    "beat": 2.0, "beats": 2.0, "upgrade": 2.5, "upgraded": 2.5, "outperform": 2.0, "bullish": 2.5,
    "surge": 2.5, "surges": 2.5, "soar": 2.5, "soars": 2.5, "rally": 2.0, "rallies": 2.0, "record": 1.5,
    "raise": 1.2, "raises": 1.5, "raised": 1.5, "buyback": 1.5, "growth": 1.2, "strong": 1.5, "tops": 1.8,
    "miss": -2.0, "misses": -2.0, "downgrade": -2.5, "downgraded": -2.5, "underperform": -2.0,
    "bearish": -2.5, "plunge": -2.8, "plunges": -2.8, "tumble": -2.5, "tumbles": -2.5, "slump": -2.3,
    "sinks": -2.2, "falls": -1.5, "cut": -1.5, "cuts": -1.5, "lawsuit": -2.0, "probe": -1.8,
    "recall": -1.8, "layoffs": -1.8, "weak": -1.8, "warns": -2.0, "selloff": -2.3, "bankruptcy": -3.0,
}
_analyzer: Optional[SentimentIntensityAnalyzer] = None


def analyzer() -> SentimentIntensityAnalyzer:
    """Singleton VADER analyzer with finance lexicon boost."""
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentIntensityAnalyzer()
        _analyzer.lexicon.update(FINANCE_LEXICON)
    return _analyzer


def score_text(text: str) -> float:
    """VADER compound score in [-1, 1]."""
    return float(analyzer().polarity_scores(text or "")["compound"]) if text else 0.0


def parse_news_item(item: dict) -> dict:
    """Normalise old (flat) and new (``content`` nested) yfinance news formats."""
    c = item.get("content") if isinstance(item.get("content"), dict) else item
    title = c.get("title") or ""
    summary = c.get("summary") or c.get("description") or ""
    prov = c.get("provider") or {}
    publisher = prov.get("displayName") if isinstance(prov, dict) else None
    publisher = publisher or c.get("publisher") or ""
    link = ((c.get("canonicalUrl") or {}).get("url") if isinstance(c.get("canonicalUrl"), dict) else None) \
        or ((c.get("clickThroughUrl") or {}).get("url") if isinstance(c.get("clickThroughUrl"), dict) else None) \
        or c.get("link") or ""
    ts = c.get("pubDate") or c.get("displayTime") or c.get("providerPublishTime")
    try:
        ts = pd.to_datetime(ts, unit="s", utc=True) if isinstance(ts, (int, float)) else pd.to_datetime(ts, utc=True)
    except Exception:
        ts = pd.NaT
    return {"published": ts, "title": title, "summary": summary, "publisher": publisher, "link": link}


def score_news(items: list[dict]) -> tuple[pd.DataFrame, float]:
    """Score each headline (title weighted 0.6, summary 0.4). Returns (table, mean score in [-1,1])."""
    rows = []
    for it in items:
        p = parse_news_item(it)
        if not p["title"]:
            continue
        t, s = score_text(p["title"]), score_text(p["summary"])
        p["score"] = 0.6 * t + 0.4 * s if p["summary"] else t
        rows.append(p)
    df = pd.DataFrame(rows, columns=["published", "title", "summary", "publisher", "link", "score"])
    return df, float(np.clip(df["score"].mean(), -1, 1)) if len(df) else 0.0


@_cache
def fetch_news(ticker: str) -> tuple[list[dict], Optional[str]]:
    """Fetch raw news items via yfinance."""
    try:
        import yfinance as yf
        news = yf.Ticker(ticker).news or []
        return list(news), None if news else "No news returned."
    except Exception as e:
        return [], f"News fetch failed: {e}"


@_cache
def fetch_analyst(ticker: str) -> tuple[dict[str, Any], Optional[str]]:
    """Fetch analyst consensus: recommendation mean/key, targets, analyst count, current price."""
    out: dict[str, Any] = {}
    err = None
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        try:
            info = t.info or {}
        except Exception as e:
            info, err = {}, f"info unavailable: {e}"
        for k_out, k_in in [("recommendation_mean", "recommendationMean"), ("recommendation_key", "recommendationKey"),
                            ("target_mean", "targetMeanPrice"), ("target_high", "targetHighPrice"),
                            ("target_low", "targetLowPrice"), ("n_analysts", "numberOfAnalystOpinions"),
                            ("current_price", "currentPrice"), ("name", "shortName")]:
            out[k_out] = info.get(k_in)
        try:
            apt = t.analyst_price_targets or {}
            out["target_mean"] = out["target_mean"] or apt.get("mean")
            out["target_high"] = out["target_high"] or apt.get("high")
            out["target_low"] = out["target_low"] or apt.get("low")
            out["current_price"] = out["current_price"] or apt.get("current")
        except Exception:
            pass
        try:
            rec = t.recommendations
            if rec is not None and len(rec):
                out["recommendations"] = rec.reset_index(drop=True).head(4).to_dict("records")
        except Exception:
            pass
    except Exception as e:
        err = f"Analyst fetch failed: {e}"
    return out, err


def analyst_score(rec_mean: Optional[float], target_mean: Optional[float], price: Optional[float]) -> float:
    """Analyst score in [-1,1]: 0.6*rating + 0.4*upside.

    rating: rec mean 1 (strong buy) -> +1, 3 -> 0, 5 (sell) -> -1.
    upside: (target/price - 1) / 0.30 clipped to [-1,1] (30% upside saturates).
    Missing parts are dropped and weights renormalised.
    """
    parts, ws = [], []
    if rec_mean:
        parts.append(float(np.clip((3 - rec_mean) / 2, -1, 1))); ws.append(0.6)
    if target_mean and price:
        parts.append(float(np.clip((target_mean / price - 1) / 0.30, -1, 1))); ws.append(0.4)
    return float(np.dot(parts, ws) / sum(ws)) if ws else 0.0


# --------------------------------------------------------------- upgrades
_finbert = None


def finbert_available() -> bool:
    """True when transformers + torch are importable."""
    try:
        import transformers, torch  # noqa: F401
        return True
    except Exception:
        return False


def score_texts_finbert(texts: list[str]) -> Optional[list[float]]:
    """ProsusAI/finbert scores (P(pos)-P(neg)); None if unavailable."""
    global _finbert
    try:
        if _finbert is None:
            from transformers import pipeline
            _finbert = pipeline("text-classification", model="ProsusAI/finbert", top_k=None)
        out = _finbert([t[:512] for t in texts])
        res = []
        for o in out:
            d = {x["label"].lower(): x["score"] for x in o}
            res.append(float(d.get("positive", 0) - d.get("negative", 0)))
        return res
    except Exception:
        return None


def score_news_v2(items: list[dict], use_finbert: bool = False, half_life_h: float = 48.0) -> tuple[pd.DataFrame, float, str]:
    """Score headlines (FinBERT if requested & available, else VADER) and recency-weight them
    (exponential decay, half-life ``half_life_h`` hours). Returns (table, weighted score, engine)."""
    df, _ = score_news(items)
    engine = "VADER+finance lexicon"
    if len(df) and use_finbert:
        fb = score_texts_finbert((df["title"] + ". " + df["summary"].fillna("")).tolist())
        if fb is not None:
            df["score"] = fb; engine = "FinBERT"
    if df.empty:
        return df, 0.0, engine
    now = pd.Timestamp.now(tz="UTC")
    age = (now - pd.to_datetime(df["published"], utc=True)).dt.total_seconds().div(3600).fillna(24 * 7)
    df["weight"] = 0.5 ** (age.clip(lower=0) / half_life_h)
    w = df["weight"].sum()
    return df, float(np.clip((df["score"] * df["weight"]).sum() / w if w else df["score"].mean(), -1, 1)), engine


def ownership_short(info: dict) -> dict:
    """Short interest & institutional ownership fields from yfinance info (if present)."""
    keys = {"short_pct_float": "shortPercentOfFloat", "short_ratio": "shortRatio",
            "inst_pct": "heldPercentInstitutions", "insider_pct": "heldPercentInsiders", "sector": "sector"}
    return {k: info.get(v) for k, v in keys.items()}


@_cache
def fetch_info(ticker: str) -> dict:
    """Raw yfinance info dict ({} on failure)."""
    try:
        import yfinance as yf
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}
