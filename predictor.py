"""Ensemble scoring.

Each component produces a signal s_i in [-1, 1]:
  * pattern:  similarity-weighted mean 5-day forward return of matches / (2 * match std),
              clipped, times (2*win_rate-1 sign agreement blend).
  * news:     mean VADER score of headlines.
  * analyst:  see sentiment.analyst_score.
  * technical: EMA alignment (+-0.5), RSI tilt (+-0.3), RVOL amplifies trend (+-0.2).
Composite S = sum(w_i * s_i) / sum(w_i) in [-1, 1].
Direction: S > +0.15 Bullish, S < -0.15 Bearish, else Neutral.
Confidence = 50 + 50*|S| * agreement, where agreement = share of weight whose sign matches S.
This is a HEURISTIC score, not a calibrated probability.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

THRESHOLD = 0.15


def pattern_signal(summary: dict, horizon: int = 5) -> float:
    """Pattern signal in [-1,1] from match outcome stats."""
    if horizon not in summary:
        return 0.0
    s = summary[horizon]
    z = s["weighted_mean"] / (s["std"] + 1e-9)
    return float(np.clip(0.6 * np.tanh(z) + 0.4 * (2 * s["win_rate"] - 1), -1, 1))


def technical_signal(df: pd.DataFrame) -> float:
    """Trend score in [-1,1] from last row of indicator frame (needs EMA9/21/200, RSI14, RVOL20, Close)."""
    r = df.iloc[-1]
    c, e9, e21, e200 = r["Close"], r["EMA9"], r["EMA21"], r["EMA200"]
    align = (np.sign(c - e9) + np.sign(e9 - e21) + np.sign(e21 - e200)) / 3  # -1..1
    rsi = r["RSI14"] if pd.notna(r["RSI14"]) else 50
    rsi_t = float(np.clip((rsi - 50) / 20, -1, 1))
    if rsi > 75 or rsi < 25:  # stretched: fade a bit
        rsi_t *= -0.5
    rv = r["RVOL20"] if pd.notna(r["RVOL20"]) else 1
    rv_t = float(np.clip(rv - 1, 0, 1)) * np.sign(c - df["Close"].iloc[-2])
    return float(np.clip(0.5 * align + 0.3 * rsi_t + 0.2 * rv_t, -1, 1))


@dataclass
class Prediction:
    direction: str
    confidence: float
    composite: float
    components: dict[str, float]
    contributions: dict[str, float]
    targets: dict


def ensemble(signals: dict[str, float], weights: dict[str, float], targets: dict | None = None) -> Prediction:
    """Combine component signals with weights into a direction + heuristic confidence."""
    tw = sum(max(weights.get(k, 0), 0) for k in signals) or 1.0
    contrib = {k: max(weights.get(k, 0), 0) * float(np.clip(v, -1, 1)) / tw for k, v in signals.items()}
    S = float(sum(contrib.values()))
    if S > THRESHOLD:
        d = "Bullish"
    elif S < -THRESHOLD:
        d = "Bearish"
    else:
        d = "Neutral"
    agree_w = sum(max(weights.get(k, 0), 0) for k, v in signals.items() if np.sign(v) == np.sign(S) and v != 0)
    agreement = agree_w / tw if S != 0 else 0.0
    conf = 50 + 50 * min(abs(S) / 0.6, 1) * agreement if d != "Neutral" else 50 - 50 * min(abs(S) / THRESHOLD, 1) * 0.3
    return Prediction(d, round(float(conf), 1), S, dict(signals), contrib, targets or {})


# ---------------------------------------------------------------------------
# Manager agent (fixed weights)
# ---------------------------------------------------------------------------
MANAGER_WEIGHTS = {"Technical Pattern Similarity": 0.5, "Sentiment & Macro": 0.3, "Risk/Volatility": 0.2}


def sentiment_macro_signal(news: float, analyst: float, macro: float, sentiment_mult: float) -> float:
    """Blend news (35%), analyst (35%), macro (30%); news+analyst scaled by regime multiplier."""
    return float(np.clip(sentiment_mult * (0.35 * news + 0.35 * analyst) + 0.30 * macro, -1, 1))


def manager(pattern: float, technical: float, sent_macro: float, risk: float, pattern_summary: dict,
            mc_cone: pd.DataFrame, last_close: float, regime: str = "Unknown") -> tuple[Prediction, str]:
    """Manager agent: Technical Pattern Similarity (pattern 70% + indicator trend 30%) 50%,
    Sentiment & Macro 30%, Risk/Volatility 20%. Returns (Prediction, rationale)."""
    tech = float(np.clip(0.7 * pattern + 0.3 * technical, -1, 1))
    sig = {"Technical Pattern Similarity": tech, "Sentiment & Macro": sent_macro, "Risk/Volatility": risk}
    targets: dict = {}
    for h in (5, 10):
        if h in pattern_summary:
            s = pattern_summary[h]; targets[f"{h}d"] = (s["target_low"], s["target_mid"], s["target_high"])
    row = mc_cone[mc_cone.horizon == 30] if len(mc_cone) else mc_cone
    if len(row):
        r = row.iloc[0]; targets["30d"] = (float(r.p5), float(r.p50), float(r.p95))
    p = ensemble(sig, MANAGER_WEIGHTS, targets)
    lines = [f"**{p.direction}** with heuristic confidence {p.confidence:.0f}% (composite {p.composite:+.2f}, regime {regime})."]
    for k, v in sig.items():
        lines.append(f"- {k} ({MANAGER_WEIGHTS[k]:.0%}): {v:+.2f} → contributes {p.contributions[k]:+.3f}")
    if 5 in pattern_summary:
        s = pattern_summary[5]
        lines.append(f"- Similar setups: 5d median {s['median']:+.2%}, win-rate {s['win_rate']:.0%}.")
    strongest = max(p.contributions, key=lambda k: abs(p.contributions[k]))
    lines.append(f"- Main driver: {strongest}. Conflicting components lower confidence.")
    return p, "\n".join(lines)
