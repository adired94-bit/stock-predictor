"""Historical pattern matching (cosine similarity or DTW) on normalized candles+volume."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def window_features(df: pd.DataFrame) -> np.ndarray:
    """Per-bar features: close log-return, body, upper/lower wick (rel. to prev close), log-volume.

    Returns array (n_bars, 5). Normalisation per window happens in ``_norm``.
    """
    c = df["Close"].to_numpy(float)
    prev = np.concatenate([[c[0]], c[:-1]])
    o, h, l = (df[k].to_numpy(float) for k in ("Open", "High", "Low"))
    ret = np.log(c / prev)
    body = (c - o) / prev
    upw = (h - np.maximum(o, c)) / prev
    loww = (np.minimum(o, c) - l) / prev
    lv = np.log(df["Volume"].to_numpy(float) + 1)
    return np.column_stack([ret, body, upw, loww, lv])


def _norm(w: np.ndarray) -> np.ndarray:
    """Z-score each feature column within the window; cumulative return path added as shape."""
    path = np.cumsum(w[:, 0])[:, None]
    x = np.hstack([path, w])
    sd = x.std(axis=0)
    sd[sd == 0] = 1
    return (x - x.mean(axis=0)) / sd


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of flattened arrays (-1..1)."""
    a, b = a.ravel(), b.ravel()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d else 0.0


def dtw_distance(a: np.ndarray, b: np.ndarray, band: int | None = None) -> float:
    """Dynamic Time Warping distance (Euclidean local cost, optional Sakoe-Chiba band), numpy impl."""
    a = a.reshape(len(a), -1); b = b.reshape(len(b), -1)
    n, m = len(a), len(b)
    band = max(n, m) if band is None else max(band, abs(n - m))
    cost = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    D = np.full((n + 1, m + 1), np.inf); D[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(max(1, i - band), min(m, i + band) + 1):
            D[i, j] = cost[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m] / (n + m))


@dataclass
class PatternResult:
    matches: pd.DataFrame  # start, end, similarity, fwd5, fwd10
    current_window: tuple[pd.Timestamp, pd.Timestamp]
    summary: dict = field(default_factory=dict)
    last_close: float = float("nan")


def find_similar(df: pd.DataFrame, window: int = 20, top_k: int = 10, method: str = "cosine",
                 horizons: tuple[int, int] = (5, 10)) -> PatternResult:
    """Compare the last ``window`` bars with every historical window; return top-K non-overlapping matches.

    Similarity: cosine in [-1,1]; DTW distance d is mapped to 1/(1+d). Only windows whose
    longest forward horizon is fully observed (and ending before the current window) are eligible.
    """
    if len(df) < window * 3 + max(horizons):
        raise ValueError("Not enough history for pattern matching.")
    feats = window_features(df)
    close = df["Close"].to_numpy(float)
    n, H = len(df), max(horizons)
    cur = _norm(feats[n - window:])
    # Cheap cosine prefilter for DTW to keep it fast.
    cands = np.arange(1, n - window - H + 1 - window)  # ends before current window starts
    cos = np.array([cosine_sim(cur, _norm(feats[s:s + window])) for s in cands])
    if method == "dtw":
        pool = cands[np.argsort(-cos)[: max(top_k * 15, 150)]]
        sims = np.array([1 / (1 + dtw_distance(cur, _norm(feats[s:s + window]), band=max(2, window // 5))) for s in pool])
    else:
        pool, sims = cands, cos
    order = np.argsort(-sims)
    chosen: list[int] = []
    rows = []
    for idx in order:
        s = int(pool[idx])
        if any(abs(s - c) < window for c in chosen):
            continue
        chosen.append(s)
        e = s + window - 1
        rows.append({"start": df.index[s], "end": df.index[e], "start_i": s, "end_i": e,
                     "similarity": float(sims[idx]),
                     **{f"fwd{h}": close[e + h] / close[e] - 1 for h in horizons}})
        if len(chosen) >= top_k:
            break
    m = pd.DataFrame(rows)
    res = PatternResult(m, (df.index[n - window], df.index[-1]), last_close=float(close[-1]))
    res.summary = summarize(m, res.last_close, horizons)
    return res


def summarize(m: pd.DataFrame, last_close: float, horizons=(5, 10)) -> dict:
    """Mean/median/win-rate per horizon, similarity-weighted mean, and 25-75th pct price targets."""
    out: dict = {}
    if m.empty:
        return out
    w = np.clip(m["similarity"].to_numpy(), 1e-6, None)
    for h in horizons:
        r = m[f"fwd{h}"].to_numpy()
        p25, p75 = np.percentile(r, [25, 75])
        out[h] = {"mean": float(r.mean()), "median": float(np.median(r)),
                  "weighted_mean": float((r * w).sum() / w.sum()), "win_rate": float((r > 0).mean()),
                  "std": float(r.std()), "p25": float(p25), "p75": float(p75),
                  "target_low": last_close * (1 + p25), "target_high": last_close * (1 + p75),
                  "target_mid": last_close * (1 + float(np.median(r)))}
    return out
