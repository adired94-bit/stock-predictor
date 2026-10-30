"""Walk-forward backtest (no look-ahead) + confidence calibration.

At each test date t, only df.iloc[:t+1] is used: pattern matching (cosine), indicator trend and
Monte-Carlo risk signal are recomputed, combined with the manager weights (news/analyst/macro are not
available historically, so the Sentiment & Macro component is 0 in the backtest – documented limitation).
The prediction is compared with the realised 5/10-day forward return.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from indicators import add_indicators
from patterns import find_similar
from predictor import ensemble, pattern_signal, technical_signal, MANAGER_WEIGHTS
from risk import monte_carlo, var_cvar, risk_signal

BUCKETS = [0, 55, 60, 65, 70, 80, 101]


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    stats: dict = field(default_factory=dict)
    calibration: pd.DataFrame = field(default_factory=pd.DataFrame)
    component_hit: dict = field(default_factory=dict)


def walk_forward(df: pd.DataFrame, window: int = 20, top_k: int = 10, step: int = 5,
                 n_tests: int = 120, horizon: int = 5, min_history: int = 300) -> BacktestResult:
    """Run the pipeline on past dates using only data up to each date."""
    ind_all = add_indicators(df)  # EMAs/RSI are causal (use only past bars) so slicing is look-ahead free
    close = df["Close"].to_numpy(float)
    last_t = len(df) - 1 - horizon
    min_history = min(min_history, max(window * 3 + 20, len(df) // 2))  # adapt for short (1y) histories
    ts = list(range(last_t, min_history, -step))[:n_tests][::-1]
    rows = []
    for t in ts:
        hist = df.iloc[: t + 1]
        try:
            pr = find_similar(hist, window, top_k, "cosine")
        except ValueError:
            continue
        ps = pattern_signal(pr.summary)
        tech = technical_signal(ind_all.iloc[: t + 1])
        paths = monte_carlo(hist["Close"], horizon=10, n_paths=1000, seed=t)
        v, _ = var_cvar(paths)
        rs = risk_signal(paths, v, 2.0)
        sig = {"Technical Pattern Similarity": float(np.clip(0.7 * ps + 0.3 * tech, -1, 1)),
               "Sentiment & Macro": 0.0, "Risk/Volatility": rs}
        p = ensemble(sig, MANAGER_WEIGHTS)
        fwd = close[t + horizon] / close[t] - 1
        rows.append({"date": df.index[t], "direction": p.direction, "confidence": p.confidence,
                     "composite": p.composite, "pattern": ps, "technical": tech, "risk": rs, "fwd": fwd})
    tr = pd.DataFrame(rows)
    return _evaluate(tr, horizon)


def _evaluate(tr: pd.DataFrame, horizon: int) -> BacktestResult:
    res = BacktestResult(tr)
    if tr.empty:
        return res
    pos = tr["direction"].map({"Bullish": 1, "Bearish": -1, "Neutral": 0}).fillna(0)
    tr["signal_ret"] = pos * tr["fwd"]
    act = tr[pos != 0]
    tr["hit"] = np.where(pos != 0, np.sign(tr["fwd"]) == pos, np.nan)
    # equity: non-overlapping because step >= horizon is recommended
    tr["equity"] = (1 + tr["signal_ret"]).cumprod()
    tr["buy_hold"] = (1 + tr["fwd"]).cumprod()
    res.stats = {"n_signals": int(len(act)), "n_dates": int(len(tr)), "horizon": horizon,
                 "hit_rate": float((np.sign(act["fwd"]) == pos[pos != 0]).mean()) if len(act) else float("nan"),
                 "avg_ret_bull": float(tr.loc[pos == 1, "fwd"].mean()) if (pos == 1).any() else float("nan"),
                 "avg_ret_bear": float(tr.loc[pos == -1, "fwd"].mean()) if (pos == -1).any() else float("nan"),
                 "strategy_total": float(tr["equity"].iloc[-1] - 1), "buy_hold_total": float(tr["buy_hold"].iloc[-1] - 1)}
    res.calibration = calibration_table(tr)
    for c in ("pattern", "technical", "risk"):
        s = np.sign(tr[c]); m = s != 0
        res.component_hit[c] = float((s[m] == np.sign(tr.loc[m, "fwd"])).mean()) if m.any() else 0.5
    return res


def calibration_table(tr: pd.DataFrame) -> pd.DataFrame:
    """Confidence bucket vs realised hit rate (directional signals only), with Laplace smoothing."""
    a = tr[tr["direction"] != "Neutral"].copy()
    if a.empty:
        return pd.DataFrame(columns=["bucket", "lo", "hi", "n", "hit_rate", "calibrated"])
    a["bucket"] = pd.cut(a["confidence"], BUCKETS, right=False)
    g = a.groupby("bucket", observed=False)["hit"].agg(["count", "sum"]).reset_index()
    g["lo"] = [b.left for b in g["bucket"]]; g["hi"] = [b.right for b in g["bucket"]]
    g["n"] = g["count"].astype(int)
    g["hit_rate"] = np.where(g["n"] > 0, g["sum"] / g["n"].replace(0, np.nan), np.nan)
    g["calibrated"] = (g["sum"] + 1) / (g["n"] + 2) * 100  # Laplace -> 50% when no data
    # isotonic-like: enforce monotone non-decreasing via cumulative max
    g["calibrated"] = np.maximum.accumulate(g["calibrated"].to_numpy())
    g["bucket"] = g["bucket"].astype(str)
    return g[["bucket", "lo", "hi", "n", "hit_rate", "calibrated"]]


def calibrate(conf: float, table: pd.DataFrame, min_n: int = 5) -> tuple[float, int]:
    """Map a raw heuristic confidence to the bucket's smoothed realised hit rate. Returns (value, n)."""
    if table is None or table.empty:
        return conf, 0
    row = table[(table.lo <= conf) & (conf < table.hi)]
    if row.empty or int(row.n.iloc[0]) < min_n:
        return conf, int(row.n.iloc[0]) if len(row) else 0
    return float(row.calibrated.iloc[0]), int(row.n.iloc[0])


def adjusted_weights(component_hit: dict, base: float = 0.5) -> dict[str, float]:
    """Multiplier per component: hit<=50% -> 0.25, scales to 1.0 at hit>=60%."""
    return {k: float(np.clip(0.25 + (h - base) / 0.10 * 0.75, 0.25, 1.0)) for k, h in component_hit.items()}
