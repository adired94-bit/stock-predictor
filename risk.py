"""Risk engine: Monte Carlo cones, VaR/CVaR, ATR stops snapped to S/R zones."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder ATR."""
    pc = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def monte_carlo(close: pd.Series, horizon: int = 30, n_paths: int = 5000, method: str = "bootstrap",
                lookback: int = 252, seed: int | None = 42) -> np.ndarray:
    """Simulate price paths (n_paths, horizon+1) from last close.

    bootstrap: resample last ``lookback`` daily log-returns with replacement.
    gbm: normal log-returns with historical mu/sigma.
    """
    r = np.log(close / close.shift(1)).dropna().to_numpy()[-lookback:]
    rng = np.random.default_rng(seed)
    if method == "gbm":
        sims = rng.normal(r.mean(), r.std(), (n_paths, horizon))
    else:
        sims = rng.choice(r, size=(n_paths, horizon), replace=True)
    paths = float(close.iloc[-1]) * np.exp(np.cumsum(sims, axis=1))
    return np.hstack([np.full((n_paths, 1), float(close.iloc[-1])), paths])


def cone(paths: np.ndarray, horizons=(5, 10, 30)) -> pd.DataFrame:
    """5th/50th/95th percentile prices at each horizon."""
    rows = [{"horizon": h, "p5": np.percentile(paths[:, h], 5), "p50": np.percentile(paths[:, h], 50),
             "p95": np.percentile(paths[:, h], 95), "prob_up": float((paths[:, h] > paths[0, 0]).mean())}
            for h in horizons if h < paths.shape[1]]
    return pd.DataFrame(rows)


def var_cvar(paths: np.ndarray, horizon: int = 1, level: float = 0.95) -> tuple[float, float]:
    """Historical-sim VaR and Expected Shortfall as positive loss fractions at ``horizon`` days."""
    ret = paths[:, horizon] / paths[:, 0] - 1
    q = np.percentile(ret, (1 - level) * 100)
    return float(-q), float(-ret[ret <= q].mean())


def sr_levels(df: pd.DataFrame, order: int = 5, lookback: int = 250) -> tuple[list[float], list[float]]:
    """Swing-low supports and swing-high resistances (local extrema over ±order bars)."""
    d = df.tail(lookback)
    h, l = d["High"].to_numpy(), d["Low"].to_numpy()
    res = [h[i] for i in range(order, len(d) - order) if h[i] == h[i - order:i + order + 1].max()]
    sup = [l[i] for i in range(order, len(d) - order) if l[i] == l[i - order:i + order + 1].min()]
    return sorted(set(sup)), sorted(set(res))


@dataclass
class TradePlan:
    side: str
    entry: float
    stop: float
    target: float
    atr: float
    rr: float
    stop_snapped: bool
    target_snapped: bool


def trade_plan(df: pd.DataFrame, side: str = "long", stop_mult: float = 1.5, tp_mult: float = 3.0,
               extra_levels: list[float] | None = None) -> TradePlan:
    """ATR stops/targets; snap to nearest S/R within 0.5 ATR (stop placed just beyond the level)."""
    a = float(atr(df).iloc[-1]); px = float(df["Close"].iloc[-1])
    sup, res = sr_levels(df)
    levels = sorted(set(sup + res + (extra_levels or [])))
    sgn = 1 if side == "long" else -1
    stop, tgt = px - sgn * stop_mult * a, px + sgn * tp_mult * a

    def snap(x: float, beyond: float) -> tuple[float, bool]:
        near = [lv for lv in levels if abs(lv - x) <= 0.5 * a]
        if not near:
            return x, False
        lv = min(near, key=lambda v: abs(v - x))
        return lv + beyond, True

    stop, ss = snap(stop, -sgn * 0.1 * a)  # beyond support (long) / resistance (short)
    tgt, ts = snap(tgt, -sgn * 0.05 * a)  # just in front of level
    risk = abs(px - stop); rew = abs(tgt - px)
    return TradePlan(side, px, stop, tgt, a, rew / risk if risk else float("nan"), ss, ts)


def risk_signal(paths: np.ndarray, var95: float, rr: float, horizon: int = 10) -> float:
    """Risk/vol sub-score -1..1: MC skew (prob_up & median drift) minus tail-risk penalty, plus R:R bonus."""
    ret = paths[:, horizon] / paths[:, 0] - 1
    drift = np.tanh(np.median(ret) / (ret.std() + 1e-9) * 3)
    pu = 2 * (ret > 0).mean() - 1
    tail = -np.clip((var95 - 0.03) / 0.07, 0, 1)  # 1d VaR above 3% penalised
    rrb = np.clip((rr - 1.5) / 1.5, -1, 1) if np.isfinite(rr) else 0
    return float(np.clip(0.35 * drift + 0.35 * pu + 0.2 * tail + 0.1 * rrb, -1, 1))
