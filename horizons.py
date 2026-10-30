"""Multi-horizon forecast (1h, 4h, 1d, 1w, 1m, 1q) using a state-conditioned analog method:
the current state (trend vs EMA, RSI side) is matched to all past bars with the same state, and their
forward h-bar returns give probability up, percentile range and a walk-forward hit rate (no look-ahead)."""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from indicators import rsi

HORIZONS = [  # label, frame, bars ahead, reliability
    ("שעתי", "1h", 1, "נמוכה"), ("4 שעות", "4h", 1, "נמוכה"), ("יומי", "1d", 1, "בינונית-נמוכה"),
    ("שבועי", "1d", 5, "בינונית"), ("חודשי", "1d", 21, "בינונית"), ("רבעוני", "1d", 63, "בינונית-נמוכה (טווח ארוך)")]


def states(close: pd.Series) -> pd.Series:
    e = close.ewm(span=20, adjust=False).mean()
    e50 = close.ewm(span=50, adjust=False).mean()
    r = rsi(close)
    return (close > e).astype(int) * 4 + (e > e50).astype(int) * 2 + (r > 50).astype(int)


def analog_forecast(close: pd.Series, h: int, min_n: int = 20) -> dict:
    close = close.dropna().astype(float)
    s = states(close)
    fwd = np.log(close.shift(-h) / close)
    cur = s.iloc[-1]
    same = fwd[(s == cur)].dropna()
    pool = same if len(same) >= min_n else fwd.dropna()
    px = float(close.iloc[-1])
    if len(pool) < 5:
        return {}
    p_up = float((pool > 0).mean())
    q = np.percentile(pool, [5, 50, 95])
    # walk-forward: at each test bar use only returns fully known before it
    hits = []
    idx = np.arange(len(close))
    tests = idx[max(60, len(close) - 300 * h): len(close) - h: max(1, h)]
    fv, sv = fwd.to_numpy(), s.to_numpy()
    for t in tests[-200:]:
        m = (idx < t - h) & (sv == sv[t]) & ~np.isnan(fv)
        if m.sum() < min_n:
            continue
        pred = np.sign(np.nanmean(fv[m]) or 1e-12)
        if fv[t] != 0 and not np.isnan(fv[t]):
            hits.append(np.sign(fv[t]) == pred)
    hr = float(np.mean(hits)) if hits else np.nan
    edge = abs(p_up - 0.5) * 2
    conf = 50 + 50 * edge * min(1, len(pool) / 100)
    if hits and hr < 0.52:  # no proven edge -> pull confidence toward 50
        conf = 50 + (conf - 50) * 0.5
    d = "עולה" if p_up >= 0.55 else "יורד" if p_up <= 0.45 else "ניטרלי"
    return {"direction": d, "p_up": p_up, "lo": px * np.exp(q[0]), "mid": px * np.exp(q[1]), "hi": px * np.exp(q[2]),
            "conf": conf, "n": len(pool), "hit": hr, "n_tests": len(hits)}


def fetch_hourly(ticker: str) -> pd.Series:
    import yfinance as yf
    d = yf.Ticker(ticker).history(period="6mo", interval="1h", auto_adjust=False)
    return d["Close"].dropna() if len(d) else pd.Series(dtype=float)


def multi_horizon(daily_close: pd.Series, hourly_close: pd.Series | None) -> pd.DataFrame:
    rows = []
    for label, frame, h, rel in HORIZONS:
        try:
            if frame == "1d":
                s = daily_close
            elif hourly_close is None or len(hourly_close) < 100:
                s = None
            elif frame == "4h":
                s = hourly_close.resample("4h").last().dropna()
            else:
                s = hourly_close
            f = analog_forecast(s, h) if s is not None else {}
        except Exception:
            f = {}
        rows.append({"טווח": label, **f, "אמינות": rel})
    return pd.DataFrame(rows)


def market_open_now() -> bool:
    ny = dt.datetime.now(ZoneInfo("America/New_York"))
    return ny.weekday() < 5 and dt.time(9, 30) <= ny.time() <= dt.time(16, 0)


def to_display(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for r in df.to_dict("records"):
        if r.get("p_up") is None or (isinstance(r.get("p_up"), float) and np.isnan(r["p_up"])):
            out.append({"טווח": r["טווח"], "כיוון": "אין נתונים", "סיכוי עלייה/ירידה": "—", "טווח מחיר (5%–50%–95%)": "—",
                        "ביטחון": "—", "דיוק בעבר": "—", "אמינות": r["אמינות"]})
            continue
        hit = "—" if np.isnan(r["hit"]) else f"{r['hit']:.0%} (n={r['n_tests']})"
        out.append({"טווח": r["טווח"], "כיוון": {"עולה": "▲ עולה", "יורד": "▼ יורד"}.get(r["direction"], "■ ניטרלי"),
                    "סיכוי עלייה/ירידה": f"{r['p_up']:.0%} / {1 - r['p_up']:.0%}",
                    "טווח מחיר (5%–50%–95%)": f"{r['lo']:,.2f} – {r['mid']:,.2f} – {r['hi']:,.2f}",
                    "ביטחון": f"{r['conf']:.0f}%", "דיוק בעבר": hit, "אמינות": r["אמינות"]})
    return pd.DataFrame(out)
