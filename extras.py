"""Options, intraday, events, peers, gaps and data-quality helpers. All degrade gracefully."""
from __future__ import annotations

import datetime as dt
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    import streamlit as st
    _cache = st.cache_data(ttl=1800, show_spinner=False)
except Exception:  # pragma: no cover
    def _cache(f):  # type: ignore
        return f

# ----------------------------------------------------------------- options
def max_pain(calls: pd.DataFrame, puts: pd.DataFrame) -> float:
    """Strike minimising total intrinsic value paid to option holders (OI weighted)."""
    strikes = np.union1d(calls["strike"], puts["strike"])
    c_oi = calls.set_index("strike")["openInterest"].fillna(0); p_oi = puts.set_index("strike")["openInterest"].fillna(0)
    ck, pk = c_oi.index.to_numpy(float), p_oi.index.to_numpy(float)
    pain = [float((np.clip(s - ck, 0, None) * c_oi.to_numpy()).sum() + (np.clip(pk - s, 0, None) * p_oi.to_numpy()).sum())
            for s in strikes]
    return float(strikes[int(np.argmin(pain))])


def options_metrics(calls: pd.DataFrame, puts: pd.DataFrame, spot: float, dte: int) -> dict[str, Any]:
    """ATM IV, ATM straddle implied move, P/C ratios (volume & OI), max pain, and a signal -1..1."""
    out: dict[str, Any] = {"dte": dte}
    k = calls.iloc[(calls["strike"] - spot).abs().argsort()[:1]]
    kp = puts.iloc[(puts["strike"] - spot).abs().argsort()[:1]]
    mid = lambda r: float(((r["bid"] + r["ask"]) / 2).iloc[0]) if float(r["ask"].iloc[0]) > 0 else float(r["lastPrice"].iloc[0])
    straddle = mid(k) + mid(kp)
    out["atm_iv"] = float(np.nanmean([k["impliedVolatility"].iloc[0], kp["impliedVolatility"].iloc[0]]))
    out["implied_move_pct"] = straddle / spot
    out["pcr_volume"] = float(puts["volume"].fillna(0).sum() / max(calls["volume"].fillna(0).sum(), 1))
    out["pcr_oi"] = float(puts["openInterest"].fillna(0).sum() / max(calls["openInterest"].fillna(0).sum(), 1))
    out["max_pain"] = max_pain(calls, puts)
    # contrarian-light: high PCR bearish positioning -> mildly bearish; pull toward max pain
    pcr_sig = float(np.clip((0.9 - out["pcr_volume"]) / 0.6, -1, 1))
    mp_sig = float(np.clip((out["max_pain"] / spot - 1) / 0.05, -1, 1))
    out["signal"] = 0.7 * pcr_sig + 0.3 * mp_sig
    return out


@_cache
def fetch_options(ticker: str, spot: float) -> tuple[dict, Optional[str]]:
    """Nearest-expiry option chain metrics via yfinance."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        exps = t.options
        if not exps:
            return {}, "No listed options."
        today = dt.date.today()
        exp = next((e for e in exps if (dt.date.fromisoformat(e) - today).days >= 1), exps[0])
        ch = t.option_chain(exp)
        m = options_metrics(ch.calls, ch.puts, spot, (dt.date.fromisoformat(exp) - today).days)
        m["expiry"] = exp
        return m, None
    except Exception as e:
        return {}, f"Options unavailable: {e}"

# ----------------------------------------------------------------- intraday
@_cache
def fetch_intraday(ticker: str, interval: str = "15m") -> tuple[pd.DataFrame, Optional[str]]:
    """Last 60 days of intraday bars (yfinance limit)."""
    try:
        import yfinance as yf
        df = yf.download(ticker, period="60d", interval=interval, progress=False, auto_adjust=True, threads=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return (df, None) if len(df) else (df, "No intraday data.")
    except Exception as e:
        return pd.DataFrame(), f"Intraday unavailable: {e}"


def intraday_analysis(df: pd.DataFrame, or_minutes: int = 30) -> dict[str, Any]:
    """Session-anchored VWAP, opening range of the latest session, time-of-day mean return/hit-rate."""
    d = df.copy()
    idx = d.index.tz_convert("America/New_York") if getattr(d.index, "tz", None) else d.index
    d.index = idx
    d["session"] = d.index.date
    tp = (d["High"] + d["Low"] + d["Close"]) / 3
    d["VWAP"] = (tp * d["Volume"]).groupby(d["session"]).cumsum() / d["Volume"].groupby(d["session"]).cumsum()
    last = d[d["session"] == d["session"].iloc[-1]]
    start = last.index[0]
    orr = last[last.index < start + pd.Timedelta(minutes=or_minutes)]
    d["ret"] = d.groupby("session")["Close"].pct_change()
    tod = d.groupby(d.index.strftime("%H:%M"))["ret"].agg(mean="mean", hit=lambda x: (x > 0).mean(), n="count").reset_index(names="time")
    return {"frame": d, "or_high": float(orr["High"].max()), "or_low": float(orr["Low"].min()),
            "last": float(last["Close"].iloc[-1]), "vwap": float(last["VWAP"].iloc[-1]), "tod": tod,
            "signal": float(np.clip((last["Close"].iloc[-1] / last["VWAP"].iloc[-1] - 1) / 0.01, -1, 1))}

# ----------------------------------------------------------------- events
# Labeled, hardcoded high-impact US macro dates (verify against official calendars).
MACRO_EVENTS = {
    "2026-11-06": "Jobs report (NFP)", "2026-11-12": "CPI", "2026-12-04": "Jobs report (NFP)",
    "2026-12-09": "FOMC decision", "2026-12-10": "CPI", "2027-01-08": "Jobs report (NFP)",
    "2027-01-13": "CPI", "2027-01-27": "FOMC decision", "2027-03-17": "FOMC decision",
}


@_cache
def fetch_events(ticker: str) -> tuple[dict, Optional[str]]:
    """Next earnings / ex-dividend from yfinance calendar."""
    try:
        import yfinance as yf
        cal = yf.Ticker(ticker).calendar or {}
        if isinstance(cal, pd.DataFrame):
            cal = cal.to_dict()
        e = cal.get("Earnings Date"); e = e[0] if isinstance(e, list) and e else e
        return {"earnings": pd.to_datetime(e).date() if e else None,
                "ex_dividend": pd.to_datetime(cal["Ex-Dividend Date"]).date() if cal.get("Ex-Dividend Date") else None}, None
    except Exception as ex:
        return {}, f"Calendar unavailable: {ex}"


def event_risk(events: dict, today: dt.date, window: int = 5) -> tuple[list[dict], float]:
    """Events within ``window`` days and a confidence multiplier (earnings 0.7, macro 0.9 each, floor 0.5)."""
    hits, mult = [], 1.0
    for name in ("earnings", "ex_dividend"):
        d = events.get(name)
        if d and 0 <= (d - today).days <= window:
            hits.append({"date": d, "event": name.replace("_", "-").title(), "type": "company"})
            mult *= 0.7 if name == "earnings" else 0.95
    for ds, lbl in MACRO_EVENTS.items():
        d = dt.date.fromisoformat(ds)
        if 0 <= (d - today).days <= window:
            hits.append({"date": d, "event": lbl, "type": "macro (hardcoded)"}); mult *= 0.9
    return hits, max(mult, 0.5)

# ----------------------------------------------------------------- peers
SEMIS = ["NVDA", "AMD", "SMH", "QQQ"]
SECTOR_ETF = {"Technology": "XLK", "Financial Services": "XLF", "Healthcare": "XLV", "Energy": "XLE",
              "Consumer Cyclical": "XLY", "Consumer Defensive": "XLP", "Industrials": "XLI",
              "Communication Services": "XLC", "Utilities": "XLU", "Real Estate": "XLRE", "Basic Materials": "XLB"}
SEMI_TICKERS = {"MU", "NVDA", "AMD", "INTC", "AVGO", "QCOM", "TSM", "ASML", "AMAT", "LRCX", "KLAC", "WDC", "SNDK", "MRVL", "TXN", "ON", "SMCI", "ARM"}


def default_peers(ticker: str, sector: Optional[str] = None) -> list[str]:
    """Semis peer list for chip names, otherwise sector ETF + SPY."""
    if ticker.upper() in SEMI_TICKERS or (sector or "") == "Semiconductors":
        return [p for p in SEMIS if p != ticker.upper()]
    return [SECTOR_ETF.get(sector or "", "SPY"), "SPY", "QQQ"]


def peer_stats(target: pd.Series, peers: dict[str, pd.Series], window: int = 60) -> tuple[pd.DataFrame, float]:
    """Rolling beta/correlation (last ``window`` days) and 5d peer moves; signal = beta-weighted 5d peer move."""
    r = target.pct_change()
    rows = []
    for name, s in peers.items():
        pr = s.pct_change()
        j = pd.concat([r, pr], axis=1, join="inner").dropna().tail(window)
        if len(j) < 20:
            continue
        corr = float(j.iloc[:, 0].corr(j.iloc[:, 1])); beta = float(j.cov().iloc[0, 1] / j.iloc[:, 1].var())
        rows.append({"peer": name, "corr": corr, "beta": beta, "ret_1d": float(s.iloc[-1] / s.iloc[-2] - 1),
                     "ret_5d": float(s.iloc[-1] / s.iloc[-6] - 1)})
    t = pd.DataFrame(rows)
    if t.empty:
        return t, 0.0
    sig = float(np.clip((t["corr"].clip(lower=0) * t["ret_5d"]).sum() / max(t["corr"].clip(lower=0).sum(), 1e-9) / 0.05, -1, 1))
    return t, sig

# ----------------------------------------------------------------- gaps
def gap_analysis(df: pd.DataFrame, tol: float = 0.5, max_days: int = 20) -> dict[str, Any]:
    """Gap frequency, fill rate and time-to-fill for gaps similar (±tol*size) to the latest gap."""
    o, c, h, l = (df[k].to_numpy(float) for k in ("Open", "Close", "High", "Low"))
    gaps = o[1:] / c[:-1] - 1
    last_gap = float(gaps[-1])
    res: dict[str, Any] = {"last_gap": last_gap, "freq_up_1pct": float((gaps > 0.01).mean()),
                           "freq_down_1pct": float((gaps < -0.01).mean())}
    size = abs(last_gap)
    sel = [i for i, g in enumerate(gaps[:-1]) if np.sign(g) == np.sign(last_gap) and size * (1 - tol) <= abs(g) <= size * (1 + tol) and abs(g) > 0.002]
    fills = []
    for i in sel:
        b = i + 1; prev = c[i]
        days = None
        for k in range(b, min(b + max_days, len(df))):
            if (last_gap > 0 and l[k] <= prev) or (last_gap < 0 and h[k] >= prev):
                days = k - b; break
        fills.append(days)
    n = len(fills); filled = [d for d in fills if d is not None]
    res.update({"n_similar": n, "fill_rate": len(filled) / n if n else float("nan"),
                "fill_same_day": float(np.mean([d == 0 for d in fills])) if n else float("nan"),
                "median_days_to_fill": float(np.median(filled)) if filled else float("nan")})
    return res

# ----------------------------------------------------------------- data quality
def data_quality(df: pd.DataFrame) -> dict[str, Any]:
    """Bar count, missing business days, zero-volume bars, stale last bar, extreme moves."""
    bd = pd.bdate_range(df.index[0], df.index[-1])
    r = df["Close"].pct_change().abs()
    return {"bars": len(df), "first": df.index[0].date(), "last": df.index[-1].date(),
            "missing_bdays(~holidays)": int(len(bd) - len(df)), "zero_volume_bars": int((df["Volume"] == 0).sum()),
            "days_since_last_bar": int((pd.Timestamp.today().normalize() - df.index[-1]).days),
            "moves_over_20pct": int((r > 0.2).sum())}
