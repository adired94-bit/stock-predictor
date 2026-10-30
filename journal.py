"""Prediction journal: save predictions, auto-evaluate when the horizon matures, persist to a private
GitHub repo (st.secrets GITHUB_TOKEN + JOURNAL_REPO) or fall back to a local CSV."""
from __future__ import annotations

import base64
import datetime as dt
import io
import json
import os
import uuid
from typing import Optional

import numpy as np
import pandas as pd

COLS = ["id", "ticker", "created", "horizon", "frame", "bars", "kind", "direction", "price", "entry", "stop", "target",
        "lo", "hi", "p_up", "status", "result", "evaluated", "actual", "ret", "note"]
HBARS = {"שעתי": ("1h", 1), "4 שעות": ("4h", 1), "יומי": ("1d", 1), "שבועי": ("1d", 5), "חודשי": ("1d", 21), "רבעוני": ("1d", 63)}
TRADE_BARS = 10  # trade plan evaluated over 10 trading days
LOCAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".journal", "journal.csv")
HEB = {"hit": "✅ עבד", "miss": "❌ לא עבד", "partial": "🟡 חלקית", "pending": "⏳ ממתין"}


# ---------------------------------------------------------------- storage
class LocalStore:
    kind = "local"

    def __init__(self, path: str = LOCAL):
        self.path = path

    def load(self) -> pd.DataFrame:
        if os.path.exists(self.path):
            return _norm(pd.read_csv(self.path))
        return pd.DataFrame(columns=COLS)

    def save(self, df: pd.DataFrame, msg: str = "") -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        df[COLS].to_csv(self.path, index=False)


class GitHubStore:
    """Stores journal.csv in a (private) repo via the GitHub contents API."""
    kind = "github"

    def __init__(self, token: str, repo: str, path: str = "journal.csv", branch: str = "main"):
        self.token, self.repo, self.path, self.branch = token, repo, path, branch
        self.url = f"https://api.github.com/repos/{repo}/contents/{path}"
        self._sha: Optional[str] = None

    def _h(self):
        return {"Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json"}

    def load(self) -> pd.DataFrame:
        import requests
        r = requests.get(self.url, headers=self._h(), params={"ref": self.branch}, timeout=15)
        if r.status_code == 404:
            self._sha = None
            return pd.DataFrame(columns=COLS)
        r.raise_for_status()
        j = r.json(); self._sha = j["sha"]
        raw = base64.b64decode(j["content"]).decode("utf-8")
        return _norm(pd.read_csv(io.StringIO(raw))) if raw.strip() else pd.DataFrame(columns=COLS)

    def save(self, df: pd.DataFrame, msg: str = "update journal") -> None:
        import requests
        if self._sha is None:
            self.load()
        body = {"message": msg, "branch": self.branch,
                "content": base64.b64encode(df[COLS].to_csv(index=False).encode()).decode()}
        if self._sha:
            body["sha"] = self._sha
        r = requests.put(self.url, headers=self._h(), data=json.dumps(body), timeout=20)
        r.raise_for_status()
        self._sha = r.json()["content"]["sha"]


def get_store(secrets=None):
    """GitHub store when secrets are configured, else local file."""
    try:
        if secrets is not None and "GITHUB_TOKEN" in secrets:
            return GitHubStore(secrets["GITHUB_TOKEN"], secrets.get("JOURNAL_REPO", "adired94-bit/stock-predictor-journal"))
    except Exception:
        pass
    return LocalStore()


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    for c in COLS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[COLS].copy()
    for c in ("id", "ticker", "created", "horizon", "frame", "kind", "direction", "status", "result", "evaluated", "note"):
        df[c] = df[c].astype(object)
    return df


# ---------------------------------------------------------------- entries
def make_entries(ticker: str, price: float, horizons: pd.DataFrame, plan: Optional[dict], now: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    now = now or pd.Timestamp.now(tz="UTC")
    rows = []
    for r in horizons.to_dict("records"):
        if r.get("p_up") is None or pd.isna(r.get("p_up")):
            continue
        frame, bars = HBARS[r["טווח"]]
        rows.append({"id": uuid.uuid4().hex[:10], "ticker": ticker, "created": now.isoformat(), "horizon": r["טווח"], "frame": frame,
                     "bars": bars, "kind": "forecast", "direction": r["direction"], "price": price, "lo": r["lo"], "hi": r["hi"],
                     "p_up": r["p_up"], "status": "pending"})
    if plan and plan.get("side") in ("LONG", "SHORT"):
        rows.append({"id": uuid.uuid4().hex[:10], "ticker": ticker, "created": now.isoformat(), "horizon": "עסקה (10 ימים)",
                     "frame": "1d", "bars": TRADE_BARS, "kind": "trade", "direction": plan["side"], "price": price,
                     "entry": plan["entry"], "stop": plan["stop"], "target": plan["tp1"], "status": "pending"})
    return _norm(pd.DataFrame(rows))


# ---------------------------------------------------------------- evaluation (pure)
def evaluate(row: dict, bars: pd.DataFrame) -> dict:
    """bars = OHLC bars strictly AFTER the prediction time, in the row's frame.
    Returns updates {status, result, actual, ret}. status stays 'pending' until `bars` rows exist."""
    n = int(row["bars"])
    if bars is None or len(bars) < n:
        return {"status": "pending"}
    w = bars.iloc[:n]
    price = float(row["price"])
    actual = float(w["Close"].iloc[-1])
    ret = actual / price - 1
    if row["kind"] == "trade":
        long = row["direction"] == "LONG"
        stop, tgt = float(row["stop"]), float(row["target"])
        res = None
        for _, b in w.iterrows():
            hit_stop = b["Low"] <= stop if long else b["High"] >= stop
            hit_tgt = b["High"] >= tgt if long else b["Low"] <= tgt
            if hit_stop:  # conservative: stop first if both in same bar
                res = "miss"; break
            if hit_tgt:
                res = "hit"; break
        if res is None:
            res = "partial" if (ret > 0) == long and ret != 0 else "miss"
        txt = {"hit": "הגיע ליעד לפני הסטופ", "miss": "נגע בסטופ / הלך נגד", "partial": "לא הגיע ליעד אבל ברווח"}[res]
    else:
        d = row["direction"]
        inside = float(row["lo"]) <= actual <= float(row["hi"])
        if d == "ניטרלי":
            res = "hit" if inside else "miss"
            txt = "נשאר בטווח" if inside else "יצא מהטווח"
        else:
            right = ret > 0 if d == "עולה" else ret < 0
            res = ("hit" if inside else "partial") if right else "miss"
            txt = "כיוון נכון" + ("" if inside else ", מחוץ לטווח") if right else "כיוון שגוי"
    return {"status": res, "result": txt, "actual": actual, "ret": ret,
            "evaluated": pd.Timestamp.now(tz="UTC").isoformat()}


def fetch_bars_after(ticker: str, frame: str, since: pd.Timestamp) -> pd.DataFrame:
    import yfinance as yf
    since = pd.Timestamp(since)
    if frame == "1d":
        d = yf.Ticker(ticker).history(start=(since - pd.Timedelta(days=3)).date(), interval="1d", auto_adjust=False)
        if not len(d):
            return d
        idx = d.index.tz_convert("UTC") if d.index.tz is not None else d.index.tz_localize("UTC")
        # a daily bar counts only if its session started after the prediction day (US close 21:00 UTC approx)
        return d[idx.normalize() > since.normalize()]
    d = yf.Ticker(ticker).history(period="1mo", interval="1h", auto_adjust=False)
    if not len(d):
        return d
    if frame == "4h":
        d = d.resample("4h").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    idx = d.index.tz_convert("UTC")
    return d[idx > since]


def evaluate_all(df: pd.DataFrame, fetch=fetch_bars_after) -> tuple[pd.DataFrame, int]:
    df = df.copy(); n = 0
    cache = {}
    for i, r in df[df["status"] == "pending"].iterrows():
        key = (r["ticker"], r["frame"], r["created"])
        try:
            if key not in cache:
                cache[key] = fetch(r["ticker"], r["frame"], pd.Timestamp(r["created"]))
            up = evaluate(r.to_dict(), cache[key])
        except Exception:
            continue
        if up["status"] != "pending":
            for k, v in up.items():
                df.at[i, k] = v
            n += 1
    return df, n


def stats(df: pd.DataFrame) -> dict:
    done = df[df["status"].isin(["hit", "miss", "partial"])].copy()
    if done.empty:
        return {"n": 0}
    done["score"] = done["status"].map({"hit": 1.0, "partial": 0.5, "miss": 0.0})
    grp = lambda col: done.groupby(col).agg(n=("score", "size"), success=("score", "mean"), avg_ret=("ret", "mean"))
    side = done.assign(side=np.where(done["direction"].isin(["LONG", "עולה"]), "לונג/עולה",
                                     np.where(done["direction"].isin(["SHORT", "יורד"]), "שורט/יורד", "ניטרלי")))
    done = done.sort_values("evaluated")
    done["cum"] = done["score"].expanding().mean()
    return {"n": len(done), "success": done["score"].mean(), "hit_rate": (done["status"] == "hit").mean(),
            "by_ticker": grp("ticker"), "by_horizon": grp("horizon"),
            "by_side": side.groupby("side").agg(n=("score", "size"), success=("score", "mean"), avg_ret=("ret", "mean")),
            "curve": done[["evaluated", "cum"]]}
