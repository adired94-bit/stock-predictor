"""Home / Market dashboard: overview tiles, news, movers, setups, watchlist, events.
Every section is independent and wrapped in try/except so one failing feed never breaks the page."""
from __future__ import annotations

import datetime as dt
import html
from typing import Callable, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

TZ = ZoneInfo("Asia/Jerusalem")
MAIN_TILES = [("S&P 500", "^GSPC"), ("נאסד\"ק", "^IXIC"), ("ביטקוין", "BTC-USD"), ("VIX · מדד הפחד", "^VIX"),
              ("תשואת אג\"ח 10 שנים", "^TNX"), ("זהב", "GC=F")]
MORE_TILES = [("דאו ג'ונס", "^DJI"), ("ראסל 2000", "^RUT"), ("מדד הדולר", "DX-Y.NYB"), ("נפט", "CL=F"), ("את'ריום", "ETH-USD")]
OVERVIEW = MAIN_TILES + MORE_TILES
CURATED = ["MU", "MSTR", "NVDA", "AMD", "COIN", "HOOD", "SOFI", "PLTR", "SMCI", "TSM", "AVGO", "MARA", "RIOT", "SOUN",
           "AAPL", "MSFT", "META", "GOOGL", "AMZN", "TSLA", "INTC", "ARM", "IBIT", "AFRM", "UPST", "PYPL", "CLSK", "IONQ"]
DEFAULT_WATCH = ["MU", "MSTR", "NVDA", "AAPL", "BTC-USD", "IBIT"]
RSS = {"Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
       "Google News · Markets": "https://news.google.com/rss/search?q=stock+market&hl=en-US&gl=US&ceid=US:en"}
TTL = 300

CSS = """<style>
.tile{background:#161b26;border-radius:16px;padding:14px 16px;margin-bottom:4px;direction:rtl;text-align:right}
.tile .n{font-size:.95rem;color:#9aa4b2}.tile .p{font-size:1.5rem;font-weight:800;direction:ltr;text-align:right}
.up{color:#22c55e}.dn{color:#ef4444}.badge{border-radius:8px;padding:1px 7px;font-size:.72rem;font-weight:700}
.bpos{background:#14532d;color:#86efac}.bneg{background:#7f1d1d;color:#fca5a5}.bneu{background:#374151;color:#d1d5db}
.news{direction:rtl;text-align:right;padding:8px 0;border-bottom:1px solid #222}.news a{color:#e5e7eb;text-decoration:none;font-weight:600}
.meta{font-size:.75rem;color:#9aa4b2}.card{direction:rtl;text-align:right;background:#161b26;border-left:4px solid #3b82f6;border-radius:10px;padding:8px 12px;margin-bottom:6px}
div.stButton>button{min-height:2.6rem}
</style>"""


# ------------------------------------------------------------------ data (cached, pure where possible)
@st.cache_data(ttl=TTL, show_spinner=False)
def fetch_daily(tickers: tuple[str, ...], period: str = "3mo") -> dict[str, pd.DataFrame]:
    import yfinance as yf
    raw = yf.download(list(tickers), period=period, interval="1d", group_by="ticker", auto_adjust=False,
                      progress=False, threads=True)
    out = {}
    for t in tickers:
        try:
            d = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
            d = d.dropna(subset=["Close"])
            if len(d) >= 2:
                out[t] = d
        except Exception:
            pass
    return out


def stats_from(d: pd.DataFrame) -> dict:
    """Price, daily %, gap %, RVOL, RSI14, 20d-high breakout from a daily OHLCV frame."""
    c, o, v = d["Close"].astype(float), d["Open"].astype(float), d["Volume"].astype(float)
    price, prev = c.iloc[-1], c.iloc[-2]
    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14).mean(); dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14).mean()
    rsi = float(100 - 100 / (1 + up.iloc[-1] / dn.iloc[-1])) if dn.iloc[-1] > 0 else 100.0
    avgv = v.iloc[-21:-1].mean()
    hi20 = d["High"].astype(float).iloc[-21:-1].max()
    return {"price": float(price), "chg": float(price / prev - 1) * 100, "gap": float(o.iloc[-1] / prev - 1) * 100,
            "rvol": float(v.iloc[-1] / avgv) if avgv and avgv > 0 else np.nan, "rsi": rsi,
            "breakout": bool(price > hi20), "volume": float(v.iloc[-1]), "spark": c.iloc[-30:].tolist(),
            "ret5": float(price / c.iloc[-6] - 1) * 100 if len(c) > 6 else np.nan}


def scan(tickers: list[str]) -> pd.DataFrame:
    data = fetch_daily(tuple(tickers))
    rows = []
    for t, d in data.items():
        try:
            rows.append({"ticker": t, **stats_from(d)})
        except Exception:
            pass
    return pd.DataFrame(rows)


def setups(df: pd.DataFrame) -> list[tuple[str, str]]:
    """Plain-language flags: gaps >3%, RVOL>2, RSI extremes, 20-day-high breakouts."""
    out = []
    for r in df.itertuples():
        f = []
        if abs(r.gap) >= 3:
            f.append(f"גאפ {'למעלה' if r.gap > 0 else 'למטה'} {abs(r.gap):.1f}%")
        if r.rvol == r.rvol and r.rvol >= 2:
            f.append(f"ווליום כבד (פי {r.rvol:.1f})")
        if r.rsi >= 70:
            f.append(f"קנויה יתר (RSI {r.rsi:.0f})")
        elif r.rsi <= 30:
            f.append(f"מכורה יתר (RSI {r.rsi:.0f})")
        if r.breakout:
            f.append("פריצה מעל שיא 20 יום")
        if f:
            out.append((r.ticker, ", ".join(f)))
    return out


@st.cache_data(ttl=TTL, show_spinner=False)
def screener(name: str) -> pd.DataFrame:
    import yfinance as yf
    res = yf.screen(name, count=15)
    q = res.get("quotes", []) if isinstance(res, dict) else []
    return pd.DataFrame([{"ticker": x.get("symbol"), "name": x.get("shortName", ""), "price": x.get("regularMarketPrice"),
                          "chg": x.get("regularMarketChangePercent"), "volume": x.get("regularMarketVolume")} for x in q])


@st.cache_data(ttl=TTL, show_spinner=False)
def market_news(limit: int = 15) -> pd.DataFrame:
    import urllib.request
    import xml.etree.ElementTree as ET
    from sentiment import score_text
    rows = []
    for src, url in RSS.items():
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            root = ET.fromstring(urllib.request.urlopen(req, timeout=10).read())
            for e in root.iter("item"):
                if len([r for r in rows if r["src"] == src]) >= limit:
                    break
                title = e.findtext("title") or ""
                ts = pd.to_datetime(e.findtext("pubDate"), utc=True, errors="coerce")
                rows.append({"title": title, "link": e.findtext("link") or "", "publisher": e.findtext("source") or src,
                             "published": ts, "sent": score_text(title), "src": src})
        except Exception:
            pass
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.drop_duplicates("title").sort_values("published", ascending=False).head(limit)


def _rss(url: str, limit: int = 5) -> list[dict]:
    import urllib.request
    import xml.etree.ElementTree as ET
    from sentiment import score_text
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    root = ET.fromstring(urllib.request.urlopen(req, timeout=10).read())
    out = []
    for e in list(root.iter("item"))[:limit]:
        title = e.findtext("title") or ""
        out.append({"title": title, "link": e.findtext("link") or "", "publisher": e.findtext("source") or "Google News",
                    "published": pd.to_datetime(e.findtext("pubDate"), utc=True, errors="coerce"), "sent": score_text(title), "summary": ""})
    return out


@st.cache_data(ttl=TTL, show_spinner=False)
def ticker_news(tickers: tuple[str, ...], per: int = 3) -> pd.DataFrame:
    from sentiment import fetch_news, parse_news_item, score_text
    rows = []
    for t in tickers:
        items, _ = fetch_news(t)
        if not items:  # yfinance news empty -> Google News RSS for the ticker
            try:
                q = t.replace("-USD", "") + "+stock"
                rows += [{"ticker": t, **x} for x in _rss(f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en", per)]
            except Exception:
                pass
            continue
        for it in items[:per]:
            p = parse_news_item(it)
            if p["title"]:
                rows.append({"ticker": t, **p, "sent": score_text(p["title"] + ". " + p["summary"])})
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600, show_spinner=False)
def earnings_dates(tickers: tuple[str, ...]) -> pd.DataFrame:
    from extras import fetch_events
    rows = []
    for t in tickers:
        if "-" in t or t in ("IBIT", "SPY", "QQQ"):
            continue
        try:
            ev, _ = fetch_events(t)
            if ev.get("earnings"):
                rows.append({"date": ev["earnings"], "event": f"{t} earnings"})
        except Exception:
            pass
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ rendering helpers
def _cls(x):
    return "up" if x >= 0 else "dn"


def spark(vals: list[float], color: str) -> go.Figure:
    f = go.Figure(go.Scatter(y=vals, mode="lines", line=dict(color=color, width=2), hoverinfo="skip"))
    f.update_layout(height=45, margin=dict(l=0, r=0, t=0, b=0), xaxis_visible=False, yaxis_visible=False,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
    return f


def badge(s: float) -> str:
    if s > 0.15:
        return '<span class="badge bpos">חיובי</span>'
    if s < -0.15:
        return '<span class="badge bneg">שלילי</span>'
    return '<span class="badge bneu">ניטרלי</span>'


def _ago(ts) -> str:
    if ts is None or pd.isna(ts):
        return ""
    return pd.Timestamp(ts).tz_convert(TZ).strftime("%d/%m %H:%M")


def _news_list(df: pd.DataFrame, show_ticker: bool = False):
    for r in df.itertuples():
        tk = f"<b>{r.ticker}</b> · " if show_ticker else ""
        st.markdown(f'<div class="news">{badge(r.sent)} <a href="{html.escape(r.link)}" target="_blank">'
                    f'{html.escape(r.title)}</a><div class="meta">{tk}{html.escape(str(r.publisher))} · {_ago(r.published)}</div></div>',
                    unsafe_allow_html=True)


def _section(title: str, fn: Callable, *a):
    try:
        fn(*a)
    except Exception as e:  # friendly fallback
        st.info(f"{title}: הנתונים לא זמינים כרגע. נסה לרענן בעוד דקה.")


def _rows_with_buttons(df: pd.DataFrame, on_analyze: Callable[[str], None], key: str, n: int = 8):
    for i, r in enumerate(df.head(n).itertuples()):
        c1, c2 = st.columns([3, 1])
        name = f" · {r.name[:22]}" if "name" in df.columns and isinstance(getattr(r, "name", ""), str) and r.name else ""
        vol = f" · RVOL {r.rvol:.1f}×" if "rvol" in df.columns and r.rvol == r.rvol else ""
        c1.markdown(f"**{r.ticker}**{name}  \n${r.price:,.2f} <span class='{_cls(r.chg)}'>{r.chg:+.2f}%</span>{vol}",
                    unsafe_allow_html=True)
        c2.button("נתח", key=f"{key}_{i}_{r.ticker}", on_click=on_analyze, args=(r.ticker,), width="stretch")


# ------------------------------------------------------------------ sections
def _tiles(items, data, ncol=2):
    cols = st.columns(ncol)
    for i, (name, t) in enumerate(items):
        if t not in data:
            continue
        s = stats_from(data[t])
        unit = "%" if t == "^TNX" else ""
        with cols[i % ncol]:
            st.markdown(f'<div class="tile"><div class="n">{name}</div><div class="p">{s["price"]:,.2f}{unit} '
                        f'<span class="{_cls(s["chg"])}" style="font-size:.85rem">{s["chg"]:+.2f}%</span></div></div>',
                        unsafe_allow_html=True)
            st.plotly_chart(spark(s["spark"], "#22c55e" if s["chg"] >= 0 else "#ef4444"), key=f"sp_{t}",
                            config={"displayModeBar": False, "staticPlot": True}, width="stretch")


def sec_overview():
    data = fetch_daily(tuple(t for _, t in OVERVIEW))
    if not data:
        raise RuntimeError("no data")
    _tiles(MAIN_TILES, data)
    with st.expander("עוד מדדים"):
        _tiles(MORE_TILES, data)


def sec_movers(on_analyze, key):
    cur = scan(CURATED)
    tabs = st.tabs(["🚀 עולות", "🔻 יורדות", "🔥 הכי פעילות", "📊 ווליום חריג", "💻 טק/פינטק/קריפטו"])
    src_note = []
    for tab, name, asc in zip(tabs[:3], ["day_gainers", "day_losers", "most_actives"], [False, True, None]):
        with tab:
            try:
                df = screener(name).dropna(subset=["price", "chg"])
                if df.empty:
                    raise ValueError
                st.caption("כל השוק האמריקאי")
            except Exception:
                df = cur.sort_values("chg" if asc is not None else "volume", ascending=bool(asc))
                st.caption("מתוך רשימת המעקב (הסורק לא זמין)")
                src_note.append(name)
            _rows_with_buttons(df, on_analyze, f"{key}{name}")
    with tabs[3]:
        st.caption("ווליום היום מול ממוצע 20 יום")
        _rows_with_buttons(cur.sort_values("rvol", ascending=False), on_analyze, f"{key}rvol")
    with tabs[4]:
        st.caption("מניות טק, פינטק וקריפטו נבחרות")
        _rows_with_buttons(cur.sort_values("chg", ascending=False), on_analyze, f"{key}cur", n=len(cur))


def sec_setups(on_analyze, key):
    s = setups(scan(CURATED))
    if not s:
        st.caption("אין כרגע סטאפים חזקים.")
    for i, (t, txt) in enumerate(s[:12]):
        c1, c2 = st.columns([3, 1])
        c1.markdown(f'<div class="card"><b>{t}</b>: {txt}</div>', unsafe_allow_html=True)
        c2.button("נתח", key=f"{key}set_{i}_{t}", on_click=on_analyze, args=(t,), width="stretch")


def sec_watchlist(on_analyze, key):
    wl = st.session_state.setdefault("watchlist", list(DEFAULT_WATCH))
    txt = st.text_input("ערוך רשימת מעקב (מופרד בפסיקים)", ", ".join(wl), key=f"{key}wl_in")
    new = [x.strip().upper() for x in txt.split(",") if x.strip()]
    if new != wl:
        st.session_state["watchlist"] = wl = new
    df = scan(wl)
    if df.empty:
        raise RuntimeError("no data")
    def verdict(r):  # cheap trend read, not the full model
        sc = (r.chg > 0) + (r.ret5 > 0) + (50 < r.rsi < 70) - (r.rsi > 75) - (r.rsi < 30) * 0
        return "🟢 מגמה חיובית" if sc >= 2 else "🔴 חלשה" if sc <= 0 else "🟡 מעורבת"
    df["mini"] = [verdict(r) for r in df.itertuples()]
    for i, r in enumerate(df.itertuples()):
        c1, c2 = st.columns([3, 1])
        c1.markdown(f"**{r.ticker}** ${r.price:,.2f} <span class='{_cls(r.chg)}'>{r.chg:+.2f}%</span> · 5 ימים <span dir='ltr'>{r.ret5:+.1f}%</span> · {r.mini}",
                    unsafe_allow_html=True)
        c2.button("נתח", key=f"{key}wl_{i}_{r.ticker}", on_click=on_analyze, args=(r.ticker,), width="stretch")
    st.caption("הערכה מהירה לפי מגמה. לחץ נתח לחיזוי מלא.")


def sec_events():
    from extras import MACRO_EVENTS
    today = dt.datetime.now(TZ).date()
    rows = [{"date": pd.to_datetime(d).date(), "event": n} for d, n in MACRO_EVENTS.items()]
    try:
        e = earnings_dates(tuple(st.session_state.get("watchlist", DEFAULT_WATCH)))
        rows += e.to_dict("records")
    except Exception:
        st.caption("תאריכי דוחות לא זמינים כרגע.")
    df = pd.DataFrame(rows)
    df = df[(df["date"] >= today) & (df["date"] <= today + dt.timedelta(days=60))].sort_values("date")
    if df.empty:
        st.caption("אין אירועים ב-60 הימים הקרובים.")
    for r in df.itertuples():
        days = (r.date - today).days
        st.markdown(f"📅 **{r.date:%a %d %b}** · {r.event} <span class='meta'>({'היום' if days == 0 else f'בעוד {days} ימים'})</span>",
                    unsafe_allow_html=True)
    st.caption("תאריכי המאקרו מרשימה קבועה – כדאי לוודא בלוח הרשמי.")


def render_home(on_analyze: Callable[[str], None], key: str = "h"):
    st.markdown(CSS, unsafe_allow_html=True)
    c1, c2 = st.columns([3, 1])
    c1.markdown("## 🌍 השוק היום")
    if c2.button("🔄 רענן", key=f"{key}refresh", width="stretch"):
        st.cache_data.clear()
    st.caption(f"עודכן לאחרונה: {dt.datetime.now(TZ):%d/%m/%Y %H:%M} (שעון ישראל) · מתרענן כל 5 דקות")
    _section("סקירת שוק", sec_overview)
    st.markdown("### ⚡ סטאפים למעקב")
    _section("סטאפים", sec_setups, on_analyze, key)
    st.markdown("### 📈 מניות מעניינות")
    _section("מניות מעניינות", sec_movers, on_analyze, key)
    st.markdown("### ⭐ רשימת המעקב שלי")
    _section("רשימת מעקב", sec_watchlist, on_analyze, key)
    st.markdown("### 📰 חדשות השוק")
    def _mn():
        df = market_news()
        if df.empty:
            raise RuntimeError("no headlines")
        _news_list(df.head(10))
    _section("חדשות", _mn)
    with st.expander("📰 חדשות על המניות שלי"):
        def _tn():
            df = ticker_news(tuple(st.session_state.get("watchlist", DEFAULT_WATCH)))
            if df.empty:
                raise RuntimeError("no headlines")
            _news_list(df, show_ticker=True)
        _section("חדשות מניות", _tn)
    st.markdown("### 📅 אירועים קרובים")
    _section("אירועים", sec_events)
