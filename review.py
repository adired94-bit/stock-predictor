"""Hebrew full review: chart analysis sentences + a decisive LONG / SHORT / stay-out trade plan."""
from __future__ import annotations

import numpy as np
import pandas as pd

from indicators import rsi, volume_profile
from risk import atr, sr_levels


def macd(c: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    m = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    s = m.ewm(span=9, adjust=False).mean()
    return m, s, m - s


def bollinger(c: pd.Series, n: int = 20, k: float = 2.0):
    mid = c.rolling(n).mean(); sd = c.rolling(n).std()
    return mid + k * sd, mid, mid - k * sd


def fib_levels(df: pd.DataFrame, lookback: int = 120) -> dict:
    d = df.tail(lookback)
    hi, lo = float(d.High.max()), float(d.Low.min())
    up = d.High.idxmax() > d.Low.idxmin()  # swing up: retrace from high
    lv = {}
    for r in (0.236, 0.382, 0.5, 0.618, 0.786):
        lv[f"{r:.3f}"] = hi - (hi - lo) * r if up else lo + (hi - lo) * r
    return {"high": hi, "low": lo, "swing_up": bool(up), "levels": lv}


def trend_structure(df: pd.DataFrame, order: int = 5) -> str:
    h, l = df.High.tail(120).to_numpy(), df.Low.tail(120).to_numpy()
    ph = [h[i] for i in range(order, len(h) - order) if h[i] == h[i - order:i + order + 1].max()]
    pl = [l[i] for i in range(order, len(l) - order) if l[i] == l[i - order:i + order + 1].min()]
    if len(ph) >= 2 and len(pl) >= 2:
        if ph[-1] > ph[-2] and pl[-1] > pl[-2]:
            return "up"
        if ph[-1] < ph[-2] and pl[-1] < pl[-2]:
            return "down"
    return "range"


def divergence(df: pd.DataFrame, r: pd.Series, n: int = 20) -> str | None:
    c = df.Close
    if len(c) < 2 * n:
        return None
    a, b = c.iloc[-n:], c.iloc[-2 * n:-n]
    ra, rb = r.iloc[-n:], r.iloc[-2 * n:-n]
    if a.max() > b.max() and ra.max() < rb.max() - 3:
        return "bear"
    if a.min() < b.min() and ra.min() > rb.min() + 3:
        return "bull"
    return None


def chart_analysis(df: pd.DataFrame) -> dict:
    """Returns indicator values + list of (title, Hebrew sentence)."""
    c = df.Close.astype(float); px = float(c.iloc[-1])
    e9, e21, e50, e200 = (c.ewm(span=n, adjust=False).mean().iloc[-1] for n in (9, 21, 50, 200))
    r = rsi(c); rv = float(r.iloc[-1])
    m, s, hst = macd(c)
    bu, bm, bl = bollinger(c)
    fib = fib_levels(df)
    tr = trend_structure(df)
    div = divergence(df, r)
    sup, res = sr_levels(df)
    a = float(atr(df).iloc[-1])
    near_sup = max([x for x in sup if x < px], default=None)
    near_res = min([x for x in res if x > px], default=None)
    _, poc = volume_profile(df.tail(250))
    wk = c.resample("W").last().dropna()
    wtrend = "up" if len(wk) > 30 and wk.ewm(span=10).mean().iloc[-1] > wk.ewm(span=30).mean().iloc[-1] else "down"
    out = []
    out.append(("מגמה", {"up": "מגמה עולה – שיאים ושפלים עולים.", "down": "מגמה יורדת – שיאים ושפלים יורדים.",
                         "range": "אין מגמה ברורה – המניה בדשדוש."}[tr]))
    ema_txt = ("המחיר מעל כל הממוצעים (9/21/50/200) – חוזק." if px > max(e9, e21, e50, e200) else
               "המחיר מתחת לכל הממוצעים – חולשה." if px < min(e9, e21, e50, e200) else
               f"המחיר בין הממוצעים: {'מעל' if px > e200 else 'מתחת'} לממוצע 200 ו{'מעל' if px > e21 else 'מתחת'} לממוצע 21.")
    out.append(("ממוצעים נעים", ema_txt))
    out.append(("תמיכה / התנגדות", (f"תמיכה קרובה {near_sup:,.2f}" if near_sup else "אין תמיכה קרובה מתחת") +
               (f", התנגדות קרובה {near_res:,.2f}." if near_res else ", אין התנגדות קרובה מעל (שיא חדש).")))
    fl = fib["levels"]
    nearest = min(fl.items(), key=lambda kv: abs(kv[1] - px))
    out.append(("פיבונאצ'י", f"תנועה {'עולה' if fib['swing_up'] else 'יורדת'} {fib['low']:,.2f}–{fib['high']:,.2f}; "
                             f"המחיר ליד רמת {float(nearest[0]):.1%} ({nearest[1]:,.2f})."))
    out.append(("RSI", f"RSI {rv:.0f} – " + ("קנוי יתר, סיכון לתיקון." if rv > 70 else "מכור יתר, סיכוי לריבאונד." if rv < 30
                                            else "מומנטום חיובי." if rv > 50 else "מומנטום חלש.")))
    out.append(("MACD", "MACD מעל קו האות – מומנטום עולה." if m.iloc[-1] > s.iloc[-1] else "MACD מתחת לקו האות – מומנטום יורד."))
    bw = (bu.iloc[-1] - bl.iloc[-1]) / bm.iloc[-1]
    btxt = ("נגיעה ברצועה העליונה – מתוח למעלה." if px >= bu.iloc[-1] else "נגיעה ברצועה התחתונה – מתוח למטה." if px <= bl.iloc[-1]
            else "בתוך הרצועות.")
    out.append(("בולינגר", btxt + (" הרצועות צרות – צפויה תנועה חדה." if bw < 0.08 else "")))
    out.append(("סטיות", {"bear": "סטייה דובית: שיא חדש במחיר בלי שיא ב-RSI – אזהרה.",
                         "bull": "סטייה שורית: שפל חדש במחיר בלי שפל ב-RSI – סימן לחוזק.", None: "אין סטייה בולטת."}[div]))
    out.append(("פרופיל ווליום", f"אזור המסחר הכבד (POC) ב-{poc:,.2f} – המחיר {'מעליו (תומך)' if px > poc else 'מתחתיו (מתנגד)'}."))
    out.append(("טווחי זמן", f"יומי: {'עולה' if tr == 'up' else 'יורד' if tr == 'down' else 'דשדוש'} · "
                             f"שבועי: {'עולה' if wtrend == 'up' else 'יורד'}."))
    score = ((tr == "up") - (tr == "down")) + (px > e50) - (px < e50) + (m.iloc[-1] > s.iloc[-1]) * 2 - 1 + \
            (wtrend == "up") * 2 - 1 + {"bull": 1, "bear": -1, None: 0}[div]
    return {"items": out, "px": px, "atr": a, "rsi": rv, "near_sup": near_sup, "near_res": near_res, "sup": sup, "res": res,
            "fib": fib, "poc": poc, "trend": tr, "wtrend": wtrend, "score": float(score), "ema21": float(e21),
            "bb": (bu, bm, bl), "macd": (m, s, hst), "rsi_series": r}


def full_review(df: pd.DataFrame, ca: dict, direction: str, conf: float, min_conf: float = 55.0) -> dict:
    """Decisive plan. direction in Bullish/Bearish/Neutral from the model; conf 0-100.
    Long and short are mirror images: long buys pullback to support / breakout, short sells bounce into resistance / breakdown."""
    px, a = ca["px"], ca["atr"]
    hi1, lo1 = float(df.High.iloc[-1]), float(df.Low.iloc[-1])
    side = "LONG" if direction == "Bullish" and conf >= min_conf else "SHORT" if direction == "Bearish" and conf >= min_conf else "OUT"
    sgn = 1 if side == "LONG" else -1
    res = sorted(x for x in ca["res"] if x > px); sup = sorted((x for x in ca["sup"] if x < px), reverse=True)
    out = {"side": side, "conf": conf}
    if side == "OUT":
        out.update(heb="לא להיכנס", why_out=f"הכיוון לא ברור מספיק (ביטחון {conf:.0f}%, נדרש {min_conf:.0f}%+).",
                   long_trigger=max(hi1, res[0] if res else hi1) + 0.1 * a, short_trigger=min(lo1, sup[0] if sup else lo1) - 0.1 * a)
        return out
    if side == "LONG":
        zone_lo = max(sup[0] if sup and sup[0] > px - 1.5 * a else px - 0.7 * a, px - 1.0 * a)
        zone = (zone_lo, px)
        trigger = (f"כניסה בתיקון לאזור {zone[0]:,.2f}–{zone[1]:,.2f}, או בפריצה מעל {hi1:,.2f} (שיא היום) עם ווליום. "
                   "עדיף לחכות 15–30 דקות אחרי הפתיחה לאישור.")
        stop = min(zone[0] - 1.0 * a, (sup[1] if len(sup) > 1 and sup[1] > zone[0] - 2 * a else zone[0] - 1.0 * a) - 0.1 * a)
    else:
        zone_hi = min(res[0] if res and res[0] < px + 1.5 * a else px + 0.7 * a, px + 1.0 * a)
        zone = (px, zone_hi)
        trigger = (f"כניסה לשורט בריבאונד לאזור {zone[0]:,.2f}–{zone[1]:,.2f} (התנגדות), או בשבירה מתחת ל-{lo1:,.2f} (שפל היום) עם ווליום. "
                   "עדיף לחכות 15–30 דקות אחרי הפתיחה לאישור.")
        stop = max(zone[1] + 1.0 * a, (res[1] if len(res) > 1 and res[1] < zone[1] + 2 * a else zone[1] + 1.0 * a) + 0.1 * a)
    entry = sum(zone) / 2
    risk = abs(entry - stop)
    lv = res if side == "LONG" else sup
    tp1 = next((x for x in lv if abs(x - entry) >= 1.5 * risk), entry + sgn * 1.5 * risk)
    tp2 = next((x for x in lv if abs(x - entry) >= 2.8 * risk and x != tp1), entry + sgn * 3 * risk)
    shares = int(100 / risk) if risk > 0 else 0  # $100 risk = 1% of $10k
    out.update(heb={"LONG": "לונג", "SHORT": "שורט"}[side], zone=zone, entry=entry, trigger=trigger, stop=stop,
               invalid=f"התרחיש מבוטל ב{'סגירה מתחת' if side == 'LONG' else 'סגירה מעל'} ל-{stop:,.2f}.",
               tp1=tp1, tp2=tp2, rr1=abs(tp1 - entry) / risk, rr2=abs(tp2 - entry) / risk, risk_pct=risk / entry * 100,
               size=f"סיכון של 1% מהתיק: על כל $10,000 – {shares} מניות (סיכון ${risk:,.2f} למניה). "
                    "מימוש חצי ב-TP1 והזזת הסטופ לכניסה.")
    return out
