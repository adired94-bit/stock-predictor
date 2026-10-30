"""Candlestick + classic chart pattern detection (pure pandas/numpy, no TA-Lib), with Hebrew names
and per-ticker historical follow-through stats."""
from __future__ import annotations

import numpy as np
import pandas as pd

# name -> (Hebrew name, bias: +1 bullish / -1 bearish / 0 neutral, reliability, short Hebrew meaning)
CANDLES = {
    "doji": ("דוג'י", 0, "נמוכה", "היסוס – קונים ומוכרים בתיקו"),
    "hammer": ("פטיש", 1, "בינונית", "דחיית מחירים נמוכים אחרי ירידה – סימן להיפוך למעלה"),
    "inverted_hammer": ("פטיש הפוך", 1, "נמוכה", "ניסיון ראשון של קונים אחרי ירידה"),
    "hanging_man": ("איש תלוי", -1, "נמוכה", "פטיש בראש עלייה – אזהרה למוכרים"),
    "shooting_star": ("כוכב נופל", -1, "בינונית", "דחיית מחירים גבוהים אחרי עלייה"),
    "bull_engulfing": ("בליעה שורית", 1, "גבוהה", "נר ירוק בולע את האדום הקודם – קונים השתלטו"),
    "bear_engulfing": ("בליעה דובית", -1, "גבוהה", "נר אדום בולע את הירוק הקודם – מוכרים השתלטו"),
    "piercing": ("קו חודר", 1, "בינונית", "פתיחה נמוכה וסגירה מעל אמצע הנר האדום"),
    "dark_cloud": ("ענן כהה", -1, "בינונית", "פתיחה גבוהה וסגירה מתחת לאמצע הנר הירוק"),
    "morning_star": ("כוכב בוקר", 1, "גבוהה", "היפוך שלושה נרות מלמטה למעלה"),
    "evening_star": ("כוכב ערב", -1, "גבוהה", "היפוך שלושה נרות מלמעלה למטה"),
    "three_white_soldiers": ("שלושה חיילים לבנים", 1, "גבוהה", "שלושה נרות ירוקים חזקים ברצף"),
    "three_black_crows": ("שלושה עורבים שחורים", -1, "גבוהה", "שלושה נרות אדומים חזקים ברצף"),
    "bull_harami": ("הראמי שורי", 1, "נמוכה", "נר קטן בתוך נר אדום גדול – הירידה נחלשת"),
    "bear_harami": ("הראמי דובי", -1, "נמוכה", "נר קטן בתוך נר ירוק גדול – העלייה נחלשת"),
    "bull_marubozu": ("מרובוזו ירוק", 1, "בינונית", "נר ירוק מלא בלי פתילים – שליטה מלאה של קונים"),
    "bear_marubozu": ("מרובוזו אדום", -1, "בינונית", "נר אדום מלא בלי פתילים – שליטה מלאה של מוכרים"),
    "tweezer_top": ("פינצטה עליונה", -1, "נמוכה", "שני שיאים זהים – התנגדות"),
    "tweezer_bottom": ("פינצטה תחתונה", 1, "נמוכה", "שני שפלים זהים – תמיכה"),
    "spinning_top": ("סביבון", 0, "נמוכה", "גוף קטן ופתילים משני הצדדים – חוסר החלטה"),
}


def detect_candles(df: pd.DataFrame, trend_n: int = 5) -> pd.DataFrame:
    """Boolean frame (index=df.index, columns=CANDLES keys)."""
    o, h, l, c = (df[k].astype(float) for k in ("Open", "High", "Low", "Close"))
    rng = (h - l).replace(0, np.nan)
    body = (c - o).abs()
    up_w = h - np.maximum(o, c)
    lo_w = np.minimum(o, c) - l
    avg_body = body.rolling(14, min_periods=1).mean().shift(1).fillna(body)
    green, red = c > o, c < o
    o1, h1, l1, c1 = o.shift(1), h.shift(1), l.shift(1), c.shift(1)
    o2, c2 = o.shift(2), c.shift(2)
    b1, b2 = body.shift(1), body.shift(2)
    green1, red1, green2, red2 = green.shift(1, fill_value=False), red.shift(1, fill_value=False), green.shift(2, fill_value=False), red.shift(2, fill_value=False)
    down = c.shift(1) < c.shift(trend_n + 1)   # prior downtrend
    up = c.shift(1) > c.shift(trend_n + 1)
    small = body <= 0.1 * rng
    hammer_shape = (lo_w >= 2 * body) & (up_w <= 0.25 * rng) & (body > 0.05 * rng)
    inv_shape = (up_w >= 2 * body) & (lo_w <= 0.25 * rng) & (body > 0.05 * rng)
    big = body > avg_body
    p = {
        "doji": small,
        "hammer": hammer_shape & down,
        "inverted_hammer": inv_shape & down,
        "hanging_man": hammer_shape & up,
        "shooting_star": inv_shape & up,
        "bull_engulfing": green & red1 & (c >= o1) & (o <= c1) & (body > b1),
        "bear_engulfing": red & green1 & (o >= c1) & (c <= o1) & (body > b1),
        "piercing": green & red1 & (o < c1) & (c > (o1 + c1) / 2) & (c < o1) & (b1 > avg_body),
        "dark_cloud": red & green1 & (o > c1) & (c < (o1 + c1) / 2) & (c > o1) & (b1 > avg_body),
        "morning_star": red2 & (b2 > avg_body) & (b1 < 0.5 * b2) & green & (c > (o2 + c2) / 2),
        "evening_star": green2 & (b2 > avg_body) & (b1 < 0.5 * b2) & red & (c < (o2 + c2) / 2),
        "three_white_soldiers": green & green1 & green2 & (c > c1) & (c1 > c2) & big & (b1 > 0.6 * avg_body) & (b2 > 0.6 * avg_body)
                                & (up_w < 0.3 * body),
        "three_black_crows": red & red1 & red2 & (c < c1) & (c1 < c2) & big & (b1 > 0.6 * avg_body) & (b2 > 0.6 * avg_body)
                             & (lo_w < 0.3 * body),
        "bull_harami": red1 & (b1 > avg_body) & green & (o > c1) & (c < o1) & (body < 0.6 * b1),
        "bear_harami": green1 & (b1 > avg_body) & red & (o < c1) & (c > o1) & (body < 0.6 * b1),
        "bull_marubozu": green & (body >= 0.95 * rng) & big,
        "bear_marubozu": red & (body >= 0.95 * rng) & big,
        "tweezer_top": up & green1 & red & ((h - h1).abs() <= 0.002 * h),
        "tweezer_bottom": down & red1 & green & ((l - l1).abs() <= 0.002 * l),
        "spinning_top": ~small & (body <= 0.3 * rng) & (up_w > body) & (lo_w > body),
    }
    return pd.DataFrame({k: v.fillna(False).astype(bool) for k, v in p.items()}, index=df.index)


def candle_table(df: pd.DataFrame, lookback: int = 10, horizon: int = 5) -> pd.DataFrame:
    """Latest patterns (last `lookback` bars) + this ticker's historical 5-day follow-through."""
    flags = detect_candles(df)
    fwd = df["Close"].shift(-horizon) / df["Close"] - 1
    rows = []
    recent = flags.tail(lookback)
    for k in flags.columns:
        hits = recent.index[recent[k]]
        if not len(hits):
            continue
        heb, bias, rel, meaning = CANDLES[k]
        hist = fwd[flags[k]].dropna()
        if bias == 0:
            win = np.nan  # neutral patterns have no direction to score
        else:
            win = ((hist * bias) > 0).mean() if len(hist) else np.nan
        rows.append({"תאריך": hits[-1].date(), "תבנית": heb, "כיוון": {1: "שורי ▲", -1: "דובי ▼", 0: "ניטרלי"}[bias],
                     "אמינות": rel, "משמעות": meaning, "הצלחה היסטורית (5 ימים)": win, "מדגם": len(hist),
                     "תשואה ממוצעת 5 ימים": hist.mean() if len(hist) else np.nan, "_key": k, "_bias": bias})
    t = pd.DataFrame(rows)
    return t.sort_values("תאריך", ascending=False) if len(t) else t


# ---------------------------------------------------------------- classic chart patterns
def _pivots(s: pd.Series, order: int, kind: str) -> list[int]:
    a = s.to_numpy(float)
    f = np.greater_equal if kind == "high" else np.less_equal
    return [i for i in range(order, len(a) - order)
            if f(a[i], a[i - order:i + order + 1]).all()]


def chart_patterns(df: pd.DataFrame, lookback: int = 120, order: int = 5, tol: float = 0.03) -> list[dict]:
    """Double top/bottom, head & shoulders (and inverse), triangles, flags, range breakouts. Hebrew names."""
    d = df.tail(lookback)
    h, l, c = d["High"], d["Low"], d["Close"]
    px = float(c.iloc[-1])
    out = []
    ph, pl = _pivots(h, order, "high"), _pivots(l, order, "low")
    if len(ph) >= 2:
        a, b = h.iloc[ph[-2]], h.iloc[ph[-1]]
        neck = l.iloc[ph[-2]:ph[-1] + 1].min()
        if abs(a - b) / max(a, b) < tol and min(a, b) - neck > 0.03 * px:
            st = "אושרה (שבירה מתחת לצוואר)" if px < neck else "בהתהוות"
            out.append({"תבנית": "פסגה כפולה", "כיוון": "דובי ▼", "סטטוס": st, "רמה": round(float(neck), 2),
                        "הסבר": f"שני שיאים ליד {max(a, b):,.2f}; שבירה מתחת ל-{neck:,.2f} מאשרת ירידה"})
    if len(pl) >= 2:
        a, b = l.iloc[pl[-2]], l.iloc[pl[-1]]
        neck = h.iloc[pl[-2]:pl[-1] + 1].max()
        if abs(a - b) / max(a, b) < tol and neck - max(a, b) > 0.03 * px:
            st = "אושרה (פריצה מעל הצוואר)" if px > neck else "בהתהוות"
            out.append({"תבנית": "תחתית כפולה", "כיוון": "שורי ▲", "סטטוס": st, "רמה": round(float(neck), 2),
                        "הסבר": f"שני שפלים ליד {min(a, b):,.2f}; פריצה מעל {neck:,.2f} מאשרת עלייה"})
    if len(ph) >= 3:
        ls, hd, rs = (h.iloc[i] for i in ph[-3:])
        if hd > ls * 1.02 and hd > rs * 1.02 and abs(ls - rs) / hd < tol:
            neck = l.iloc[ph[-3]:ph[-1] + 1].min()
            out.append({"תבנית": "ראש וכתפיים", "כיוון": "דובי ▼", "סטטוס": "אושרה" if px < neck else "בהתהוות",
                        "רמה": round(float(neck), 2), "הסבר": f"ראש {hd:,.2f} בין שתי כתפיים; קו צוואר {neck:,.2f}"})
    if len(pl) >= 3:
        ls, hd, rs = (l.iloc[i] for i in pl[-3:])
        if hd < ls * 0.98 and hd < rs * 0.98 and abs(ls - rs) / ls < tol:
            neck = h.iloc[pl[-3]:pl[-1] + 1].max()
            out.append({"תבנית": "ראש וכתפיים הפוך", "כיוון": "שורי ▲", "סטטוס": "אושרה" if px > neck else "בהתהוות",
                        "רמה": round(float(neck), 2), "הסבר": f"שפל {hd:,.2f} בין שתי כתפיים; קו צוואר {neck:,.2f}"})
    # triangles from last 3 pivot highs/lows slopes
    if len(ph) >= 2 and len(pl) >= 2:
        sh = (h.iloc[ph[-1]] - h.iloc[ph[-2]]) / h.iloc[ph[-2]]
        sl = (l.iloc[pl[-1]] - l.iloc[pl[-2]]) / l.iloc[pl[-2]]
        name = None
        if abs(sh) < 0.015 and sl > 0.02:
            name, bias = "משולש עולה", "שורי ▲"
        elif sh < -0.02 and abs(sl) < 0.015:
            name, bias = "משולש יורד", "דובי ▼"
        elif sh < -0.02 and sl > 0.02:
            name, bias = "משולש סימטרי", "ניטרלי (לפי כיוון הפריצה)"
        if name:
            out.append({"תבנית": name, "כיוון": bias, "סטטוס": "בהתהוות", "רמה": round(float(h.iloc[ph[-1]]), 2),
                        "הסבר": f"התכנסות בין {l.iloc[pl[-1]]:,.2f} ל-{h.iloc[ph[-1]]:,.2f}"})
    # flags: strong 10-bar pole then tight 5-bar consolidation
    if len(c) > 20:
        pole = c.iloc[-6] / c.iloc[-16] - 1
        cons = (h.iloc[-5:].max() - l.iloc[-5:].min()) / px
        if abs(pole) > 0.10 and cons < abs(pole) / 2:
            out.append({"תבנית": "דגל שורי" if pole > 0 else "דגל דובי", "כיוון": "שורי ▲" if pole > 0 else "דובי ▼",
                        "סטטוס": "בהתהוות", "רמה": round(float(h.iloc[-5:].max() if pole > 0 else l.iloc[-5:].min()), 2),
                        "הסבר": f"תנועה חדה של {pole:+.0%} ואחריה דשדוש צר"})
    # range breakout (20 bars)
    hi20, lo20 = float(h.iloc[-21:-1].max()), float(l.iloc[-21:-1].min())
    if px > hi20:
        out.append({"תבנית": "פריצה מטווח", "כיוון": "שורי ▲", "סטטוס": "פעילה", "רמה": round(hi20, 2),
                    "הסבר": f"סגירה מעל שיא 20 ימים ({hi20:,.2f})"})
    elif px < lo20:
        out.append({"תבנית": "שבירה מטווח", "כיוון": "דובי ▼", "סטטוס": "פעילה", "רמה": round(lo20, 2),
                    "הסבר": f"סגירה מתחת לשפל 20 ימים ({lo20:,.2f})"})
    return out
