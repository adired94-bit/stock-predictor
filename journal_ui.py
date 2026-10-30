"""Streamlit page for the prediction journal (Hebrew)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import journal as J


def store():
    if "jstore" not in st.session_state:
        try:
            sec = st.secrets if len(st.secrets) else None
        except Exception:
            sec = None
        st.session_state["jstore"] = J.get_store(sec)
    return st.session_state["jstore"]


def load() -> pd.DataFrame:
    try:
        return store().load()
    except Exception as e:
        st.error(f"טעינת היומן נכשלה ({type(e).__name__}). בדוק את ה-Secrets.")
        return pd.DataFrame(columns=J.COLS)


def save_prediction(ticker, price, horizons_df, plan):
    try:
        s = store(); df = s.load()
        new = J.make_entries(ticker, price, horizons_df, plan)
        s.save(pd.concat([df, new], ignore_index=True), f"add {ticker}")
        st.success(f"נשמרו {len(new)} חיזויים למעקב ✔")
    except Exception as e:
        st.error(f"השמירה נכשלה ({type(e).__name__}).")


def render_journal():
    st.markdown("## 📒 יומן חיזויים – האם החיזוי עבד?")
    s = store()
    if s.kind == "local":
        st.warning("היומן נשמר זמנית בשרת בלבד ועלול להימחק באתחול. כדי לשמור לתמיד – הוסף GITHUB_TOKEN ב-Secrets (ראה README). "
                   "בינתיים אפשר להוריד CSV ולהעלות אותו שוב.")
        up = st.file_uploader("העלאת יומן (CSV)", type="csv")
        if up is not None and st.button("טען את הקובץ ליומן"):
            s.save(J._norm(pd.read_csv(up)), "upload"); st.success("נטען.")
    df = load()
    if st.button("🔄 בדוק תוצאות עכשיו", type="primary", width="stretch"):
        with st.spinner("בודק מחירים בפועל…"):
            df, n = J.evaluate_all(df)
            if n:
                s.save(df, f"evaluate {n}")
        st.success(f"עודכנו {n} חיזויים." if n else "אין חיזויים שהבשילו עדיין.")
    if df.empty:
        st.info("עדיין אין חיזויים. נתח מניה ולחץ 'שמור למעקב'.")
        return
    stt = J.stats(df)
    c = st.columns(3)
    c[0].metric("חיזויים שנבדקו", stt["n"]); c[1].metric("הצלחה כוללת", f"{stt['success']:.0%}" if stt["n"] else "—",
                                                       help="עבד = 1, חלקית = 0.5, לא עבד = 0")
    c[2].metric("ממתינים", int((df["status"] == "pending").sum()))
    if 0 < stt["n"] < 30:
        st.caption("⚠️ פחות מ-30 חיזויים שנבדקו – אחוז ההצלחה עוד לא אמין.")
    if stt["n"]:
        fmt = {"success": "{:.0%}", "avg_ret": "{:+.2%}"}
        ren = {"n": "מדגם", "success": "הצלחה", "avg_ret": "תשואת מחיר ממוצעת"}
        t1, t2, t3, t4 = st.tabs(["לפי מניה", "לפי טווח", "לונג מול שורט", "גרף דיוק"])
        t1.dataframe(stt["by_ticker"].style.format(fmt).relabel_index(list(ren.values()), axis=1), width="stretch")
        t2.dataframe(stt["by_horizon"].style.format(fmt).relabel_index(list(ren.values()), axis=1), width="stretch")
        t3.dataframe(stt["by_side"].style.format(fmt).relabel_index(list(ren.values()), axis=1), width="stretch")
        cv = stt["curve"]
        f = go.Figure(go.Scatter(x=pd.to_datetime(cv["evaluated"]), y=cv["cum"], mode="lines+markers"))
        f.update_layout(template="plotly_dark", height=300, yaxis_tickformat=".0%", margin=dict(l=5, r=5, t=10, b=5))
        t4.plotly_chart(f, width="stretch")
    f1, f2, f3 = st.columns(3)
    tk = f1.multiselect("מניה", sorted(df["ticker"].dropna().unique()))
    hz = f2.multiselect("טווח", sorted(df["horizon"].dropna().unique()))
    dr = f3.multiselect("כיוון", sorted(df["direction"].dropna().astype(str).unique()))
    v = df.copy()
    if tk: v = v[v.ticker.isin(tk)]
    if hz: v = v[v.horizon.isin(hz)]
    if dr: v = v[v.direction.astype(str).isin(dr)]
    show = pd.DataFrame({"מניה": v.ticker, "נוצר": pd.to_datetime(v.created, utc=True).dt.tz_convert("Asia/Jerusalem").dt.strftime("%d/%m %H:%M"),
                         "טווח": v.horizon, "כיוון": v.direction, "מחיר": v.price.round(2), "סטופ": v.stop.round(2), "יעד": v.target.round(2),
                         "בפועל": v.actual.round(2), "תשואה": (v.ret * 100).round(2).astype(str) + "%",
                         "האם עבד?": v.status.map(J.HEB), "פירוט": v.result})
    st.dataframe(show.iloc[::-1], width="stretch", hide_index=True)
    st.download_button("⬇️ הורד CSV", df.to_csv(index=False).encode("utf-8-sig"), "journal.csv", "text/csv", width="stretch")
