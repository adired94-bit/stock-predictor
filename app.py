"""Stock Movement Prediction dashboard (Streamlit). Run: streamlit run app.py

NOT FINANCIAL ADVICE. All confidence numbers are heuristic unless shown as "calibrated" (backtest-mapped).
"""
from __future__ import annotations

import datetime as dt
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import fetch_ohlcv, synthetic_ohlcv  # noqa: E402
from indicators import add_indicators, volume_profile  # noqa: E402
from patterns import find_similar  # noqa: E402
from sentiment import fetch_news, fetch_analyst, analyst_score, score_news_v2, finbert_available, fetch_info, ownership_short  # noqa: E402
from macro import fetch_macro, MacroState  # noqa: E402
from risk import monte_carlo, cone, var_cvar, trade_plan, risk_signal  # noqa: E402
from predictor import manager, pattern_signal, technical_signal, sentiment_macro_signal, MANAGER_WEIGHTS, ensemble  # noqa: E402
from backtest import walk_forward, calibrate, adjusted_weights  # noqa: E402
from extras import (fetch_options, fetch_intraday, intraday_analysis, fetch_events, event_risk,  # noqa: E402
                    default_peers, peer_stats, gap_analysis, data_quality)

st.set_page_config(page_title="חיזוי מניות", page_icon="📈", layout="wide")
DARK = "plotly_dark"
st.markdown("""<style>
.main .block-container, section.main, [data-testid="stMainBlockContainer"], [data-testid="stSidebar"] {direction: rtl; text-align: right;}
.js-plotly-plot, .stPlotlyChart, [data-testid="stDataFrame"], code {direction: ltr; text-align: left;}
.rtl{direction:rtl;text-align:right}.up{color:#22c55e}.dn{color:#ef4444}
div.stButton>button, div.stFormSubmitButton>button{min-height:2.8rem;font-size:1rem}
</style>""", unsafe_allow_html=True)
SMALL_N = 8

HELP = {
    "הפסד יומי מקסימלי (95%)": "ב-95% מהימים ההפסד לא אמור לעבור את זה.",
    "הפסד ממוצע בימים הגרועים": "הפסד ממוצע ב-5% הימים הגרועים.",
    "ATR(14)": "טווח יומי ממוצע – התנועה הרגילה ביום.",
    "סיכון:סיכוי": "רווח חלקי סיכון. מעל 2 = אטרקטיבי.",
    "ציון סיכון": "1- = מסוכן, 1+ = פרופיל נוח.",
    "דירוג ממוצע (1=קנייה חזקה)": "דירוג ממוצע: 1 קנייה חזקה … 5 מכירה.",
    "אנליסטים": "מספר אנליסטים שמכסים את המניה.",
    "יעד ממוצע": "יעד ממוצע ל-12 חודשים (לא תחזית קצרה).",
    "יעד נמוך/גבוה": "היעד הנמוך והגבוה.",
    "שורט % · מוסדיים %": "אחוז המניות בשורט; אחזקת מוסדיים.",
    "תנודתיות גלומה": "תנודתיות גלומה שנתית באופציות.",
    "תנועה צפויה": "התנועה ששוק האופציות מתמחר עד הפקיעה.",
    "מקס כאב": "המחיר שבו מחזיקי האופציות מפסידים הכי הרבה.",
    "אחוז הצלחה": "אחוז התחזיות שצדקו אחרי 5 ימים.",
    "ממוצע 5 ימים אחרי עלייה": "תשואה ממוצעת אחרי תחזית עלייה.",
    "ממוצע 5 ימים אחרי ירידה": "תשואה ממוצעת אחרי תחזית ירידה (שלילי = טוב).",
    "אסטרטגיה": "תשואה כוללת לפי האותות (בלי עמלות).",
    "קנה והחזק": "תשואה מהחזקה פשוטה באותם תאריכים.",
}
from streamlit.delta_generator import DeltaGenerator as _DG  # noqa: E402
_orig_metric = _DG.metric
def _metric(self, label, *a, **k):  # inject help text for every metric
    k.setdefault("help", HELP.get(label))
    return _orig_metric(self, label, *a, **k)
_DG.metric = _metric

from home import render_home, ticker_news  # noqa: E402
from review import chart_analysis, full_review  # noqa: E402
from horizons import multi_horizon, fetch_hourly, to_display, market_open_now  # noqa: E402
from journal_ui import render_journal, save_prediction  # noqa: E402
from candles import CANDLES, detect_candles, candle_table, chart_patterns  # noqa: E402

st.session_state.setdefault("ticker_in", "MU")


@st.cache_data(ttl=300, show_spinner=False)
def fetch_hourly_c(t: str):
    return fetch_hourly(t)


def analyze_ticker(t: str):
    """Tap-to-analyze callback from the home dashboard (runs before widgets are rebuilt)."""
    st.session_state["ticker_in"] = t
    st.session_state["analyzed"] = True
    st.session_state["auto_run"] = True
    st.session_state["page"] = "main"


# ============================================================== sidebar
with st.sidebar:
    st.title("📈 חיזוי מניות")
    st.caption("כתוב סימול מניה ולחץ **נתח**. כל השאר מוגדר מראש.")
    with st.form("controls"):
        ticker = st.text_input("סימול מניה", key="ticker_in", help="למשל MU, NVDA, SPY, BTC-USD").strip().upper()
        today = dt.date.today()
        years = st.select_slider("היסטוריה", ["1y", "2y", "5y", "10y"], "5y", help="כמה אחורה לחפש דפוסים דומים")
        with st.expander("הגדרות מתקדמות"):
            c1, c2 = st.columns(2)
            start = c1.date_input("התחלה", None, help="ריק = לפי היסטוריה")
            end = c2.date_input("סיום", today)
            window = st.slider("חלון דפוס (ימים)", 10, 60, 20, help="אורך הדפוס האחרון להשוואה")
            top_k = st.slider("מספר התאמות", 3, 30, 10, help="כמה דפוסים דומים לקחת")
            method = st.selectbox("שיטת דמיון", ["cosine", "dtw"], help="Cosine מהיר; DTW גמיש יותר (איטי)")
            mc_method = st.selectbox("Monte Carlo", ["bootstrap", "gbm"], help="Bootstrap = תשואות אמיתיות מהעבר; GBM = התפלגות נורמלית")
            n_paths = st.select_slider("מספר סימולציות", [2000, 5000, 10000, 20000], 5000)
            use_finbert = st.checkbox("FinBERT (אם מותקן)", value=False, disabled=not finbert_available(),
                                      help="pip install transformers torch to enable")
            peers_txt = st.text_input("מניות מקבילות (פסיק; ריק = אוטומטי)", "")
            run_bt = st.checkbox("הרץ בדיקה לאחור", True, help="בודק את המודל על העבר כדי לכייל את הביטחון")
            auto_w = st.checkbox("הורד משקל לרכיבים חלשים", True)
            demo = st.checkbox("מצב הדגמה (נתונים סינתטיים)", False)
        go_btn = st.form_submit_button("🔍 נתח", type="primary", width="stretch")
    j1, j2 = st.columns(2)
    if j1.button("🏠 בית", width="stretch"):
        st.session_state["page"] = "main"; st.session_state["analyzed"] = False
    if j2.button("📒 יומן", width="stretch"):
        st.session_state["page"] = "journal"
    if start is None:
        start = end - dt.timedelta(days=365 * int(years[:-1]))
    st.caption("משקלות: דפוסים 50% · סנטימנט ומאקרו 30% · סיכון 20%")

st.caption("⚠️ כלי לימודי – **לא ייעוץ השקעות**.")

# ============================================================== data
if go_btn:
    st.session_state["analyzed"] = True
    st.session_state["page"] = "main"
if st.session_state.get("page") == "journal":
    render_journal()
    st.stop()
if st.session_state.pop("auto_run", False):
    go_btn = True
if not st.session_state.get("analyzed"):
    st.info("👆 לחץ על **>>** למעלה כדי לפתוח את התפריט, כתוב סימול ולחץ **נתח** – או לחץ **נתח** ליד כל מניה למטה.")
    render_home(analyze_ticker)
    st.stop()
if st.button("🏠 חזרה למסך הבית", key="back_home"):
    st.session_state["analyzed"] = False
    st.rerun()
st.markdown(f"## 🔍 {ticker}")
notes: list[str] = []
if demo:
    df = synthetic_ohlcv(1260, seed=7)
else:
    df, err = fetch_ohlcv(ticker, start, end)
    if err:
        st.error(f"לא נמצאו נתונים לסימול הזה ({err}). בדוק את הסימול.")
        st.stop()
if len(df) < window * 3 + 40:
    st.error("אין מספיק היסטוריה – הרחב את טווח התאריכים."); st.stop()

ind = add_indicators(df)
last = float(df["Close"].iloc[-1])
offline = demo

with st.spinner("מחפש דפוסים דומים…"):
    pr = find_similar(df, window, top_k, method)
ps = pattern_signal(pr.summary)
tech = technical_signal(ind)

info = {} if offline else fetch_info(ticker)
own = ownership_short(info)
news_items, news_err = ([], "demo") if offline else fetch_news(ticker)
news_df, news_score, engine = score_news_v2(news_items, use_finbert)
an, an_err = ({}, "demo") if offline else fetch_analyst(ticker)
a_score = analyst_score(an.get("recommendation_mean"), an.get("target_mean"), an.get("current_price") or last)
macro = MacroState("Unknown", None, 0.0, notes=["demo"]) if offline else fetch_macro(end)
opt, opt_err = ({}, "demo") if offline else fetch_options(ticker, last)
peers = [p.strip().upper() for p in peers_txt.split(",") if p.strip()] or default_peers(ticker, own.get("sector"))
peer_series = {} if offline else {p: s["Close"] for p in peers if len(s := fetch_ohlcv(p, start, end)[0])}
peer_tbl, peer_sig = peer_stats(df["Close"], peer_series)
events, ev_err = ({}, "demo") if offline else fetch_events(ticker)
ev_hits, ev_mult = event_risk(events, today)
gaps = gap_analysis(df)

# short interest tilt: high short % -> slight bearish sentiment (and squeeze risk noted)
si = own.get("short_pct_float") or 0
si_sig = -float(np.clip((si - 0.05) / 0.15, 0, 1)) if si else 0.0
news_blend = float(np.clip(0.8 * news_score + 0.2 * si_sig, -1, 1))
macro_sig = float(np.clip(0.6 * macro.score + 0.4 * peer_sig, -1, 1))
sent_macro = sentiment_macro_signal(news_blend, a_score, macro_sig, macro.sentiment_mult)
if opt:
    sent_macro = float(np.clip(0.8 * sent_macro + 0.2 * opt["signal"], -1, 1))

paths = monte_carlo(df["Close"], 30, n_paths, mc_method)
mc = cone(paths)
var95, es95 = var_cvar(paths, 1)
side = "long" if (0.7 * ps + 0.3 * tech) >= 0 else "short"
plan = trade_plan(df, side, extra_levels=[volume_profile(df.tail(250))[1]])
rsk = risk_signal(paths, var95, plan.rr)
if opt and opt.get("atm_iv"):  # implied vs realised vol: rich IV penalises risk score
    rv = float(np.log(df.Close).diff().tail(20).std() * np.sqrt(252))
    rsk = float(np.clip(rsk - 0.3 * np.clip(opt["atm_iv"] / max(rv, 1e-6) - 1.2, 0, 1), -1, 1))

# ============================================================== backtest & manager
bt = None
weights = dict(MANAGER_WEIGHTS)
if run_bt:
    with st.spinner("בודק את המודל על העבר…"):
        bt = walk_forward(df, window, top_k, step=5, n_tests=150)
    if auto_w and bt.component_hit:
        mult = adjusted_weights(bt.component_hit)
        tm = 0.7 * mult.get("pattern", 1) + 0.3 * mult.get("technical", 1)
        weights["Technical Pattern Similarity"] *= tm
        weights["Risk/Volatility"] *= mult.get("risk", 1)

pred, rationale = manager(ps, tech, sent_macro, rsk, pr.summary, mc, last, macro.regime)
if weights != MANAGER_WEIGHTS:  # recompute with adjusted weights, keep targets
    p2 = ensemble(pred.components, weights, pred.targets)
    rationale += f"\n- Auto-weights from backtest: { {k: round(v, 2) for k, v in weights.items()} }"
    pred = p2
raw_conf = pred.confidence
cal_conf, cal_n = calibrate(raw_conf, bt.calibration) if bt is not None else (raw_conf, 0)
final_conf = 50 + (cal_conf - 50) * ev_mult
if ev_hits:
    rationale += f"\n- Event risk within 5 days ({', '.join(e['event'] for e in ev_hits)}): confidence shrunk ×{ev_mult:.2f}."

# ============================================================== main card (Hebrew, simple)
ca = chart_analysis(df)
rv_ = full_review(df, ca, pred.direction, final_conf)
ctab = candle_table(df, lookback=5)
cpat = chart_patterns(df)
name_ = "DEMO" if demo else ticker
SIDE_COL = {"LONG": "#16a34a", "SHORT": "#dc2626", "OUT": "#64748b"}[rv_["side"]]
if rv_["side"] == "OUT":
    rows_html = (f"<div>⏸️ <b>מה עושים</b><br>מחכים. {rv_['why_out']}</div>"
                 f"<div>🟢 <b>לונג רק מעל</b><br>{rv_['long_trigger']:,.2f}</div>"
                 f"<div>🔴 <b>שורט רק מתחת</b><br>{rv_['short_trigger']:,.2f}</div>")
    when_ = f"לונג רק בסגירה מעל {rv_['long_trigger']:,.2f}, שורט רק בשבירה מתחת ל-{rv_['short_trigger']:,.2f}."
else:
    rows_html = (f"<div>💵 <b>כניסה</b><br>{rv_['zone'][0]:,.2f}–{rv_['zone'][1]:,.2f}</div>"
                 f"<div>🛑 <b>סטופ</b><br>{rv_['stop']:,.2f} ({rv_['risk_pct']:.1f}%)</div>"
                 f"<div>🎯 <b>יעד 1</b><br>{rv_['tp1']:,.2f} (1:{rv_['rr1']:.1f})</div>"
                 f"<div>🏁 <b>יעד 2</b><br>{rv_['tp2']:,.2f} (1:{rv_['rr2']:.1f})</div>")
    when_ = rv_["trigger"]
candle_line = ""
if len(ctab):
    c0 = ctab.iloc[0]
    candle_line = f"<div style='font-size:14px;opacity:.85;margin-top:8px'>🕯️ נר אחרון: {c0['תבנית']} ({c0['כיוון']}) – {c0['משמעות']}</div>"
st.markdown(f"""
<div class="rtl" style="border-radius:16px;padding:18px 20px;background:linear-gradient(135deg,{SIDE_COL}33,#0f172a);border:2px solid {SIDE_COL}">
<div style="font-size:14px;opacity:.8"><span dir="ltr">{name_} · {last:,.2f}</span></div>
<div style="font-size:38px;font-weight:800;color:{SIDE_COL}">{rv_['heb']} · {final_conf:.0f}% ביטחון</div>
<div style="font-size:16px;margin:6px 0 12px">⏱️ <b>מתי להיכנס:</b> {when_}</div>
<div style="display:flex;gap:22px;flex-wrap:wrap;font-size:16px">{rows_html}</div>
{candle_line}
</div>""", unsafe_allow_html=True)
if rv_["side"] != "OUT":
    st.caption(rv_["invalid"] + " " + rv_["size"])
st.caption("ביטחון 50% = הטלת מטבע. זו הערכה סטטיסטית, לא הבטחה ולא ייעוץ השקעות.")
if ev_hits:
    st.warning("⚠️ אירוע קרוב: " + ", ".join(f"{e['event']} {e['date']}" for e in ev_hits))

# ---- multi-horizon table
st.markdown("### ⏱️ תחזית לפי טווחי זמן")
try:
    hourly = None if demo else fetch_hourly_c(ticker)
    mh = multi_horizon(df["Close"], hourly)
    st.dataframe(to_display(mh), width="stretch", hide_index=True)
    st.caption("שעתי ו-4 שעות = אמינות נמוכה. 'דיוק בעבר' = בדיקה לאחור על המניה הזו (n = מספר בדיקות)."
               + ("" if market_open_now() else " השוק סגור עכשיו – התחזית השעתית מתייחסת למסחר הבא."))
except Exception as e_:
    mh = None
    st.info(f"תחזית לפי טווחים לא זמינה ({type(e_).__name__}).")
if not demo and st.button("💾 שמור למעקב", width="stretch"):
    save_prediction(ticker, last, mh if mh is not None else pd.DataFrame(columns=["טווח"]), rv_)

colA, colB = st.columns(2)
with colA:
    st.markdown("### 📊 נתוני אמת")
    try:
        prev = float(df["Close"].iloc[-2]); vol = float(df["Volume"].iloc[-1]); avgv = float(df["Volume"].tail(21).iloc[:-1].mean())
        y = df.tail(252)
        lines = [f"מחיר: <b dir='ltr'>{last:,.2f}</b> (<span class='{'up' if last >= prev else 'dn'}' dir='ltr'>{last / prev - 1:+.2%}</span> היום)",
                 f"ווליום: <span dir='ltr'>{vol / avgv:.1f}×</span> מהממוצע" if avgv else "ווליום: אין נתון",
                 f"טווח 52 שבועות: <span dir='ltr'>{y.Low.min():,.2f} – {y.High.max():,.2f}</span>"]
        if an.get("n_analysts"):
            tm = an.get("target_mean")
            lines.append(f"אנליסטים ({an['n_analysts']}): {an.get('recommendation_key') or ''}" +
                         (f", יעד ממוצע <span dir='ltr'>{tm:,.2f} ({tm / last - 1:+.0%})</span>" if tm else ""))
        if events.get("earnings"):
            lines.append(f"דוח רבעוני הבא: <span dir='ltr'>{events['earnings']}</span>")
        st.markdown("<br>".join(lines), unsafe_allow_html=True)
        hd_ = news_df if not news_df.empty else (pd.DataFrame() if demo else ticker_news((ticker,), 3))
        if not hd_.empty:
            st.markdown("**כותרות אחרונות:**")
            for r_ in hd_.sort_values("published", ascending=False).head(3).itertuples():
                st.markdown(f"- [{r_.title}]({r_.link})" if getattr(r_, "link", "") else f"- {r_.title}")
    except Exception as e_:
        st.info(f"חלק מהנתונים לא זמינים כרגע ({type(e_).__name__}).")
with colB:
    st.markdown("### 🔮 חיזוי האפליקציה")
    try:
        lines = []
        for h_, lbl in ((5, "5 ימים"), (10, "10 ימים"), (30, "30 ימים")):
            row_ = mc[mc.horizon == h_]
            if len(row_):
                r0 = row_.iloc[0]
                lines.append(f"{lbl}: טווח <span dir='ltr'>{r0.p5:,.2f} – {r0.p95:,.2f}</span> · סיכוי לעלייה {r0.prob_up:.0%}")
        if pr.summary and 5 in pr.summary:
            s5 = pr.summary[5]
            lines.append(f"דפוסים דומים בעבר: {len(pr.matches)} · עלו אחרי 5 ימים ב-{s5['win_rate']:.0%} מהמקרים"
                         + (" (מדגם קטן)" if len(pr.matches) < SMALL_N else ""))
        lines.append(f"כיוון המודל: {({'Bullish': 'עולה', 'Bearish': 'יורד', 'Neutral': 'ניטרלי'})[pred.direction]} · מצב שוק: {macro.regime}")
        if bt is not None and bt.stats.get("n_signals"):
            lines.append(f"דיוק המודל בעבר על המניה: {bt.stats['hit_rate']:.0%} (מדגם {bt.stats['n_signals']})")
        st.markdown("<br>".join(lines), unsafe_allow_html=True)
    except Exception as e_:
        st.info(f"החיזוי חלקי ({type(e_).__name__}).")

st.markdown("#### 🔎 פרטים מלאים (לחץ לפתיחה)")
with st.expander("📈 ניתוח גרף ונרות יפניים"):
    try:
        for t_, s_ in ca["items"]:
            st.markdown(f"**{t_}:** {s_}")
        view2 = df.tail(180)
        fl = detect_candles(df).tail(180)
        bu_, bm_, bl_ = (x.tail(180) for x in ca["bb"])
        f2 = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03)
        f2.add_trace(go.Candlestick(x=view2.index, open=view2.Open, high=view2.High, low=view2.Low, close=view2.Close, name="מחיר"), 1, 1)
        f2.add_trace(go.Scatter(x=view2.index, y=bu_, line=dict(width=1, color="#64748b"), name="בולינגר"), 1, 1)
        f2.add_trace(go.Scatter(x=view2.index, y=bl_, line=dict(width=1, color="#64748b"), showlegend=False), 1, 1)
        f2.add_trace(go.Scatter(x=view2.index, y=view2.Close.ewm(span=21).mean(), line=dict(width=1.2, color="#22d3ee"), name="EMA21"), 1, 1)
        for k_, v_ in ca["fib"]["levels"].items():
            f2.add_hline(y=v_, line_dash="dot", line_color="#eab308", opacity=0.5, annotation_text=f"Fib {k_}", row=1, col=1)
        for lv_ in [ca["near_sup"], ca["near_res"]]:
            if lv_:
                f2.add_hline(y=lv_, line_color="#94a3b8", row=1, col=1)
        for k_ in fl.columns:
            idx_ = fl.index[fl[k_]]
            if len(idx_) and CANDLES[k_][1] != 0:
                bias_ = CANDLES[k_][1]
                yv = view2.loc[idx_, "Low"] * 0.98 if bias_ > 0 else view2.loc[idx_, "High"] * 1.02
                f2.add_trace(go.Scatter(x=idx_, y=yv, mode="markers", name=CANDLES[k_][0], hovertext=CANDLES[k_][3],
                                        marker=dict(symbol="triangle-up" if bias_ > 0 else "triangle-down", size=9,
                                                    color="#22c55e" if bias_ > 0 else "#ef4444")), 1, 1)
        f2.add_trace(go.Scatter(x=view2.index, y=ca["rsi_series"].tail(180), name="RSI", line=dict(color="#a78bfa")), 2, 1)
        m_, s_, h_ = (x.tail(180) for x in ca["macd"])
        f2.add_trace(go.Bar(x=view2.index, y=h_, name="MACD", marker_color="#38bdf8"), 3, 1)
        f2.update_layout(template=DARK, height=750, xaxis_rangeslider_visible=False, showlegend=False, margin=dict(l=5, r=5, t=10, b=5))
        st.plotly_chart(f2, width="stretch")
        st.markdown("**תבניות נרות אחרונות (10 ימים) והביצועים שלהן בעבר על המניה:**")
        ct10 = candle_table(df, lookback=10)
        if len(ct10):
            show = ct10.drop(columns=["_key", "_bias"])
            st.dataframe(show.style.format({"הצלחה היסטורית (5 ימים)": "{:.0%}", "תשואה ממוצעת 5 ימים": "{:+.2%}"}, na_rep="—"),
                         width="stretch", hide_index=True)
            st.caption("מדגם מתחת ל-10 = סטטיסטיקה לא אמינה.")
        else:
            st.caption("לא זוהו תבניות נרות ב-10 הימים האחרונים.")
        st.markdown("**תבניות גרף קלאסיות:**")
        if cpat:
            st.dataframe(pd.DataFrame(cpat), width="stretch", hide_index=True)
        else:
            st.caption("לא זוהתה תבנית גרף קלאסית כרגע.")
    except Exception as e_:
        st.info(f"ניתוח הגרף לא זמין ({type(e_).__name__}).")

with st.expander("🧮 כל הניתוחים המתקדמים"):
    tabs = st.tabs(["גרף ודפוסים", "חיזוי", "סיכון ומונטה קרלו", "סנטימנט ומאקרו", "אופציות · אירועים · מקבילות · גאפים",
                    "תוך-יומי", "בדיקה לאחור", "איכות נתונים"])

    # ---------------------------------------------------------------- chart
    with tabs[0]:
        st.caption("💡 גרף מחיר עם ממוצעים. כתום = השבועות האחרונים; ירוק = תקופות בעבר שנראו דומות.")
        view = ind.tail(500)
        fig = make_subplots(rows=2, cols=2, shared_xaxes=False, column_widths=[0.85, 0.15], row_heights=[0.75, 0.25],
                            specs=[[{}, {}], [{}, None]], horizontal_spacing=0.01, vertical_spacing=0.03)
        fig.add_trace(go.Candlestick(x=view.index, open=view.Open, high=view.High, low=view.Low, close=view.Close, name="OHLC"), 1, 1)
        for n_, col in ((9, "#facc15"), (21, "#22d3ee"), (200, "#a78bfa")):
            fig.add_trace(go.Scatter(x=view.index, y=view[f"EMA{n_}"], name=f"EMA{n_}", line=dict(width=1.2, color=col)), 1, 1)
        fig.add_trace(go.Scatter(x=view.index, y=view.VWAP20, name="VWAP20", line=dict(width=1, dash="dot", color="white")), 1, 1)
        fig.add_trace(go.Bar(x=view.index, y=view.Volume, name="Volume", marker_color="#475569"), 2, 1)
        for _, r in pr.matches.iterrows():
            if r.start >= view.index[0]:
                fig.add_vrect(x0=r.start, x1=r.end, fillcolor="#22c55e", opacity=0.15, line_width=0, row=1, col=1)
        fig.add_vrect(x0=pr.current_window[0], x1=pr.current_window[1], fillcolor="#f97316", opacity=0.2, line_width=0, row=1, col=1)
        prof, poc = volume_profile(view)
        fig.add_trace(go.Bar(y=prof.price, x=prof.volume, orientation="h", marker_color="#64748b", name="Vol profile"), 1, 2)
        fig.add_hline(y=poc, line_dash="dash", line_color="#f43f5e", row=1, col=2)
        fig.update_yaxes(matches="y", row=1, col=2, showticklabels=False)
        fig.update_layout(template=DARK, height=700, xaxis_rangeslider_visible=False, legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Orange = current window; green = matched setups within view. POC ≈ {poc:,.2f}. "
                   "VWAP20 = rolling 20-day volume-weighted typical price (daily bars).")
        # overlay matched vs current
        ov = go.Figure()
        for _, r in pr.matches.iterrows():
            seg = df.Close.iloc[r.start_i: r.end_i + 11].to_numpy()
            ov.add_trace(go.Scatter(y=seg / seg[0] - 1, mode="lines", line=dict(width=1), opacity=0.5,
                                    name=f"{r.start:%Y-%m-%d} ({r.similarity:.2f})"))
        cur = df.Close.iloc[-window:].to_numpy()
        ov.add_trace(go.Scatter(y=cur / cur[0] - 1, line=dict(width=4, color="#f97316"), name="Current"))
        ov.add_vline(x=window - 1, line_dash="dash")
        ov.update_layout(template=DARK, height=380, title="דפוסים דומים מול הנוכחי (+10 ימים)", yaxis_tickformat=".0%")
        st.plotly_chart(ov, width="stretch")
        st.dataframe(pr.matches.drop(columns=["start_i", "end_i"]).style.format({"similarity": "{:.3f}", "fwd5": "{:+.2%}", "fwd10": "{:+.2%}"}), width="stretch")

    # ---------------------------------------------------------------- prediction
    with tabs[1]:
        st.caption("💡 למה זו ההמלצה: כל רכיב מצביע מ-1- (ירידה) עד 1+ (עלייה).")
        a, b = st.columns([1, 1])
        with a:
            st.markdown("### נימוקים"); st.markdown(rationale)
            st.markdown(f"**Confidence pipeline:** raw heuristic {raw_conf:.0f}% → calibrated {cal_conf:.0f}% (bucket n={cal_n}) → "
                        f"event-adjusted {final_conf:.0f}%.")
            st.caption("Raw confidence = 50 + 50·min(|S|/0.6,1)·agreement (S = weighted composite in [-1,1]; "
                       "agreement = weight share of components agreeing with S). Neutral if |S| ≤ 0.15. Heuristic, not a probability.")
            tg = pd.DataFrame([{"horizon": k, "low": v[0], "mid": v[1], "high": v[2]} for k, v in pred.targets.items()])
            st.dataframe(tg.style.format({"low": "{:,.2f}", "mid": "{:,.2f}", "high": "{:,.2f}"}), width="stretch")
            st.caption("5/10d: 25th–75th pct of similar-setup outcomes applied to last close. 30d: Monte Carlo 5th–95th pct.")
        with b:
            k = list(pred.contributions)
            bar = go.Figure(go.Bar(x=[pred.contributions[x] for x in k], y=k, orientation="h",
                                   marker_color=["#22c55e" if pred.contributions[x] > 0 else "#ef4444" for x in k]))
            bar.update_layout(template=DARK, height=300, title="תרומת כל רכיב")
            st.plotly_chart(bar, width="stretch")
            sub = {"Pattern": ps, "Technical trend": tech, "News (recency-wtd)": news_score, "Analyst": a_score,
                   "Macro+peers": macro_sig, "Options": opt.get("signal", 0.0), "Short interest": si_sig, "Risk": rsk}
            sb = go.Figure(go.Bar(x=list(sub.values()), y=list(sub), orientation="h", marker_color="#38bdf8"))
            sb.update_layout(template=DARK, height=340, title="אותות משנה (-1..+1)", xaxis_range=[-1, 1])
            st.plotly_chart(sb, width="stretch")
        h5 = go.Figure()
        for h, col in ((5, "#38bdf8"), (10, "#a78bfa")):
            h5.add_trace(go.Histogram(x=pr.matches[f"fwd{h}"], name=f"{h}d", opacity=0.6, marker_color=col, nbinsx=15))
        h5.update_layout(template=DARK, barmode="overlay", height=300, title=f"Forward outcomes of {len(pr.matches)} similar setups", xaxis_tickformat=".0%")
        st.plotly_chart(h5, width="stretch")
        if pr.summary:
            st.dataframe(pd.DataFrame(pr.summary).T[["mean", "median", "win_rate", "p25", "p75"]].style.format("{:+.2%}"))

    # ---------------------------------------------------------------- risk
    with tabs[2]:
        st.caption("💡 אלפי תרחישים מדומים לפי התנודתיות: הקונוס המוצלל = איפה המחיר כנראה יהיה.")
        pc = np.percentile(paths, [5, 25, 50, 75, 95], axis=0); x = np.arange(paths.shape[1])
        fan = go.Figure()
        fan.add_trace(go.Scatter(x=x, y=pc[4], line=dict(width=0), showlegend=False))
        fan.add_trace(go.Scatter(x=x, y=pc[0], fill="tonexty", fillcolor="rgba(56,189,248,0.15)", line=dict(width=0), name="5–95%"))
        fan.add_trace(go.Scatter(x=x, y=pc[3], line=dict(width=0), showlegend=False))
        fan.add_trace(go.Scatter(x=x, y=pc[1], fill="tonexty", fillcolor="rgba(56,189,248,0.35)", line=dict(width=0), name="25–75%"))
        fan.add_trace(go.Scatter(x=x, y=pc[2], line=dict(color="#f97316"), name="Median"))
        fan.add_hline(y=plan.stop, line_color="#ef4444", annotation_text="stop"); fan.add_hline(y=plan.target, line_color="#22c55e", annotation_text="target")
        fan.update_layout(template=DARK, height=450, title=f"קונוס מונטה קרלו ({n_paths:,} סימולציות)", xaxis_title="ימי מסחר")
        st.plotly_chart(fan, width="stretch")
        st.dataframe(mc.style.format({"p5": "{:,.2f}", "p50": "{:,.2f}", "p95": "{:,.2f}", "prob_up": "{:.0%}"}), width="stretch")
        r = st.columns(5)
        r[0].metric("הפסד יומי מקסימלי (95%)", f"{var95:.2%}"); r[1].metric("הפסד ממוצע בימים הגרועים", f"{es95:.2%}")
        r[2].metric("ATR(14)", f"{plan.atr:,.2f}"); r[3].metric("סיכון:סיכוי", f"{plan.rr:.2f}")
        r[4].metric("ציון סיכון", f"{rsk:+.2f}")
        st.markdown(f"**{plan.side.upper()} plan:** entry ≈ {plan.entry:,.2f} · stop {plan.stop:,.2f} "
                    f"({'snapped to S/R' if plan.stop_snapped else '1.5×ATR'}) · target {plan.target:,.2f} "
                    f"({'snapped to S/R' if plan.target_snapped else '3×ATR'}). Overnight gaps can skip stops.")

    # ---------------------------------------------------------------- sentiment
    with tabs[3]:
        st.caption("💡 מצב הרוח בחדשות, דירוגי אנליסטים ומצב השוק (מדד הפחד VIX).")
        g1, g2, g3 = st.columns(3)
        for col, val, title in ((g1, news_score, f"News ({engine})"), (g2, a_score, "אנליסטים"), (g3, sent_macro, "Sentiment & Macro")):
            gf = go.Figure(go.Indicator(mode="gauge+number", value=val, title={"text": title},
                                        gauge={"axis": {"range": [-1, 1]}, "bar": {"color": "#f8fafc"},
                                               "steps": [{"range": [-1, -0.2], "color": "#7f1d1d"}, {"range": [-0.2, 0.2], "color": "#334155"},
                                                         {"range": [0.2, 1], "color": "#14532d"}]}))
            gf.update_layout(template=DARK, height=250, margin=dict(t=50, b=0)); col.plotly_chart(gf, width="stretch")
        st.markdown(f"**Regime:** {macro.regime} (VIX {macro.vix if macro.vix else 'n/a'}, 20d VIX change {macro.vix_trend:+.0%}); "
                    f"news/analyst weight × {macro.sentiment_mult}. Index trend vs EMA50: "
                    + ", ".join(f"{k} {v:+.1%}" for k, v in macro.index_trends.items()))
        ac = st.columns(5)
        ac[0].metric("דירוג ממוצע (1=קנייה חזקה)", an.get("recommendation_mean") or "n/a", an.get("recommendation_key") or "")
        ac[1].metric("אנליסטים", an.get("n_analysts") or "n/a")
        ac[2].metric("יעד ממוצע", f"{an['target_mean']:,.2f}" if an.get("target_mean") else "n/a",
                     f"{an['target_mean'] / last - 1:+.1%}" if an.get("target_mean") else None)
        ac[3].metric("יעד נמוך/גבוה", f"{an.get('target_low') or 'n/a'} / {an.get('target_high') or 'n/a'}")
        ac[4].metric("שורט % · מוסדיים %", f"{si:.1%}" if si else "n/a", f"inst {own['inst_pct']:.0%}" if own.get("inst_pct") else None)
        if an.get("recommendations"):
            st.dataframe(pd.DataFrame(an["recommendations"]), width="stretch")
        if news_df.empty:
            st.info("אין חדשות זמינות.")
        else:
            st.dataframe(news_df[["published", "publisher", "title", "score", "weight"]].sort_values("published", ascending=False)
                         .style.format({"score": "{:+.2f}", "weight": "{:.2f}"}), width="stretch")

    # ---------------------------------------------------------------- options/events/peers/gaps
    with tabs[4]:
        st.caption("💡 מה שוק האופציות מצפה, אירועים קרובים, מניות מקבילות וסגירת גאפים.")
        o1, o2 = st.columns(2)
        with o1:
            st.subheader("אופציות (פקיעה קרובה)")
            if opt:
                oc = st.columns(3)
                oc[0].metric("תנודתיות גלומה", f"{opt['atm_iv']:.0%}"); oc[1].metric("תנועה צפויה", f"±{opt['implied_move_pct']:.1%}", opt.get("expiry"))
                oc[2].metric("מקס כאב", f"{opt['max_pain']:,.2f}")
                st.write(f"P/C volume {opt['pcr_volume']:.2f} · P/C OI {opt['pcr_oi']:.2f} · options signal {opt['signal']:+.2f}")
            else:
                st.info(opt_err)
            st.subheader("אירועים (5 ימים)")
            st.write(f"Next earnings: {events.get('earnings', 'n/a')} · ex-div: {events.get('ex_dividend', 'n/a')}" if events else ev_err)
            st.dataframe(pd.DataFrame(ev_hits) if ev_hits else pd.DataFrame({"info": ["No flagged events"]}), width="stretch")
            st.caption("Macro dates are a hardcoded, labeled list (FOMC/CPI/NFP) – verify with official calendars.")
        with o2:
            st.subheader(f"מקבילות: {', '.join(peers)}")
            if peer_tbl.empty:
                st.info("אין נתוני מקבילות.")
            else:
                st.dataframe(peer_tbl.style.format({"corr": "{:.2f}", "beta": "{:.2f}", "ret_1d": "{:+.2%}", "ret_5d": "{:+.2%}"}), width="stretch")
                st.write(f"Peer momentum signal: {peer_sig:+.2f} (60d correlation/beta)")
            st.subheader("ניתוח גאפים")
            st.write(f"Latest gap {gaps['last_gap']:+.2%}. Gaps >1%: up {gaps['freq_up_1pct']:.1%} / down {gaps['freq_down_1pct']:.1%} of days.")
            st.write(f"Similar gaps: n={gaps['n_similar']}, fill rate (≤20d) {gaps['fill_rate']:.0%}, same-day fill {gaps['fill_same_day']:.0%}, "
                     f"median days to fill {gaps['median_days_to_fill']}" if gaps["n_similar"] else "No similar gaps in history.")
            if 0 < gaps["n_similar"] < SMALL_N:
                st.warning("מדגם גאפים קטן.")

    # ---------------------------------------------------------------- intraday
    with tabs[5]:
        st.caption("💡 מסחר תוך-יומי: VWAP וטווח 30 הדקות הראשונות.")
        if offline:
            st.info("לא זמין במצב הדגמה.")
        else:
            iv = st.radio("Interval", ["5m", "15m"], index=1, horizontal=True)
            idf, ierr = fetch_intraday(ticker, iv)
            if ierr:
                st.info(ierr)
            else:
                ia = intraday_analysis(idf)
                f = ia["frame"].tail(5 * (78 if iv == "5m" else 26))
                ff = go.Figure(go.Candlestick(x=f.index, open=f.Open, high=f.High, low=f.Low, close=f.Close, name="OHLC"))
                ff.add_trace(go.Scatter(x=f.index, y=f.VWAP, name="Session VWAP", line=dict(color="#facc15")))
                ff.add_hline(y=ia["or_high"], line_dash="dot", annotation_text="OR high"); ff.add_hline(y=ia["or_low"], line_dash="dot", annotation_text="OR low")
                ff.update_layout(template=DARK, height=500, xaxis_rangeslider_visible=False, xaxis_rangebreaks=[dict(bounds=["sat", "mon"]), dict(bounds=[16, 9.5], pattern="hour")])
                st.plotly_chart(ff, width="stretch")
                st.write(f"Last {ia['last']:,.2f} vs VWAP {ia['vwap']:,.2f} · 30-min opening range {ia['or_low']:,.2f}–{ia['or_high']:,.2f}")
                tod = go.Figure(go.Bar(x=ia["tod"].time, y=ia["tod"]["mean"], marker_color="#38bdf8"))
                tod.update_layout(template=DARK, height=300, title="תשואה ממוצעת לפי שעה (ET, 60 ימים)", yaxis_tickformat=".2%")
                st.plotly_chart(tod, width="stretch")

    # ---------------------------------------------------------------- backtest
    with tabs[6]:
        st.caption("💡 איך המודל היה מצליח בעבר, רק עם מידע שהיה זמין באותו יום.")
        if bt is None or bt.trades.empty:
            st.info("הבדיקה לאחור כבויה או שאין מספיק נתונים.")
        else:
            s = bt.stats
            bc = st.columns(5)
            bc[0].metric("אחוז הצלחה", f"{s['hit_rate']:.1%}", f"n={s['n_signals']} signals"); bc[1].metric("ממוצע 5 ימים אחרי עלייה", f"{s['avg_ret_bull']:+.2%}")
            bc[2].metric("ממוצע 5 ימים אחרי ירידה", f"{s['avg_ret_bear']:+.2%}"); bc[3].metric("אסטרטגיה", f"{s['strategy_total']:+.1%}")
            bc[4].metric("קנה והחזק", f"{s['buy_hold_total']:+.1%}")
            if s["n_signals"] < 30:
                st.warning("פחות מ-30 אותות – אחוז ההצלחה לא אמין סטטיסטית.")
            eq = go.Figure([go.Scatter(x=bt.trades.date, y=bt.trades.equity, name="אסטרטגיה"),
                            go.Scatter(x=bt.trades.date, y=bt.trades.buy_hold, name="Buy & hold")])
            eq.update_layout(template=DARK, height=350, title="עקומת הון (החזקות של 5 ימים, בלי עמלות)")
            st.plotly_chart(eq, width="stretch")
            cal = bt.calibration
            cf = go.Figure([go.Bar(x=cal.bucket, y=cal.hit_rate * 100, name="Realized hit %", text=cal.n, textposition="outside"),
                            go.Scatter(x=cal.bucket, y=cal.calibrated, name="Calibrated (smoothed, monotone)", mode="lines+markers"),
                            go.Scatter(x=cal.bucket, y=(cal.lo + cal.hi.clip(upper=100)) / 2, name="Ideal (raw)", line=dict(dash="dot"))])
            cf.update_layout(template=DARK, height=350, title="כיול: ביטחון מול הצלחה בפועל")
            st.plotly_chart(cf, width="stretch")
            st.dataframe(cal, width="stretch")
            st.write("Component directional hit rates:", {k: f"{v:.0%}" for k, v in bt.component_hit.items()})
            st.caption("No look-ahead: each date uses only bars up to that date. News/analyst/macro/options are not available "
                       "historically, so the backtest scores the pattern+technical+risk parts only.")

    # ---------------------------------------------------------------- data quality
    with tabs[7]:
        st.caption("💡 אילו מקורות נתונים נטענו והאם יש חוסרים.")
        dq = data_quality(df)
        st.json({**{k: str(v) for k, v in dq.items()},
                 "pattern_matches": len(pr.matches), "news_items": len(news_df), "news_engine": engine,
                 "analyst": "ok" if an.get("n_analysts") else (an_err or "missing"), "options": "ok" if opt else opt_err,
                 "macro_regime": macro.regime, "peers_loaded": list(peer_series), "events": "ok" if events else ev_err,
                 "backtest_signals": bt.stats.get("n_signals") if bt is not None else None, "source": "synthetic" if demo else "yfinance"})
