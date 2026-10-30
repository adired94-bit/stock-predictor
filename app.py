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

st.set_page_config(page_title="Stock Movement Predictor", page_icon="📈", layout="wide")
DARK = "plotly_dark"
SMALL_N = 8

HELP = {
    "1d VaR 95%": "On 95% of days the loss should not exceed this (from simulation).",
    "1d ES 95%": "Average loss on the worst 5% of days (Expected Shortfall).",
    "ATR(14)": "Average daily price range over 14 days – typical daily move.",
    "R:R": "Reward ÷ risk of the suggested stop/target. Above 2 is attractive.",
    "Risk score": "-1 = risky/negative skew, +1 = favorable volatility profile.",
    "Rec. mean (1=SB,5=S)": "Analyst average rating: 1 Strong Buy … 5 Sell.",
    "Analysts": "Number of analysts covering the stock.",
    "Target mean": "Average 12-month analyst price target (not a short-term forecast).",
    "Target low/high": "Lowest and highest analyst 12-month targets.",
    "Short % float · Inst %": "Share of float sold short; institutional ownership.",
    "ATM IV": "Implied volatility of at-the-money options (annualized).",
    "Implied move": "Move the options market prices in by expiry (ATM straddle ÷ price).",
    "Max pain": "Strike where option holders lose the most at expiry.",
    "Hit rate": "Share of past directional calls that were right after 5 days.",
    "Avg 5d after Bullish": "Average 5-day return after past Bullish calls.",
    "Avg 5d after Bearish": "Average 5-day return after past Bearish calls (negative = good).",
    "Strategy": "Total return following the signals in the backtest (no costs).",
    "Buy & hold (same dates)": "Total return just holding over the same dates.",
}
from streamlit.delta_generator import DeltaGenerator as _DG  # noqa: E402
_orig_metric = _DG.metric
def _metric(self, label, *a, **k):  # inject help text for every metric
    k.setdefault("help", HELP.get(label))
    return _orig_metric(self, label, *a, **k)
_DG.metric = _metric

# ============================================================== sidebar
with st.sidebar:
    st.title("📈 Predictor")
    st.caption("Type a ticker and press **Analyze**. Everything else has sensible defaults.")
    with st.form("controls"):
        ticker = st.text_input("Ticker", "MU", help="Stock or ETF symbol, e.g. MU, NVDA, SPY").strip().upper()
        today = dt.date.today()
        years = st.select_slider("History", ["1y", "2y", "5y", "10y"], "5y", help="How far back to search for similar setups")
        with st.expander("Advanced settings"):
            c1, c2 = st.columns(2)
            start = c1.date_input("Start (overrides History)", None, help="Leave empty to use History")
            end = c2.date_input("End", today)
            window = st.slider("Pattern window (days)", 10, 60, 20, help="Length of the recent price pattern to compare")
            top_k = st.slider("Top-K matches", 3, 30, 10, help="How many similar historical setups to use")
            method = st.selectbox("Similarity", ["cosine", "dtw"], help="Cosine = fast; DTW = tolerant to timing shifts (slower)")
            mc_method = st.selectbox("Monte Carlo", ["bootstrap", "gbm"], help="Bootstrap resamples real past daily returns; GBM assumes normal returns")
            n_paths = st.select_slider("MC paths", [2000, 5000, 10000, 20000], 5000)
            use_finbert = st.checkbox("FinBERT sentiment (if installed)", value=False, disabled=not finbert_available(),
                                      help="pip install transformers torch to enable")
            peers_txt = st.text_input("Peers (comma; blank = auto)", "")
            run_bt = st.checkbox("Run walk-forward backtest", True, help="Tests the model on past dates to calibrate confidence")
            auto_w = st.checkbox("Auto-reduce weak components", True)
            demo = st.checkbox("Demo mode (synthetic data, offline)", False)
        go_btn = st.form_submit_button("🔍 Analyze", type="primary", width="stretch")
    if start is None:
        start = end - dt.timedelta(days=365 * int(years[:-1]))
    st.caption("Manager weights: Pattern 50% · Sentiment&Macro 30% · Risk 20%")

st.warning("⚠️ Educational tool – **not financial advice**. Confidence is a heuristic score; the 'calibrated' "
           "value maps it to past walk-forward hit rates and is only as good as the sample size shown.")

# ============================================================== data
if go_btn:
    st.session_state["analyzed"] = True
if not st.session_state.get("analyzed"):
    st.info("👈 Type a ticker in the sidebar and press **Analyze**.")
    st.stop()
notes: list[str] = []
if demo:
    df = synthetic_ohlcv(1260, seed=7)
else:
    df, err = fetch_ohlcv(ticker, start, end)
    if err:
        st.error(f"{err} — switch on *Demo mode* to explore with synthetic data.")
        st.stop()
if len(df) < window * 3 + 40:
    st.error("Not enough history for this window; widen the date range."); st.stop()

ind = add_indicators(df)
last = float(df["Close"].iloc[-1])
offline = demo

with st.spinner("Pattern matching…"):
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
    with st.spinner("Walk-forward backtest…"):
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

# ============================================================== verdict card
color = {"Bullish": "#16a34a", "Bearish": "#dc2626", "Neutral": "#64748b"}[pred.direction]
t5 = pred.targets.get("5d") or pred.targets.get("10d")
t30 = pred.targets.get("30d")
driver = max(pred.contributions, key=lambda k: abs(pred.contributions[k]))
plain = {"Bullish": "leans UP", "Bearish": "leans DOWN", "Neutral": "has no clear edge"}[pred.direction]
sentence = (f"{ticker if not demo else 'DEMO'} {plain} for the next 1–2 weeks, mainly because of the "
            f"{driver.lower()} signal; confidence is {final_conf:.0f}% (50% = coin flip).")
st.markdown(f"""
<div style="border-radius:16px;padding:20px 24px;background:linear-gradient(135deg,{color}33,#0f172a);border:2px solid {color}">
<div style="font-size:14px;opacity:.8">VERDICT · {ticker if not demo else 'DEMO'} · last {last:,.2f}</div>
<div style="font-size:40px;font-weight:800;color:{color}">{pred.direction.upper()} · {final_conf:.0f}%</div>
<div style="font-size:17px;margin:6px 0 12px">{sentence}</div>
<div style="display:flex;gap:28px;flex-wrap:wrap;font-size:16px">
<div>🎯 <b>5–10d range</b><br>{f"{t5[0]:,.2f} – {t5[2]:,.2f}" if t5 else "n/a"}</div>
<div>📅 <b>30d range</b><br>{f"{t30[0]:,.2f} – {t30[2]:,.2f}" if t30 else "n/a"}</div>
<div>🛑 <b>Stop ({plan.side})</b><br>{plan.stop:,.2f}</div>
<div>✅ <b>Take-profit</b><br>{plan.target:,.2f}</div>
<div>⚖️ <b>Risk:Reward</b><br>{plan.rr:.2f}</div>
<div>🌡️ <b>Market</b><br>{macro.regime}</div>
</div></div>""", unsafe_allow_html=True)
st.caption("Confidence is a heuristic score adjusted by past backtest results and upcoming events – not a guarantee. "
           "Stop/take-profit are ATR-based levels snapped to nearby support/resistance. Not financial advice.")
m = st.columns(4)
m[0].metric("Raw confidence", f"{raw_conf:.0f}%", help="Model score before calibration")
m[1].metric("Calibration sample", cal_n, help="How many past signals had similar confidence; <10 = unreliable")
m[2].metric("Similar setups", len(pr.matches), help="Historical windows that looked like the last weeks")
m[3].metric("Composite score", f"{pred.composite:+.2f}", help="-1 = strongly bearish, +1 = strongly bullish")
if len(pr.matches) < SMALL_N:
    st.info(f"Only {len(pr.matches)} pattern matches – statistics are noisy.")
if bt is not None and cal_n < 10:
    st.info(f"Calibration bucket has n={cal_n} (<10): confidence is mostly uncalibrated.")
if ev_hits:
    st.warning("Upcoming events: " + ", ".join(f"{e['event']} {e['date']}" for e in ev_hits))
st.markdown("#### Details")
tabs = st.tabs(["Chart & Patterns", "Prediction", "Risk & Monte Carlo", "Sentiment & Macro", "Options · Events · Peers · Gaps",
                "Intraday", "Backtest", "Data quality"])

# ---------------------------------------------------------------- chart
with tabs[0]:
    st.caption("💡 Price chart with moving averages. Orange = the last weeks; green = past periods that looked similar.")
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
    ov.update_layout(template=DARK, height=380, title="Matched setups vs current (normalized, +10d outcome)", yaxis_tickformat=".0%")
    st.plotly_chart(ov, width="stretch")
    st.dataframe(pr.matches.drop(columns=["start_i", "end_i"]).style.format({"similarity": "{:.3f}", "fwd5": "{:+.2%}", "fwd10": "{:+.2%}"}), width="stretch")

# ---------------------------------------------------------------- prediction
with tabs[1]:
    st.caption("💡 Why the verdict: each component votes from -1 (down) to +1 (up); bars show how much each one pushed.")
    a, b = st.columns([1, 1])
    with a:
        st.markdown("### Rationale"); st.markdown(rationale)
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
        bar.update_layout(template=DARK, height=300, title="Component contributions")
        st.plotly_chart(bar, width="stretch")
        sub = {"Pattern": ps, "Technical trend": tech, "News (recency-wtd)": news_score, "Analyst": a_score,
               "Macro+peers": macro_sig, "Options": opt.get("signal", 0.0), "Short interest": si_sig, "Risk": rsk}
        sb = go.Figure(go.Bar(x=list(sub.values()), y=list(sub), orientation="h", marker_color="#38bdf8"))
        sb.update_layout(template=DARK, height=340, title="Sub-signals (-1..+1)", xaxis_range=[-1, 1])
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
    st.caption("💡 5,000+ simulated futures based on past volatility: the shaded cone is where price will likely be.")
    pc = np.percentile(paths, [5, 25, 50, 75, 95], axis=0); x = np.arange(paths.shape[1])
    fan = go.Figure()
    fan.add_trace(go.Scatter(x=x, y=pc[4], line=dict(width=0), showlegend=False))
    fan.add_trace(go.Scatter(x=x, y=pc[0], fill="tonexty", fillcolor="rgba(56,189,248,0.15)", line=dict(width=0), name="5–95%"))
    fan.add_trace(go.Scatter(x=x, y=pc[3], line=dict(width=0), showlegend=False))
    fan.add_trace(go.Scatter(x=x, y=pc[1], fill="tonexty", fillcolor="rgba(56,189,248,0.35)", line=dict(width=0), name="25–75%"))
    fan.add_trace(go.Scatter(x=x, y=pc[2], line=dict(color="#f97316"), name="Median"))
    fan.add_hline(y=plan.stop, line_color="#ef4444", annotation_text="stop"); fan.add_hline(y=plan.target, line_color="#22c55e", annotation_text="target")
    fan.update_layout(template=DARK, height=450, title=f"Monte Carlo cone ({n_paths:,} {mc_method} paths)", xaxis_title="trading days")
    st.plotly_chart(fan, width="stretch")
    st.dataframe(mc.style.format({"p5": "{:,.2f}", "p50": "{:,.2f}", "p95": "{:,.2f}", "prob_up": "{:.0%}"}), width="stretch")
    r = st.columns(5)
    r[0].metric("1d VaR 95%", f"{var95:.2%}"); r[1].metric("1d ES 95%", f"{es95:.2%}")
    r[2].metric("ATR(14)", f"{plan.atr:,.2f}"); r[3].metric("R:R", f"{plan.rr:.2f}")
    r[4].metric("Risk score", f"{rsk:+.2f}")
    st.markdown(f"**{plan.side.upper()} plan:** entry ≈ {plan.entry:,.2f} · stop {plan.stop:,.2f} "
                f"({'snapped to S/R' if plan.stop_snapped else '1.5×ATR'}) · target {plan.target:,.2f} "
                f"({'snapped to S/R' if plan.target_snapped else '3×ATR'}). Overnight gaps can skip stops.")

# ---------------------------------------------------------------- sentiment
with tabs[3]:
    st.caption("💡 Mood of news headlines, analyst ratings, and the overall market (VIX fear index).")
    g1, g2, g3 = st.columns(3)
    for col, val, title in ((g1, news_score, f"News ({engine})"), (g2, a_score, "Analysts"), (g3, sent_macro, "Sentiment & Macro")):
        gf = go.Figure(go.Indicator(mode="gauge+number", value=val, title={"text": title},
                                    gauge={"axis": {"range": [-1, 1]}, "bar": {"color": "#f8fafc"},
                                           "steps": [{"range": [-1, -0.2], "color": "#7f1d1d"}, {"range": [-0.2, 0.2], "color": "#334155"},
                                                     {"range": [0.2, 1], "color": "#14532d"}]}))
        gf.update_layout(template=DARK, height=250, margin=dict(t=50, b=0)); col.plotly_chart(gf, width="stretch")
    st.markdown(f"**Regime:** {macro.regime} (VIX {macro.vix if macro.vix else 'n/a'}, 20d VIX change {macro.vix_trend:+.0%}); "
                f"news/analyst weight × {macro.sentiment_mult}. Index trend vs EMA50: "
                + ", ".join(f"{k} {v:+.1%}" for k, v in macro.index_trends.items()))
    ac = st.columns(5)
    ac[0].metric("Rec. mean (1=SB,5=S)", an.get("recommendation_mean") or "n/a", an.get("recommendation_key") or "")
    ac[1].metric("Analysts", an.get("n_analysts") or "n/a")
    ac[2].metric("Target mean", f"{an['target_mean']:,.2f}" if an.get("target_mean") else "n/a",
                 f"{an['target_mean'] / last - 1:+.1%}" if an.get("target_mean") else None)
    ac[3].metric("Target low/high", f"{an.get('target_low') or 'n/a'} / {an.get('target_high') or 'n/a'}")
    ac[4].metric("Short % float · Inst %", f"{si:.1%}" if si else "n/a", f"inst {own['inst_pct']:.0%}" if own.get("inst_pct") else None)
    if an.get("recommendations"):
        st.dataframe(pd.DataFrame(an["recommendations"]), width="stretch")
    if news_df.empty:
        st.info(f"No news available ({news_err}).")
    else:
        st.dataframe(news_df[["published", "publisher", "title", "score", "weight"]].sort_values("published", ascending=False)
                     .style.format({"score": "{:+.2f}", "weight": "{:.2f}"}), width="stretch")

# ---------------------------------------------------------------- options/events/peers/gaps
with tabs[4]:
    st.caption("💡 What options traders expect, upcoming events, related stocks, and how often gaps get filled.")
    o1, o2 = st.columns(2)
    with o1:
        st.subheader("Options (nearest expiry)")
        if opt:
            oc = st.columns(3)
            oc[0].metric("ATM IV", f"{opt['atm_iv']:.0%}"); oc[1].metric("Implied move", f"±{opt['implied_move_pct']:.1%}", opt.get("expiry"))
            oc[2].metric("Max pain", f"{opt['max_pain']:,.2f}")
            st.write(f"P/C volume {opt['pcr_volume']:.2f} · P/C OI {opt['pcr_oi']:.2f} · options signal {opt['signal']:+.2f}")
        else:
            st.info(opt_err)
        st.subheader("Events (next 5 days)")
        st.write(f"Next earnings: {events.get('earnings', 'n/a')} · ex-div: {events.get('ex_dividend', 'n/a')}" if events else ev_err)
        st.dataframe(pd.DataFrame(ev_hits) if ev_hits else pd.DataFrame({"info": ["No flagged events"]}), width="stretch")
        st.caption("Macro dates are a hardcoded, labeled list (FOMC/CPI/NFP) – verify with official calendars.")
    with o2:
        st.subheader(f"Peers: {', '.join(peers)}")
        if peer_tbl.empty:
            st.info("Peer data unavailable.")
        else:
            st.dataframe(peer_tbl.style.format({"corr": "{:.2f}", "beta": "{:.2f}", "ret_1d": "{:+.2%}", "ret_5d": "{:+.2%}"}), width="stretch")
            st.write(f"Peer momentum signal: {peer_sig:+.2f} (60d correlation/beta)")
        st.subheader("Gap analysis")
        st.write(f"Latest gap {gaps['last_gap']:+.2%}. Gaps >1%: up {gaps['freq_up_1pct']:.1%} / down {gaps['freq_down_1pct']:.1%} of days.")
        st.write(f"Similar gaps: n={gaps['n_similar']}, fill rate (≤20d) {gaps['fill_rate']:.0%}, same-day fill {gaps['fill_same_day']:.0%}, "
                 f"median days to fill {gaps['median_days_to_fill']}" if gaps["n_similar"] else "No similar gaps in history.")
        if 0 < gaps["n_similar"] < SMALL_N:
            st.warning("Small gap sample.")

# ---------------------------------------------------------------- intraday
with tabs[5]:
    st.caption("💡 Today's minute-level action: VWAP (average traded price) and the first-30-minute range.")
    if offline:
        st.info("Intraday unavailable in demo mode.")
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
            tod.update_layout(template=DARK, height=300, title="Mean return by time of day (ET, last 60d)", yaxis_tickformat=".2%")
            st.plotly_chart(tod, width="stretch")

# ---------------------------------------------------------------- backtest
with tabs[6]:
    st.caption("💡 How this model would have done in the past, using only data available at each date.")
    if bt is None or bt.trades.empty:
        st.info("Backtest disabled or insufficient data.")
    else:
        s = bt.stats
        bc = st.columns(5)
        bc[0].metric("Hit rate", f"{s['hit_rate']:.1%}", f"n={s['n_signals']} signals"); bc[1].metric("Avg 5d after Bullish", f"{s['avg_ret_bull']:+.2%}")
        bc[2].metric("Avg 5d after Bearish", f"{s['avg_ret_bear']:+.2%}"); bc[3].metric("Strategy", f"{s['strategy_total']:+.1%}")
        bc[4].metric("Buy & hold (same dates)", f"{s['buy_hold_total']:+.1%}")
        if s["n_signals"] < 30:
            st.warning("Fewer than 30 signals – hit rate is statistically weak.")
        eq = go.Figure([go.Scatter(x=bt.trades.date, y=bt.trades.equity, name="Strategy"),
                        go.Scatter(x=bt.trades.date, y=bt.trades.buy_hold, name="Buy & hold")])
        eq.update_layout(template=DARK, height=350, title="Equity curve (non-overlapping 5d holds, no costs)")
        st.plotly_chart(eq, width="stretch")
        cal = bt.calibration
        cf = go.Figure([go.Bar(x=cal.bucket, y=cal.hit_rate * 100, name="Realized hit %", text=cal.n, textposition="outside"),
                        go.Scatter(x=cal.bucket, y=cal.calibrated, name="Calibrated (smoothed, monotone)", mode="lines+markers"),
                        go.Scatter(x=cal.bucket, y=(cal.lo + cal.hi.clip(upper=100)) / 2, name="Ideal (raw)", line=dict(dash="dot"))])
        cf.update_layout(template=DARK, height=350, title="Calibration: raw confidence bucket vs realized hit rate (bar labels = n)")
        st.plotly_chart(cf, width="stretch")
        st.dataframe(cal, width="stretch")
        st.write("Component directional hit rates:", {k: f"{v:.0%}" for k, v in bt.component_hit.items()})
        st.caption("No look-ahead: each date uses only bars up to that date. News/analyst/macro/options are not available "
                   "historically, so the backtest scores the pattern+technical+risk parts only.")

# ---------------------------------------------------------------- data quality
with tabs[7]:
    st.caption("💡 Which data sources loaded and any gaps in the data.")
    dq = data_quality(df)
    st.json({**{k: str(v) for k, v in dq.items()},
             "pattern_matches": len(pr.matches), "news_items": len(news_df), "news_engine": engine,
             "analyst": "ok" if an.get("n_analysts") else (an_err or "missing"), "options": "ok" if opt else opt_err,
             "macro_regime": macro.regime, "peers_loaded": list(peer_series), "events": "ok" if events else ev_err,
             "backtest_signals": bt.stats.get("n_signals") if bt is not None else None, "source": "synthetic" if demo else "yfinance"})
