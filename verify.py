"""End-to-end module verification per ticker x range. Usage: python verify.py MU,NVDA"""
import datetime as dt, sys, traceback, warnings
warnings.filterwarnings("ignore")
import pandas as pd
from data import fetch_ohlcv
from indicators import add_indicators, volume_profile
from patterns import find_similar
from sentiment import fetch_news, score_news_v2, fetch_analyst, analyst_score, fetch_info
from macro import fetch_macro
from extras import fetch_options, fetch_intraday, intraday_analysis, fetch_events, peer_stats, default_peers, gap_analysis
from risk import monte_carlo, cone, var_cvar, trade_plan, risk_signal
from backtest import walk_forward
from predictor import manager, pattern_signal, technical_signal, sentiment_macro_signal

TICKERS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["MU", "MSTR", "NVDA", "AAPL", "TSLA", "SOUN", "SPY", "TSM"]
end = dt.date.today(); rows = []
macro = fetch_macro(end)
for t in TICKERS:
    for rn, days in {"1y": 365, "5y": 1826}.items():
        r = {"ticker": t, "range": rn}
        def step(name, fn):
            try:
                v = fn(); r[name] = "LIMITED" if v is False else "PASS"; return v
            except Exception as e:
                r[name] = f"FAIL:{type(e).__name__}:{str(e)[:60]}"; traceback.print_exc()
        start = end - dt.timedelta(days=days)
        df = step("data", lambda: fetch_ohlcv(t, start, end)[0])
        if df is None or df.empty:
            rows.append(r); continue
        r["bars"] = len(df)
        ind = step("indicators", lambda: (add_indicators(df), volume_profile(df))[0])
        pr = step("pattern_cos", lambda: find_similar(df, 20, 10, "cosine"))
        step("pattern_dtw", lambda: find_similar(df, 20, 10, "dtw"))
        news = step("news", lambda: score_news_v2(fetch_news(t)[0]))
        an = step("analyst", lambda: fetch_analyst(t)[0])
        a_s = analyst_score((an or {}).get("recommendation_mean"), (an or {}).get("target_mean"), float(df.Close.iloc[-1]))
        r["macro"] = macro.regime
        step("options", lambda: fetch_options(t, float(df.Close.iloc[-1]))[0] or False)
        step("intraday", lambda: intraday_analysis(fetch_intraday(t, "15m")[0]))
        step("events", lambda: fetch_events(t)[0] or False)
        step("peers", lambda: peer_stats(df.Close, {p: s.Close for p in default_peers(t, fetch_info(t).get("sector"))
                                                    if len(s := fetch_ohlcv(p, start, end)[0])})[0].shape[0] > 0 or False)
        step("gaps", lambda: gap_analysis(df))
        paths = step("montecarlo", lambda: monte_carlo(df.Close, 30, 5000))
        tp = step("risk_plan", lambda: trade_plan(df))
        bt = step("backtest", lambda: walk_forward(df, 20, 10, step=5, n_tests=150))
        r["bt_n"] = bt.stats.get("n_signals") if bt else None
        r["bt_hit"] = round(bt.stats["hit_rate"], 2) if bt and bt.stats else None
        if pr is not None and paths is not None and tp is not None:
            v, _ = var_cvar(paths)
            p = step("ensemble", lambda: manager(pattern_signal(pr.summary), technical_signal(ind),
                     sentiment_macro_signal(news[1] if news else 0, a_s, macro.score, macro.sentiment_mult),
                     risk_signal(paths, v, tp.rr), pr.summary, cone(paths), pr.last_close, macro.regime)[0])
            if p: r["pred"] = f"{p.direction} {p.confidence:.0f}"
        rows.append(r); print(r, flush=True)
out = pd.DataFrame(rows); out.to_csv("verification_results.csv", index=False)
print(out.to_string())
