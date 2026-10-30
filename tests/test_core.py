import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data import synthetic_ohlcv
from indicators import add_indicators, rsi, volume_profile
from patterns import cosine_sim, dtw_distance, find_similar
from predictor import ensemble, manager, technical_signal, pattern_signal
from risk import monte_carlo, cone, var_cvar, trade_plan, risk_signal
from macro import classify_regime, macro_from_frames
from sentiment import analyst_score, score_news

df = synthetic_ohlcv(800, seed=1)

def test_indicators():
    x = add_indicators(df)
    assert x["RSI14"].dropna().between(0, 100).all()
    assert x["RVOL20"].dropna().gt(0).all()
    prof, poc = volume_profile(df)
    assert df["Low"].min() <= poc <= df["High"].max()

def test_similarity():
    a = np.random.default_rng(0).normal(size=(20, 3))
    assert abs(cosine_sim(a, a) - 1) < 1e-9 and dtw_distance(a, a) == 0
    assert dtw_distance(a, a[::-1]) > 0

def test_patterns_both_methods():
    for m in ("cosine", "dtw"):
        r = find_similar(df, 20, 5, m)
        assert len(r.matches) == 5
        s = sorted(r.matches.start_i)
        assert all(b - a >= 20 for a, b in zip(s, s[1:]))
        assert r.summary[5]["target_low"] <= r.summary[5]["target_high"]

def test_ensemble_direction():
    w = {"a": 1, "b": 1}
    assert ensemble({"a": 0.8, "b": 0.6}, w).direction == "Bullish"
    assert ensemble({"a": -0.8, "b": -0.6}, w).direction == "Bearish"
    p = ensemble({"a": 0.05, "b": -0.05}, w)
    assert p.direction == "Neutral" and 0 <= p.confidence <= 100

def test_risk_and_manager():
    paths = monte_carlo(df["Close"], 30, 5000)
    c = cone(paths)
    assert (c.p5 <= c.p50).all() and (c.p50 <= c.p95).all()
    v, es = var_cvar(paths)
    assert es >= v
    tp = trade_plan(df)
    assert tp.stop < tp.entry < tp.target
    r = find_similar(df, 20, 5)
    p, why = manager(pattern_signal(r.summary), technical_signal(add_indicators(df)), 0.1,
                     risk_signal(paths, v, tp.rr), r.summary, c, r.last_close)
    assert p.direction in {"Bullish", "Bearish", "Neutral"} and "30d" in p.targets and why

def test_macro_and_sentiment():
    assert classify_regime(12, 0) == "Calm" and classify_regime(30, 0) == "Stress"
    assert classify_regime(None, 0) == "Unknown"
    assert macro_from_frames(None, {}).regime == "Unknown"
    assert analyst_score(1, 130, 100) > 0.5 and analyst_score(5, 70, 100) < -0.5
    t, s = score_news([{"title": "Stock surges after earnings beat"},
                       {"content": {"title": "Shares plunge on downgrade", "summary": ""}}])
    assert len(t) == 2 and t.score.iloc[0] > 0 > t.score.iloc[1]

import datetime as dt
import pandas as pd
from backtest import walk_forward, calibrate, adjusted_weights
from extras import max_pain, options_metrics, intraday_analysis, event_risk, peer_stats, gap_analysis, data_quality, default_peers

def test_backtest_no_lookahead_and_calibration():
    r = walk_forward(df, step=10, n_tests=20, min_history=300)
    assert r.stats["n_dates"] > 5
    # no look-ahead: forward return recomputed from future bar only
    t = r.trades.iloc[0]; i = df.index.get_loc(t.date)
    assert abs(t.fwd - (df.Close.iloc[i + 5] / df.Close.iloc[i] - 1)) < 1e-12
    v, n = calibrate(70, r.calibration, min_n=1)
    assert 0 <= v <= 100
    assert all(0.25 <= x <= 1 for x in adjusted_weights(r.component_hit).values())

def test_options_and_extras():
    strikes = np.arange(80, 121, 5.0)
    calls = pd.DataFrame({"strike": strikes, "openInterest": 100.0, "volume": 50.0, "bid": 1.0, "ask": 1.2, "lastPrice": 1.1, "impliedVolatility": 0.4})
    puts = calls.copy()
    assert max_pain(calls, puts) == 100.0
    m = options_metrics(calls, puts, 100, 7)
    assert abs(m["pcr_volume"] - 1) < 1e-9 and m["implied_move_pct"] > 0
    idx = pd.date_range("2026-10-01 09:30", periods=26 * 5, freq="15min", tz="America/New_York")
    intr = synthetic_ohlcv(len(idx)).set_index(idx)
    a = intraday_analysis(intr)
    assert a["or_low"] <= a["or_high"] and len(a["tod"])
    hits, mult = event_risk({"earnings": dt.date(2026, 11, 2)}, dt.date(2026, 10, 30))
    assert hits and mult < 1
    t, s = peer_stats(df.Close, {"P": df.Close * 1.01})
    assert abs(t["corr"].iloc[0] - 1) < 1e-9 and -1 <= s <= 1
    g = gap_analysis(df); assert "fill_rate" in g
    assert data_quality(df)["bars"] == len(df)
    assert "NVDA" in default_peers("MU")
