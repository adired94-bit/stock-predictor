import os, sys
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from candles import detect_candles, chart_patterns
from horizons import analog_forecast, multi_horizon
from review import chart_analysis, full_review
import journal as J


def mk(rows, pre="down"):
    """Prepend a 6-bar trend context then the given (o,h,l,c) candles."""
    base = [100 - 2 * i for i in range(6)] if pre == "down" else [100 + 2 * i for i in range(6)]
    ctx = [(b + (1 if pre == "down" else -1), b + 1.5, b - 1.5, b) for b in base]
    data = ctx + rows
    idx = pd.date_range("2024-01-01", periods=len(data), freq="B")
    return pd.DataFrame(data, columns=["Open", "High", "Low", "Close"], index=idx).assign(Volume=1e6)


def last(df, k):
    return bool(detect_candles(df)[k].iloc[-1])


def test_doji():            assert last(mk([(90, 92, 88, 90.05)]), "doji")
def test_hammer():          assert last(mk([(89, 89.6, 85, 89.5)]), "hammer")
def test_inverted_hammer(): assert last(mk([(89, 93, 88.9, 89.6)]), "inverted_hammer")
def test_hanging_man():     assert last(mk([(111, 111.6, 107, 111.5)], "up"), "hanging_man")
def test_shooting_star():   assert last(mk([(111, 115, 110.9, 111.6)], "up"), "shooting_star")
def test_bull_engulfing():  assert last(mk([(90, 90.5, 87.5, 88), (87.5, 91.5, 87, 91)]), "bull_engulfing")
def test_bear_engulfing():  assert last(mk([(110, 112.5, 109.5, 112), (112.5, 113, 109, 109.5)], "up"), "bear_engulfing")
def test_piercing():        assert last(mk([(95, 95.5, 89.5, 90), (89, 94, 88.5, 93.5)]), "piercing")
def test_dark_cloud():      assert last(mk([(105, 110.5, 104.5, 110), (111, 111.5, 106, 106.5)], "up"), "dark_cloud")
def test_morning_star():    assert last(mk([(96, 96.5, 89.5, 90), (89.5, 90, 88.5, 89.2), (89.5, 94.5, 89, 94)]), "morning_star")
def test_evening_star():    assert last(mk([(104, 110.5, 103.5, 110), (110.5, 111.5, 110, 110.8), (110.5, 111, 105.5, 106)], "up"), "evening_star")
def test_three_white():     assert last(mk([(90, 93.2, 89.8, 93), (93, 96.2, 92.8, 96), (96, 99.2, 95.8, 99)]), "three_white_soldiers")
def test_three_black():     assert last(mk([(110, 110.2, 106.8, 107), (107, 107.2, 103.8, 104), (104, 104.2, 100.8, 101)], "up"), "three_black_crows")
def test_bull_harami():     assert last(mk([(95, 95.5, 87.5, 88), (89, 92, 88.5, 91)]), "bull_harami")
def test_bear_harami():     assert last(mk([(104, 112.5, 103.5, 112), (111, 111.5, 108, 109)], "up"), "bear_harami")
def test_marubozu():        assert last(mk([(90, 96, 90, 96)]), "bull_marubozu")
def test_bear_marubozu():   assert last(mk([(112, 112, 106, 106)], "up"), "bear_marubozu")
def test_tweezer_top():     assert last(mk([(109, 113, 108.5, 112.5), (112.5, 113.01, 109.5, 110)], "up"), "tweezer_top")
def test_tweezer_bottom():  assert last(mk([(91, 91.5, 87, 87.5), (87.5, 90, 87.01, 89.5)]), "tweezer_bottom")
def test_spinning_top():    assert last(mk([(90, 92, 88, 90.6)]), "spinning_top")
def test_no_false_doji():   assert not last(mk([(90, 96, 90, 96)]), "doji")


def walk(n=400, drift=0.0, seed=0):
    r = np.random.default_rng(seed).normal(drift, 0.02, n)
    c = 100 * np.exp(np.cumsum(r)); o = np.r_[c[0], c[:-1]]
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({"Open": o, "High": np.maximum(o, c) * 1.01, "Low": np.minimum(o, c) * 0.99, "Close": c, "Volume": 1e6}, index=idx)


def test_review_long_short_mirror():
    d = walk(400, 0.003, 1); ca = chart_analysis(d)
    L = full_review(d, ca, "Bullish", 70); S = full_review(d, ca, "Bearish", 70); O = full_review(d, ca, "Bullish", 50)
    assert L["side"] == "LONG" and L["stop"] < L["entry"] < L["tp1"] <= L["tp2"]
    assert S["side"] == "SHORT" and S["stop"] > S["entry"] > S["tp1"] >= S["tp2"]
    assert O["side"] == "OUT" and O["long_trigger"] > O["short_trigger"]


def test_chart_patterns_runs():
    assert isinstance(chart_patterns(walk()), list)


def test_horizons():
    d = walk(800, 0.001, 2)
    f = analog_forecast(d.Close, 5)
    assert 0 <= f["p_up"] <= 1 and f["lo"] < f["mid"] < f["hi"]
    t = multi_horizon(d.Close, None)
    assert len(t) == 6 and t.iloc[0].get("p_up") is None or np.isnan(t.iloc[0].get("p_up", np.nan))


# ---------------- journal evaluation
def bars(closes, highs=None, lows=None):
    idx = pd.date_range("2025-01-02", periods=len(closes), freq="B")
    return pd.DataFrame({"Open": closes, "High": highs or [c * 1.001 for c in closes], "Low": lows or [c * 0.999 for c in closes], "Close": closes}, index=idx)


FC = dict(kind="forecast", bars=1, price=100.0, lo=97.0, hi=103.0)


def test_eval_pending():          assert J.evaluate({**FC, "direction": "עולה", "bars": 5}, bars([101, 102]))["status"] == "pending"
def test_eval_hit_up():           assert J.evaluate({**FC, "direction": "עולה"}, bars([101]))["status"] == "hit"
def test_eval_partial_up():       assert J.evaluate({**FC, "direction": "עולה"}, bars([108]))["status"] == "partial"
def test_eval_miss_up():          assert J.evaluate({**FC, "direction": "עולה"}, bars([99]))["status"] == "miss"
def test_eval_hit_down():         assert J.evaluate({**FC, "direction": "יורד"}, bars([98]))["status"] == "hit"
def test_eval_neutral():          assert J.evaluate({**FC, "direction": "ניטרלי"}, bars([110]))["status"] == "miss"
def test_eval_weekly_uses_5th():  assert J.evaluate({**FC, "direction": "עולה", "bars": 5}, bars([95, 96, 97, 98, 101, 80]))["status"] == "hit"

TR = dict(kind="trade", bars=3, price=100.0, entry=100.0)


def test_trade_long_target():  assert J.evaluate({**TR, "direction": "LONG", "stop": 95, "target": 106}, bars([102, 105, 104], highs=[103, 107, 105], lows=[100, 103, 103]))["status"] == "hit"
def test_trade_long_stop():    assert J.evaluate({**TR, "direction": "LONG", "stop": 95, "target": 106}, bars([97, 94, 108], highs=[100, 98, 109], lows=[96, 93, 100]))["status"] == "miss"
def test_trade_long_partial(): assert J.evaluate({**TR, "direction": "LONG", "stop": 95, "target": 110}, bars([101, 102, 103]))["status"] == "partial"
def test_trade_short_target(): assert J.evaluate({**TR, "direction": "SHORT", "stop": 104, "target": 95}, bars([99, 96, 94], lows=[98, 95.5, 93]))["status"] == "hit"
def test_trade_same_bar_stop_first(): assert J.evaluate({**TR, "direction": "SHORT", "stop": 104, "target": 95}, bars([100, 100, 100], highs=[105, 101, 101], lows=[94, 99, 99]))["status"] == "miss"


def test_local_store_and_stats(tmp_path):
    s = J.LocalStore(str(tmp_path / "j.csv"))
    h = pd.DataFrame([{"טווח": "יומי", "direction": "עולה", "p_up": 0.6, "lo": 97, "hi": 103}])
    e = J.make_entries("TEST", 100.0, h, {"side": "LONG", "entry": 100, "stop": 95, "tp1": 106})
    s.save(e); d = s.load(); assert len(d) == 2
    fake = lambda t, f, since: bars([101] * 12, highs=[107] * 12)
    d2, n = J.evaluate_all(d, fetch=fake); assert n == 2
    st = J.stats(d2); assert st["n"] == 2 and st["success"] == 1.0


def test_get_store_fallback():
    assert J.get_store(None).kind == "local"
    assert J.get_store({"GITHUB_TOKEN": "x", "JOURNAL_REPO": "a/b"}).kind == "github"
