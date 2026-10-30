"""Streamlit AppTest end-to-end check (Hebrew UI). Usage: python verify_app.py"""
import re, sys, time, warnings
warnings.filterwarnings("ignore")
from streamlit.testing.v1 import AppTest

fail = 0
rows = []


def side_of(at):
    for m in at.markdown:
        b = m.value
        if "ביטחון</div>" in b or "% ביטחון" in b:
            for s in ("לונג", "שורט", "לא להיכנס"):
                if f">{s} ·" in b:
                    return s
    return "?"


def report(name, at, t0, extra=""):
    global fail
    ex = [str(e.value)[:200] for e in at.exception]
    fail += bool(ex)
    hz = [d for d in at.dataframe if "טווח" in str(getattr(d.value, "columns", []))]
    rows.append((name, side_of(at), len(ex), len(hz), extra, f"{time.time() - t0:.0f}s"))
    print(rows[-1], ex, flush=True)


# 1) home
t0 = time.time(); at = AppTest.from_file("app.py", default_timeout=900); at.run()
btns = [b for b in at.button if b.label == "נתח"]
report("HOME", at, t0, f"analyze-buttons={len(btns)} infos={[i.value[:40] for i in at.info]}")
# 2) tap-to-analyze from home watchlist
for t in ["MU", "MSTR", "NVDA", "AAPL"]:
    t0 = time.time(); at = AppTest.from_file("app.py", default_timeout=900); at.run()
    b = [b for b in at.button if b.key and b.key.startswith("hwl_") and b.key.endswith("_" + t)]
    if not b:
        print("no button for", t); fail += 1; continue
    b[0].click(); at.run()
    report(f"tap {t}", at, t0, f"ticker_in={at.session_state['ticker_in']}")
    if t == "MU":  # save to journal (local fallback) then open journal page
        sv = [x for x in at.button if x.label == "💾 שמור למעקב"]; sv[0].click(); at.run()
        report("save MU", at, t0, f"success={[s.value for s in at.success]}")
        [x for x in at.button if x.label == "📒 יומן"][0].click(); at.run()
        report("journal", at, t0, f"warn={len(at.warning)}")
# 3) typed ticker
for t in ["TSLA", "SOUN"]:
    t0 = time.time(); at = AppTest.from_file("app.py", default_timeout=900); at.run()
    at.text_input(key="ticker_in").set_value(t); [b for b in at.button if "נתח" in b.label and b.key is None or b.label == "🔍 נתח"][0].click(); at.run()
    report(f"typed {t}", at, t0)
print("\nname | side | exceptions | horizon-table | extra | time")
for r in rows:
    print(" | ".join(map(str, r)))
print("APP FAILURES:", fail)
sys.exit(1 if fail else 0)
