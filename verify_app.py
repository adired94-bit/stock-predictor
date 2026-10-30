"""Streamlit AppTest smoke test. Usage: python verify_app.py MU,NVDA"""
import sys, time, warnings
warnings.filterwarnings("ignore")
from streamlit.testing.v1 import AppTest
cases = [(t, y) for t in sys.argv[1].split(",") for y in ("1y", "5y")] + [("ZZZZNOTREAL", "1y"), ("DEMO", "5y")]
fail = 0
for t, y in cases:
    s = time.time(); at = AppTest.from_file("app.py", default_timeout=900); at.run()
    if t == "DEMO":
        [c for c in at.checkbox if "Demo" in c.label][0].check()
    else:
        at.text_input[0].set_value(t)
    at.select_slider[0].set_value(y)
    at.button[0].click(); at.run()
    ex = [str(e.value)[:300] for e in at.exception]; err = [e.value[:70] for e in at.error]
    fail += bool(ex)
    verdict = [m.body[:0] for m in []]
    print(f"{t:12s} {y} exceptions={ex} errors={err} tabs={len(at.tabs)} {time.time()-s:.0f}s", flush=True)
print("APP FAILURES:", fail)
