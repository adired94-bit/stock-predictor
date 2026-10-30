<div dir="rtl">

# 📈 חיזוי מניות – מדריך מהיר

> ⚠️ כלי לימודי בלבד – **לא ייעוץ השקעות**. ביטחון 50% = הטלת מטבע.

## איך משתמשים (בטלפון)
1. פותחים את הקישור של האפליקציה. מסך הבית מראה: מדדים, סטאפים, מניות עולות/יורדות, רשימת מעקב, חדשות ואירועים.
2. לוחצים **נתח** ליד מניה, או פותחים את התפריט (**>>** למעלה), כותבים סימול ולוחצים **🔍 נתח**.
3. למעלה מופיע כרטיס אחד: **לונג / שורט / לא להיכנס**, אזור כניסה, מתי להיכנס, סטופ, יעדים וביטחון.
4. מתחתיו: טבלת **תחזית לפי טווחי זמן** (שעתי עד רבעוני), **נתוני אמת** ו**חיזוי האפליקציה**.
5. גרפים, נרות יפניים וכל הניתוחים המתקדמים – בחלונות הנפתחים למטה.
6. **💾 שמור למעקב** שומר את החיזוי, ו-**📒 יומן** בודק אחר כך אם הוא עבד.

## שמירת היומן לתמיד (חד-פעמי)
בלי ההגדרה הזו היומן נשמר זמנית ועלול להימחק כשהשרת מתאתחל.
1. ב-GitHub: Settings → Developer settings → Personal access tokens → **Fine-grained tokens** → Generate new token.
   - Repository access: **Only select repositories** → `stock-predictor-journal`.
   - Permissions → Repository permissions → **Contents: Read and write**. שמור את הטוקן.
2. ב-Streamlit Cloud: **Manage app** → **Settings** → **Secrets**, הדבק:
```toml
GITHUB_TOKEN = "github_pat_..."
JOURNAL_REPO = "adired94-bit/stock-predictor-journal"
```
3. **Save**. האפליקציה תתאתחל והיומן יישמר בריפו הפרטי. לעולם אל תשים את הטוקן בקוד.

## הרצה במחשב
`./run.sh` (מק/לינוקס) או `run.bat` (ווינדוס), או: `pip install -r requirements.txt && streamlit run app.py`.

</div>

---

# 📈 Stock Movement Predictor

> ⚠️ Educational tool – **not financial advice**. The confidence number is a heuristic score (optionally
> mapped to past backtest hit rates); markets are mostly noise and past patterns may not repeat.

## Quickstart (3 steps)
1. Install Python 3.10+ (python.org; on Windows tick “Add Python to PATH”).
2. Start it: **Windows** double-click `run.bat` · **Mac/Linux** run `./run.sh`
   (first run creates `.venv` and installs everything, takes 1–3 min).
3. Your browser opens at http://localhost:8501 → type a ticker → **Analyze**.

The top **Verdict card** says Bullish/Bearish/Neutral, confidence, 5–10 day and 30 day ranges, stop,
take-profit and risk:reward in one sentence. Everything advanced is in the tabs below it.

Manual: `pip install -r requirements.txt && streamlit run app.py`.
Optional FinBERT sentiment: `pip install transformers torch`.

## How it works
| Module | What it does |
|---|---|
| `home.py` | Hebrew home dashboard: market tiles + sparklines, Yahoo screeners (gainers/losers/most active) with fallback, RVOL, setups, watchlist, RSS news with VADER, events. 5-min cache |
| `review.py` | Hebrew chart analysis (trend, EMAs, S/R, Fibonacci, RSI, MACD, Bollinger, divergence, POC, daily/weekly) and the LONG/SHORT/stay-out plan (entry zone, trigger, stop, TP1/TP2, R:R, 1% risk sizing) |
| `candles.py` | 20 candlestick patterns + double top/bottom, H&S, triangles, flags, range breakouts; per-ticker 5-day follow-through |
| `horizons.py` | 1h/4h/1d/1w/1m/1q forecast: state-conditioned analogs, p(up), 5/50/95% range, walk-forward hit rate |
| `journal.py`, `journal_ui.py` | prediction journal; auto-evaluates hit/miss/partial; GitHub contents-API storage via secrets, local CSV fallback |
| `data.py` | yfinance daily OHLCV (cached), synthetic demo data |
| `indicators.py` | EMA 9/21/200, RSI14, rolling 20-day VWAP (daily bars), RVOL20, volume profile/POC |
| `patterns.py` | last N candles+volume → z-scored features; cosine or DTW (numpy) vs all history; top-K non-overlapping matches; 5/10-day outcomes, 25–75th pct targets |
| `sentiment.py` | yfinance news (old & new formats), VADER + finance lexicon or FinBERT, recency-weighted (48h half-life); analyst rating/targets; short interest & institutions |
| `macro.py` | ^VIX level/trend + S&P/Nasdaq vs EMA50 → regime Calm/Normal/Elevated/Stress; shrinks news/analyst weight in stress |
| `risk.py` | Monte Carlo (bootstrap/GBM, 5k+ paths) cones 5/10/30d, VaR/ES 95%, ATR(14) stop/target snapped to swing S/R or POC, R:R |
| `extras.py` | options (ATM IV, implied move, P/C vol & OI, max pain), intraday 5m/15m (session VWAP, opening range, time-of-day stats), events (earnings, ex-div, hardcoded labeled FOMC/CPI/NFP list), peers (semis default / sector ETF, 60d corr & beta), gap fill stats, data quality |
| `backtest.py` | walk-forward, no look-ahead; hit rate, avg return per signal, equity vs buy&hold, calibration table (Laplace-smoothed, monotone) and component hit rates for auto-weighting |
| `predictor.py` | manager: **Pattern 50% · Sentiment&Macro 30% · Risk 20%** |
| `app.py` | Streamlit dark dashboard |

**Confidence**: S = weighted sum of sub-scores (−1..1). Neutral if |S| ≤ 0.15. Raw = 50 + 50·min(|S|/0.6,1)·agreement.
Then mapped to the realized hit rate of past signals in the same confidence bucket (if n ≥ 5), then shrunk toward 50%
when earnings/macro events are within 5 days.

## Tests / verification
`pytest -q` (offline, synthetic) · `python verify.py MU,NVDA` (live module matrix → verification_results.csv) ·
`python verify_app.py MU,NVDA` (headless Streamlit AppTest).

## Limitations
Backtest scores only pattern+technical+risk (news/analyst/options history isn't available free); yfinance is unofficial
and may rate-limit; intraday limited to 60 days; macro event dates are a hardcoded list; no trading costs in backtest.
