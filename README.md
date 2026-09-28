# UFC Oracle

A local app that turns your five Kaggle notebooks into one working system: it scrapes current UFC data, predicts upcoming fights (winner, method and the most likely exact outcomes), backtests itself honestly on past events, and learns from the fights it gets wrong.

## Quick start

1. Install Python 3.10 or newer.
2. Double-click `run.bat` (Windows) or run `./run.sh` (macOS/Linux). This makes a virtual environment, installs the requirements and opens the app in your browser.
3. In the app, open **Data & model** and get data in one of two ways:
   - **Scrape UFCStats.** The first full run fetches ~8,000 fights and takes a while (roughly 15 to 30 minutes, depending on your connection). Enter e.g. `150` in "most recent N events" for a quick first test. Later updates only fetch new events.
   - **Import the Kaggle CSV** your notebooks load (`UFC_full_data_silver.csv`), then press update to add anything newer.
4. Press **Train model**.
5. Open **Fight cards** for the announced events, or **Head to head** for any two fighters.

After each event, press **Update data + learn from last event** in the sidebar. That scrapes the results, grades the predictions it made, re-learns and retrains.

Optional: add a free key from The Odds API in the sidebar (or set `ODDS_API_KEY`) to see the betting market next to each prediction.

## New in v2.3: pricing desk, live market monitor, shareable reports

**Pricing desk** tab: every upcoming fight priced from the model as fair American odds and compared with the
vig-free market consensus and the best book on each side (The Odds API returns 25+ books). Shows edge, EV per
dollar and a fractional-Kelly stake on your bankroll, flags bets above your edge threshold, and prices the props
books offer: winner by KO/TKO, submission or decision; goes the distance; over/under 1.5 and 2.5 rounds.

**Live market monitor**: every odds fetch is appended to `data/odds_history.csv`, so the tab shows line movement
(open vs now, and a chart per fight against the model's number). Keep it polling from a terminal:

```
python -m ufc_predictor odds --watch 60        # snapshot the market every 60 minutes
python -m ufc_predictor price --bankroll 1000  # pricing desk in the terminal
```

**Fight cards** tab: strong/solid picks are highlighted, the safest parlays are listed, results are graded once the
event is scraped, and *Track record* backtests the last N events one retrain at a time. *Export shareable report*
writes a single self-contained HTML page (picks, results, pricing, props, track record) you can email or host.

```
python -m ufc_predictor report --track 10 --out card.html
```

**XGBoost**: install `requirements-optional.txt` (on a Mac first `brew install libomp`) and the *full* preset
blends XGBoost into the ensemble automatically; nothing else changes.

## What each tab does

**Fight cards** shows every announced bout with the win probability split, the six possible outcomes ranked (e.g. "Fighter A by KO/TKO 31%"), the chance it goes the distance, the market line if you added an odds key, and an expandable "why" panel showing the factors that drove the pick and what each model thought.

**Head to head** predicts any hypothetical matchup using both fighters' current profiles.

**Backtest** replays history. At each cutoff the model is retrained on earlier fights only and then predicts the next window, exactly as it would have in real time. You get accuracy, log loss, AUC, calibration, accuracy by year and weight class, each model's individual score, the most confident misses, and, when historical odds exist (the Kaggle CSV has them), a comparison against the betting favourite plus a flat-stake value-betting simulation. "Compare both" runs it with and without the learning mechanisms, so you can see whether learning from mistakes actually helps on real data.

**Learning** shows the prediction ledger (every card prediction, saved before the fight and graded after), live accuracy, current ensemble weights and calibration, and two settings: how hard to lean on past mistakes, and how fast old fights fade.

## How it learns from mistakes

Three mechanisms, all using only predictions that were made *before* the result was known:

1. **Mistake weighting.** Each graded fight gets a hardness score, |actual − predicted|. The next training run gives those fights more weight, `1 + alpha × hardness`.
2. **Ensemble re-weighting.** The models (logistic regression, random forest, XGBoost, LightGBM) are blended, and each one's vote is re-weighted by its recent log loss using the Hedge algorithm, so whichever model has been reading fights best gets more say.
3. **Re-calibration.** If "70%" picks have really been winning 64% of the time, a temperature factor softens future probabilities to match.

A caution: some upsets are just luck. Leaning too hard on mistakes teaches the model noise. Keep the mistake slider moderate and check "Compare both" in the backtest before trusting a higher setting.

## What changed from the notebooks

- **Leakage removed.** The LR/RF notebook built `diff_` features from the fight's own stats (strikes landed in that same fight), which partly hands the model the answer. Every feature here uses only fights that happened before the bout.
- **Honest evaluation.** Random 80/20 splits let future fights train the model that predicts the past. The backtest here is walk-forward.
- **Corner bias removed.** UFCStats lists the winner first. Training data is mirrored and predictions are averaged over both corner orders, so a fighter's probability never depends on which side they're listed on.
- **Scraper fixes.** `title_fight` was always False (" Bout" was stripped before checking for "Title Bout"). Fights not yet fought were stored as "No Contest". Each fight page was downloaded twice. All fixed; updates are incremental; upcoming cards are scraped.
- **Leaner features.** The silver-to-gold notebook's cumulative-sum idea is kept, but with career and last-5 windows plus Elo ratings, strength of schedule, streaks, layoff, age and physicals, rather than 13 windows × hundreds of metrics. Much faster, which makes repeated backtests practical.
- **Method model.** A second model predicts KO/TKO, submission or decision given who wins, which produces the ranked outcomes.

## What accuracy to expect

MMA is noisy. Betting favourites win roughly 65% of UFC fights, and good public models tend to land around 60 to 70% accuracy. If a model claims 85%+, it is almost certainly leaking the result. Log loss and calibration are better guides than raw accuracy, and the backtest shows all three against an Elo baseline and, where available, the market.

Other limits: UFC debutants have no UFC history, so those picks are flagged as less reliable. The number of rounds for future bouts isn't published by UFCStats, so main events and title fights are assumed to be 5 rounds. The value-betting simulation uses historical lines you might not have been able to get, so treat its ROI as optimistic. This is a statistical tool, not betting advice.

## Command line

```
python -m ufc_predictor update [--max-events 50]
python -m ufc_predictor import UFC_full_data_silver.csv
python -m ufc_predictor train --preset balanced
python -m ufc_predictor backtest --start 2023-01-01 --mode compare
python -m ufc_predictor predict --log
python -m ufc_predictor cycle
```

## Files

```
app.py                     Streamlit interface
ufc_predictor/scraper.py   UFCStats, UFC rankings, The Odds API; Kaggle import
ufc_predictor/features.py  leak-free features, Elo
ufc_predictor/models.py    ensemble + method model, calibration, Hedge
ufc_predictor/pipeline.py  training, predictions, matchups, market comparison
ufc_predictor/backtest.py  walk-forward backtests and metrics
ufc_predictor/learning.py  ledger, grading, re-learning, retraining
tests/                     synthetic end-to-end test, parser tests, UI smoke test
data/, models/             created on first run
```

Run the tests with `python tests/test_pipeline.py` and `python tests/test_parsers.py`.
