"""Paths and defaults. Override the home folder with the UFC_HOME env variable."""
import os
from pathlib import Path

ROOT = Path(os.getenv("UFC_HOME", Path(__file__).resolve().parent.parent))
DATA = ROOT / "data"
MODELS = ROOT / "models"

FIGHTS_CSV = DATA / "fights.csv"          # one row per completed bout (meta + totals)
FIGHTERS_CSV = DATA / "fighters.csv"      # one row per fighter (bio)
EVENTS_CSV = DATA / "events.csv"          # fully scraped completed events
UPCOMING_CSV = DATA / "upcoming.csv"      # announced bouts
RANKINGS_CSV = DATA / "rankings.csv"
ODDS_CSV = DATA / "odds.csv"
BACKTEST_CSV = DATA / "backtest_latest.csv"
HARDNESS_CSV = DATA / "hardness.csv"
LEDGER_DB = DATA / "ledger.sqlite"

MODEL_FILE = MODELS / "predictor.joblib"
LEARNING_JSON = MODELS / "learning_state.json"

DEFAULTS = {
    "half_life_years": 5.0,     # recency weighting of training fights
    "hardness_alpha": 0.5,      # how much extra weight past mistakes get
    "hedge_eta": 0.05,          # speed of ensemble re-weighting
    "hedge_window": 600,        # graded predictions used for re-weighting
    "upcoming_events": 4,       # how many announced cards to pull
    "scrape_workers": 8,
}


def ensure_dirs() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
