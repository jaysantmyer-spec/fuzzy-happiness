"""
Walk-forward backtesting.

The notebooks evaluated with a random 80/20 split, which lets fights from the
future train the model that predicts the past. Here the model is retrained at
each cutoff using ONLY fights before it, then predicts the next window, exactly
as it would have in real time.

`adaptive=True` switches on the three learning-from-mistakes mechanisms, each
using only predictions that were already graded at that point in time:
  1. Hardness weighting - fights the model got wrong get more training weight.
  2. Hedge re-weighting of the ensemble members by their recent log loss.
  3. Temperature re-calibration of the blended probability.
Run `compare()` to see whether they actually help on your data.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config
from .models import blend, fit_temperature, hedge_weights
from .pipeline import train_predictor

Progress = Optional[Callable[[float, str], None]]


def _ll(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def learning_signals(past: pd.DataFrame, eta: float, window: int):
    """Ensemble weights + temperature + hardness from already-graded predictions."""
    if past is None or len(past) < 150:
        return None, 1.0, {}
    recent = past.tail(window)
    comp = recent[[c for c in recent.columns if c.startswith("comp_")]]
    comp.columns = [c[5:] for c in comp.columns]
    weights = hedge_weights(comp, recent["y"], eta=eta)
    temp = fit_temperature(blend(comp, weights), recent["y"])
    hardness = dict(zip(past["fight_url"], np.abs(past["y"] - past["p_A"])))
    return weights, temp, hardness


def walk_forward(feat: pd.DataFrame, start, end=None, step_days: int = 91,
                 learners: list[str] | None = None, adaptive: bool = False,
                 half_life_years: float | None = None, hardness_alpha: float | None = None,
                 eta: float | None = None, window: int | None = None,
                 progress: Progress = None) -> pd.DataFrame:
    hl = config.DEFAULTS["half_life_years"] if half_life_years is None else half_life_years
    alpha = config.DEFAULTS["hardness_alpha"] if hardness_alpha is None else hardness_alpha
    eta = config.DEFAULTS["hedge_eta"] if eta is None else eta
    window = config.DEFAULTS["hedge_window"] if window is None else window

    data = feat[feat["y"].notna() & feat["A_n_fights"].notna()]
    start = pd.Timestamp(start)
    end = pd.Timestamp(end) if end is not None else data["event_date"].max()
    test = data[(data["event_date"] >= start) & (data["event_date"] <= end)]
    edges = list(pd.date_range(start, end + pd.Timedelta(days=1), freq=f"{step_days}D"))
    if not edges or edges[-1] <= end:
        edges.append(end + pd.Timedelta(days=1))

    preds: list[pd.DataFrame] = []
    n_steps = len(edges) - 1
    for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        rows = test[(test["event_date"] >= lo) & (test["event_date"] < hi)]
        if rows.empty:
            continue
        if progress:
            progress(k / max(n_steps, 1), f"Training on fights before {lo.date()} → predicting {len(rows)} fights")
        past = pd.concat(preds, ignore_index=True) if preds else None
        weights, temp, hard = learning_signals(past, eta, window) if adaptive else (None, 1.0, {})
        model = train_predictor(feat, learners, cutoff=lo, half_life_years=hl, hardness=hard,
                                hardness_alpha=alpha if adaptive else 0.0, weights=weights, temperature=temp)
        P = model.predict(rows, explain=False)
        P["y"] = rows["y"].to_numpy()
        P["actual_method"] = rows["method"].to_numpy()
        P["winner"] = np.where(P["y"] == 1, P["A_name"], P["B_name"])
        if "f_1_odds" in rows and "f_2_odds" in rows:
            oa, ob = pd.to_numeric(rows["f_1_odds"], errors="coerce"), pd.to_numeric(rows["f_2_odds"], errors="coerce")
            ok = (oa > 1) & (ob > 1)
            P["odds_A"], P["odds_B"] = oa.where(ok).to_numpy(), ob.where(ok).to_numpy()
            P["market_p_A"] = ((1 / oa) / (1 / oa + 1 / ob)).where(ok).to_numpy()
        P["train_cutoff"] = lo
        preds.append(P)
    if progress:
        progress(1.0, "Backtest complete")
    if not preds:
        return pd.DataFrame()
    bt = pd.concat(preds, ignore_index=True)
    bt["correct"] = ((bt["p_A"] >= 0.5).astype(float) == bt["y"]).astype(int)
    bt["log_loss"] = _ll(bt["p_A"], bt["y"])
    return bt


# --------------------------------------------------------------------------- metrics
def summarize(bt: pd.DataFrame, edge: float = 0.05) -> dict:
    if bt is None or bt.empty:
        return {}
    y, p = bt["y"].to_numpy(float), bt["p_A"].to_numpy(float)
    s = {
        "fights": int(len(bt)),
        "accuracy": float(((p >= 0.5) == y).mean()),
        "log_loss": float(_ll(p, y).mean()),
        "brier": float(((p - y) ** 2).mean()),
        # predictions are corner-symmetric, so score both orientations (UFCStats lists winners first)
        "auc": float(roc_auc_score(np.r_[y, 1 - y], np.r_[p, 1 - p])),
        "elo_accuracy": float(((bt["elo_p_A"] >= 0.5) == y).mean()),
        "elo_log_loss": float(_ll(bt["elo_p_A"], y).mean()),
        "models": {},
    }
    for c in [c for c in bt.columns if c.startswith("comp_")]:
        s["models"][c[5:]] = {"accuracy": float(((bt[c] >= 0.5) == y).mean()),
                              "log_loss": float(_ll(bt[c], y).mean())}
    hit = bt["correct"] == 1
    mcols = {m: (f"A_{m}", f"B_{m}") for m in ["KO/TKO", "Submission", "Decision"]}
    if hit.any() and bt["actual_method"].notna().any():
        sub = bt[hit & bt["actual_method"].notna()]
        side = np.where(sub["y"] == 1, 0, 1)
        probs = np.column_stack([sub[mcols[m][0]].where(side == 0, sub[mcols[m][1]]) for m in mcols])
        guess = np.array(list(mcols))[probs.argmax(axis=1)]
        s["method_accuracy_given_winner"] = float((guess == sub["actual_method"].to_numpy()).mean())
        exact = bt["top_outcome"] == (bt["winner"] + " by " + bt["actual_method"].fillna("?"))
        s["exact_outcome_hit_rate"] = float(exact.mean())
    if "market_p_A" in bt and bt["market_p_A"].notna().sum() >= 30:
        mk = bt[bt["market_p_A"].notna()]
        ym = mk["y"].to_numpy(float)
        s["market"] = {
            "fights": int(len(mk)),
            "market_accuracy": float(((mk["market_p_A"] >= 0.5) == ym).mean()),
            "model_accuracy_same_fights": float(((mk["p_A"] >= 0.5) == ym).mean()),
            "market_log_loss": float(_ll(mk["market_p_A"], ym).mean()),
            "model_log_loss_same_fights": float(_ll(mk["p_A"], ym).mean()),
        }
        s["betting"] = simulate_bets(mk, edge)
    return s


def simulate_bets(mk: pd.DataFrame, edge: float = 0.05) -> dict:
    """Flat 1-unit bets whenever model probability beats the vig-free market by `edge`."""
    ea = mk["p_A"] - mk["market_p_A"]
    bet_a, bet_b = ea > edge, -ea > edge
    pnl = np.where(bet_a, np.where(mk["y"] == 1, mk["odds_A"] - 1, -1.0), 0.0) + \
          np.where(bet_b, np.where(mk["y"] == 0, mk["odds_B"] - 1, -1.0), 0.0)
    n = int(bet_a.sum() + bet_b.sum())
    return {"edge_threshold": edge, "bets": n, "profit_units": float(pnl.sum()),
            "roi": float(pnl.sum() / n) if n else float("nan")}


def calibration_table(bt: pd.DataFrame, bins=(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 1.0)) -> pd.DataFrame:
    pick = np.maximum(bt["p_A"], 1 - bt["p_A"])
    b = pd.cut(pick, list(bins), include_lowest=True)
    t = pd.DataFrame({"bucket": b.astype(str), "predicted": pick, "correct": bt["correct"]})
    return t.groupby("bucket", observed=True).agg(fights=("correct", "size"), predicted=("predicted", "mean"),
                                                  actual=("correct", "mean")).reset_index()


def breakdown(bt: pd.DataFrame, by: str) -> pd.DataFrame:
    d = bt.copy()
    if by == "year":
        d["year"] = pd.to_datetime(d["event_date"]).dt.year
    if by == "favourite_gap":
        d["favourite_gap"] = pd.cut(np.maximum(d["p_A"], 1 - d["p_A"]), [0.5, 0.6, 0.7, 1.0],
                                    include_lowest=True).astype(str)
    return d.groupby(by, observed=True).agg(fights=("correct", "size"), accuracy=("correct", "mean"),
                                            log_loss=("log_loss", "mean")).reset_index()


def rolling_accuracy(bt: pd.DataFrame, n: int = 200) -> pd.DataFrame:
    d = bt.sort_values("event_date")
    return pd.DataFrame({"event_date": pd.to_datetime(d["event_date"]),
                         "model": d["correct"].rolling(n, min_periods=50).mean(),
                         "elo baseline": ((d["elo_p_A"] >= 0.5) == d["y"]).astype(float)
                         .rolling(n, min_periods=50).mean()}).set_index("event_date")


def compare(feat: pd.DataFrame, start, end=None, progress: Progress = None, **kw):
    """Same backtest with and without the learning mechanisms."""
    half = (lambda f, m: progress(f * 0.5, "[static] " + m)) if progress else None
    half2 = (lambda f, m: progress(0.5 + f * 0.5, "[adaptive] " + m)) if progress else None
    static = walk_forward(feat, start, end, adaptive=False, progress=half, **kw)
    adaptive = walk_forward(feat, start, end, adaptive=True, progress=half2, **kw)
    return static, adaptive


# --------------------------------------------------------------------------- per-event slate backtest
def event_backtest(feat: pd.DataFrame, n_events: int = 10, learners: list[str] | None = None,
                   progress: Progress = None) -> pd.DataFrame:
    """Retrain before each of the last `n_events` events (using only earlier fights) and predict that
    whole slate, exactly as the app would have on fight night. Returns one row per fight with
    everything the card report needs to grade it: pick, confidence, outcomes, p_decision, winner, method."""
    from .pipeline import train_predictor  # noqa: F811  (local import keeps module load light)
    data = feat[feat["y"].notna() & feat["A_n_fights"].notna()]
    events = (data.groupby(["event_date", "event_name"]).size().reset_index()
              .sort_values("event_date").tail(n_events))
    preds = []
    for k, (_, ev) in enumerate(events.iterrows()):
        rows = data[(data["event_date"] == ev["event_date"]) & (data["event_name"] == ev["event_name"])]
        if progress:
            progress(k / max(len(events), 1), f"{ev['event_name']}: training on fights before {ev['event_date'].date()}")
        model = train_predictor(feat, learners, cutoff=ev["event_date"])
        P = model.predict(rows, explain=False)
        P["y"] = rows["y"].to_numpy()
        P["actual_method"] = rows["method"].to_numpy()
        P["winner"] = np.where(P["y"] == 1, P["A_name"], P["B_name"])
        preds.append(P)
    if progress:
        progress(1.0, "Slate backtest complete")
    if not preds:
        return pd.DataFrame()
    bt = pd.concat(preds, ignore_index=True)
    bt["correct"] = ((bt["p_A"] >= 0.5).astype(float) == bt["y"]).astype(int)
    bt["log_loss"] = _ll(bt["p_A"], bt["y"])
    return bt
