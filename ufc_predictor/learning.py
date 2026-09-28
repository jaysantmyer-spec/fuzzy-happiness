"""
Learning from mistakes.

The loop
--------
1. Every prediction you make for a real card is saved in a ledger (SQLite).
2. After the event, `grade()` matches each prediction to the real result.
3. `update_state()` pools graded ledger predictions with backtest predictions and
   re-learns: ensemble weights (Hedge on log loss), calibration temperature, and
   a per-fight 'hardness' score = |actual - predicted|.
4. `retrain()` fits the production model with recency x hardness sample weights,
   so fights it misjudged count more next time.

`run_cycle()` does update-data -> grade -> update_state -> retrain in one go.

Caveat worth knowing: upsets are partly luck, so leaning too hard on mistakes
fits noise. Keep `hardness_alpha` moderate and verify with backtest.compare().
"""
from __future__ import annotations

import json
import sqlite3
from typing import Callable, Optional

import numpy as np
import pandas as pd

from . import config, store
from .features import METHODS, build_features
from .models import blend, fit_temperature, hedge_weights
from .pipeline import Predictor, load_data, train_predictor

Progress = Optional[Callable[[float, str], None]]

LEDGER_COLS = ["fight_key", "created_at", "model_version", "event_name", "event_date", "A_name", "A_url",
               "B_name", "B_url", "p_A", "components", "method_probs", "top_outcome", "market_p_A",
               "actual_winner_side", "actual_method", "graded_at", "correct", "log_loss"]


# --------------------------------------------------------------------------- state
def load_state() -> dict:
    base = {"weights": {}, "temperature": 1.0, "hardness_alpha": config.DEFAULTS["hardness_alpha"],
            "half_life_years": config.DEFAULTS["half_life_years"], "learners": None, "history": []}
    if config.LEARNING_JSON.exists():
        base.update(json.loads(config.LEARNING_JSON.read_text()))
    return base


def save_state(state: dict) -> None:
    config.ensure_dirs()
    state["history"] = state.get("history", [])[-200:]
    config.LEARNING_JSON.write_text(json.dumps(state, indent=2, default=str))


def _log(state: dict, action: str, **detail) -> None:
    state.setdefault("history", []).append({"at": pd.Timestamp.now().isoformat(timespec="seconds"),
                                            "action": action, **detail})


# --------------------------------------------------------------------------- ledger
def _db():
    config.ensure_dirs()
    con = sqlite3.connect(config.LEDGER_DB)
    con.execute(f"CREATE TABLE IF NOT EXISTS ledger ({', '.join(c + ' TEXT' for c in LEDGER_COLS)}, "
                "PRIMARY KEY (fight_key))")
    return con


def fight_key(a_url: str, b_url: str, date) -> str:
    x, y = sorted([str(a_url), str(b_url)])
    return f"{x}|{y}|{pd.Timestamp(date).date()}"


def log_predictions(preds: pd.DataFrame, model_version: str = "") -> int:
    """Store pre-fight predictions. Re-logging overwrites until the fight is graded."""
    if preds.empty:
        return 0
    con = _db()
    n = 0
    for _, r in preds.iterrows():
        if str(r["fight_url"]).startswith("custom:"):
            continue
        key = fight_key(r["A_url"], r["B_url"], r["event_date"])
        graded = con.execute("SELECT graded_at FROM ledger WHERE fight_key=?", (key,)).fetchone()
        if graded and graded[0]:
            continue
        comps = {c[5:]: float(r[c]) for c in r.index if c.startswith("comp_")}
        meth = {f"{s}_{m}": float(r[f"{s}_{m}"]) for s in "AB" for m in METHODS}
        vals = [key, pd.Timestamp.now().isoformat(timespec="seconds"), model_version, r.get("event_name"),
                str(pd.Timestamp(r["event_date"]).date()), r["A_name"], r["A_url"], r["B_name"], r["B_url"],
                float(r["p_A"]), json.dumps(comps), json.dumps(meth), r.get("top_outcome"),
                None if pd.isna(r.get("market_p_A", np.nan)) else float(r["market_p_A"]),
                None, None, None, None, None]
        con.execute(f"INSERT OR REPLACE INTO ledger VALUES ({','.join('?' * len(LEDGER_COLS))})", vals)
        n += 1
    con.commit()
    con.close()
    return n


def read_ledger() -> pd.DataFrame:
    con = _db()
    df = pd.read_sql("SELECT * FROM ledger ORDER BY event_date DESC", con)
    con.close()
    for c in ("p_A", "market_p_A", "correct", "log_loss"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def grade() -> dict:
    """Match ungraded ledger rows with completed fights (same pair, within 3 days)."""
    fights = store.read(config.FIGHTS_CSV)
    con = _db()
    todo = pd.read_sql("SELECT * FROM ledger WHERE graded_at IS NULL", con)
    graded = 0
    if not todo.empty and not fights.empty:
        fights["pair"] = [tuple(sorted(x)) for x in zip(fights["f_1_url"].astype(str), fights["f_2_url"].astype(str))]
        from .features import method_label
        fights["m"] = method_label(fights["result"] if "result" in fights else pd.Series(index=fights.index))
        for _, r in todo.iterrows():
            pair = tuple(sorted([r["A_url"], r["B_url"]]))
            d = pd.Timestamp(r["event_date"])
            hit = fights[(fights["pair"] == pair) & ((fights["event_date"] - d).abs() <= pd.Timedelta(days=3))]
            if hit.empty:
                continue
            f = hit.iloc[-1]
            w = str(f["winner"])
            a_is_f1 = f["f_1_url"] == r["A_url"]
            a_name_real = f["f_1_name"] if a_is_f1 else f["f_2_name"]
            b_name_real = f["f_2_name"] if a_is_f1 else f["f_1_name"]
            side = "A" if w == a_name_real else "B" if w == b_name_real else "none"
            p = float(r["p_A"])
            if side == "none":
                correct, ll = None, None
            else:
                y = 1.0 if side == "A" else 0.0
                correct = int((p >= 0.5) == (y == 1))
                q = min(max(p, 1e-6), 1 - 1e-6)
                ll = float(-(y * np.log(q) + (1 - y) * np.log(1 - q)))
            con.execute("UPDATE ledger SET actual_winner_side=?, actual_method=?, graded_at=?, correct=?, "
                        "log_loss=? WHERE fight_key=?",
                        (side, None if pd.isna(f["m"]) else f["m"], pd.Timestamp.now().isoformat(timespec="seconds"),
                         correct, ll, r["fight_key"]))
            graded += 1
    con.commit()
    remaining = con.execute("SELECT COUNT(*) FROM ledger WHERE graded_at IS NULL").fetchone()[0]
    con.close()
    return {"graded_now": graded, "still_pending": remaining}


def graded_as_oof() -> pd.DataFrame:
    """Ledger rows in the same shape as backtest predictions (for pooling)."""
    lg = read_ledger()
    lg = lg[lg["actual_winner_side"].isin(["A", "B"])]
    if lg.empty:
        return pd.DataFrame()
    fights = store.read(config.FIGHTS_CSV)
    fights["pair"] = [tuple(sorted(x)) for x in zip(fights["f_1_url"].astype(str), fights["f_2_url"].astype(str))]
    lookup = {(p, str(pd.Timestamp(d).date())): u for p, d, u in zip(fights["pair"], fights["event_date"], fights["fight_url"])}
    rows = []
    for _, r in lg.iterrows():
        comps = json.loads(r["components"] or "{}")
        key = (tuple(sorted([r["A_url"], r["B_url"]])), r["event_date"])
        row = {"fight_url": lookup.get(key, r["fight_key"]), "event_date": pd.Timestamp(r["event_date"]),
               "p_A": r["p_A"], "y": 1.0 if r["actual_winner_side"] == "A" else 0.0, "source": "ledger"}
        row.update({f"comp_{k}": v for k, v in comps.items()})
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- learning
def pooled_oof() -> pd.DataFrame:
    parts = []
    bt = store.read(config.BACKTEST_CSV)
    if not bt.empty:
        bt["source"] = "backtest"
        parts.append(bt)
    lg = graded_as_oof()
    if not lg.empty:
        parts.append(lg)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    # ledger (real, pre-fight) predictions win over backtest duplicates
    df["_pri"] = (df["source"] == "ledger").astype(int)
    df = df.sort_values(["_pri"]).drop_duplicates("fight_url", keep="last").drop(columns="_pri")
    return df.sort_values("event_date").reset_index(drop=True)


def update_state(state: dict | None = None) -> dict:
    state = state or load_state()
    oof = pooled_oof()
    if len(oof) < 150:
        _log(state, "update_skipped", reason=f"only {len(oof)} graded predictions (need 150)")
        save_state(state)
        return state
    comp_cols = [c for c in oof.columns if c.startswith("comp_")]
    recent = oof.tail(config.DEFAULTS["hedge_window"]).dropna(subset=comp_cols)
    if len(recent) >= 150 and comp_cols:
        comp = recent[comp_cols].copy()
        comp.columns = [c[5:] for c in comp.columns]
        old = dict(state.get("weights", {}))
        state["weights"] = hedge_weights(comp, recent["y"], eta=config.DEFAULTS["hedge_eta"])
        state["temperature"] = fit_temperature(blend(comp, state["weights"]), recent["y"])
        _log(state, "reweighted", old_weights=old, new_weights=state["weights"],
             temperature=state["temperature"], n=len(recent))
    hard = np.abs(oof["y"] - oof["p_A"])
    store.write(pd.DataFrame({"fight_url": oof["fight_url"], "hardness": hard, "source": oof["source"]}),
                config.HARDNESS_CSV)
    save_state(state)
    return state


def load_hardness() -> dict:
    h = store.read(config.HARDNESS_CSV)
    return {} if h.empty else dict(zip(h["fight_url"], h["hardness"]))


def retrain(learners: list[str] | None = None, state: dict | None = None, use_hardness: bool = True) -> Predictor:
    state = state or load_state()
    fights, fighters = load_data(include_upcoming=False)
    feat = build_features(fights, fighters)
    learners = learners or state.get("learners")
    pred = train_predictor(feat, learners, half_life_years=state["half_life_years"],
                           hardness=load_hardness() if use_hardness else None,
                           hardness_alpha=state["hardness_alpha"] if use_hardness else 0.0,
                           weights=state.get("weights"), temperature=state.get("temperature", 1.0))
    # keep only weights for learners that exist in this model
    pred.weights = {k: v for k, v in pred.weights.items() if k in pred.win.models}
    pred.save()
    state["learners"] = list(pred.win.learners)
    _log(state, "retrained", **{k: pred.meta[k] for k in ("n_train", "data_through", "version")})
    save_state(state)
    return pred


def run_cycle(progress: Progress = None, scrape: bool = True, odds_api_key: str | None = None) -> dict:
    say = progress or (lambda f, m: None)
    out = {}
    if scrape:
        from .scraper import update_all
        out["update"] = update_all(progress=lambda f, m: say(0.6 * f, m), odds_api_key=odds_api_key)
    say(0.62, "Grading past predictions…")
    out["grade"] = grade()
    say(0.7, "Re-learning ensemble weights and calibration…")
    st = update_state()
    out["weights"], out["temperature"] = st.get("weights"), st.get("temperature")
    say(0.8, "Retraining production model with mistake weighting…")
    out["model"] = retrain(state=st).meta
    say(1.0, "Learning cycle complete")
    return out


# --------------------------------------------------------------------------- analysis
def mistake_report(bt: pd.DataFrame, top: int = 15) -> dict:
    if bt is None or bt.empty:
        return {}
    d = bt.copy()
    d["pick_prob"] = np.maximum(d["p_A"], 1 - d["p_A"])
    d["picked"] = np.where(d["p_A"] >= 0.5, d["A_name"], d["B_name"])
    misses = d[d["correct"] == 0].sort_values("pick_prob", ascending=False)
    cols = ["event_date", "A_name", "B_name", "picked", "pick_prob", "winner", "actual_method"]
    d["debut_involved"] = np.where((d["A_n_fights"] == 0) | (d["B_n_fights"] == 0), "debut involved", "both experienced")
    return {
        "confident_misses": misses[[c for c in cols if c in misses]].head(top),
        "by_weight_class": d.groupby("weight_class").agg(fights=("correct", "size"),
                                                         accuracy=("correct", "mean")).query("fights >= 15")
                            .sort_values("accuracy").reset_index(),
        "by_experience": d.groupby("debut_involved").agg(fights=("correct", "size"),
                                                         accuracy=("correct", "mean")).reset_index(),
        "by_method": d.groupby("actual_method").agg(fights=("correct", "size"),
                                                    accuracy=("correct", "mean")).reset_index(),
        "overconfidence": float((d.loc[d["pick_prob"] >= 0.7, "correct"].mean()
                                 if (d["pick_prob"] >= 0.7).any() else np.nan)),
    }
