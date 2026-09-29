"""Glue: load data -> features -> train -> predict (cards, custom match-ups)."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd

from . import config, store
from .features import (CONTEXT_FEATURES, DIFF_FEATURES, METHODS, build_features, label, method_matrix)
from .models import MethodModel, WinModel, blend, preset

SHORT = {"KO/TKO": "KO/TKO", "Submission": "Submission", "Decision": "Decision"}


# --------------------------------------------------------------------------- data
def load_data(include_upcoming: bool = True, extra: pd.DataFrame | None = None):
    fights = store.read(config.FIGHTS_CSV)
    fighters = store.read(config.FIGHTERS_CSV)
    parts = [fights]
    if include_upcoming:
        up = store.read(config.UPCOMING_CSV)
        if not up.empty:
            up = up[~up["fight_url"].isin(fights.get("fight_url", pd.Series(dtype=str)))].copy()
            up["winner"] = np.nan
            parts.append(up)
    if extra is not None and not extra.empty:
        parts.append(extra)
    allf = pd.concat([p for p in parts if not p.empty], ignore_index=True) if any(
        not p.empty for p in parts) else pd.DataFrame()
    return allf, fighters


def features_now(extra: pd.DataFrame | None = None) -> pd.DataFrame:
    fights, fighters = load_data(True, extra)
    if fights.empty:
        return pd.DataFrame()
    return build_features(fights, fighters)


def training_rows(feat: pd.DataFrame, before=None) -> pd.DataFrame:
    t = feat[feat["y"].notna()]
    t = t[t["A_n_fights"].notna()]
    if before is not None:
        t = t[t["event_date"] < pd.Timestamp(before)]
    return t


def recency_weights(dates: pd.Series, ref, half_life_years: float) -> np.ndarray:
    if not half_life_years or half_life_years <= 0:
        return np.ones(len(dates))
    age_years = (pd.Timestamp(ref) - pd.to_datetime(dates)).dt.days.to_numpy(float) / 365.25
    return 0.5 ** (np.clip(age_years, 0, None) / half_life_years)


def hardness_weights(fight_urls: pd.Series, hardness: dict | None, alpha: float) -> np.ndarray:
    """1 + alpha * |y - p_predicted| for fights the model predicted before seeing them."""
    if not hardness or alpha <= 0:
        return np.ones(len(fight_urls))
    h = fight_urls.map(hardness).fillna(0.0).to_numpy(float)
    return 1.0 + alpha * np.clip(h, 0, 1)


# --------------------------------------------------------------------------- predictor
@dataclass
LAST_LOAD_ERROR: str | None = None  # set by Predictor.load when a saved model can't be read


class Predictor:
    win: WinModel
    method: MethodModel
    weights: dict = field(default_factory=dict)
    temperature: float = 1.0
    meta: dict = field(default_factory=dict)

    def save(self, path=config.MODEL_FILE):
        config.ensure_dirs()
        joblib.dump(self, path)

    @staticmethod
    def load(path=config.MODEL_FILE) -> "Predictor | None":
        """Returns None if there is no model, or if the saved one was made with an incompatible
        library version (a pickle from a different scikit-learn / Python build). Callers may retrain."""
        global LAST_LOAD_ERROR
        LAST_LOAD_ERROR = None
        try:
            return joblib.load(path)
        except FileNotFoundError:
            return None
        except Exception as ex:  # ModuleNotFoundError / AttributeError from version drift
            LAST_LOAD_ERROR = f"{type(ex).__name__}: {ex}"
            return None

    def predict(self, rows: pd.DataFrame, explain: bool = True) -> pd.DataFrame:
        if rows.empty:
            return pd.DataFrame()
        D, C = rows[DIFF_FEATURES], rows[CONTEXT_FEATURES]
        comp = self.win.predict_components(D, C)
        p = blend(comp, self.weights, self.temperature)
        mA = self.method.predict(method_matrix(rows, np.ones(len(rows), bool)))
        mB = self.method.predict(method_matrix(rows, np.zeros(len(rows), bool)))
        out = pd.DataFrame({
            "fight_url": rows["fight_url"].to_numpy(),
            "event_name": rows.get("event_name", pd.Series(index=rows.index, dtype=object)).to_numpy(),
            "event_date": rows["event_date"].to_numpy(),
            "weight_class": rows.get("weight_class", pd.Series(index=rows.index, dtype=object)).to_numpy(),
            "num_rounds": rows["num_rounds"].to_numpy(),
            "A_name": rows["f_1_name"].to_numpy(), "B_name": rows["f_2_name"].to_numpy(),
            "A_url": rows["f_1_url"].to_numpy(), "B_url": rows["f_2_url"].to_numpy(),
            "p_A": p, "p_B": 1 - p,
            "elo_p_A": (rows["elo_edge"] + 0.5).to_numpy(),
            "A_n_fights": rows["A_n_fights"].to_numpy(), "B_n_fights": rows["B_n_fights"].to_numpy(),
        })
        for c in comp.columns:
            out[f"comp_{c}"] = comp[c].to_numpy()
        for m in METHODS:
            out[f"A_{SHORT[m]}"] = p * mA[m].to_numpy()
            out[f"B_{SHORT[m]}"] = (1 - p) * mB[m].to_numpy()
        out["p_decision"] = out["A_Decision"] + out["B_Decision"]
        out["p_finish"] = 1 - out["p_decision"]
        out["pick"] = np.where(p >= 0.5, out["A_name"], out["B_name"])
        out["pick_prob"] = np.maximum(p, 1 - p)
        out["confidence"] = pd.cut(out["pick_prob"], [0, 0.55, 0.62, 0.70, 1.01],
                                   labels=["Coin flip", "Lean", "Solid", "Strong"]).astype(str)
        spread = comp.max(axis=1) - comp.min(axis=1)
        out["model_spread"] = spread.to_numpy()
        tops = []
        for i in range(len(out)):
            cand = [(f"{out.at[i, 'A_name']} by {m}", out.at[i, f"A_{m}"]) for m in SHORT.values()] + \
                   [(f"{out.at[i, 'B_name']} by {m}", out.at[i, f"B_{m}"]) for m in SHORT.values()]
            tops.append(sorted(cand, key=lambda t: -t[1]))
        out["outcomes"] = tops
        out["top_outcome"] = [t[0][0] for t in tops]
        out["top_outcome_prob"] = [t[0][1] for t in tops]
        if explain:
            ex = self.win.explain(D, C)
            out["why"] = [[_why(rows.iloc[i], f, c) for f, c in e] for i, e in enumerate(ex)]
        return out


def _why(row: pd.Series, feat: str, contrib: float) -> dict:
    base = feat.replace("diff_", "")
    a = row.get(f"A_{base}", np.nan)
    b = row.get(f"B_{base}", np.nan)
    if base == "elo_edge":
        a, b = row.get("A_elo"), row.get("B_elo")
    return {"feature": label(feat), "favours": "A" if contrib > 0 else "B",
            "strength": abs(contrib), "A": a, "B": b}


def train_predictor(feat: pd.DataFrame, learners: list[str] | None = None, cutoff=None,
                    half_life_years: float | None = None, hardness: dict | None = None,
                    hardness_alpha: float = 0.0, weights: dict | None = None,
                    temperature: float = 1.0, seed: int = 42) -> Predictor:
    half_life_years = config.DEFAULTS["half_life_years"] if half_life_years is None else half_life_years
    tr = training_rows(feat, cutoff)
    if len(tr) < 200:
        raise ValueError(f"Only {len(tr)} completed fights with history - need more data.")
    ref = cutoff or tr["event_date"].max()
    w = recency_weights(tr["event_date"], ref, half_life_years) * \
        hardness_weights(tr["fight_url"], hardness, hardness_alpha)
    win = WinModel(learners or preset("balanced"), seed).fit(tr[DIFF_FEATURES], tr[CONTEXT_FEATURES],
                                                             tr["y"].astype(int), w)
    mt = tr[tr["method"].isin(METHODS)]
    method = MethodModel(seed).fit(method_matrix(mt, mt["y"].to_numpy() == 1), mt["method"],
                                   recency_weights(mt["event_date"], ref, half_life_years))
    return Predictor(win, method, weights or {}, temperature, {
        "trained_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "n_train": int(len(tr)), "data_through": str(pd.Timestamp(tr["event_date"].max()).date()),
        "learners": list(win.learners), "half_life_years": half_life_years,
        "hardness_alpha": hardness_alpha, "version": pd.Timestamp.now().strftime("%Y%m%d-%H%M%S"),
    })


# --------------------------------------------------------------------------- upcoming / custom
def upcoming_rows(feat: pd.DataFrame) -> pd.DataFrame:
    up = store.read(config.UPCOMING_CSV)
    if up.empty or feat.empty:
        return pd.DataFrame()
    return feat[feat["fight_url"].isin(up["fight_url"]) & ~feat["completed"]].copy()


def custom_matchup(a_url: str, b_url: str, fighters: pd.DataFrame, num_rounds: int = 3,
                   title: bool = False, date=None, weight_class: str | None = None) -> pd.DataFrame:
    names = fighters.drop_duplicates("fighter_url").set_index("fighter_url")["fighter_name"]
    date = pd.Timestamp(date or pd.Timestamp.today().normalize())
    return pd.DataFrame([{
        "fight_url": f"custom:{a_url}|{b_url}", "event_name": "Custom match-up", "event_date": date,
        "f_1_name": names.get(a_url, a_url), "f_1_url": a_url, "f_2_name": names.get(b_url, b_url),
        "f_2_url": b_url, "winner": np.nan, "num_rounds": num_rounds, "title_fight": title,
        "weight_class": weight_class, "bout_order": 99,
        "gender": "F" if weight_class and "women" in weight_class.lower() else "M",
    }])


def predict_custom(pred: Predictor, a_url: str, b_url: str, **kw) -> pd.DataFrame:
    fighters = store.read(config.FIGHTERS_CSV)
    row = custom_matchup(a_url, b_url, fighters, **kw)
    if not kw.get("weight_class"):
        # infer weight class from the fighters' most recent bouts
        fights = store.read(config.FIGHTS_CSV)
        recent = fights[(fights["f_1_url"].isin([a_url, b_url])) | (fights["f_2_url"].isin([a_url, b_url]))]
        if not recent.empty and "weight_class" in recent:
            row["weight_class"] = recent.sort_values("event_date")["weight_class"].iloc[-1]
            row["gender"] = recent.sort_values("event_date").get("gender", pd.Series(["M"])).iloc[-1]
    feat = features_now(extra=row)
    return pred.predict(feat[feat["fight_url"] == row["fight_url"].iloc[0]])


# --------------------------------------------------------------------------- odds
def _key(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z]", "", s)


def attach_market(preds: pd.DataFrame) -> pd.DataFrame:
    """Add vig-free market probabilities (median across books) and model edge."""
    odds = store.read(config.ODDS_CSV)
    out = preds.copy()
    out["market_p_A"] = np.nan
    if odds.empty or preds.empty:
        return out
    odds = odds.dropna(subset=["odds_1", "odds_2"])
    rows = []
    for _, r in odds.iterrows():
        rows.append((_key(r["fighter_1"]), _key(r["fighter_2"]), float(r["odds_1"]), float(r["odds_2"])))
        rows.append((_key(r["fighter_2"]), _key(r["fighter_1"]), float(r["odds_2"]), float(r["odds_1"])))
    o = pd.DataFrame(rows, columns=["a", "b", "oa", "ob"]).groupby(["a", "b"]).median().reset_index()
    o["market_p_A"] = (1 / o["oa"]) / (1 / o["oa"] + 1 / o["ob"])
    out["a"], out["b"] = out["A_name"].map(_key), out["B_name"].map(_key)
    out = out.drop(columns="market_p_A").merge(o[["a", "b", "market_p_A", "oa", "ob"]], on=["a", "b"], how="left")
    out = out.rename(columns={"oa": "odds_A", "ob": "odds_B"}).drop(columns=["a", "b"])
    out["edge_A"] = out["p_A"] - out["market_p_A"]
    return out


# --------------------------------------------------------------------------- manually entered cards
_VS_RX = re.compile(r"\s+(?:vs\.?|v\.?|versus|-|–|—)\s+", re.I)


def _name_index(fights: pd.DataFrame, fighters: pd.DataFrame) -> dict:
    """normalised name -> (url, display name), preferring the most recently active fighter."""
    long = pd.concat([fights[["f_1_url", "f_1_name", "event_date"]].set_axis(["url", "name", "d"], axis=1),
                      fights[["f_2_url", "f_2_name", "event_date"]].set_axis(["url", "name", "d"], axis=1)])
    long = long.dropna().sort_values("d")
    idx = {}
    for url, name in zip(long["url"], long["name"]):
        idx[_key(name)] = (url, name)          # later (more recent) rows overwrite earlier ones
    for url, name in zip(fighters.get("fighter_url", []), fighters.get("fighter_name", [])):
        idx.setdefault(_key(name), (url, name))
    return idx


def match_fighter(name: str, idx: dict) -> tuple[str | None, str | None, str]:
    """Return (url, display name, note). Exact match on the normalised name, then a close match."""
    import difflib
    k = _key(name)
    if k in idx:
        return idx[k][0], idx[k][1], "ok"
    close = difflib.get_close_matches(k, list(idx), n=1, cutoff=0.85)
    if close:
        return idx[close[0]][0], idx[close[0]][1], f"matched to {idx[close[0]][1]}"
    return None, None, "not found"


def card_from_text(text: str, event_name: str, event_date) -> tuple[pd.DataFrame, list[str]]:
    """
    Turn pasted lines like 'Fighter A vs Fighter B' into upcoming bouts.
    The first line is the main event (5 rounds). Add '(title)' or '*' to a line for a title fight,
    or '(5)' to force five rounds. Returns the bouts and a list of human-readable notes.
    """
    fights = store.read(config.FIGHTS_CSV)
    fighters = store.read(config.FIGHTERS_CSV)
    if fights.empty:
        raise ValueError("Import or scrape fight history first, so the names can be matched.")
    idx = _name_index(fights, fighters)
    date = pd.Timestamp(event_date).normalize()
    rows, notes = [], []
    order = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        title = bool(re.search(r"\(title\)|\*|title", line, re.I))
        five = bool(re.search(r"\(5\)|5 ?rounds?", line, re.I))
        clean = re.sub(r"\(.*?\)|\*|title fight|title|\d+ ?rounds?", "", line, flags=re.I).strip(" ,;")
        parts = _VS_RX.split(clean, maxsplit=1)
        if len(parts) != 2:
            notes.append(f"Skipped '{line}': write it as 'Fighter A vs Fighter B'.")
            continue
        (au, an, anote), (bu, bn, bnote) = match_fighter(parts[0], idx), match_fighter(parts[1], idx)
        if au is None or bu is None:
            who = parts[0] if au is None else parts[1]
            notes.append(f"Skipped '{line}': '{who}' is not in the fight history (debutant, or spelled differently).")
            continue
        for nm, note in ((parts[0], anote), (parts[1], bnote)):
            if note.startswith("matched"):
                notes.append(f"'{nm}' {note}.")
        recent = fights[(fights["f_1_url"].isin([au, bu])) | (fights["f_2_url"].isin([au, bu]))].sort_values("event_date")
        wc = recent["weight_class"].iloc[-1] if not recent.empty and "weight_class" in recent else None
        rows.append({
            "event_url": f"manual:{_key(event_name)}:{date.date()}", "fight_url": f"manual:{au}|{bu}|{date.date()}",
            "bout_order": order, "f_1_name": an, "f_1_url": au, "f_2_name": bn, "f_2_url": bu,
            "weight_class": wc, "title_fight": title, "event_name": event_name, "event_date": date,
            "event_location": None, "num_rounds": 5 if (order == 0 or title or five) else 3,
            "gender": "F" if wc and "women" in str(wc).lower() else "M",
        })
        order += 1
    return pd.DataFrame(rows), notes


def save_manual_card(card: pd.DataFrame) -> int:
    """Replace any earlier version of this event in upcoming.csv and drop cards whose date has passed."""
    up = store.read(config.UPCOMING_CSV)
    if not up.empty:
        up = up[up["event_url"] != card["event_url"].iloc[0]]
        up = up[pd.to_datetime(up["event_date"]) >= pd.Timestamp.today().normalize() - pd.Timedelta(days=1)]
    out = pd.concat([up, card], ignore_index=True) if not up.empty else card
    store.write(out, config.UPCOMING_CSV)
    return len(card)
