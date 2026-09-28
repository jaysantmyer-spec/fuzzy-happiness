"""
Leak-free pre-fight features.

Borrowed from `silver-to-gold-ufc-data-processing.ipynb`: the long format
(two rows per fight) and exclusive cumulative sums instead of pandas rolling.

Deliberate differences
----------------------
* Only information available BEFORE the bout is used. (The LR/RF notebook built
  `diff_` features from the fight's own strike totals, which leaks the result.)
* A compact feature set (career + last-5 windows, Elo, physicals, activity)
  instead of ~13 windows x hundreds of metrics: similar signal, a fraction of
  the size, so walk-forward backtests run in minutes.
* Every model feature is a difference A - B (antisymmetric) or symmetric
  context, so swapping the corners exactly mirrors the prediction. UFCStats
  lists the winner first, and this design stops a model from exploiting that.
* Upcoming bouts are appended as rows with no result; because every statistic
  is shifted, they automatically receive each fighter's up-to-date profile.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STATS = ["knockdowns", "sig_strikes_succ", "sig_strikes_att", "total_strikes_succ",
         "total_strikes_att", "takedown_succ", "takedown_att", "submission_att",
         "reversals", "ctrl_time_sec"]
METHODS = ["KO/TKO", "Submission", "Decision"]
WINDOWS = {"car": None, "l5": 5}

RATE_NAMES = ["slpm", "sapm", "sig_diff_pm", "str_acc", "str_def", "td15", "td_acc", "td_def",
              "td_abs15", "sub15", "ctrl_pct", "ctrl_abs_pct", "kd15", "kd_abs15",
              "win_rate", "finish_rate", "ko_win_rate", "sub_win_rate", "ko_loss_rate",
              "sub_loss_rate", "dec_rate"]
FIGHTER_FEATURES = ([f"{r}_{w}" for w in WINDOWS for r in RATE_NAMES] +
                    ["win_rate_l3", "n_fights", "win_streak", "loss_streak", "days_since",
                     "avg_minutes", "elo", "elo_trend", "opp_elo_avg", "age", "height",
                     "reach", "reach_ratio", "southpaw", "switch"])
DIFF_FEATURES = [f"diff_{f}" for f in FIGHTER_FEATURES] + ["elo_edge"]
CONTEXT_FEATURES = ["num_rounds", "title_fight", "is_female", "weight_lbs", "stance_mismatch"]
METHOD_BASE = ["ko_win_rate_car", "sub_win_rate_car", "finish_rate_car", "ko_loss_rate_car",
               "sub_loss_rate_car", "dec_rate_car", "kd15_car", "kd_abs15_car", "sub15_car",
               "td15_car", "slpm_car", "sapm_car", "str_def_car", "n_fights", "age",
               "avg_minutes", "elo"]
METHOD_FEATURES = [f"W_{f}" for f in METHOD_BASE] + [f"L_{f}" for f in METHOD_BASE] + \
                  ["num_rounds", "is_female", "weight_lbs"]

LABELS = {
    "slpm": "sig. strikes landed / min", "sapm": "sig. strikes absorbed / min",
    "sig_diff_pm": "striking differential / min", "str_acc": "striking accuracy",
    "str_def": "striking defence", "td15": "takedowns / 15 min", "td_acc": "takedown accuracy",
    "td_def": "takedown defence", "td_abs15": "takedowns conceded / 15 min",
    "sub15": "submission attempts / 15 min", "ctrl_pct": "control time share",
    "ctrl_abs_pct": "time controlled by opponents", "kd15": "knockdowns / 15 min",
    "kd_abs15": "knockdowns absorbed / 15 min", "win_rate": "win rate",
    "finish_rate": "finish rate", "ko_win_rate": "KO win rate", "sub_win_rate": "submission win rate",
    "ko_loss_rate": "KO loss rate", "sub_loss_rate": "submission loss rate",
    "dec_rate": "decision rate", "n_fights": "UFC experience", "win_streak": "win streak",
    "loss_streak": "losing streak", "days_since": "layoff (days)", "avg_minutes": "avg fight length",
    "elo": "Elo rating", "elo_trend": "Elo trend (last 3)", "opp_elo_avg": "strength of schedule",
    "age": "age", "height": "height", "reach": "reach", "reach_ratio": "reach / height",
    "southpaw": "southpaw", "switch": "switch stance", "elo_edge": "Elo win expectancy",
}
WINDOW_LABELS = {"car": "career", "l5": "last 5", "l3": "last 3"}


def label(feature: str) -> str:
    f = feature.replace("diff_", "")
    for w, wl in WINDOW_LABELS.items():
        if f.endswith("_" + w):
            return f"{LABELS.get(f[:-len(w) - 1], f)} ({wl})"
    return LABELS.get(f, f)


# --------------------------------------------------------------------------- basics
def weight_lbs(wc: pd.Series) -> pd.Series:
    s = wc.fillna("").astype(str).str.lower()
    out = pd.Series(np.nan, index=wc.index)
    for key, lbs in [("strawweight", 115), ("flyweight", 125), ("bantamweight", 135),
                     ("featherweight", 145), ("light heavyweight", 205), ("lightweight", 155),
                     ("welterweight", 170), ("middleweight", 185), ("heavyweight", 265)]:
        out[out.isna() & s.str.contains(key)] = lbs
    return out


def method_label(result: pd.Series) -> pd.Series:
    r = result.fillna("").astype(str).str.upper()
    out = pd.Series(np.nan, index=result.index, dtype=object)
    out[r.str.contains("DEC")] = "Decision"
    out[r.str.contains("SUB")] = "Submission"
    out[r.str.contains("KO")] = "KO/TKO"
    return out


def fight_minutes(df: pd.DataFrame) -> pd.Series:
    fr = pd.to_numeric(df["finish_round"], errors="coerce") if "finish_round" in df else pd.Series(np.nan, index=df.index)
    ft = df["finish_time"] if "finish_time" in df else pd.Series(None, index=df.index, dtype=object)
    t = ft.astype(str).str.extract(r"(\d+):(\d+)")
    return (fr - 1) * 5 + t[0].astype(float) + t[1].astype(float) / 60


def _num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def prepare(fights: pd.DataFrame) -> pd.DataFrame:
    f = fights.copy()
    f["event_date"] = pd.to_datetime(f["event_date"], errors="coerce")
    f = f.dropna(subset=["event_date", "f_1_url", "f_2_url"])
    f = f[f["f_1_url"] != f["f_2_url"]]
    if "bout_order" not in f:
        f["bout_order"] = 0
    f["bout_order"] = _num(f["bout_order"]).fillna(0)
    # chronological; within a card the main event (order 0) happens last
    f = f.sort_values(["event_date", "bout_order"], ascending=[True, False], kind="mergesort")
    f = f.drop_duplicates("fight_url", keep="last").reset_index(drop=True)
    f["_idx"] = np.arange(len(f))
    w = f["winner"].astype(str) if "winner" in f else pd.Series("nan", index=f.index)
    lw = w.str.lower()
    side = np.where(w == f["f_1_name"].astype(str), "f_1",
           np.where(w == f["f_2_name"].astype(str), "f_2",
           np.where(lw.str.contains("draw"), "draw",
           np.where(lw.str.contains("no contest"), "nc", "none"))))
    f["_side"] = side
    f["minutes"] = fight_minutes(f)
    f["method"] = method_label(f["result"] if "result" in f else pd.Series(None, index=f.index, dtype=object))
    f.loc[~f["_side"].isin(["f_1", "f_2"]), "method"] = np.nan
    for p in ("f_1", "f_2"):
        for s in STATS:
            if f"{p}_{s}" not in f:
                f[f"{p}_{s}"] = np.nan
    return f


# --------------------------------------------------------------------------- Elo
def compute_elo(f: pd.DataFrame, k: float = 32.0, finish_bonus: float = 0.25, init: float = 1500.0):
    ratings, counts = {}, {}
    pre1, pre2 = np.empty(len(f)), np.empty(len(f))
    for i, (a, b, side, m) in enumerate(zip(f["f_1_url"], f["f_2_url"], f["_side"], f["method"])):
        ra, rb = ratings.get(a, init), ratings.get(b, init)
        pre1[i], pre2[i] = ra, rb
        if side not in ("f_1", "f_2", "draw"):
            continue
        ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
        sa = 1.0 if side == "f_1" else 0.0 if side == "f_2" else 0.5
        kk = k * (1 + finish_bonus if m in ("KO/TKO", "Submission") else 1.0)
        ka = kk * (1.5 if counts.get(a, 0) < 5 else 1.0)
        kb = kk * (1.5 if counts.get(b, 0) < 5 else 1.0)
        ratings[a] = ra + ka * (sa - ea)
        ratings[b] = rb + kb * (ea - sa)
        counts[a], counts[b] = counts.get(a, 0) + 1, counts.get(b, 0) + 1
    return pre1, pre2


# --------------------------------------------------------------------------- long format
def _long(f: pd.DataFrame, elo1, elo2) -> pd.DataFrame:
    frames = []
    for me, op, my_elo, op_elo in (("f_1", "f_2", elo1, elo2), ("f_2", "f_1", elo2, elo1)):
        side, m = f["_side"], f["method"]
        done = side.isin(["f_1", "f_2", "draw"])
        win, loss = side == me, side == op
        has = done & _num(f[f"{me}_sig_strikes_att"]).notna() & f["minutes"].notna()
        d = {
            "_idx": f["_idx"].to_numpy(), "side": me, "date": f["event_date"].to_numpy(),
            "fighter": f[f"{me}_url"].to_numpy(), "elo": my_elo, "opp_elo": op_elo,
            "done": done.astype(float), "win": win.astype(float), "loss": loss.astype(float),
            "ko_win": (win & (m == "KO/TKO")).astype(float), "sub_win": (win & (m == "Submission")).astype(float),
            "ko_loss": (loss & (m == "KO/TKO")).astype(float), "sub_loss": (loss & (m == "Submission")).astype(float),
            "dec": (done & (m == "Decision")).astype(float),
            "stat_min": f["minutes"].where(has, 0).fillna(0).to_numpy(),
            "min_all": f["minutes"].where(done, 0).fillna(0).to_numpy(),
            "opp_elo_done": np.where(done, op_elo, 0.0),
        }
        for s in STATS:
            d[f"own_{s}"] = _num(f[f"{me}_{s}"]).where(has, 0).fillna(0).to_numpy()
            d[f"opp_{s}"] = _num(f[f"{op}_{s}"]).where(has, 0).fillna(0).to_numpy()
        frames.append(pd.DataFrame(d))
    L = pd.concat(frames, ignore_index=True)
    return L.sort_values(["fighter", "_idx"], kind="mergesort").reset_index(drop=True)


def _div(a, b):
    b = b.astype(float)
    return a / b.where(b > 0)


def _rates(S: pd.DataFrame, w: str) -> dict:
    sm = S["stat_min"]
    slpm, sapm = _div(S["own_sig_strikes_succ"], sm), _div(S["opp_sig_strikes_succ"], sm)
    n = S["done"]
    return {
        f"slpm_{w}": slpm, f"sapm_{w}": sapm, f"sig_diff_pm_{w}": slpm - sapm,
        f"str_acc_{w}": _div(S["own_sig_strikes_succ"], S["own_sig_strikes_att"]),
        f"str_def_{w}": 1 - _div(S["opp_sig_strikes_succ"], S["opp_sig_strikes_att"]),
        f"td15_{w}": 15 * _div(S["own_takedown_succ"], sm),
        f"td_acc_{w}": _div(S["own_takedown_succ"], S["own_takedown_att"]),
        f"td_def_{w}": 1 - _div(S["opp_takedown_succ"], S["opp_takedown_att"]),
        f"td_abs15_{w}": 15 * _div(S["opp_takedown_succ"], sm),
        f"sub15_{w}": 15 * _div(S["own_submission_att"], sm),
        f"ctrl_pct_{w}": _div(S["own_ctrl_time_sec"], sm * 60),
        f"ctrl_abs_pct_{w}": _div(S["opp_ctrl_time_sec"], sm * 60),
        f"kd15_{w}": 15 * _div(S["own_knockdowns"], sm),
        f"kd_abs15_{w}": 15 * _div(S["opp_knockdowns"], sm),
        f"win_rate_{w}": (S["win"] + 1) / (n + 2),           # Laplace-smoothed
        f"finish_rate_{w}": _div(S["ko_win"] + S["sub_win"], n),
        f"ko_win_rate_{w}": _div(S["ko_win"], n), f"sub_win_rate_{w}": _div(S["sub_win"], n),
        f"ko_loss_rate_{w}": _div(S["ko_loss"], n), f"sub_loss_rate_{w}": _div(S["sub_loss"], n),
        f"dec_rate_{w}": _div(S["dec"], n),
    }


def _fighter_features(L: pd.DataFrame, fighters: pd.DataFrame) -> pd.DataFrame:
    sum_cols = [c for c in L.columns if c not in ("_idx", "side", "date", "fighter", "elo", "opp_elo")]
    X = L[sum_cols].astype(float)
    g = L["fighter"]
    C = X.groupby(g, sort=False).cumsum() - X            # exclusive: fights strictly before
    out = {}
    for w, n in WINDOWS.items():
        S = C if n is None else C - C.groupby(g, sort=False).shift(n).fillna(0)
        out.update(_rates(S, w))
    S3 = C - C.groupby(g, sort=False).shift(3).fillna(0)
    out["win_rate_l3"] = (S3["win"] + 1) / (S3["done"] + 2)
    out["n_fights"] = C["done"]
    out["avg_minutes"] = _div(C["min_all"], C["done"])
    out["opp_elo_avg"] = _div(C["opp_elo_done"], C["done"])
    out["elo"] = L["elo"]
    out["elo_trend"] = L["elo"] - L.groupby(g, sort=False)["elo"].shift(3)

    # streaks prior to the bout
    for col, name in (("win", "win_streak"), ("loss", "loss_streak")):
        s = L[col].astype(int)
        block = (s == 0).astype(int).groupby(g, sort=False).cumsum()
        run = s.groupby([g, block], sort=False).cumsum()
        out[name] = run.groupby(g, sort=False).shift(1).fillna(0)
    prev_date = L.groupby(g, sort=False)["date"].shift(1)
    out["days_since"] = (pd.to_datetime(L["date"]) - pd.to_datetime(prev_date)).dt.days

    F = pd.DataFrame(out)
    F.insert(0, "_idx", L["_idx"].to_numpy())
    F.insert(1, "side", L["side"].to_numpy())
    F["fighter"] = L["fighter"].to_numpy()
    F["date"] = L["date"].to_numpy()

    # physicals / age from the fighter table (static attributes -> no leakage)
    if fighters is not None and not fighters.empty:
        bio = fighters.drop_duplicates("fighter_url").set_index("fighter_url")
        get = lambda c: F["fighter"].map(bio[c]) if c in bio else pd.Series(np.nan, index=F.index)
        dob = pd.to_datetime(get("fighter_dob"), errors="coerce")
        F["age"] = (pd.to_datetime(F["date"]) - dob).dt.days / 365.25
        F["height"] = _num(get("fighter_height_cm"))
        F["reach"] = _num(get("fighter_reach_cm"))
        stance = get("fighter_stance").fillna("").astype(str).str.lower()
        F["southpaw"] = (stance == "southpaw").astype(float)
        F["switch"] = (stance == "switch").astype(float)
    else:
        for c in ("age", "height", "reach", "southpaw", "switch"):
            F[c] = np.nan
    F["reach_ratio"] = _div(F["reach"], F["height"])
    return F


# --------------------------------------------------------------------------- public
def build_features(fights: pd.DataFrame, fighters: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Returns one row per bout with meta columns, A_*/B_* fighter features
    (A = f_1 corner), diff_* model features, context features, the target
    `y` (1 = A won, 0 = B won, NaN otherwise) and the winning `method`.
    """
    f = prepare(fights)
    e1, e2 = compute_elo(f)
    L = _long(f, e1, e2)
    F = _fighter_features(L, fighters)

    A = F[F["side"] == "f_1"].set_index("_idx")[FIGHTER_FEATURES].add_prefix("A_")
    B = F[F["side"] == "f_2"].set_index("_idx")[FIGHTER_FEATURES].add_prefix("B_")
    out = f.set_index("_idx").join(A).join(B)

    diffs = {f"diff_{c}": out[f"A_{c}"] - out[f"B_{c}"] for c in FIGHTER_FEATURES}
    diffs["elo_edge"] = 1.0 / (1.0 + 10 ** ((out["B_elo"] - out["A_elo"]) / 400.0)) - 0.5
    col = lambda name, default: out[name] if name in out else pd.Series(default, index=out.index)
    ctx = {
        "num_rounds": _num(col("num_rounds", np.nan)).where(lambda s: s > 0),
        "title_fight": col("title_fight", False).astype(str).str.lower().isin(["true", "1", "1.0"]).astype(float),
        "is_female": (col("gender", "M").astype(str).str.upper() == "F").astype(float),
        "weight_lbs": weight_lbs(col("weight_class", None)),
        "stance_mismatch": (out["A_southpaw"] != out["B_southpaw"]).astype(float),
    }
    ctx_df = pd.DataFrame(ctx, index=out.index)
    out = out.drop(columns=[c for c in ctx_df.columns if c in out.columns])
    out = pd.concat([out, pd.DataFrame(diffs, index=out.index), ctx_df], axis=1)
    out["num_rounds"] = out["num_rounds"].fillna(3)
    out["y"] = np.select([out["_side"] == "f_1", out["_side"] == "f_2"], [1.0, 0.0], np.nan)
    out["completed"] = out["_side"].isin(["f_1", "f_2", "draw", "nc"])
    return out.reset_index()


def method_matrix(feat: pd.DataFrame, winner_is_a) -> pd.DataFrame:
    """Winner-oriented features for the method-of-victory model."""
    winner_is_a = np.asarray(winner_is_a, dtype=bool)
    cols = {}
    for base in METHOD_BASE:
        a, b = feat[f"A_{base}"].to_numpy(float), feat[f"B_{base}"].to_numpy(float)
        cols[f"W_{base}"] = np.where(winner_is_a, a, b)
        cols[f"L_{base}"] = np.where(winner_is_a, b, a)
    for c in ("num_rounds", "is_female", "weight_lbs"):
        cols[c] = feat[c].to_numpy(float)
    return pd.DataFrame(cols, index=feat.index)[METHOD_FEATURES]
