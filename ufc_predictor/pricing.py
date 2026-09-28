"""
Pricing desk: turns model probabilities into prices and compares them with the live market.

  fair_american(p)            probability -> fair American odds
  price_card(card, odds)      per-fight moneyline pricing: fair vs best book, edge, EV, Kelly stake
  prop_prices(row)            winner-by-method, goes-the-distance and round-total prices for one fight
  snapshot_odds(api_key)      fetch the market and APPEND it to data/odds_history.csv (for line movement)
  line_movement(history, a, b)  consensus vig-free price over time for one fight
  alerts(priced, threshold)   fights where the model edge is above the threshold

Everything here is a price the model believes in, not a guarantee. Books re-price constantly;
the history file is what lets you see whether the market is moving toward or away from the model.
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

from . import config, store

ODDS_HISTORY_CSV = config.DATA / "odds_history.csv"

# Share of finishes that happen in each round (UFC, 2015 onward). Used to price round totals.
FINISH_ROUND_SHARE = {3: {1: 0.511, 2: 0.324, 3: 0.165},
                      5: {1: 0.361, 2: 0.273, 3: 0.185, 4: 0.100, 5: 0.082}}


# --------------------------------------------------------------------------- odds maths
def _key(name) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z]", "", s)


def fair_american(p: float) -> str:
    p = float(min(max(p, 1e-6), 1 - 1e-6))
    return f"-{100 * p / (1 - p):.0f}" if p >= 0.5 else f"+{100 * (1 - p) / p:.0f}"


def decimal_to_american(d: float) -> str:
    d = float(d)
    if d <= 1:
        return "–"
    return f"+{(d - 1) * 100:.0f}" if d >= 2 else f"-{100 / (d - 1):.0f}"


def implied(d: float) -> float:
    return 1.0 / float(d)


def ev_per_unit(p: float, d: float) -> float:
    """Expected profit per 1 unit staked at decimal odds d with true probability p."""
    return p * (d - 1) - (1 - p)


def kelly(p: float, d: float, fraction: float = 0.25) -> float:
    """Fractional Kelly stake as a share of bankroll (0 when there is no edge)."""
    b = d - 1
    if b <= 0:
        return 0.0
    f = (p * b - (1 - p)) / b
    return max(0.0, f * fraction)


# --------------------------------------------------------------------------- market table
def book_table(odds: pd.DataFrame) -> pd.DataFrame:
    """Long table: one row per (fighter a, opponent b, bookmaker) with a's decimal price."""
    if odds is None or odds.empty:
        return pd.DataFrame(columns=["a", "b", "book", "d", "fetched_at"])
    o = odds.dropna(subset=["odds_1", "odds_2"])
    rows = []
    for _, r in o.iterrows():
        f = r.get("fetched_at")
        rows.append((_key(r["fighter_1"]), _key(r["fighter_2"]), r["bookmaker"], float(r["odds_1"]), f))
        rows.append((_key(r["fighter_2"]), _key(r["fighter_1"]), r["bookmaker"], float(r["odds_2"]), f))
    return pd.DataFrame(rows, columns=["a", "b", "book", "d", "fetched_at"])


def price_card(card: pd.DataFrame, odds: pd.DataFrame | None = None, bankroll: float = 1000.0,
               kelly_fraction: float = 0.25, edge_threshold: float = 0.04) -> pd.DataFrame:
    """One row per side of every fight: fair price, consensus market, best book, edge, EV, stake."""
    if odds is None:
        odds = store.read(config.ODDS_CSV)
    bt = book_table(odds)
    if not bt.empty and bt["fetched_at"].notna().any():
        latest = bt["fetched_at"].max()
        bt = bt[bt["fetched_at"] == latest]
    rows = []
    for _, r in card.reset_index(drop=True).iterrows():
        for side, other in (("A", "B"), ("B", "A")):
            name, opp = r[f"{side}_name"], r[f"{other}_name"]
            p = float(r[f"p_{side}"])
            sub = bt[(bt["a"] == _key(name)) & (bt["b"] == _key(opp))]
            best_d, best_book, n_books, cons_p = np.nan, None, 0, np.nan
            if not sub.empty:
                i = sub["d"].idxmax()
                best_d, best_book, n_books = float(sub.at[i, "d"]), sub.at[i, "book"], int(len(sub))
                # consensus vig-free: median of implied probabilities, normalised against the other side
                other_sub = bt[(bt["a"] == _key(opp)) & (bt["b"] == _key(name))]
                if not other_sub.empty:
                    ia, ib = np.median(1 / sub["d"]), np.median(1 / other_sub["d"])
                    cons_p = ia / (ia + ib)
            edge = p - cons_p if pd.notna(cons_p) else np.nan
            ev = ev_per_unit(p, best_d) if pd.notna(best_d) else np.nan
            stake = kelly(p, best_d, kelly_fraction) if pd.notna(best_d) else 0.0
            rows.append({
                "bout": f"{r['A_name']} vs {r['B_name']}", "fighter": name, "opponent": opp,
                "p_model": p, "fair": fair_american(p),
                "p_market": cons_p, "market": fair_american(cons_p) if pd.notna(cons_p) else "–",
                "best_odds": decimal_to_american(best_d) if pd.notna(best_d) else "–", "best_dec": best_d,
                "best_book": best_book or "–", "books": n_books,
                "edge": edge, "ev": ev, "kelly": stake, "stake": round(stake * bankroll, 0),
                "bet": bool(pd.notna(edge) and edge >= edge_threshold and ev > 0),
            })
    return pd.DataFrame(rows)


def prop_prices(r: pd.Series) -> pd.DataFrame:
    """Fair prices for the props a book usually offers on a fight."""
    R = int(r.get("num_rounds") or 3)
    share = FINISH_ROUND_SHARE.get(R, FINISH_ROUND_SHARE[3])
    p_fin = float(r["p_finish"])
    rows = [("Moneyline", r["A_name"], float(r["p_A"])), ("Moneyline", r["B_name"], float(r["p_B"]))]
    for m, lbl in (("KO", "KO/TKO"), ("Sub", "Submission"), ("Dec", "Decision")):
        for side in ("A", "B"):
            col = f"{side}_{m}" if f"{side}_{m}" in r.index else f"{side}_{lbl}"
            if col in r.index:
                rows.append(("Method", f"{r[f'{side}_name']} by {lbl}", float(r[col])))
    rows.append(("Distance", "Goes the distance: Yes", 1 - p_fin))
    rows.append(("Distance", "Goes the distance: No", p_fin))
    cum = 0.0
    for k in range(1, R):
        cum += share.get(k, 0.0)
        p_under = p_fin * cum  # fight ends before the midpoint line of round k+1 ~ ends in rounds 1..k
        rows.append(("Rounds", f"Under {k}.5 rounds", p_under))
        rows.append(("Rounds", f"Over {k}.5 rounds", 1 - p_under))
    out = pd.DataFrame(rows, columns=["market", "selection", "p"])
    out["fair"] = out["p"].map(fair_american)
    return out


def alerts(priced: pd.DataFrame, threshold: float = 0.04) -> pd.DataFrame:
    if priced is None or priced.empty:
        return pd.DataFrame()
    a = priced[(priced["edge"] >= threshold) & (priced["ev"] > 0)].copy()
    return a.sort_values("edge", ascending=False)


# --------------------------------------------------------------------------- market history
def snapshot_odds(api_key: str) -> pd.DataFrame:
    """Fetch the market now, keep it as the current odds file AND append it to the history."""
    from .scraper import fetch_odds
    snap = fetch_odds(api_key)
    if snap.empty:
        return snap
    store.write(snap, config.ODDS_CSV)
    hist = store.read(ODDS_HISTORY_CSV)
    hist = pd.concat([hist, snap], ignore_index=True) if not hist.empty else snap
    store.write(hist, ODDS_HISTORY_CSV)
    return snap


def load_history() -> pd.DataFrame:
    h = store.read(ODDS_HISTORY_CSV)
    if h.empty:
        cur = store.read(config.ODDS_CSV)
        return cur
    return h


def line_movement(history: pd.DataFrame, a_name: str, b_name: str) -> pd.DataFrame:
    """Consensus vig-free probability of `a_name` at each fetch time, plus best available price."""
    bt = book_table(history)
    if bt.empty:
        return pd.DataFrame()
    a, b = _key(a_name), _key(b_name)
    sa, sb = bt[(bt["a"] == a) & (bt["b"] == b)], bt[(bt["a"] == b) & (bt["b"] == a)]
    if sa.empty:
        return pd.DataFrame()
    rows = []
    for t, g in sa.groupby("fetched_at"):
        gb = sb[sb["fetched_at"] == t]
        ia = np.median(1 / g["d"])
        ib = np.median(1 / gb["d"]) if not gb.empty else 1 - ia
        rows.append({"fetched_at": pd.to_datetime(t), "p_market": ia / (ia + ib),
                     "best_dec": float(g["d"].max()), "books": int(len(g))})
    return pd.DataFrame(rows).sort_values("fetched_at")


def movement_summary(history: pd.DataFrame, card: pd.DataFrame) -> pd.DataFrame:
    """Open vs current market price for each fight on the card."""
    rows = []
    for _, r in card.iterrows():
        mv = line_movement(history, r["A_name"], r["B_name"])
        if mv.empty:
            continue
        first, last = mv.iloc[0], mv.iloc[-1]
        rows.append({"bout": f"{r['A_name']} vs {r['B_name']}", "fighter": r["A_name"],
                     "open": first["p_market"], "now": last["p_market"], "move": last["p_market"] - first["p_market"],
                     "model": float(r["p_A"]), "snapshots": len(mv),
                     "since": first["fetched_at"], "latest": last["fetched_at"]})
    return pd.DataFrame(rows)
