"""
Shareable card report: highlighted picks, safest parlays, standalone HTML export.

    python -m ufc_predictor report [--event "name"] [--out card.html]

The HTML it writes is self-contained (inline CSS, Google Fonts only), so it can be
emailed as an attachment, dropped on any static host, or published as a link.
"""
from __future__ import annotations

import html as _html
import re
import unicodedata

import numpy as np
import pandas as pd

HIGHLIGHT = ("Strong", "Solid")
RED, BLUE, GOLD = "#C8323C", "#3B6FC4", "#E3B341"


# --------------------------------------------------------------------------- parlays
def fair_american(p: float) -> str:
    """Fair (no-vig) American odds implied by probability p."""
    p = float(min(max(p, 1e-6), 1 - 1e-6))
    return f"-{100 * p / (1 - p):.0f}" if p >= 0.5 else f"+{100 * (1 - p) / p:.0f}"


def _legs(card: pd.DataFrame) -> list[dict]:
    legs = []
    for i, r in card.reset_index(drop=True).iterrows():
        bout = f"{r['A_name']} vs {r['B_name']}"
        legs.append({"fight": i, "bout": bout, "type": "ML", "p": float(r["pick_prob"]),
                     "label": f"{r['pick']} to win"})
        legs.append({"fight": i, "bout": bout, "type": "OVER", "p": float(r["p_decision"]),
                     "label": "Goes the distance"})
        legs.append({"fight": i, "bout": bout, "type": "UNDER", "p": float(r["p_finish"]),
                     "label": "Doesn't go the distance"})
    return legs


def _best(legs: list[dict], n: int, allow=("ML", "OVER", "UNDER")) -> dict | None:
    pool = sorted([l for l in legs if l["type"] in allow], key=lambda l: -l["p"])
    chosen, used = [], set()
    for l in pool:
        if l["fight"] in used:
            continue
        chosen.append(l)
        used.add(l["fight"])
        if len(chosen) == n:
            break
    if len(chosen) < n:
        return None
    p = float(np.prod([l["p"] for l in chosen]))
    return {"legs": chosen, "p": p, "odds": fair_american(p)}


def safest_parlays(card: pd.DataFrame) -> list[dict]:
    """Three parlays, safest first. Legs are one per fight; probabilities assume independence."""
    legs = _legs(card)
    out = []
    for name, n, allow, note in [
        ("Safest 2-leg", 2, ("ML", "OVER", "UNDER"), "The two single highest-probability legs on the card."),
        ("Distance double", 2, ("OVER", "UNDER"), "Both legs are round-total plays: goes / doesn't go the distance."),
        ("Safest 3-leg", 3, ("ML", "OVER", "UNDER"), "Adds the next safest leg. Lower hit rate, better price."),
    ]:
        p = _best(legs, n, allow)
        if p:
            p.update(name=name, note=note)
            out.append(p)
    return out


# --------------------------------------------------------------------------- results
def results_from_fights(card: pd.DataFrame, fights: pd.DataFrame) -> dict:
    """Look up finished results for a card in the fights table. Keyed by (A_name, B_name)."""
    out = {}
    if fights is None or fights.empty:
        return out
    idx = {}
    for _, f in fights.iterrows():
        idx[frozenset((_key(f["f_1_name"]), _key(f["f_2_name"])))] = f
    for _, r in card.iterrows():
        f = idx.get(frozenset((_key(r["A_name"]), _key(r["B_name"]))))
        if f is None or pd.isna(f.get("winner")) or not str(f.get("winner")):
            continue
        res = str(f.get("result") or "")
        method = "Decision" if "Decision" in res else "KO/TKO" if "KO" in res else \
                 "Submission" if "Submission" in res else res or "Result"
        rd = f.get("finish_round")
        out[(r["A_name"], r["B_name"])] = {
            "winner": str(f["winner"]), "method": method, "detail": res,
            "round": int(rd) if pd.notna(rd) else None, "time": str(f.get("finish_time") or ""),
        }
    return out


def grade(card: pd.DataFrame, results: dict) -> pd.DataFrame:
    """Attach result columns to the card: res_winner, res_method, res_round, res_time, res_note,
    pick_hit, distance_hit (True if fight went the distance), outcome_hit."""
    card = card.copy()
    cols = {k: [] for k in ("res_winner", "res_method", "res_round", "res_time", "res_note",
                            "pick_hit", "went_distance", "outcome_hit")}
    for _, r in card.iterrows():
        res = results.get((r["A_name"], r["B_name"])) or {}
        w = res.get("winner")
        cols["res_winner"].append(w)
        cols["res_method"].append(res.get("method"))
        cols["res_round"].append(res.get("round"))
        cols["res_time"].append(res.get("time"))
        cols["res_note"].append(res.get("note"))
        if not w:
            cols["pick_hit"].append(None); cols["went_distance"].append(None); cols["outcome_hit"].append(None)
            continue
        m = res.get("method") or ""
        cols["pick_hit"].append(_key(w) == _key(r["pick"]))
        cols["went_distance"].append(m == "Decision")
        cols["outcome_hit"].append(_key(r["top_outcome"]) == _key(f"{w} by {m}"))
    for k, v in cols.items():
        card[k] = v
    return card


def leg_hit(leg: dict, graded: pd.DataFrame):
    r = graded.iloc[leg["fight"]]
    if r.get("pick_hit") is None or pd.isna(r.get("pick_hit")):
        return None
    if leg["type"] == "ML":
        return bool(r["pick_hit"])
    if leg["type"] == "OVER":
        return bool(r["went_distance"])
    return not bool(r["went_distance"])


# --------------------------------------------------------------------------- html report
def _key(name) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z]", "", s)


CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Barlow:wght@400;500;600&display=swap');
:root {{
  --bg:#F5F3EE; --panel:#FFFFFF; --ink:#1B1D20; --muted:#646A73; --line:#DCD8CF; --track:#ECE9E1;
  --red:{RED}; --blue:{BLUE}; --gold:{GOLD}; --gold-ink:#7A5A0A; --gold-bg:#FBF3DC;
  --hit-bg:#DDF3E4; --hit-ink:#1E6B3A; --miss-bg:#F7DDDD; --miss-ink:#9B2C2C;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ color-scheme:dark;
  --bg:#141618; --panel:#1E2124; --ink:#F1EFEA; --muted:#9AA1AA; --line:#2E3237; --track:#2A2E33;
  --gold-ink:#F0CB68; --gold-bg:#2A2416; --hit-bg:#1C3A27; --hit-ink:#7FD79B; --miss-bg:#43201F; --miss-ink:#F09C9C; }} }}
:root[data-theme="dark"] {{ color-scheme:dark;
  --bg:#141618; --panel:#1E2124; --ink:#F1EFEA; --muted:#9AA1AA; --line:#2E3237; --track:#2A2E33;
  --gold-ink:#F0CB68; --gold-bg:#2A2416; --hit-bg:#1C3A27; --hit-ink:#7FD79B; --miss-bg:#43201F; --miss-ink:#F09C9C; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font-family:'Barlow',system-ui,sans-serif; font-size:15px; line-height:1.45; }}
.wrap {{ max-width:760px; margin:0 auto; padding-block:28px 48px; padding-inline:16px; }}
h1,h2,h3 {{ font-family:'Barlow Condensed','Arial Narrow',sans-serif; margin:0; text-wrap:balance; }}
h1 {{ font-size:2.4rem; font-weight:700; line-height:1; }}
h2 {{ font-size:1.5rem; font-weight:600; margin-top:36px; padding-bottom:6px; border-bottom:2px solid var(--line); }}
.sub {{ color:var(--muted); margin:6px 0 20px; }}
.stats {{ display:flex; flex-wrap:wrap; gap:10px; margin:0 0 24px; }}
.stat {{ flex:1 1 140px; background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px 14px; }}
.stat b {{ display:block; font-family:'Barlow Condensed',sans-serif; font-size:2rem; font-weight:700; line-height:1; }}
.stat span {{ color:var(--muted); font-size:.8rem; text-transform:uppercase; letter-spacing:.06em; }}
.picks {{ display:flex; flex-direction:column; gap:6px; margin-bottom:8px; }}
.pick {{ display:flex; align-items:center; gap:10px; background:var(--gold-bg); border-left:4px solid var(--gold); border-radius:6px; padding:8px 12px; }}
.pick .who {{ font-family:'Barlow Condensed',sans-serif; font-size:1.2rem; font-weight:600; flex:1; }}
.pick .vs {{ color:var(--muted); font-size:.85rem; }}
.pick .p {{ font-family:'Barlow Condensed',sans-serif; font-size:1.2rem; font-weight:700; font-variant-numeric:tabular-nums; }}
.badge {{ font-size:.7rem; font-weight:600; letter-spacing:.08em; text-transform:uppercase; padding:2px 7px; border-radius:4px; background:var(--gold); color:#1B1D20; }}
.bout {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px 18px 12px; margin:12px 0; }}
.bout.hi {{ border-color:var(--gold); box-shadow:0 0 0 2px var(--gold-bg) inset; }}
.bout .meta {{ display:flex; justify-content:space-between; gap:8px; color:var(--muted); font-size:.82rem; margin-bottom:4px; }}
.bout .names {{ display:flex; justify-content:space-between; gap:12px; font-family:'Barlow Condensed',sans-serif; font-size:1.45rem; font-weight:600; line-height:1.15; }}
.bout .red {{ color:var(--red); }} .bout .blue {{ color:var(--blue); text-align:right; }}
.bout .rank {{ font-size:.8rem; font-weight:500; color:var(--muted); margin:0 6px; }}
.bout .split {{ display:flex; height:12px; border-radius:6px; overflow:hidden; margin:8px 0 4px; }}
.bout .split .r {{ background:var(--red); }} .bout .split .b {{ background:var(--blue); }}
.bout .pcts {{ display:flex; justify-content:space-between; font-variant-numeric:tabular-nums; font-family:'Barlow Condensed',sans-serif; font-size:1.25rem; font-weight:700; }}
.bout .verdict {{ font-size:.95rem; margin:6px 0 10px; }}
.bout .verdict .tag {{ color:var(--gold-ink); font-weight:600; }}
.bout .outc {{ display:grid; grid-template-columns:minmax(150px,1.3fr) 3fr 44px; gap:4px 10px; align-items:center; font-size:.86rem; }}
.bout .track {{ background:var(--track); border-radius:3px; height:7px; }}
.bout .fill {{ height:7px; border-radius:3px; }}
.bout .num {{ text-align:right; font-variant-numeric:tabular-nums; }}
.bout .market {{ font-size:.84rem; color:var(--muted); margin-top:8px; }}
.parlays {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:12px; }}
.parlay {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px 16px; display:flex; flex-direction:column; gap:8px; }}
.parlay.top {{ border-color:var(--gold); }}
.parlay h3 {{ font-size:1.25rem; font-weight:600; display:flex; justify-content:space-between; align-items:baseline; }}
.parlay .odds {{ font-variant-numeric:tabular-nums; color:var(--gold-ink); }}
.parlay .leg {{ display:flex; justify-content:space-between; gap:8px; font-size:.9rem; border-top:1px solid var(--line); padding-top:6px; }}
.parlay .leg small {{ display:block; color:var(--muted); }}
.parlay .leg b {{ font-variant-numeric:tabular-nums; white-space:nowrap; }}
.parlay .chance {{ font-family:'Barlow Condensed',sans-serif; font-size:1.6rem; font-weight:700; line-height:1; }}
.parlay .chance small {{ font-family:'Barlow',sans-serif; font-size:.78rem; font-weight:400; color:var(--muted); margin-left:6px; }}
.parlay .note {{ font-size:.82rem; color:var(--muted); }}
.res {{ display:flex; flex-wrap:wrap; align-items:center; gap:8px; margin-top:10px; padding-top:8px; border-top:1px dashed var(--line); font-size:.9rem; }}
.res .lbl {{ color:var(--muted); font-size:.75rem; text-transform:uppercase; letter-spacing:.06em; }}
.res b {{ font-family:'Barlow Condensed',sans-serif; font-size:1.1rem; }}
.hit, .miss, .na {{ font-size:.7rem; font-weight:600; letter-spacing:.06em; text-transform:uppercase; padding:2px 7px; border-radius:4px; white-space:nowrap; }}
.hit {{ background:var(--hit-bg); color:var(--hit-ink); }} .miss {{ background:var(--miss-bg); color:var(--miss-ink); }}
.na {{ background:var(--track); color:var(--muted); }}
.parlay.won {{ border-color:var(--hit-ink); }} .parlay.lost {{ opacity:.8; }}
.stat.good b {{ color:var(--hit-ink); }}
.tbl-wrap {{ overflow-x:auto; }}
table.track {{ width:100%; border-collapse:collapse; font-size:.9rem; font-variant-numeric:tabular-nums; }}
table.track th {{ text-align:left; color:var(--muted); font-size:.72rem; text-transform:uppercase; letter-spacing:.06em; font-weight:600; padding:6px 8px; border-bottom:2px solid var(--line); }}
table.track td {{ padding:8px; border-bottom:1px solid var(--line); vertical-align:top; }}
table.track td.n {{ text-align:right; white-space:nowrap; }}
table.track td.ev {{ min-width:200px; }} table.track td:last-child {{ white-space:nowrap; }}
table.track tr.total td {{ font-weight:600; border-top:2px solid var(--line); border-bottom:none; }}
table.track .ev small {{ display:block; color:var(--muted); }}
.evpicks {{ margin:18px 0 0; }}
.evpicks h3 {{ font-size:1.15rem; font-weight:600; margin:0 0 6px; display:flex; justify-content:space-between; gap:8px; }}
.evpicks h3 span {{ color:var(--muted); font-weight:500; font-size:.9rem; }}
.pick .out {{ color:var(--muted); font-size:.85rem; white-space:nowrap; }}
table.track td.bet {{ font-weight:600; }}
.pl {{ font-family:'Barlow Condensed',sans-serif; font-size:1.6rem; font-weight:700; }}
.props {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:4px 16px; font-size:.86rem; margin-top:8px; }}
.props div {{ display:flex; justify-content:space-between; border-bottom:1px dotted var(--line); padding:2px 0; }}
.props b {{ font-variant-numeric:tabular-nums; }}
.foot {{ margin-top:36px; color:var(--muted); font-size:.82rem; line-height:1.5; }}
@media (max-width:480px) {{ .bout .outc {{ grid-template-columns:minmax(120px,1.4fr) 2fr 40px; }} h1 {{ font-size:2rem; }} }}
"""


def bout_html(r: pd.Series, ranks: dict | None = None) -> str:
    e = _html.escape
    ranks = ranks or {}
    ra, rb = ranks.get(_key(r["A_name"])), ranks.get(_key(r["B_name"]))
    rank_a = f'<span class="rank">{ra}</span>' if ra else ""
    rank_b = f'<span class="rank">{rb}</span>' if rb else ""
    wc = r.get("weight_class") or "Bout"
    hi = r["confidence"] in HIGHLIGHT
    badge = f'<span class="badge">{e(str(r["confidence"]))} pick</span>' if hi else ""
    meta = f'<span>{e(str(wc))}, {int(r["num_rounds"])} rounds</span>{badge}'
    pa = float(r["p_A"])
    outc = ""
    for name, p in r["outcomes"]:
        colour = "var(--red)" if name.startswith(str(r["A_name"]) + " by") else "var(--blue)"
        outc += (f'<div>{e(name)}</div><div class="track"><div class="fill" style="width:{p * 100:.1f}%;'
                 f'background:{colour}"></div></div><div class="num">{p:.0%}</div>')
    market = ""
    if "market_p_A" in r.index and pd.notna(r.get("market_p_A")):
        edge = pa - r["market_p_A"]
        side = r["A_name"] if edge > 0 else r["B_name"]
        market = (f'<div class="market">Market (vig-free): {e(str(r["A_name"]))} {r["market_p_A"]:.0%}. '
                  f'Model sees {abs(edge):.0%} more value on {e(str(side))}.</div>')
    debut = ""
    if (r.get("A_n_fights", 1) or 0) == 0 or (r.get("B_n_fights", 1) or 0) == 0:
        debut = " A UFC debut is involved, so treat this one with extra caution."
    conf = f'<span class="tag">{str(r["confidence"]).lower()} pick</span>' if hi else f'{str(r["confidence"]).lower()}'
    verdict = (f'<b>{e(str(r["pick"]))}</b> ({conf}). Most likely result: '
               f'<b>{e(r["top_outcome"])}</b> ({r["top_outcome_prob"]:.0%}). '
               f'Goes the distance: {r["p_decision"]:.0%}.{debut}')
    result = ""
    if r.get("res_note") and pd.notna(r.get("res_note")):
        result = f'<div class="res"><span class="lbl">Result</span><span>{e(str(r["res_note"]))}</span><span class="na">Not graded</span></div>'
    elif r.get("res_winner") and pd.notna(r.get("res_winner")):
        rd, tm = r.get("res_round"), r.get("res_time")
        when = f' R{int(rd)} {e(str(tm))}' if rd is not None and pd.notna(rd) and tm and pd.notna(tm) else ""
        pick_tag = '<span class="hit">Pick hit</span>' if r["pick_hit"] else '<span class="miss">Pick missed</span>'
        dist = "went the distance" if r["went_distance"] else "finished"
        out_tag = ' <span class="hit">Exact result hit</span>' if r.get("outcome_hit") else ""
        result = (f'<div class="res"><span class="lbl">Result</span><b>{e(str(r["res_winner"]))} by {e(str(r["res_method"]))}</b>'
                  f'<span>{when}, {dist}</span>{pick_tag}{out_tag}</div>')
    return f"""<div class="bout{' hi' if hi else ''}"><div class="meta">{meta}</div>
<div class="names"><div class="red">{e(str(r['A_name']))}{rank_a}</div><div class="blue">{rank_b}{e(str(r['B_name']))}</div></div>
<div class="split"><div class="r" style="width:{pa * 100:.1f}%"></div><div class="b" style="width:{(1 - pa) * 100:.1f}%"></div></div>
<div class="pcts"><span style="color:var(--red)">{pa:.0%}</span><span style="color:var(--blue)">{1 - pa:.0%}</span></div>
<div class="verdict">{verdict}</div><div class="outc">{outc}</div>{market}{result}</div>"""


def picks_strip_html(card: pd.DataFrame) -> str:
    e = _html.escape
    hi = card[card["confidence"].isin(HIGHLIGHT)].sort_values("pick_prob", ascending=False)
    if hi.empty:
        return '<p class="sub">No strong or solid picks on this card.</p>'
    rows = ""
    for _, r in hi.iterrows():
        opp = r["B_name"] if r["pick"] == r["A_name"] else r["A_name"]
        mark = ""
        if r.get("res_note") and pd.notna(r.get("res_note")):
            mark = '<span class="na">Not graded</span>'
        elif r.get("res_winner") and pd.notna(r.get("res_winner")):
            mark = '<span class="hit">Hit</span>' if r["pick_hit"] else '<span class="miss">Missed</span>'
        rows += (f'<div class="pick"><span class="who">{e(str(r["pick"]))}</span>'
                 f'<span class="vs">over {e(str(opp))}</span>'
                 f'<span class="badge">{e(str(r["confidence"]))}</span><span class="p">{r["pick_prob"]:.0%}</span>{mark}</div>')
    return f'<div class="picks">{rows}</div>'


def parlays_html(card: pd.DataFrame) -> str:
    e = _html.escape
    cards = ""
    graded = "pick_hit" in card.columns
    for i, p in enumerate(safest_parlays(card)):
        legs, hits = "", []
        for l in p["legs"]:
            h = leg_hit(l, card) if graded else None
            hits.append(h)
            tag = "" if h is None else ('<span class="hit">Hit</span>' if h else '<span class="miss">Miss</span>')
            legs += (f'<div class="leg"><span>{e(l["label"])}<small>{e(l["bout"])}</small></span>'
                     f'<span>{tag} <b>{l["p"]:.0%}</b></span></div>')
        state = "" if any(h is None for h in hits) else (" won" if all(hits) else " lost")
        verdict = "" if not state else (' <span class="hit">Cashed</span>' if all(hits) else ' <span class="miss">Lost</span>')
        cards += (f'<div class="parlay{" top" if i == 0 else ""}{state}"><h3>{e(p["name"])}{verdict}'
                  f'<span class="odds">{p["odds"]}</span></h3>'
                  f'<div class="chance">{p["p"]:.0%}<small>chance all legs hit</small></div>{legs}'
                  f'<div class="note">{e(p["note"])} Fair odds shown; a book will price it worse.</div></div>')
    return f'<div class="parlays">{cards}</div>'


def _slate_summary(P: pd.DataFrame) -> dict:
    """Grade one slate that has winner/actual_method columns (from backtest.event_backtest)."""
    results = {(r["A_name"], r["B_name"]): {"winner": r["winner"], "method": r["actual_method"]}
               for _, r in P.iterrows() if pd.notna(r.get("actual_method"))}
    g = grade(P.reset_index(drop=True), results)
    g = g[g["pick_hit"].notna()]
    hi = g[g["confidence"].isin(HIGHLIGHT)]
    parlays = safest_parlays(g)
    par = None
    if parlays:
        hits = [leg_hit(l, g) for l in parlays[0]["legs"]]
        par = None if any(h is None for h in hits) else all(hits)
    return {"graded": g, "n": len(g), "ok": int(g["pick_hit"].sum()), "hi_n": len(hi),
            "hi_ok": int(hi["pick_hit"].sum()), "exact": int(g["outcome_hit"].sum()),
            "fin": int((~g["went_distance"].astype(bool)).sum()), "exp_fin": float(g["p_finish"].sum()),
            "parlay": par, "parlay_odds": parlays[0]["odds"] if parlays else "", "hi": hi}


def track_record_html(bt: pd.DataFrame) -> str:
    """Scoreboard for the slates in `bt` (one retrain per event), plus every strong/solid pick graded."""
    e = _html.escape
    if bt is None or bt.empty:
        return ""
    bt = bt.copy()
    bt["event_date"] = pd.to_datetime(bt["event_date"])
    slates = [(d, n, _slate_summary(P)) for (d, n), P in
              sorted(bt.groupby(["event_date", "event_name"]), key=lambda t: t[0][0], reverse=True)]
    rows, T = "", {"n": 0, "ok": 0, "hi_n": 0, "hi_ok": 0, "exact": 0, "fin": 0, "exp_fin": 0.0, "pw": 0, "pn": 0}
    for d, name, sm in slates:
        for k in ("n", "ok", "hi_n", "hi_ok", "exact", "fin", "exp_fin"):
            T[k] += sm[k]
        ptag = '<span class="na">n/a</span>' if sm["parlay"] is None else \
               (f'<span class="hit">Cashed {sm["parlay_odds"]}</span>' if sm["parlay"] else f'<span class="miss">Lost {sm["parlay_odds"]}</span>')
        if sm["parlay"] is not None:
            T["pn"] += 1; T["pw"] += int(sm["parlay"])
        rows += (f'<tr><td class="ev">{e(str(name))}<small>{d.strftime("%d %b %Y")}</small></td>'
                 f'<td class="n">{sm["ok"]} / {sm["n"]}</td><td class="n">{sm["hi_ok"]} / {sm["hi_n"]}</td>'
                 f'<td class="n">{sm["exact"]} / {sm["n"]}</td><td class="n">{sm["fin"]} <span style="color:var(--muted)">({sm["exp_fin"]:.1f})</span></td>'
                 f'<td>{ptag}</td></tr>')
    acc = T["ok"] / T["n"] if T["n"] else 0
    hi_acc = T["hi_ok"] / T["hi_n"] if T["hi_n"] else 0
    rows += (f'<tr class="total"><td class="ev">All {len(slates)} events</td><td class="n">{T["ok"]} / {T["n"]} ({acc:.0%})</td>'
             f'<td class="n">{T["hi_ok"]} / {T["hi_n"]} ({hi_acc:.0%})</td><td class="n">{T["exact"]} / {T["n"]} ({T["exact"] / max(T["n"], 1):.0%})</td>'
             f'<td class="n">{T["fin"]} <span style="color:var(--muted)">({T["exp_fin"]:.1f})</span></td><td>{T["pw"]} of {T["pn"]} cashed</td></tr>')
    table = (f'<div class="tbl-wrap"><table class="track"><thead><tr><th>Event</th><th>Winners</th>'
             f'<th>Strong / solid</th><th>Exact result</th><th>Finishes (expected)</th><th>Safest 2-leg</th></tr></thead>'
             f'<tbody>{rows}</tbody></table></div>')
    picks = ""
    for d, name, sm in slates:
        hi = sm["hi"].sort_values("pick_prob", ascending=False)
        items = ""
        for _, r in hi.iterrows():
            opp = r["B_name"] if r["pick"] == r["A_name"] else r["A_name"]
            mark = '<span class="hit">Hit</span>' if r["pick_hit"] else '<span class="miss">Missed</span>'
            items += (f'<div class="pick"><span class="who">{e(str(r["pick"]))}</span><span class="vs">over {e(str(opp))}</span>'
                      f'<span class="badge">{e(str(r["confidence"]))}</span><span class="p">{r["pick_prob"]:.0%}</span>'
                      f'<span class="out">{e(str(r["res_winner"]))} by {e(str(r["res_method"]))}</span>{mark}</div>')
        if not items:
            items = '<p class="sub">No strong or solid picks on this card.</p>'
        picks += (f'<div class="evpicks"><h3>{e(str(name))}<span>{d.strftime("%d %b %Y")} · {sm["hi_ok"]} of {sm["hi_n"]} hit</span></h3>'
                  f'<div class="picks">{items}</div></div>')
    return (f'<h2>Track record: last {len(slates)} events</h2>'
            f'<p class="sub">Walk-forward backtest. For each event the model was retrained using only fights before that date, '
            f'then graded on the whole slate. Overall: <b>{acc:.0%}</b> of winners called, <b>{hi_acc:.0%}</b> of strong / solid picks hit, '
            f'safest 2-leg parlay cashed <b>{T["pw"]} of {T["pn"]}</b> times.</p>'
            f'{table}<h2>Strong and solid picks, event by event</h2>{picks}')


def pricing_html(priced: pd.DataFrame, graded: pd.DataFrame | None = None, bankroll: float = 1000.0) -> str:
    """Pricing desk table: fair vs market vs best book, edge, stake. Grades the flagged bets if results exist."""
    e = _html.escape
    if priced is None or priced.empty:
        return ""
    winners = {}
    if graded is not None and "res_winner" in graded.columns:
        for _, r in graded.iterrows():
            if r.get("res_winner") and pd.notna(r.get("res_winner")):
                winners[f"{r['A_name']} vs {r['B_name']}"] = _key(r["res_winner"])
    rows, staked, pnl, nbet, nwin = "", 0.0, 0.0, 0, 0
    for _, r in priced.iterrows():
        flag, res = "", ""
        if r["bet"]:
            flag = '<span class="badge">Bet</span>'
            w = winners.get(r["bout"])
            if w:
                nbet += 1; staked += r["stake"]
                if w == _key(r["fighter"]):
                    nwin += 1; pnl += r["stake"] * (r["best_dec"] - 1); res = '<span class="hit">Won</span>'
                else:
                    pnl -= r["stake"]; res = '<span class="miss">Lost</span>'
        edge = f'{r["edge"]:+.0%}' if pd.notna(r["edge"]) else "–"
        ev = f'{r["ev"]:+.2f}' if pd.notna(r["ev"]) else "–"
        stake = f'${r["stake"]:,.0f}' if r["bet"] else "–"
        rows += (f'<tr><td class="ev">{e(str(r["fighter"]))}<small>{e(str(r["bout"]))}</small></td>'
                 f'<td class="n">{r["p_model"]:.0%} <span style="color:var(--muted)">{r["fair"]}</span></td>'
                 f'<td class="n">{r["market"]}</td><td class="n">{r["best_odds"]}<small style="display:block;color:var(--muted)">{e(str(r["best_book"]))}</small></td>'
                 f'<td class="n">{edge}</td><td class="n">{ev}</td><td class="n bet">{stake} {flag} {res}</td></tr>')
    summary = ""
    if nbet:
        roi = pnl / staked if staked else 0
        summary = (f'<div class="stats"><div class="stat"><b>{nwin} / {nbet}</b><span>Flagged bets won</span></div>'
                   f'<div class="stat"><b>${staked:,.0f}</b><span>Total staked</span></div>'
                   f'<div class="stat{" good" if pnl > 0 else ""}"><b class="pl">{"+" if pnl >= 0 else "−"}${abs(pnl):,.0f}</b><span>Profit / loss ({roi:+.0%} ROI)</span></div></div>')
    table = (f'<div class="tbl-wrap"><table class="track"><thead><tr><th>Side</th><th>Model (fair)</th><th>Market</th>'
             f'<th>Best book</th><th>Edge</th><th>EV / $1</th><th>Stake</th></tr></thead><tbody>{rows}</tbody></table></div>')
    return (f'<h2>Pricing desk</h2><p class="sub">Model probability turned into a fair price, against the vig-free market '
            f'consensus and the best available book price. A bet is flagged when the edge is 4 points or more and EV is positive; '
            f'stakes are quarter-Kelly on a ${bankroll:,.0f} bankroll.</p>{summary}{table}')


def props_html(card: pd.DataFrame) -> str:
    """Fair prices for method, distance and round-total props, one block per fight."""
    from .pricing import prop_prices
    e = _html.escape
    blocks = ""
    for _, r in card.iterrows():
        pp = prop_prices(r)
        pp = pp[pp["market"] != "Moneyline"]
        items = "".join(f'<div><span>{e(str(x["selection"]))}</span><b>{x["fair"]}</b></div>' for _, x in pp.iterrows())
        blocks += f'<div class="evpicks"><h3>{e(str(r["A_name"]))} vs {e(str(r["B_name"]))}</h3><div class="props">{items}</div></div>'
    return f'<h2>Prop prices</h2><p class="sub">Fair prices for the props books usually offer. Round totals use the historical share of finishes that land in each round.</p>{blocks}'


def report_html(card: pd.DataFrame, event_label: str, ranks: dict | None = None,
                model_meta: dict | None = None, results: dict | None = None,
                track: pd.DataFrame | None = None, priced: pd.DataFrame | None = None,
                bankroll: float = 1000.0, props: bool = True) -> str:
    e = _html.escape
    card = card.reset_index(drop=True)
    if results:
        card = grade(card, results)
    graded = card[card["pick_hit"].notna()] if "pick_hit" in card.columns else card.iloc[0:0]
    score = ""
    if len(graded):
        hi_g = graded[graded["confidence"].isin(HIGHLIGHT)]
        n_ok, n_hi_ok = int(graded["pick_hit"].sum()), int(hi_g["pick_hit"].sum())
        ex_ok = int(graded["outcome_hit"].sum())
        fin = int((~graded["went_distance"].astype(bool)).sum())
        score = (f'<h2>How the model did</h2><div class="stats">'
                 f'<div class="stat{" good" if n_ok / len(graded) >= .6 else ""}"><b>{n_ok} / {len(graded)}</b><span>Winners called</span></div>'
                 f'<div class="stat{" good" if len(hi_g) and n_hi_ok == len(hi_g) else ""}"><b>{n_hi_ok} / {len(hi_g)}</b><span>Strong / solid picks hit</span></div>'
                 f'<div class="stat"><b>{ex_ok} / {len(graded)}</b><span>Exact results called</span></div>'
                 f'<div class="stat"><b>{fin}</b><span>Finishes (model expected {graded["p_finish"].sum():.1f})</span></div></div>')
    n_hi = int(card["confidence"].isin(HIGHLIGHT).sum())
    bouts = "".join(bout_html(r, ranks) for _, r in card.iterrows())
    meta = model_meta or {}
    trained = f"Model trained {meta.get('trained_at', '')[:10]} on {meta.get('n_train', 0):,} fights. " if meta else ""
    title = re.sub(r"\s*\(.*\)$", "", event_label).strip() or "Fight card"
    return f"""<title>UFC Oracle: {e(title)}</title>
<style>{CSS}</style>
<div class="wrap">
<h1>{e(event_label)}</h1>
<p class="sub">UFC Oracle model card{", graded against the official results" if len(graded) else ""}. Fights listed main event first.</p>
<div class="stats"><div class="stat"><b>{len(card)}</b><span>Bouts</span></div>
<div class="stat"><b>{n_hi}</b><span>Strong / solid picks</span></div>
<div class="stat"><b>{card['p_finish'].sum():.1f}</b><span>Expected finishes</span></div></div>
{score}
<h2>Strong and solid picks</h2>
{picks_strip_html(card)}
<h2>Every bout</h2>
{bouts}
<h2>Safest parlays</h2>
{parlays_html(card)}
{pricing_html(priced, card if len(graded) else None, bankroll) if priced is not None else ""}
{props_html(card) if props else ""}
{track_record_html(track) if track is not None else ""}
<p class="foot">{trained}Parlay probabilities multiply each leg's model probability and assume the legs are independent.
"Strong" means the pick is at 70% or more; "Solid" is 62 to 70%. This is a statistical tool, not betting advice.</p>
</div>"""
