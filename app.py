"""
UFC Oracle - Streamlit front end.   Run:  streamlit run app.py
"""
from __future__ import annotations

import html
import os
import re
import tempfile
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from ufc_predictor import backtest, config, learning, scraper, store
from ufc_predictor.models import LEARNER_NAMES, available_learners, preset
from ufc_predictor.pipeline import Predictor, attach_market, features_now, predict_custom, upcoming_rows
from ufc_predictor import pricing, report

st.set_page_config(page_title="UFC Oracle", page_icon="🥊", layout="wide")
config.ensure_dirs()

RED, BLUE, GOLD = "#B3202A", "#1F4E9A", "#E3B341"
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Barlow:wght@400;500&display=swap');
html, body, [class*="css"] {{ font-family: 'Barlow', system-ui, sans-serif; }}
h1, h2, h3 {{ font-family: 'Barlow Condensed', 'Arial Narrow', sans-serif; letter-spacing: .01em; }}
.bout {{ border: 1px solid #D5DAE1; border-radius: 10px; padding: 14px 18px 12px; margin: 12px 0 4px; background: #FFFFFF; }}
.bout .meta {{ color: #5B6470; font-size: .85rem; margin-bottom: 4px; }}
.bout .names {{ display: flex; justify-content: space-between; gap: 12px;
  font-family: 'Barlow Condensed', 'Arial Narrow', sans-serif; font-size: 1.45rem; font-weight: 600; line-height: 1.15; }}
.bout .red {{ color: {RED}; }} .bout .blue {{ color: {BLUE}; text-align: right; }}
.bout .rank {{ font-size: .8rem; font-weight: 500; color: #5B6470; margin-left: 6px; }}
.bout .split {{ display: flex; height: 14px; border-radius: 7px; overflow: hidden; margin: 8px 0 4px; }}
.bout .split .r {{ background: {RED}; }} .bout .split .b {{ background: {BLUE}; }}
.bout .pcts {{ display: flex; justify-content: space-between; font-variant-numeric: tabular-nums;
  font-family: 'Barlow Condensed', sans-serif; font-size: 1.25rem; font-weight: 700; }}
.bout .verdict {{ font-size: .95rem; margin: 6px 0 10px; }}
.bout .outc {{ display: grid; grid-template-columns: minmax(180px, 1.3fr) 3fr 48px; gap: 4px 12px;
  align-items: center; font-size: .88rem; }}
.bout .track {{ background: #EEF0F3; border-radius: 3px; height: 7px; }}
.bout .fill {{ height: 7px; border-radius: 3px; }}
.bout .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
.bout .market {{ font-size: .85rem; color: #3A414B; margin-top: 8px; }}
.bout.hi {{ border: 2px solid {GOLD}; background: #FFFDF5; }}
.badge {{ font-size: .7rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; padding: 2px 7px;
  border-radius: 4px; background: {GOLD}; color: #1B1D20; margin-left: 8px; vertical-align: middle; }}
.picks {{ display: flex; flex-direction: column; gap: 6px; margin: 4px 0 12px; }}
.pick {{ display: flex; align-items: center; gap: 10px; background: #FBF3DC; border-left: 4px solid {GOLD};
  border-radius: 6px; padding: 8px 12px; }}
.pick .who {{ font-family: 'Barlow Condensed', sans-serif; font-size: 1.2rem; font-weight: 600; flex: 1; }}
.pick .vs {{ color: #5B6470; font-size: .85rem; }}
.pick .p {{ font-family: 'Barlow Condensed', sans-serif; font-size: 1.2rem; font-weight: 700; }}
.parlays {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }}
.parlay {{ background: #FFFFFF; border: 1px solid #D5DAE1; border-radius: 10px; padding: 14px 16px;
  display: flex; flex-direction: column; gap: 8px; }}
.parlay.top {{ border-color: {GOLD}; }}
.parlay h3 {{ font-size: 1.25rem; font-weight: 600; margin: 0; display: flex; justify-content: space-between; }}
.parlay .odds {{ color: #7A5A0A; }}
.parlay .leg {{ display: flex; justify-content: space-between; gap: 8px; font-size: .9rem; border-top: 1px solid #ECE9E1; padding-top: 6px; }}
.parlay .leg small {{ display: block; color: #5B6470; }}
.parlay .hit {{ font-family: 'Barlow Condensed', sans-serif; font-size: 1.6rem; font-weight: 700; line-height: 1; }}
.parlay .hit small {{ font-family: 'Barlow', sans-serif; font-size: .78rem; font-weight: 400; color: #5B6470; margin-left: 6px; }}
.parlay .note {{ font-size: .82rem; color: #5B6470; }}
</style>""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- cached loaders
def _mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


def data_sig():
    return tuple(_mtime(p) for p in (config.FIGHTS_CSV, config.FIGHTERS_CSV, config.UPCOMING_CSV, config.ODDS_CSV))


def model_sig():
    return _mtime(config.MODEL_FILE)


@st.cache_resource(show_spinner=False)
def get_predictor(sig):
    return Predictor.load()


@st.cache_data(show_spinner="Building features…")
def get_features(sig):
    return features_now()


@st.cache_data(show_spinner="Predicting upcoming cards…")
def card_predictions(dsig, msig):
    pred = get_predictor(msig)
    feat = get_features(dsig)
    if pred is None or feat.empty:
        return pd.DataFrame()
    rows = upcoming_rows(feat)
    return attach_market(pred.predict(rows)) if not rows.empty else pd.DataFrame()


@st.cache_data(show_spinner=False)
def fighter_options(dsig):
    fighters = store.read(config.FIGHTERS_CSV)
    fights = store.read(config.FIGHTS_CSV)
    if fighters.empty or fights.empty:
        return pd.DataFrame()
    long = pd.concat([fights[["f_1_url", "event_date", "winner", "f_1_name"]].set_axis(["url", "d", "w", "n"], axis=1),
                      fights[["f_2_url", "event_date", "winner", "f_2_name"]].set_axis(["url", "d", "w", "n"], axis=1)])
    long["win"] = long["w"] == long["n"]
    agg = long.groupby("url").agg(last=("d", "max"), wins=("win", "sum"), n=("win", "size"))
    f = fighters.drop_duplicates("fighter_url").set_index("fighter_url").join(agg, how="inner")
    f["label"] = f["fighter_name"] + "  (" + f["wins"].astype(int).astype(str) + "-" + \
                 (f["n"] - f["wins"]).astype(int).astype(str) + " UFC, last " + f["last"].dt.strftime("%b %Y") + ")"
    return f.sort_values("last", ascending=False)


@st.cache_data(show_spinner=False)
def rankings_lookup(sig):
    r = store.read(config.RANKINGS_CSV)
    if r.empty:
        return {}
    r = r[~r["weightclass"].str.contains("Pound", case=False, na=False)]
    return {_key(n): ("C" if int(k) == 0 else f"#{int(k)}") for n, k in zip(r["fighter"], r["rank"])}


def _key(name) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z]", "", s)


def progress_bar():
    bar = st.progress(0.0, text="Starting…")
    return lambda f, m: bar.progress(float(min(max(f, 0.0), 1.0)), text=m)


def pct(x) -> str:
    return "–" if x is None or pd.isna(x) else f"{x:.0%}"


# --------------------------------------------------------------------------- bout card
def bout_html(r: pd.Series, ranks: dict) -> str:
    e = html.escape
    ra, rb = ranks.get(_key(r["A_name"])), ranks.get(_key(r["B_name"]))
    rank_a = f'<span class="rank">{ra}</span>' if ra else ""
    rank_b = f'<span class="rank">{rb}</span>' if rb else ""
    wc = r.get("weight_class") or "Bout"
    meta = f"{e(str(wc))}, {int(r['num_rounds'])} rounds"
    if r.get("event_name") and r["event_name"] != "Custom match-up":
        meta = f"{e(str(r['event_name']))} &nbsp;|&nbsp; " + meta
    hi = r["confidence"] in report.HIGHLIGHT
    if hi:
        meta += f'<span class="badge">{e(str(r["confidence"]))} pick</span>'
    pa = float(r["p_A"])
    outc = ""
    for name, p in r["outcomes"]:
        colour = RED if name.startswith(str(r["A_name"]) + " by") else BLUE
        outc += (f'<div>{e(name)}</div><div class="track"><div class="fill" style="width:{p * 100:.1f}%;'
                 f'background:{colour}"></div></div><div class="num">{p:.0%}</div>')
    market = ""
    if "market_p_A" in r and pd.notna(r.get("market_p_A")):
        edge = pa - r["market_p_A"]
        side = r["A_name"] if edge > 0 else r["B_name"]
        market = (f'<div class="market">Market (vig-free): {e(str(r["A_name"]))} {pct(r["market_p_A"])}. '
                  f'Model sees {abs(edge):.0%} more value on {e(str(side))}.</div>')
    debut = ""
    if (r.get("A_n_fights", 1) or 0) == 0 or (r.get("B_n_fights", 1) or 0) == 0:
        debut = " A UFC debut is involved, so treat this one with extra caution."
    verdict = (f'<b>{e(str(r["pick"]))}</b> ({r["confidence"].lower()} pick). Most likely result: '
               f'<b>{e(r["top_outcome"])}</b> ({r["top_outcome_prob"]:.0%}). '
               f'Goes the distance: {r["p_decision"]:.0%}.{debut}')
    return f"""<div class="bout{' hi' if hi else ''}"><div class="meta">{meta}</div>
<div class="names"><div class="red">{e(str(r['A_name']))}{rank_a}</div><div class="blue">{rank_b}{e(str(r['B_name']))}</div></div>
<div class="split"><div class="r" style="width:{pa * 100:.1f}%"></div><div class="b" style="width:{(1 - pa) * 100:.1f}%"></div></div>
<div class="pcts"><span style="color:{RED}">{pa:.0%}</span><span style="color:{BLUE}">{1 - pa:.0%}</span></div>
<div class="verdict">{verdict}</div><div class="outc">{outc}</div>{market}</div>"""


def show_bout(r: pd.Series, ranks: dict, key: str):
    st.markdown(bout_html(r, ranks), unsafe_allow_html=True)
    with st.expander("Why the model leans this way", expanded=False):
        why = r.get("why") or []
        if why:
            rows = [{"Factor": w["feature"], "Favours": r["A_name"] if w["favours"] == "A" else r["B_name"],
                     r["A_name"]: _fmt(w["A"]), r["B_name"]: _fmt(w["B"]), "Weight": round(w["strength"], 3)}
                    for w in why]
            st.dataframe(pd.DataFrame(rows), hide_index=True)
        comps = {LEARNER_NAMES.get(c[5:], c[5:]): r[c] for c in r.index if c.startswith("comp_")}
        st.caption("Each model's probability for " + str(r["A_name"]) + ": " +
                   ", ".join(f"{k} {v:.0%}" for k, v in comps.items()) +
                   f". Elo alone: {r['elo_p_A']:.0%}.")


def _fmt(v):
    if v is None or pd.isna(v):
        return "–"
    v = float(v)
    return f"{v:.0f}" if abs(v) >= 100 else f"{v:.2f}"


# --------------------------------------------------------------------------- sidebar
status = store.data_status()
pred = get_predictor(model_sig())
state = learning.load_state()

with st.sidebar:
    st.title("UFC Oracle")
    if status["fights"]:
        st.write(f"**{status['fights']:,}** fights, **{status['fighters']:,}** fighters")
        st.write(f"Results through **{pd.Timestamp(status['last_date']).date()}**")
        st.write(f"**{status['upcoming']}** announced bouts")
    else:
        st.warning("No data yet. Open the *Data & model* tab to scrape UFCStats or import the Kaggle CSV.")
    if pred:
        st.write(f"Model trained {pred.meta.get('trained_at', '')[:10]} on {pred.meta.get('n_train', 0):,} fights")
    else:
        st.info("No trained model yet.")
    st.divider()
    odds_key = st.text_input("The Odds API key (optional)", value=os.getenv("ODDS_API_KEY", ""), type="password",
                             help="Adds live betting lines so the model can be compared with the market.")
    auto_log = st.checkbox("Save card predictions to the ledger automatically", value=True,
                           help="The ledger is what the model grades itself against after each event.")
    if st.button("Update data + learn from last event", type="primary"):
        cb = progress_bar()
        try:
            out = learning.run_cycle(cb, scrape=True, odds_api_key=odds_key or None)
            st.success(f"Graded {out['grade']['graded_now']} predictions, retrained on "
                       f"{out['model']['n_train']:,} fights.")
            st.cache_data.clear()
            st.cache_resource.clear()
        except Exception as ex:
            st.error(f"Cycle failed: {ex}")

tab_card, tab_price, tab_h2h, tab_bt, tab_learn, tab_data = st.tabs(
    ["Fight cards", "Pricing desk", "Head to head", "Backtest", "Learning", "Data & model"])

ranks = rankings_lookup(_mtime(config.RANKINGS_CSV))

# --------------------------------------------------------------------------- fight cards
with tab_card:
    if pred is None:
        st.info("Train a model first (Data & model tab).")
    else:
        P = card_predictions(data_sig(), model_sig())
        if P.empty:
            st.info("No upcoming bouts on file. Paste the card in the Data & model tab (Option C).")
        else:
            P["event_label"] = P["event_name"].astype(str) + " (" + pd.to_datetime(P["event_date"]).dt.strftime("%d %b %Y") + ")"
            ev = st.selectbox("Event", P["event_label"].unique())
            card = P[P["event_label"] == ev].reset_index(drop=True)
            if auto_log:
                learning.log_predictions(card, pred.meta.get("version", ""))
            c1, c2, c3 = st.columns(3)
            c1.metric("Bouts", len(card))
            c2.metric("Strong / solid picks", int(card["confidence"].isin(["Strong", "Solid"]).sum()))
            c3.metric("Expected finishes", f"{card['p_finish'].sum():.1f}")
            st.subheader("Strong and solid picks")
            st.markdown(report.picks_strip_html(card), unsafe_allow_html=True)
            st.subheader("Every bout")
            for i, r in card.iterrows():
                show_bout(r, ranks, f"card{i}")
            st.subheader("Safest parlays")
            st.markdown(report.parlays_html(card), unsafe_allow_html=True)
            st.caption("Legs are one per fight. The hit chance multiplies the model's leg probabilities and assumes "
                       "independence; the odds shown are fair odds, so a book will pay less.")
            results = report.results_from_fights(card, store.read(config.FIGHTS_CSV))
            if results:
                st.subheader("How the model did")
                st.caption(f"Results found for {len(results)} of {len(card)} bouts. "
                           "Run *Update data + learn from last event* in the sidebar after a card to pull them in.")
            st.subheader("Track record")
            n_ev = st.slider("Events to backtest", 4, 20, 10, key="track_n")
            if st.button("Backtest the last events (retrains once per event, about 10 s each)"):
                cb = progress_bar()
                st.session_state["track"] = backtest.event_backtest(get_features(data_sig()), n_ev,
                                                                     pred.meta.get("learners"), progress=cb)
            track = st.session_state.get("track")
            if track is not None and not track.empty:
                st.markdown(report.track_record_html(track), unsafe_allow_html=True)
            page = report.report_html(card, ev, ranks, pred.meta, results=results, track=track,
                                      priced=pricing.price_card(card))
            st.download_button("Export shareable report (HTML)", page, file_name="ufc_card_report.html",
                               mime="text/html", help="A single self-contained page you can email or host as a link.")
            cols = ["event_date", "A_name", "B_name", "p_A", "p_B", "pick", "confidence", "top_outcome",
                    "top_outcome_prob", "p_decision"] + [c for c in ("market_p_A", "edge_A") if c in card]
            st.download_button("Download card predictions (CSV)", card[cols].to_csv(index=False),
                               file_name="ufc_card_predictions.csv")

# --------------------------------------------------------------------------- pricing desk
with tab_price:
    st.write("Turns the model's probabilities into prices and compares them with the live market: fair odds, "
             "vig-free consensus, the best book on each side, edge, EV and a fractional-Kelly stake.")
    if pred is None:
        st.info("Train a model first (Data & model tab).")
    else:
        P = card_predictions(data_sig(), model_sig())
        if P.empty:
            st.info("No upcoming bouts on file.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            bankroll = c1.number_input("Bankroll ($)", 50.0, 1_000_000.0, 1000.0, step=50.0)
            kf = c2.select_slider("Kelly fraction", options=[0.1, 0.25, 0.5, 1.0], value=0.25,
                                  format_func=lambda v: {0.1: "1/10", 0.25: "Quarter", 0.5: "Half", 1.0: "Full"}[v])
            thr = c3.slider("Edge threshold", 0.0, 0.15, 0.04, 0.01, format="%.2f")
            if c4.button("Fetch odds now", help="Pulls the market from The Odds API and adds a snapshot to the history."):
                if not odds_key:
                    st.warning("Add your Odds API key in the sidebar first.")
                else:
                    try:
                        snap = pricing.snapshot_odds(odds_key)
                        st.success(f"{len(snap)} lines from {snap['bookmaker'].nunique()} books saved.")
                        st.cache_data.clear()
                    except Exception as ex:
                        st.error(f"Odds fetch failed: {ex}")
            P["event_label"] = P["event_name"].astype(str) + " (" + pd.to_datetime(P["event_date"]).dt.strftime("%d %b %Y") + ")"
            ev_p = st.selectbox("Event", P["event_label"].unique(), key="price_event")
            card_p = P[P["event_label"] == ev_p].reset_index(drop=True)
            priced = pricing.price_card(card_p, bankroll=bankroll, kelly_fraction=kf, edge_threshold=thr)
            odds_df = store.read(config.ODDS_CSV)
            if odds_df.empty:
                st.info("No market odds on file yet. Add an Odds API key in the sidebar and press *Fetch odds now*.")
            else:
                st.caption(f"Market as of {pd.to_datetime(odds_df['fetched_at']).max():%d %b %Y %H:%M} UTC, "
                           f"{odds_df['bookmaker'].nunique()} books.")
            al = pricing.alerts(priced, thr)
            st.subheader(f"Edges above {thr:.0%}: {len(al)}")
            if not al.empty:
                st.dataframe(al[["fighter", "opponent", "fair", "market", "best_odds", "best_book", "edge", "ev", "stake"]]
                             .rename(columns={"best_odds": "best price", "best_book": "book", "ev": "EV / $1"})
                             .style.format({"edge": "{:+.1%}", "EV / $1": "{:+.2f}", "stake": "${:,.0f}"}), hide_index=True)
            st.subheader("Every side")
            st.dataframe(priced[["bout", "fighter", "p_model", "fair", "market", "best_odds", "best_book", "books", "edge", "ev", "stake", "bet"]]
                         .rename(columns={"p_model": "model", "best_odds": "best price", "best_book": "book", "ev": "EV / $1"})
                         .style.format({"model": "{:.0%}", "edge": "{:+.1%}", "EV / $1": "{:+.2f}", "stake": "${:,.0f}"}), hide_index=True)
            st.subheader("Line movement")
            hist = pricing.load_history()
            mv = pricing.movement_summary(hist, card_p)
            if mv.empty or mv["snapshots"].max() < 2:
                st.caption("Needs at least two odds snapshots. Each *Fetch odds now* (or `python -m ufc_predictor odds --watch 60`) adds one.")
            else:
                st.dataframe(mv[["bout", "fighter", "open", "now", "move", "model", "snapshots"]]
                             .style.format({"open": "{:.0%}", "now": "{:.0%}", "move": "{:+.1%}", "model": "{:.0%}"}), hide_index=True)
                pick_b = st.selectbox("Chart a fight", mv["bout"], key="mv_bout")
                rr = card_p[(card_p["A_name"] + " vs " + card_p["B_name"]) == pick_b].iloc[0]
                lm = pricing.line_movement(hist, rr["A_name"], rr["B_name"]).set_index("fetched_at")
                lm["model"] = float(rr["p_A"])
                st.line_chart(lm[["p_market", "model"]].rename(columns={"p_market": f"market: {rr['A_name']}"}))
            st.subheader("Prop prices")
            for i, r in card_p.iterrows():
                with st.expander(f"{r['A_name']} vs {r['B_name']}"):
                    st.dataframe(pricing.prop_prices(r).style.format({"p": "{:.1%}"}), hide_index=True)
            st.download_button("Download pricing (CSV)", priced.to_csv(index=False), file_name="ufc_pricing.csv")

# --------------------------------------------------------------------------- head to head
with tab_h2h:
    opts = fighter_options(data_sig())
    if pred is None or opts.empty:
        st.info("Needs data and a trained model.")
    else:
        labels = dict(zip(opts.index, opts["label"]))
        c1, c2 = st.columns(2)
        a = c1.selectbox("Red corner", opts.index, format_func=labels.get, index=0)
        b = c2.selectbox("Blue corner", opts.index, format_func=labels.get, index=1 if len(opts) > 1 else 0)
        c3, c4, c5 = st.columns([1, 1, 2])
        rounds = c3.radio("Rounds", [3, 5], horizontal=True)
        title = c4.checkbox("Title fight")
        date = c5.date_input("Fight date", value=pd.Timestamp.today().date())
        if a == b:
            st.warning("Pick two different fighters.")
        elif st.button("Predict this fight", type="primary"):
            with st.spinner("Computing both fighters' current profiles…"):
                res = predict_custom(pred, a, b, num_rounds=rounds, title=title, date=date)
            if res.empty:
                st.error("Couldn't build features for this pairing.")
            else:
                show_bout(attach_market(res).iloc[0], ranks, "h2h")

# --------------------------------------------------------------------------- backtest
with tab_bt:
    st.write("Replays history: at each cutoff the model is retrained on earlier fights only, then predicts the "
             "next window, exactly as it would have in real time.")
    if status["fights"] < 500:
        st.info("Needs at least a few hundred fights of data.")
    else:
        last = pd.Timestamp(status["last_date"]).date()
        c1, c2, c3 = st.columns(3)
        start = c1.date_input("Start", value=(pd.Timestamp(last) - pd.DateOffset(years=2)).date())
        end = c2.date_input("End", value=last)
        step = c3.select_slider("Retrain every", options=[30, 61, 91, 182, 365], value=91,
                                format_func=lambda d: {30: "month", 61: "2 months", 91: "quarter",
                                                       182: "6 months", 365: "year"}[d])
        c4, c5 = st.columns(2)
        pset = c4.selectbox("Models", ["fast", "balanced", "full"], index=0,
                            format_func=lambda p: f"{p}: " + ", ".join(LEARNER_NAMES[l] for l in preset(p)))
        mode = c5.radio("Mode", ["With learning", "Static", "Compare both"], horizontal=True,
                        help="'With learning' re-weights mistakes, ensemble members and calibration as it goes.")
        if st.button("Run backtest", type="primary"):
            feat = get_features(data_sig())
            cb = progress_bar()
            kw = dict(step_days=step, learners=preset(pset), progress=cb)
            if mode == "Compare both":
                s0, s1 = backtest.compare(feat, start, end, **kw)
                st.session_state["bt"], st.session_state["bt_static"] = s1, s0
            else:
                st.session_state["bt"] = backtest.walk_forward(feat, start, end, adaptive=mode != "Static", **kw)
                st.session_state.pop("bt_static", None)
            store.write(st.session_state["bt"], config.BACKTEST_CSV)

        bt = st.session_state.get("bt")
        if bt is None:
            saved = store.read(config.BACKTEST_CSV)
            bt = saved if not saved.empty else None
        if bt is not None and not bt.empty:
            s = backtest.summarize(bt)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", pct(s["accuracy"]), f"{(s['accuracy'] - s['elo_accuracy']) * 100:+.1f} pts vs Elo")
            m2.metric("Log loss", f"{s['log_loss']:.3f}", f"{s['log_loss'] - s['elo_log_loss']:+.3f} vs Elo",
                      delta_color="inverse")
            m3.metric("AUC", f"{s['auc']:.3f}")
            m4.metric("Fights tested", f"{s['fights']:,}")
            if "exact_outcome_hit_rate" in s:
                st.caption(f"Method correct when the winner was right: {pct(s['method_accuracy_given_winner'])}. "
                           f"Exact top outcome (winner + method): {pct(s['exact_outcome_hit_rate'])}.")
            if "bt_static" in st.session_state:
                s0 = backtest.summarize(st.session_state["bt_static"])
                st.subheader("Does learning from mistakes help?")
                cmp = pd.DataFrame({"Static": [s0["accuracy"], s0["log_loss"], s0["brier"]],
                                    "With learning": [s["accuracy"], s["log_loss"], s["brier"]]},
                                   index=["Accuracy", "Log loss (lower is better)", "Brier (lower is better)"])
                st.dataframe(cmp.style.format("{:.4f}"))
            if "market" in s:
                mk = s["market"]
                st.subheader("Against the betting market")
                st.write(f"On {mk['fights']:,} fights with odds: model {pct(mk['model_accuracy_same_fights'])} vs "
                         f"favourite {pct(mk['market_accuracy'])}; log loss {mk['model_log_loss_same_fights']:.3f} "
                         f"vs {mk['market_log_loss']:.3f}.")
                bet = s["betting"]
                st.write(f"Flat-stake value bets (edge > {bet['edge_threshold']:.0%}): {bet['bets']} bets, "
                         f"{bet['profit_units']:+.1f} units, ROI {bet['roi']:+.1%}. Historical odds are closing "
                         "lines you couldn't always get, so read this as optimistic.")
            st.subheader("Rolling accuracy (200 fights)")
            st.line_chart(backtest.rolling_accuracy(bt), color=[BLUE, RED])
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Calibration")
                cal = backtest.calibration_table(bt)
                st.dataframe(cal.style.format({"predicted": "{:.1%}", "actual": "{:.1%}"}), hide_index=True)
                st.caption("When it says 70%, it should be right about 70% of the time.")
            with c2:
                st.subheader("Each model on its own")
                st.dataframe(pd.DataFrame(s["models"]).T.rename(index=LEARNER_NAMES)
                             .style.format({"accuracy": "{:.1%}", "log_loss": "{:.4f}"}))
            c3, c4 = st.columns(2)
            c3.dataframe(backtest.breakdown(bt, "year").style.format({"accuracy": "{:.1%}", "log_loss": "{:.3f}"}),
                         hide_index=True)
            c4.dataframe(backtest.breakdown(bt, "weight_class").style.format({"accuracy": "{:.1%}", "log_loss": "{:.3f}"}),
                         hide_index=True)
            rep = learning.mistake_report(bt)
            st.subheader("Most confident misses")
            st.dataframe(rep["confident_misses"].style.format({"pick_prob": "{:.0%}"}), hide_index=True)
            st.download_button("Download backtest predictions (CSV)",
                               bt.drop(columns=["outcomes"], errors="ignore").to_csv(index=False),
                               file_name="ufc_backtest.csv")

# --------------------------------------------------------------------------- learning
with tab_learn:
    st.write("How the model learns from its mistakes: every card prediction is saved before the fight. After the "
             "event, results are matched to those predictions. The misses then get extra weight in the next "
             "training run, ensemble members that did better get more say, and the probabilities are "
             "re-calibrated. Your backtest predictions are pooled in too.")
    c1, c2, c3 = st.columns(3)
    if c1.button("Grade saved predictions"):
        st.write(learning.grade())
    if c2.button("Re-learn from mistakes + retrain", type="primary"):
        with st.spinner("Updating weights and retraining…"):
            stt = learning.update_state()
            learning.retrain(state=stt)
        st.cache_resource.clear()
        st.cache_data.clear()
        st.success("Model updated.")
        state = learning.load_state()

    st.subheader("Settings")
    s1, s2 = st.columns(2)
    alpha = s1.slider("How hard to lean on past mistakes", 0.0, 2.0, float(state["hardness_alpha"]), 0.1,
                      help="0 ignores mistakes. Too high chases upsets that were just luck; check with the "
                           "'Compare both' backtest.")
    hl = s2.slider("Recency half-life (years)", 1.0, 15.0, float(state["half_life_years"]), 0.5,
                   help="A fight this many years old counts half as much as a fight from today.")
    if (alpha, hl) != (state["hardness_alpha"], state["half_life_years"]):
        state["hardness_alpha"], state["half_life_years"] = alpha, hl
        learning.save_state(state)
        st.caption("Saved. Takes effect at the next retrain.")

    w = state.get("weights") or {}
    if w:
        st.subheader("Current ensemble weights")
        st.bar_chart(pd.Series(w).rename(index=LEARNER_NAMES), color=BLUE)
        st.caption(f"Calibration temperature: {state.get('temperature', 1.0):.2f} "
                   "(below 1 means raw probabilities were over-confident and are being softened).")

    lg = learning.read_ledger()
    st.subheader("Prediction ledger")
    if lg.empty:
        st.info("Nothing saved yet. Predictions are saved when you open a fight card.")
    else:
        g = lg[lg["correct"].notna()]
        if len(g):
            k1, k2, k3 = st.columns(3)
            k1.metric("Graded live predictions", len(g))
            k2.metric("Live accuracy", pct(g["correct"].mean()))
            k3.metric("Live log loss", f"{g['log_loss'].mean():.3f}")
        view = lg[["event_date", "event_name", "A_name", "B_name", "p_A", "top_outcome", "actual_winner_side",
                   "actual_method", "correct"]].copy()
        view["actual_winner"] = np.where(view["actual_winner_side"] == "A", view["A_name"],
                                         np.where(view["actual_winner_side"] == "B", view["B_name"], ""))
        st.dataframe(view.drop(columns="actual_winner_side").style.format({"p_A": "{:.0%}"}), hide_index=True)
    hist = state.get("history", [])
    if hist:
        with st.expander("Learning history"):
            st.dataframe(pd.DataFrame(hist[::-1]), hide_index=True)

# --------------------------------------------------------------------------- data & model
with tab_data:
    st.subheader("1. Get data")
    st.write("**Option A:** scrape UFCStats. A first full run takes a while (~8,000 fights); later runs only fetch "
             "new events. **Option B:** import the Kaggle *UFC_full_data_silver.csv* your notebooks use, then "
             "run an update to add anything newer.")
    c1, c2 = st.columns(2)
    with c1:
        limit = st.number_input("Only the most recent N new events (0 = all)", 0, 1000, 0,
                                help="Handy for a quick first test run.")
        if st.button("Scrape / update from UFCStats", type="primary"):
            cb = progress_bar()
            try:
                out = scraper.update_all(cb, max_events=int(limit) or None, odds_api_key=odds_key or None)
                st.success(out)
                st.cache_data.clear()
            except Exception as ex:
                st.error(f"Update failed: {ex}")
    with c2:
        up = st.file_uploader("Import Kaggle silver CSV", type=["csv"])
        if up is not None and st.button("Import this file"):
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                tmp.write(up.getbuffer())
            try:
                st.success(scraper.import_silver_csv(tmp.name))
                st.cache_data.clear()
            except Exception as ex:
                st.error(f"Import failed: {ex}")

    st.markdown("**Option C: paste the card.** UFCStats now blocks scrapers, so for the announced bouts open "
                "ufc.com or Tapology, and paste one bout per line as `Fighter A vs Fighter B`. Put the main event "
                "first; add `(title)` for a title fight.")
    cc1, cc2 = st.columns([2, 1])
    ev_name = cc1.text_input("Event name", placeholder="UFC 332: Someone vs Someone")
    ev_date = cc2.date_input("Event date", value=pd.Timestamp.today().normalize() + pd.Timedelta(days=3))
    card_txt = st.text_area("Bouts, one per line", height=180,
                            placeholder="Islam Makhachev vs Jack Della Maddalena (title)\nMarlon Vera vs Giga Chikadze")
    if st.button("Add this card"):
        try:
            from ufc_predictor.pipeline import card_from_text, save_manual_card
            card, notes = card_from_text(card_txt, ev_name.strip() or "Pasted card", ev_date)
            for n in notes:
                st.warning(n)
            if card.empty:
                st.error("No bouts were added.")
            else:
                st.success(f"Added {save_manual_card(card)} bouts to {card['event_name'].iloc[0]}. Open Fight cards.")
                st.cache_data.clear()
        except Exception as ex:
            st.error(f"Could not add the card: {ex}")

    st.subheader("2. Train")
    pset = st.selectbox("Model set", ["balanced", "fast", "full"],
                        format_func=lambda p: f"{p}: " + ", ".join(LEARNER_NAMES[l] for l in preset(p)),
                        key="train_preset")
    missing = [n for n in ("xgb", "lgbm") if n not in available_learners()]
    if missing:
        st.caption("Not installed: " + ", ".join(LEARNER_NAMES[m] for m in missing) +
                   ". sklearn's gradient boosting fills in; `pip install xgboost lightgbm` to add them.")
    use_h = st.checkbox("Use mistake weighting from backtests / ledger", value=True)
    if st.button("Train model", type="primary", disabled=status["fights"] < 500):
        with st.spinner("Training…"):
            stt = learning.load_state()
            stt["learners"] = preset(pset)
            p = learning.retrain(state=stt, use_hardness=use_h)
        st.success(f"Trained on {p.meta['n_train']:,} fights (through {p.meta['data_through']}).")
        st.cache_resource.clear()
        st.cache_data.clear()

    st.subheader("Data preview")
    f = store.read(config.FIGHTS_CSV)
    if not f.empty:
        st.dataframe(f.sort_values("event_date", ascending=False).head(50), hide_index=True)
    r = store.read(config.RANKINGS_CSV)
    if not r.empty:
        with st.expander("Official UFC rankings"):
            st.dataframe(r, hide_index=True)
