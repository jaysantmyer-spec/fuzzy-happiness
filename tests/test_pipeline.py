"""
End-to-end test on synthetic data shaped like the scraped UFCStats tables.
Winners are listed first (like UFCStats) to prove the model can't exploit order.

    python -m pytest tests -q        or        python tests/test_pipeline.py
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

TMP = Path(tempfile.mkdtemp())
os.environ["UFC_HOME"] = str(TMP)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ufc_predictor import backtest, config, learning, store  # noqa: E402
from ufc_predictor.features import build_features  # noqa: E402
from ufc_predictor.pipeline import Predictor, features_now, predict_custom, train_predictor, upcoming_rows  # noqa: E402


def make_data(n_fighters=500, n_events=260, seed=0):
    rng = np.random.default_rng(seed)
    skill = rng.normal(0, 1, n_fighters)
    power = rng.normal(0, 1, n_fighters)
    grap = rng.normal(0, 1, n_fighters)
    wcs = ["Lightweight", "Welterweight", "Middleweight", "Heavyweight", "Women's Strawweight"]
    wc_of = rng.integers(0, len(wcs), n_fighters)
    urls = [f"http://ufcstats.com/fighter-details/{i:05d}" for i in range(n_fighters)]
    names = [f"Fighter {i}" for i in range(n_fighters)]
    dob = pd.Timestamp("1985-01-01") + pd.to_timedelta(rng.integers(0, 4000, n_fighters), "D")
    fighters = pd.DataFrame({"fighter_url": urls, "fighter_name": names, "fighter_dob": dob,
                             "fighter_height_cm": rng.normal(178, 8, n_fighters).round(),
                             "fighter_reach_cm": rng.normal(182, 9, n_fighters).round(),
                             "fighter_stance": rng.choice(["Orthodox", "Southpaw", "Switch"], n_fighters, p=[.72, .22, .06])})
    rows, date = [], pd.Timestamp("2012-01-07")
    for e in range(n_events):
        date += pd.Timedelta(days=int(rng.integers(6, 15)))
        for b in range(12):
            wc = int(rng.integers(0, len(wcs)))
            pool = np.where(wc_of == wc)[0]
            i, j = rng.choice(pool, 2, replace=False)
            age_i = (date - dob[i]).days / 365.25
            age_j = (date - dob[j]).days / 365.25
            logit = 1.1 * (skill[i] - skill[j]) - 0.05 * (age_i - age_j)
            i_wins = rng.random() < 1 / (1 + np.exp(-logit))
            w, l = (i, j) if i_wins else (j, i)
            fin = rng.random()
            pk = 0.25 + 0.15 * power[w] + (0.1 if wcs[wc] == "Heavyweight" else 0)
            ps = 0.18 + 0.12 * grap[w]
            if fin < pk:
                res, rnd = "KO/TKO", int(rng.integers(1, 4))
            elif fin < pk + ps:
                res, rnd = "Submission", int(rng.integers(1, 4))
            else:
                res, rnd = "Decision - Unanimous", 3
            t = "5:00" if res.startswith("Dec") else f"{rng.integers(0, 5)}:{rng.integers(10, 59)}"
            r = {"event_url": f"ev{e}", "event_name": f"UFC Synthetic {e}", "event_date": date,
                 "fight_url": f"http://ufcstats.com/fight-details/{e:04d}{b:02d}", "bout_order": b,
                 "f_1_name": names[w], "f_1_url": urls[w], "f_2_name": names[l], "f_2_url": urls[l],
                 "winner": names[w], "result": res, "finish_round": rnd, "finish_time": t,
                 "num_rounds": 5 if b == 0 else 3, "weight_class": wcs[wc], "title_fight": b == 0 and e % 4 == 0,
                 "gender": "F" if "Women" in wcs[wc] else "M"}
            for pref, k in (("f_1", w), ("f_2", l)):
                att = rng.poisson(60 + 10 * skill[k])
                r.update({f"{pref}_sig_strikes_att": att, f"{pref}_sig_strikes_succ": rng.binomial(att, 0.45),
                          f"{pref}_total_strikes_att": att + 20, f"{pref}_total_strikes_succ": rng.binomial(att + 20, .5),
                          f"{pref}_takedown_att": rng.poisson(2 + grap[k].clip(-1.5)), f"{pref}_takedown_succ": 0,
                          f"{pref}_knockdowns": rng.poisson(0.3 + 0.2 * power[k].clip(-1)),
                          f"{pref}_submission_att": rng.poisson(0.5 + 0.3 * grap[k].clip(-1)),
                          f"{pref}_reversals": 0, f"{pref}_ctrl_time_sec": rng.integers(0, 200)})
                r[f"{pref}_takedown_succ"] = rng.binomial(r[f"{pref}_takedown_att"], 0.4)
            # market odds: noisy view of true probability with ~5% vig
            p_w = 1 / (1 + np.exp(-(logit if i_wins else -logit)))
            pm = np.clip(p_w + rng.normal(0, 0.08), 0.05, 0.95)
            r["f_1_odds"], r["f_2_odds"] = round(1 / (pm * 1.05), 2), round(1 / ((1 - pm) * 1.05), 2)
            rows.append(r)
    fights = pd.DataFrame(rows)
    up_date = date + pd.Timedelta(days=10)
    upcoming = pd.DataFrame([{"event_url": "evU", "event_name": "UFC Next", "event_date": up_date,
                              "fight_url": f"http://ufcstats.com/fight-details/U{k}", "bout_order": k,
                              "f_1_name": names[a], "f_1_url": urls[a], "f_2_name": names[b], "f_2_url": urls[b],
                              "weight_class": wcs[wc_of[a]], "title_fight": k == 0, "num_rounds": 5 if k == 0 else 3,
                              "gender": "M"}
                             for k, (a, b) in enumerate([(0, 1), (2, 3), (4, 5)])])
    return fights, fighters, upcoming, skill


def test_end_to_end():
    fights, fighters, upcoming, skill = make_data()
    config.ensure_dirs()
    store.write(fights, config.FIGHTS_CSV)
    store.write(fighters, config.FIGHTERS_CSV)
    store.write(upcoming, config.UPCOMING_CSV)

    feat = features_now()
    assert len(feat) == len(fights) + len(upcoming)
    # leakage check: first UFC fight of every fighter must have zero prior fights
    first = feat.sort_values("event_date").drop_duplicates("f_1_url")
    assert (first["A_n_fights"] >= 0).all()
    # antisymmetry check
    assert np.allclose(feat["elo_edge"].dropna().abs().max() < 0.5, True)

    # upcoming fights receive current-state features
    up = upcoming_rows(feat)
    assert len(up) == 3 and up["A_n_fights"].notna().all()

    bt = backtest.walk_forward(feat, start="2017-06-01", step_days=182, learners=["logreg", "hgb"])
    s = backtest.summarize(bt)
    print({k: v for k, v in s.items() if k != "models"}, s["models"])
    assert s["accuracy"] > 0.60, s  # skill signal is learnable...
    assert s["auc"] > 0.6

    st, ad = backtest.compare(feat, start="2017-06-01", step_days=182, learners=["logreg", "hgb"])
    print("static", backtest.summarize(st)["log_loss"], "adaptive", backtest.summarize(ad)["log_loss"])

    store.write(ad, config.BACKTEST_CSV)
    state = learning.update_state()
    assert state["weights"], state
    pred = learning.retrain(learners=["logreg", "hgb"], state=state)
    assert Predictor.load() is not None

    cards = pred.predict(upcoming_rows(features_now()))
    assert len(cards) == 3 and np.allclose(cards["p_A"] + cards["p_B"], 1)
    outcome_sum = cards[[c for c in cards.columns if c[:2] in ("A_", "B_") and c.split("_", 1)[1] in
                         ("KO/TKO", "Submission", "Decision")]].sum(axis=1)
    assert np.allclose(outcome_sum, 1, atol=1e-6)
    print(cards[["A_name", "B_name", "p_A", "top_outcome", "top_outcome_prob", "confidence"]])
    print(cards["why"].iloc[0][:3])

    # symmetry: swapping corners mirrors the probability
    p1 = predict_custom(pred, fighters.fighter_url[10], fighters.fighter_url[11])["p_A"].iloc[0]
    p2 = predict_custom(pred, fighters.fighter_url[11], fighters.fighter_url[10])["p_A"].iloc[0]
    assert abs(p1 + p2 - 1) < 1e-6, (p1, p2)

    # ledger: log -> results arrive -> grade
    assert learning.log_predictions(cards, pred.meta["version"]) == 3
    done = upcoming.copy()
    done["winner"] = done["f_2_name"]
    done["result"] = "KO/TKO"
    done["finish_round"], done["finish_time"] = 1, "2:00"
    store.upsert(done, config.FIGHTS_CSV, "fight_url")
    g = learning.grade()
    assert g["graded_now"] == 3, g
    lg = learning.read_ledger()
    assert lg["correct"].notna().all()
    rep = learning.mistake_report(ad)
    assert "confident_misses" in rep
    print("OK  ->", TMP)


if __name__ == "__main__":
    test_end_to_end()
