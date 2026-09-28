"""
Headless usage:
    python -m ufc_predictor update [--max-events 30] [--odds-key KEY]
    python -m ufc_predictor import path/to/UFC_full_data_silver.csv
    python -m ufc_predictor train [--preset balanced]
    python -m ufc_predictor backtest --start 2023-01-01 [--mode compare] [--step 91]
    python -m ufc_predictor predict [--log]
    python -m ufc_predictor cycle            # update -> grade -> re-learn -> retrain
    python -m ufc_predictor report [--event NAME] [--out card.html] [--track 10]   # shareable HTML page
    python -m ufc_predictor price [--bankroll 1000]                    # pricing desk in the terminal
    python -m ufc_predictor odds --watch 60                            # poll the market every 60 min
"""
import argparse
import json
import logging
import os

import pandas as pd

from . import backtest, config, learning, pricing, report, scraper, store
from .models import preset
from .pipeline import Predictor, attach_market, features_now, upcoming_rows


def _p(f, m):
    print(f"[{f:5.0%}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser(prog="ufc_predictor")
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("update")
    u.add_argument("--max-events", type=int)
    u.add_argument("--odds-key", default=os.getenv("ODDS_API_KEY"))
    i = sub.add_parser("import")
    i.add_argument("csv")
    t = sub.add_parser("train")
    t.add_argument("--preset", default="balanced", choices=["fast", "balanced", "full"])
    b = sub.add_parser("backtest")
    b.add_argument("--start", required=True)
    b.add_argument("--end")
    b.add_argument("--step", type=int, default=91)
    b.add_argument("--preset", default="fast", choices=["fast", "balanced", "full"])
    b.add_argument("--mode", default="adaptive", choices=["static", "adaptive", "compare"])
    p = sub.add_parser("predict")
    p.add_argument("--log", action="store_true", help="save predictions to the ledger")
    c = sub.add_parser("cycle")
    c.add_argument("--no-scrape", action="store_true")
    c.add_argument("--odds-key", default=os.getenv("ODDS_API_KEY"))
    r = sub.add_parser("report", help="write a self-contained HTML page for one card")
    r.add_argument("--event", help="event name (substring); default is the soonest card")
    r.add_argument("--out", default="ufc_card_report.html")
    r.add_argument("--track", type=int, default=0, help="also backtest the last N events and add a track record")
    pr = sub.add_parser("price", help="fair prices vs the market for every upcoming fight")
    pr.add_argument("--bankroll", type=float, default=1000.0)
    pr.add_argument("--edge", type=float, default=0.04)
    od = sub.add_parser("odds", help="snapshot the market (appends to odds_history.csv)")
    od.add_argument("--key", default=os.getenv("ODDS_API_KEY"))
    od.add_argument("--watch", type=int, default=0, help="keep polling every N minutes")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    config.ensure_dirs()

    if a.cmd == "update":
        print(json.dumps(scraper.update_all(_p, a.max_events, odds_api_key=a.odds_key), indent=2, default=str))
    elif a.cmd == "import":
        print(scraper.import_silver_csv(a.csv))
    elif a.cmd == "train":
        st = learning.load_state()
        st["learners"] = preset(a.preset)
        print(learning.retrain(state=st).meta)
    elif a.cmd == "backtest":
        feat = features_now()
        kw = dict(step_days=a.step, learners=preset(a.preset), progress=_p)
        if a.mode == "compare":
            s0, s1 = backtest.compare(feat, a.start, a.end, **kw)
            for name, bt in (("static", s0), ("adaptive", s1)):
                s = backtest.summarize(bt)
                print(f"{name:9s} acc={s['accuracy']:.3f} logloss={s['log_loss']:.4f} auc={s['auc']:.3f}")
            store.write(s1, config.BACKTEST_CSV)
        else:
            bt = backtest.walk_forward(feat, a.start, a.end, adaptive=a.mode == "adaptive", **kw)
            store.write(bt, config.BACKTEST_CSV)
            print(json.dumps(backtest.summarize(bt), indent=2, default=str))
    elif a.cmd == "predict":
        pred = Predictor.load()
        if pred is None:
            raise SystemExit("No model yet - run `train` first.")
        P = attach_market(pred.predict(upcoming_rows(features_now()), explain=False))
        with pd.option_context("display.width", 160, "display.max_rows", 200):
            print(P[["event_date", "A_name", "B_name", "p_A", "top_outcome", "top_outcome_prob",
                     "confidence", "market_p_A"]].round(3).to_string(index=False))
        if a.log:
            print("logged:", learning.log_predictions(P, pred.meta.get("version", "")))
    elif a.cmd == "report":
        pred = Predictor.load()
        if pred is None:
            raise SystemExit("No model yet - run `train` first.")
        P = attach_market(pred.predict(upcoming_rows(features_now())))
        if P.empty:
            raise SystemExit("No upcoming bouts on file.")
        P["event_label"] = P["event_name"].astype(str) + " (" + pd.to_datetime(P["event_date"]).dt.strftime("%d %b %Y") + ")"
        labels = list(P.sort_values("event_date")["event_label"].unique())
        pick = next((l for l in labels if a.event and a.event.lower() in l.lower()), labels[0])
        card = P[P["event_label"] == pick].reset_index(drop=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            results = report.results_from_fights(card, store.read(config.FIGHTS_CSV))
            track = backtest.event_backtest(features_now(), a.track, pred.meta.get("learners"), progress=_p) if a.track else None
            fh.write(report.report_html(card, pick, model_meta=pred.meta, results=results, track=track))
        print(f"wrote {a.out}: {pick}, {len(card)} bouts, {len(results)} results")
    elif a.cmd == "price":
        pred = Predictor.load()
        if pred is None:
            raise SystemExit("No model yet - run `train` first.")
        P = pred.predict(upcoming_rows(features_now()), explain=False)
        pr = pricing.price_card(P, bankroll=a.bankroll, edge_threshold=a.edge)
        with pd.option_context("display.width", 200, "display.max_rows", 500):
            print(pr[["bout", "fighter", "fair", "market", "best_odds", "best_book", "edge", "ev", "stake", "bet"]]
                  .round(3).to_string(index=False))
    elif a.cmd == "odds":
        import time
        if not a.key:
            raise SystemExit("Pass --key or set ODDS_API_KEY.")
        while True:
            snap = pricing.snapshot_odds(a.key)
            print(f"{pd.Timestamp.utcnow():%Y-%m-%d %H:%M} UTC: {len(snap)} lines from {snap['bookmaker'].nunique() if len(snap) else 0} books")
            if not a.watch:
                break
            time.sleep(a.watch * 60)
    elif a.cmd == "cycle":
        print(json.dumps(learning.run_cycle(_p, scrape=not a.no_scrape, odds_api_key=a.odds_key),
                         indent=2, default=str))


if __name__ == "__main__":
    main()
