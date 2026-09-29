"""
Data collection, ported from `ufc-full-rankings-betting-odds-scraper.ipynb`.

Changes vs. the notebook
------------------------
* One request per fight: metadata AND totals are parsed from the same page
  (the notebook fetched every fight page twice).
* Incremental: only events/fights/fighters not already on disk are scraped.
* Upcoming cards are scraped too, so the app can predict real announced bouts.
* Fixes: `title_fight` was always False (" Bout" was stripped before checking
  for "Title Bout"); fights without a W/L/D/NC status (not yet fought) are
  skipped instead of being stored as "No Contest"; future-dated events on the
  "completed" list are ignored.
"""
from __future__ import annotations

import logging
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Callable, Iterable, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from . import config, store

log = logging.getLogger(__name__)
BASE = "http://ufcstats.com"  # fighter/fight URLs in the data use http; requests are upgraded to https
UA = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
]
STAT_KEYS = ["knockdowns", "sig_strikes_succ", "sig_strikes_att", "total_strikes_succ",
             "total_strikes_att", "takedown_succ", "takedown_att", "submission_att",
             "reversals", "ctrl_time_sec"]
TARGET_KEYS = ["head", "body", "leg", "distance", "clinch", "ground"]
Progress = Optional[Callable[[float, str], None]]

_local = threading.local()


# --------------------------------------------------------------------------- HTTP
def _session() -> requests.Session:
    s = getattr(_local, "s", None)
    if s is None:
        s = requests.Session()
        retry = Retry(total=4, backoff_factor=0.8, status_forcelist=(429, 500, 502, 503, 504),
                      allowed_methods=frozenset(["GET"]))
        s.mount("http://", HTTPAdapter(max_retries=retry))
        s.mount("https://", HTTPAdapter(max_retries=retry))
        _local.s = s
    return s


def _browser_session():
    """A session that impersonates a real Chrome at the TLS level (curl_cffi). UFCStats sits behind a
    'checking your browser' wall that blocks plain Python requests from data-centre addresses; this
    usually gets through. Returns None if curl_cffi isn't installed."""
    s = getattr(_local, "cffi", None)
    if s is None:
        try:
            from curl_cffi import requests as cffi_requests
            s = cffi_requests.Session(impersonate="chrome")
        except Exception:
            s = False
        _local.cffi = s
    return s or None


def _looks_blocked(text: str) -> bool:
    head = text[:3000].lower()
    return ("checking your browser" in head or "requires javascript" in head or "cf-chl" in head
            or "just a moment" in head)


def fetch_html(url: str, timeout: int = 20) -> str:
    url = url.replace("http://ufcstats.com", "https://ufcstats.com", 1)
    headers = {"User-Agent": random.choice(UA), "Accept-Language": "en-US,en;q=0.9",
               "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}
    bs = _browser_session()
    if bs is not None:
        try:
            r = bs.get(url, timeout=timeout, headers={"Accept-Language": "en-US,en;q=0.9"})
            if r.status_code == 200 and not _looks_blocked(r.text):
                return r.text
        except Exception as e:  # fall through to plain requests
            log.warning("browser-impersonated fetch failed for %s: %s", url, e)
    r = _session().get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    if _looks_blocked(r.text):
        raise RuntimeError("UFCStats returned a browser-verification page (bot wall). "
                           "Install curl_cffi (pip install curl_cffi) or run the update from a home connection.")
    return r.text


def get_soup(url: str, timeout: int = 20, polite: float = 0.15) -> BeautifulSoup:
    time.sleep(random.uniform(0, polite))
    return BeautifulSoup(fetch_html(url, timeout), "html.parser")


def parallel_map(fn, items: list, workers: int, progress: Progress = None, label: str = "") -> dict:
    out = {}
    if not items:
        return out
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, it): it for it in items}
        for k, fut in enumerate(as_completed(futs), 1):
            it = futs[fut]
            try:
                out[it] = fut.result()
            except Exception as e:  # keep going; failed items get retried next update
                log.warning("%s failed for %s: %s", label, it, e)
                out[it] = None
            if progress:
                progress(k / len(items), f"{label} {k}/{len(items)}")
    return out


# --------------------------------------------------------------------------- helpers
def _norm(u: str | None) -> str:
    return (u or "").strip().rstrip("/")


def _to_int(txt: str) -> int:
    try:
        return int(str(txt).strip().replace("%", "").replace("---", "0").replace("--", "0"))
    except ValueError:
        return 0


def _split_of(txt: str) -> tuple[int, int]:
    try:
        a, b = str(txt).split(" of ")
        return int(a.strip()), int(b.strip())
    except ValueError:
        return 0, 0


def _mmss(txt: str) -> int:
    try:
        m, s = str(txt).strip().split(":")
        return int(m) * 60 + int(s)
    except ValueError:
        return 0


def _get_info(label: str, soup: BeautifulSoup) -> Optional[str]:
    for p in soup.select("p.b-fight-details__text"):
        for tag in p.find_all("i", class_="b-fight-details__label"):
            if tag.get_text(strip=True) == label:
                nxt = tag.next_sibling
                while nxt is not None:
                    t = nxt.strip() if isinstance(nxt, str) else nxt.get_text(strip=True)
                    if t:
                        return t
                    nxt = nxt.next_sibling
    return None


# --------------------------------------------------------------------------- pages
DATE_RX = re.compile(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}")


class ScrapeError(RuntimeError):
    pass


def scrape_event_list(kind: str = "completed") -> pd.DataFrame:
    url = f"{BASE}/statistics/events/{kind}?page=all"
    time.sleep(random.uniform(0, 0.15))
    html_text = fetch_html(url, timeout=30)
    soup = BeautifulSoup(html_text, "html.parser")
    rows = []
    # Find events by their links rather than exact table classes, so small site changes don't break it.
    for a in soup.select("a[href*='event-details']"):
        tr = a.find_parent("tr")
        if tr is None:
            continue
        m = DATE_RX.search(tr.get_text(" ", strip=True))
        if not m:
            continue
        try:
            d = datetime.strptime(m.group(0), "%B %d, %Y")
        except ValueError:
            continue
        tds = tr.find_all("td")
        rows.append({
            "event_url": _norm(a.get("href")),
            "event_name": a.get_text(strip=True),
            "event_date": pd.Timestamp(d),
            "event_location": tds[1].get_text(" ", strip=True) if len(tds) > 1 else None,
        })
    if not rows:
        # Save what the site sent so the problem can be diagnosed, then fail loudly.
        config.ensure_dirs()
        dbg = config.DATA / f"debug_events_{kind}.html"
        dbg.write_text(html_text, encoding="utf-8")
        title = soup.title.get_text(strip=True) if soup.title else "(no title)"
        snippet = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))[:300]
        raise ScrapeError(
            f"UFCStats returned a page with no events ({url}), page title '{title}'. "
            f"Start of page text: {snippet!r}. Raw page saved to {dbg}.")
    df = pd.DataFrame(rows, columns=["event_url", "event_name", "event_date", "event_location"]).drop_duplicates("event_url")
    today = pd.Timestamp.today().normalize()
    if kind == "completed":
        df = df[df["event_date"] < today]
    else:
        df = df[df["event_date"] >= today - pd.Timedelta(days=1)].sort_values("event_date")
    return df.reset_index(drop=True)


def scrape_event_card(event_url: str) -> pd.DataFrame:
    soup = get_soup(event_url)
    rows = []
    for i, tr in enumerate(soup.select("tr.b-fight-details__table-row")):
        link = tr.get("data-link") or ""
        if "fight-details" not in link:
            a = tr.select_one("a[href*='fight-details']")
            link = a.get("href", "") if a else ""
        tds = tr.find_all("td")
        fighters = tds[1].select("a[href*='fighter-details']") if len(tds) > 1 else []
        if "fight-details" not in link or len(fighters) < 2:
            continue
        wc_td = tds[6] if len(tds) > 6 else None
        rows.append({
            "event_url": _norm(event_url),
            "fight_url": _norm(link),
            "bout_order": i,  # 0 = main event
            "f_1_name": fighters[0].get_text(strip=True), "f_1_url": _norm(fighters[0].get("href")),
            "f_2_name": fighters[1].get_text(strip=True), "f_2_url": _norm(fighters[1].get("href")),
            "weight_class": wc_td.get_text(" ", strip=True) if wc_td else None,
            "title_fight": bool(wc_td and wc_td.select_one("img[src*='belt']")),
        })
    return pd.DataFrame(rows)


def _summary_tables(soup: BeautifulSoup) -> dict:
    out = {}
    for p in soup.select("p.b-fight-details__collapse-link_tot"):
        label = p.get_text(strip=True).lower()
        tbl = p.find_next("table")
        if tbl is None:
            continue
        if "total" in label and "totals" not in out:
            out["totals"] = tbl
        elif "significant" in label and "sig" not in out:
            out["sig"] = tbl
    return out


def _cell(tds, col: int, i: int) -> str:
    try:
        return tds[col].find_all("p")[i].get_text(strip=True)
    except (IndexError, AttributeError):
        return "0"


def scrape_fight(fight_url: str) -> Optional[dict]:
    """Metadata + per-fighter totals for one bout. Returns None if not parseable."""
    soup = get_soup(fight_url)
    persons = soup.select("div.b-fight-details__person")
    if len(persons) != 2:
        return None
    names, urls, status = [], [], []
    for p in persons:
        a = p.select_one("h3 a") or p.select_one("a")
        names.append(a.get_text(strip=True) if a else None)
        urls.append(_norm(a.get("href")) if a else None)
        s = p.select_one(".b-fight-details__person-status")
        status.append(s.get_text(strip=True).upper() if s else "")
    if not all(names):
        return None
    if status[0] == "W":
        winner = names[0]
    elif status[1] == "W":
        winner = names[1]
    elif status[0] == status[1] == "D":
        winner = "Draw"
    elif status[0] == status[1] == "NC":
        winner = "No Contest"
    else:
        return {"fight_url": _norm(fight_url), "completed": False}

    title_el = soup.select_one("i.b-fight-details__fight-title")
    raw_title = title_el.get_text(" ", strip=True) if title_el else ""
    num_rounds = 0
    tf = _get_info("Time format:", soup) or ""
    m = re.search(r"(\d+)\s*Rnd", tf)
    if m:
        num_rounds = int(m.group(1))
    details = None
    for p in soup.select("p.b-fight-details__text"):
        if "Details:" in p.get_text():
            details = p.get_text(" ", strip=True).split("Details:")[-1].strip()

    rec = {
        "fight_url": _norm(fight_url), "completed": True,
        "f_1_name": names[0], "f_1_url": urls[0], "f_2_name": names[1], "f_2_url": urls[1],
        "winner": winner,
        "result": _get_info("Method:", soup),
        "result_details": details,
        "finish_round": _to_int(_get_info("Round:", soup) or 0),
        "finish_time": _get_info("Time:", soup),
        "num_rounds": num_rounds,
        "referee": _get_info("Referee:", soup),
        "weight_class": re.sub(r"\s*Bout$", "", raw_title).strip() or None,
        "title_fight": "title" in raw_title.lower(),
        "gender": "F" if "women" in raw_title.lower() else "M",
    }
    tables = _summary_tables(soup)
    if "totals" in tables:
        tr = tables["totals"].select_one("tbody tr")
        tds = tr.find_all("td") if tr else []
        for i, pref in enumerate(("f_1", "f_2")):
            ss, sa = _split_of(_cell(tds, 2, i))
            ts, ta = _split_of(_cell(tds, 4, i))
            ds, da = _split_of(_cell(tds, 5, i))
            rec.update({
                f"{pref}_knockdowns": _to_int(_cell(tds, 1, i)),
                f"{pref}_sig_strikes_succ": ss, f"{pref}_sig_strikes_att": sa,
                f"{pref}_total_strikes_succ": ts, f"{pref}_total_strikes_att": ta,
                f"{pref}_takedown_succ": ds, f"{pref}_takedown_att": da,
                f"{pref}_submission_att": _to_int(_cell(tds, 7, i)),
                f"{pref}_reversals": _to_int(_cell(tds, 8, i)),
                f"{pref}_ctrl_time_sec": _mmss(_cell(tds, 9, i)),
            })
    if "sig" in tables:
        tr = tables["sig"].select_one("tbody tr")
        tds = tr.find_all("td") if tr else []
        for i, pref in enumerate(("f_1", "f_2")):
            for j, key in enumerate(TARGET_KEYS, start=3):
                s, a = _split_of(_cell(tds, j, i))
                rec[f"{pref}_{key}_succ"], rec[f"{pref}_{key}_att"] = s, a
    return rec


def scrape_fighter(fighter_url: str) -> Optional[dict]:
    soup = get_soup(fighter_url)
    t = soup.select_one("span.b-content__title-highlight")
    name = t.get_text(" ", strip=True) if t else None
    if not name:
        return None
    rec = {"fighter_url": _norm(fighter_url), "fighter_name": name,
           "fighter_height_cm": None, "fighter_weight_lbs": None, "fighter_reach_cm": None,
           "fighter_stance": None, "fighter_dob": None}
    nn = soup.select_one("p.b-content__Nickname")
    rec["fighter_nickname"] = nn.get_text(strip=True) if nn else None
    r = soup.select_one("span.b-content__title-record")
    m = re.search(r"(\d+)-(\d+)-(\d+)", r.get_text(" ", strip=True) if r else "")
    if m:
        rec["record_w"], rec["record_l"], rec["record_d"] = map(int, m.groups())
    for li in soup.select("li.b-list__box-list-item"):
        txt = li.get_text(" ", strip=True)
        if txt.startswith("Height:"):
            mm = re.search(r"(\d+)\s*'\s*(\d+)", txt)
            if mm:
                rec["fighter_height_cm"] = round((int(mm.group(1)) * 12 + int(mm.group(2))) * 2.54)
        elif txt.startswith("Weight:"):
            mm = re.search(r"(\d+)", txt)
            rec["fighter_weight_lbs"] = int(mm.group(1)) if mm else None
        elif txt.startswith("Reach:"):
            mm = re.search(r"(\d+(?:\.\d+)?)", txt)
            rec["fighter_reach_cm"] = round(float(mm.group(1)) * 2.54) if mm else None
        elif txt.upper().startswith("STANCE:"):
            rec["fighter_stance"] = txt.split(":", 1)[1].strip() or None
        elif txt.startswith("DOB:"):
            raw = txt.split(":", 1)[1].strip()
            for fmt in ("%b %d, %Y", "%B %d, %Y"):
                try:
                    rec["fighter_dob"] = datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
                    break
                except ValueError:
                    pass
    return rec


def scrape_rankings() -> pd.DataFrame:
    soup = get_soup("https://www.ufc.com/rankings", polite=0)
    out, today = [], pd.Timestamp.today().normalize()
    for sec in soup.find_all("div", class_="view-grouping"):
        header = sec.find("div", class_="view-grouping-header")
        division = re.sub(r"\s+", " ", header.get_text()).strip() if header else "Unknown"
        champ = sec.select_one("div.rankings--athlete--champion")
        if champ:
            tag = champ.find("h5") or champ.find("a")
            nm = re.sub(r"\s+", " ", tag.get_text()).strip() if tag else ""
            if nm:
                out.append({"date": today, "weightclass": division, "fighter": nm, "rank": 0})
        cells = sec.select("td.views-field.views-field-title") or sec.select("table a[href*='/athlete/']")
        for rank, cell in enumerate(cells, 1):
            nm = re.sub(r"\s+", " ", cell.get_text()).strip()
            if nm:
                out.append({"date": today, "weightclass": division, "fighter": nm, "rank": rank})
    return pd.DataFrame(out, columns=["date", "weightclass", "fighter", "rank"]).drop_duplicates()


def fetch_odds(api_key: str) -> pd.DataFrame:
    """Current MMA moneylines from The Odds API (free tier ~500 req/month)."""
    r = requests.get("https://api.the-odds-api.com/v4/sports/mma_mixed_martial_arts/odds",
                     params={"regions": "us,uk,eu", "markets": "h2h", "oddsFormat": "decimal",
                             "apiKey": api_key}, timeout=20)
    r.raise_for_status()
    rows, now = [], pd.Timestamp.utcnow().isoformat()
    for match in r.json():
        for bk in match.get("bookmakers", []):
            for mk in bk.get("markets", []):
                oc = mk.get("outcomes") or []
                if mk.get("key") != "h2h" or len(oc) != 2:
                    continue
                rows.append({"commence_time": match.get("commence_time"), "bookmaker": bk.get("title"),
                             "fighter_1": oc[0].get("name"), "odds_1": oc[0].get("price"),
                             "fighter_2": oc[1].get("name"), "odds_2": oc[1].get("price"),
                             "fetched_at": now})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- orchestration
def update_all(progress: Progress = None, max_events: Optional[int] = None,
               refresh_upcoming_fighters: bool = True, odds_api_key: Optional[str] = None,
               rankings: bool = True, workers: Optional[int] = None) -> dict:
    """Incrementally bring the local database up to date with UFCStats."""
    config.ensure_dirs()
    workers = workers or config.DEFAULTS["scrape_workers"]
    say = progress or (lambda f, m: log.info(m))
    summary = {}

    # 1) completed events not yet fully scraped
    say(0.0, "Fetching completed events list…")
    events = scrape_event_list("completed")
    done = store.read(config.EVENTS_CSV)
    known = set(done["event_url"]) if not done.empty else set()
    new_events = events[~events["event_url"].isin(known)].sort_values("event_date")
    if max_events:
        new_events = new_events.tail(max_events)
    summary["new_events"] = len(new_events)

    fights = store.read(config.FIGHTS_CSV)
    have_fights = set(fights["fight_url"]) if not fights.empty else set()

    cards = parallel_map(scrape_event_card, new_events["event_url"].tolist(), workers,
                         lambda f, m: say(0.05 + 0.15 * f, m), "Event cards")
    card_rows = [c for c in cards.values() if c is not None and not c.empty]
    card_df = pd.concat(card_rows, ignore_index=True) if card_rows else pd.DataFrame()

    todo = [] if card_df.empty else [u for u in card_df["fight_url"] if u not in have_fights]
    details = parallel_map(scrape_fight, todo, workers, lambda f, m: say(0.2 + 0.5 * f, m), "Fights")
    recs = [d for d in details.values() if d and d.get("completed")]
    new_fights = pd.DataFrame(recs)
    if not new_fights.empty:
        meta = card_df[["fight_url", "event_url", "bout_order"]].merge(
            new_events[["event_url", "event_name", "event_date"]], on="event_url", how="left")
        new_fights = new_fights.merge(meta, on="fight_url", how="left").drop(columns=["completed"])
        fights = store.upsert(new_fights, config.FIGHTS_CSV, "fight_url")
    summary["new_fights"] = len(new_fights)

    # an event is 'done' only when every bout on its card parsed as completed
    if not card_df.empty:
        have_now = set(fights["fight_url"]) if not fights.empty else set()
        ok = card_df.groupby("event_url")["fight_url"].apply(lambda s: all(u in have_now for u in s))
        finished = new_events[new_events["event_url"].isin(ok[ok].index)]
        store.upsert(finished, config.EVENTS_CSV, "event_url")

    # 2) upcoming cards
    say(0.72, "Fetching upcoming events…")
    try:
        up_events = scrape_event_list("upcoming").head(config.DEFAULTS["upcoming_events"])
        up_cards = parallel_map(scrape_event_card, up_events["event_url"].tolist(), workers, None, "Upcoming")
        parts = [c for c in up_cards.values() if c is not None and not c.empty]
        upcoming = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if not upcoming.empty:
            upcoming = upcoming.merge(up_events, on="event_url", how="left")
            # UFCStats does not publish scheduled rounds for future bouts: main events and title fights are 5
            upcoming["num_rounds"] = ((upcoming["bout_order"] == 0) | upcoming["title_fight"]).map({True: 5, False: 3})
            upcoming["gender"] = upcoming["weight_class"].fillna("").str.contains("Women").map({True: "F", False: "M"})
        store.write(upcoming, config.UPCOMING_CSV)
        summary["upcoming_fights"] = len(upcoming)
    except Exception as e:
        log.warning("Upcoming scrape failed: %s", e)
        summary["upcoming"] = f"failed: {e}"
        upcoming = store.read(config.UPCOMING_CSV)

    # 3) fighter profiles (new fighters + everyone on upcoming cards)
    fighters = store.read(config.FIGHTERS_CSV)
    have_f = set(fighters["fighter_url"]) if not fighters.empty else set()
    need = set()
    for df in (fights, upcoming):
        if df is not None and not df.empty:
            need |= set(df["f_1_url"].dropna()) | set(df["f_2_url"].dropna())
    todo_f = sorted(u for u in need if u and u not in have_f)
    if refresh_upcoming_fighters and upcoming is not None and not upcoming.empty:
        todo_f = sorted(set(todo_f) | set(upcoming["f_1_url"]) | set(upcoming["f_2_url"]))
    prof = parallel_map(scrape_fighter, todo_f, workers, lambda f, m: say(0.75 + 0.2 * f, m), "Fighters")
    new_f = pd.DataFrame([p for p in prof.values() if p])
    if not new_f.empty:
        store.upsert(new_f, config.FIGHTERS_CSV, "fighter_url")
    summary["fighters_scraped"] = len(new_f)

    # 4) optional extras
    if rankings:
        try:
            store.write(scrape_rankings(), config.RANKINGS_CSV)
            summary["rankings"] = "updated"
        except Exception as e:
            summary["rankings"] = f"failed: {e}"
    if odds_api_key:
        try:
            store.write(fetch_odds(odds_api_key), config.ODDS_CSV)
            summary["odds"] = "updated"
        except Exception as e:
            summary["odds"] = f"failed: {e}"
    say(1.0, "Done")
    return summary


# --------------------------------------------------------------------------- Kaggle import
def import_silver_csv(path) -> dict:
    """
    Seed the database from the Kaggle 'UFC_full_data_silver.csv' used by the
    notebooks, so you don't have to scrape ~8k fights from scratch. The next
    `update_all` fills in anything newer.
    """
    df = pd.read_csv(path, low_memory=False)
    ren = {"f_1": "f_1_name", "f_2": "f_2_name", "f_1_fighter_url": "f_1_url", "f_2_fighter_url": "f_2_url"}
    df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns and v not in df.columns})
    need = ["f_1_name", "f_2_name", "winner", "event_date"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    for p in ("f_1", "f_2"):
        if f"{p}_url" not in df.columns:
            df[f"{p}_url"] = "name:" + df[f"{p}_name"].astype(str)
    if "fight_url" not in df.columns:
        df["fight_url"] = ("import:" + df["event_date"].astype(str) + ":" +
                           df["f_1_url"].astype(str) + ":" + df["f_2_url"].astype(str))
    for c in ("f_1_url", "f_2_url", "fight_url"):
        df[c] = df[c].astype(str).str.strip().str.rstrip("/")
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df = df[df["winner"].notna()]

    bio_keys = ["fighter_height_cm", "fighter_reach_cm", "fighter_weight_lbs", "fighter_stance", "fighter_dob"]
    frames = []
    for p in ("f_1", "f_2"):
        cols = {f"{p}_url": "fighter_url", f"{p}_name": "fighter_name"}
        cols.update({f"{p}_{k}": k for k in bio_keys if f"{p}_{k}" in df.columns})
        frames.append(df[list(cols)].rename(columns=cols))
    fighters = pd.concat(frames).sort_values("fighter_url").drop_duplicates("fighter_url", keep="last")

    keep = [c for c in df.columns if not any(c.startswith(f"{p}_{k}") for p in ("f_1", "f_2") for k in bio_keys)]
    keep = [c for c in keep if not re.match(r"^f_[12]_r\d_", c)]  # per-round columns are not used
    fights = df[keep]
    store.upsert(fights, config.FIGHTS_CSV, "fight_url")
    store.upsert(fighters, config.FIGHTERS_CSV, "fighter_url")
    return {"fights_imported": len(fights), "fighters_imported": len(fighters),
            "has_odds": "f_1_odds" in fights.columns}
