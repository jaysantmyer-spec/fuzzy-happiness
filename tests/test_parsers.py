"""Parser smoke tests on HTML fragments shaped like UFCStats pages (no network)."""
import sys
from pathlib import Path
from bs4 import BeautifulSoup
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ufc_predictor import scraper

FIGHT = """
<div class="b-fight-details__person"><i class="b-fight-details__person-status">W</i>
 <h3><a href="http://ufcstats.com/fighter-details/aaa/">Alex Alpha</a></h3></div>
<div class="b-fight-details__person"><i class="b-fight-details__person-status">L</i>
 <h3><a href="http://ufcstats.com/fighter-details/bbb">Ben Beta</a></h3></div>
<i class="b-fight-details__fight-title">UFC Lightweight Title Bout</i>
<p class="b-fight-details__text">
 <i class="b-fight-details__text-item_first"><i class="b-fight-details__label">Method:</i><i>KO/TKO</i></i>
 <i class="b-fight-details__text-item"><i class="b-fight-details__label">Round:</i> 2 </i>
 <i class="b-fight-details__text-item"><i class="b-fight-details__label">Time:</i> 3:14 </i>
 <i class="b-fight-details__text-item"><i class="b-fight-details__label">Time format:</i> 5 Rnd (5-5-5-5-5) </i>
 <i class="b-fight-details__text-item"><i class="b-fight-details__label">Referee:</i><span>Herb Dean</span></i></p>
<p class="b-fight-details__text">Details: Punches to Head At Distance</p>
<p class="b-fight-details__collapse-link_tot">Totals</p>
<table><tbody><tr>
 <td><p>Alex</p><p>Ben</p></td><td><p>1</p><p>0</p></td><td><p>40 of 80</p><p>20 of 60</p></td>
 <td><p>50%</p><p>33%</p></td><td><p>55 of 95</p><p>25 of 70</p></td><td><p>2 of 4</p><p>0 of 1</p></td>
 <td><p>50%</p><p>0%</p></td><td><p>1</p><p>0</p></td><td><p>0</p><p>0</p></td><td><p>3:21</p><p>0:10</p></td>
</tr></tbody></table>
<p class="b-fight-details__collapse-link_tot">Significant Strikes</p>
<table><tbody><tr>
 <td><p>A</p><p>B</p></td><td><p>40 of 80</p><p>20 of 60</p></td><td><p>50%</p><p>33%</p></td>
 <td><p>20 of 50</p><p>10 of 40</p></td><td><p>10 of 15</p><p>5 of 10</p></td><td><p>10 of 15</p><p>5 of 10</p></td>
 <td><p>30 of 60</p><p>18 of 55</p></td><td><p>5 of 10</p><p>2 of 5</p></td><td><p>5 of 10</p><p>0 of 0</p></td>
</tr></tbody></table>
"""
CARD = """<table><tr class="b-fight-details__table-row" data-link="http://ufcstats.com/fight-details/f1">
<td></td><td><p><a href="http://ufcstats.com/fighter-details/aaa">Alex Alpha</a></p>
<p><a href="http://ufcstats.com/fighter-details/bbb">Ben Beta</a></p></td><td></td><td></td><td></td><td></td>
<td><p>Lightweight <img src="/belt.png"></p></td></tr></table>"""
UPCOMING_FIGHT = FIGHT.replace(">W<", "><").replace(">L<", "><")


def run(html, fn, arg):
    orig = scraper.get_soup
    scraper.get_soup = lambda *a, **k: BeautifulSoup(html, "html.parser")
    try:
        return fn(arg)
    finally:
        scraper.get_soup = orig


def test_parsers():
    r = run(FIGHT, scraper.scrape_fight, "x")
    assert r["winner"] == "Alex Alpha" and r["title_fight"] and r["num_rounds"] == 5
    assert r["result"] == "KO/TKO" and r["finish_round"] == 2 and r["finish_time"] == "3:14"
    assert r["f_1_sig_strikes_succ"] == 40 and r["f_2_sig_strikes_att"] == 60
    assert r["f_1_ctrl_time_sec"] == 201 and r["f_1_head_succ"] == 20 and r["f_1_url"].endswith("aaa")
    assert r["weight_class"] == "UFC Lightweight Title"
    assert run(UPCOMING_FIGHT, scraper.scrape_fight, "x") == {"fight_url": "x", "completed": False}
    c = run(CARD, scraper.scrape_event_card, "ev")
    assert len(c) == 1 and bool(c.title_fight[0]) and c.f_2_name[0] == "Ben Beta"
    print("parsers OK", r["referee"], r["result_details"])


if __name__ == "__main__":
    test_parsers()
