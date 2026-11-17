"""The interactive page in a real browser, against the real server with a stand-in engine.

Skipped when Playwright or its browser is missing; CI sets KEMUDI_REQUIRE_BROWSER so that there a
missing browser is a failure, not a skip.
"""

import json
import os
import threading
import time
from pathlib import Path

import pytest

from kemudi.server import App, serve

sync_api = pytest.importorskip("playwright.sync_api") if not os.environ.get("KEMUDI_REQUIRE_BROWSER") else __import__(
    "playwright.sync_api", fromlist=["sync_playwright"]
)

SHIPPED = Path(__file__).resolve().parents[1] / "kemudi" / "calibrations" / "qwen2.5-0.5b-instruct"

PAIRED = {
    "mood": [
        {"dose": -0.3, "change": -0.19, "interval": [-0.26, -0.13], "intended": 15, "openings": 16, "clear": True},
        {"dose": 0.3, "change": 0.11, "interval": [0.05, 0.17], "intended": 14, "openings": 16, "clear": True},
        {"dose": 0.45, "change": 0.15, "interval": [0.07, 0.22], "intended": 15, "openings": 16, "clear": True},
    ],
    "speech": [{"dose": -0.45, "change": 0.0, "interval": [0.0, 0.0], "intended": 0, "openings": 16, "clear": False}],
}


def pair(first, second, fluent, works):
    corners = [
        {"doses": {first: a, second: b}, "perplexity_ratio": 1.2 if fluent or (a, b) != (1, 1) else 1.9,
         "distinct_ratio": 1.0, "fluent": fluent or (a, b) != (1, 1)}
        for a in (-1, 1) for b in (-1, 1)
    ]
    effects = [{"knob": first, "next_to": second, "still_works": works[0]}, {"knob": second, "next_to": first, "still_works": works[1]}]
    return {"knobs": [first, second], "corners": corners, "fluent_together": fluent, "effects": effects}


PAIRS = {"pairs": [pair("mood", "tension", True, (True, True)), pair("mood", "speech", False, (True, False))]}


class SlowEngine:
    """Writes word0, word1... slowly enough for a knob to be moved in the middle."""

    def __init__(self):
        self.lock = threading.Lock()
        self.knob_calls = []

    def set_gauges(self, axes):
        pass

    def set_knobs(self, axes, values):
        self.knob_calls.append(dict(values))

    def story_prompt(self, story, controls=""):
        return story

    def stream(self, prompt, max_new_tokens, temperature, seed, stop):
        for index in range(max_new_tokens):
            if stop.is_set():
                return
            time.sleep(0.05)
            yield f"word{index} ", {}


@pytest.fixture(scope="module")
def site():
    summary = json.loads((SHIPPED / "calibration.json").read_text(encoding="utf-8"))
    engine = SlowEngine()
    server = serve(App(engine, {name: object() for name in summary["axes"]}, summary, {"paired": PAIRED, "pairs": PAIRS}), 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield engine, f"http://127.0.0.1:{server.server_address[1]}/"
    server.shutdown()
    server.server_close()


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as playwright:
        try:
            chromium = playwright.chromium.launch()
        except Exception as error:
            if os.environ.get("KEMUDI_REQUIRE_BROWSER"):
                raise
            pytest.skip(f"no browser to run the page in: {str(error)[:80]}")
        yield chromium
        chromium.close()


@pytest.fixture()
def page(site, browser):
    tab = browser.new_page(viewport={"width": 1280, "height": 900})
    tab.errors = []
    tab.on("console", lambda message: tab.errors.append(message.text) if message.type == "error" else None)
    tab.on("pageerror", lambda error: tab.errors.append(str(error)))
    tab.goto(site[1])
    tab.wait_for_selector(".axis")
    yield tab
    assert tab.errors == []
    tab.close()


def turn(page, name, dose):
    page.locator(f'input[type=range][aria-label^="{name}:"]').evaluate(
        "(slider, dose) => { slider.value = dose; slider.dispatchEvent(new Event('input')) }", dose
    )


def note(page, name):
    return page.locator(".axis", has=page.locator("h3", has_text=name)).locator("[data-note]").inner_text()


def test_the_page_shows_one_card_per_knob_and_fits_a_phone(page):
    assert page.locator(".axis h3").all_inner_texts() == ["mood", "tension", "register", "speech"]
    assert "Qwen2.5-0.5B-Instruct" in page.locator("#model").inner_text()
    page.set_viewport_size({"width": 390, "height": 844})
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")


def test_the_note_under_a_knob_gives_the_measured_change_or_says_there_is_none(page):
    turn(page, "mood", 0.4)
    assert note(page, "mood") == "Measured at dose +0.45: change +0.15, 95% interval +0.07 to +0.22 (judge scale from 0 to 1)."
    turn(page, "mood", -0.1)
    assert note(page, "mood").startswith("Measured at dose -0.30: change -0.19")
    turn(page, "speech", -0.3)
    assert note(page, "speech") == "Measured at dose -0.45: no clear effect (change +0.00, 95% interval +0.00 to +0.00)."
    turn(page, "mood", 0.58)
    assert note(page, "mood") == "Past the measured fluent range: expect broken or repeating text."
    turn(page, "mood", 0)
    assert note(page, "mood") == ""


def test_two_knobs_show_what_was_measured_together_and_three_say_not_measured(page):
    combo = page.locator("#combo")
    turn(page, "mood", 0.3)
    assert combo.inner_text() == ""
    turn(page, "tension", 0.1)
    assert "mood + tension were measured together" in combo.inner_text()
    assert "stayed fluent (perplexity 1.20x of dose 0)" in combo.inner_text()
    assert "mood still works, tension still works" in combo.inner_text()
    assert "warn" not in combo.get_attribute("class")
    turn(page, "tension", 0)
    turn(page, "speech", 0.1)
    assert "did not stay fluent (perplexity 1.90x of dose 0)" in combo.inner_text()
    assert "speech has no clear effect" in combo.inner_text() and "warn" in combo.get_attribute("class")
    turn(page, "speech", -0.1)
    assert "stayed fluent" in combo.inner_text(), "the corner follows the signs of the two knobs"
    turn(page, "tension", 0.1)
    assert combo.inner_text() == "3 knobs are on. Knobs were measured alone and in pairs; three or more together were not measured."
    page.click("#reset")
    assert combo.inner_text() == ""


def test_writing_streams_text_and_a_knob_moved_mid_stream_reaches_the_engine(page, site):
    engine = site[0]
    page.fill("#tokens", "200")
    page.click("#write")
    page.wait_for_function("document.getElementById('pages').innerText.includes('word5')")
    turn(page, "mood", 0.3)
    page.wait_for_function("document.getElementById('pages').innerText.includes('word12')")
    assert engine.knob_calls[-1]["mood"] == 0.3
    page.click("#stop")
    page.wait_for_function("document.getElementById('stop').disabled")
    written = page.locator("#pages").inner_text()
    time.sleep(0.4)
    assert page.locator("#pages").inner_text() == written, "Stop ends the writing"
