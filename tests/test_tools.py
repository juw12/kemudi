"""The follow-up tools: blind rating, reproduce comparison, paired check, two knobs together."""

import importlib.util
import json
import shutil
from pathlib import Path

from kemudi import calibration

ROOT = Path(__file__).resolve().parents[1]
SHIPPED = ROOT / "kemudi" / "calibrations" / "qwen2.5-0.5b-instruct"


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read(name):
    return json.loads((SHIPPED / name).read_text(encoding="utf-8"))


def shipped_pairs(tool):
    data = calibration.load_axes_data()
    return tool.make_pairs(read("calibration.json"), read("samples.json"), data["axes"], data["evaluation_openers"])


def test_blind_pairs_hide_the_dose_and_the_key_points_at_the_right_text():
    tool = load_tool("blind_rating")
    shown, key = shipped_pairs(tool)
    samples = read("samples.json")
    assert len(shown) == len(key) == 128
    assert all(set(pair) == {"id", "question", "opening", "A", "B"} for pair in shown)
    for pair in shown:
        entry = key[pair["id"]]
        steered = samples[entry["axis"]]["steering"][str(entry["dose"])]
        intended_text = pair[entry["intended"]]
        other_text = pair["B" if entry["intended"] == "A" else "A"]
        assert (intended_text in steered) if entry["dose"] > 0 else (other_text in steered)
    assert {entry["intended"] for entry in key.values()} == {"A", "B"}
    assert shipped_pairs(tool) == (shown, key)


def test_blind_score_counts_intended_picks_and_a_tie_is_a_miss():
    tool = load_tool("blind_rating")
    _shown, key = shipped_pairs(tool)
    right = {pair_id: entry["intended"] for pair_id, entry in key.items()}
    wrong = {pair_id: "B" if pick == "A" else "A" for pair_id, pick in right.items()}
    rows = tool.score(key, {"1": right, "2": right, "3": wrong})
    assert all(row["raters"] == {"1": 16, "2": 16, "3": 0} and row["majority"] == 16 and row["passes"] for row in rows)
    tied = tool.score(key, {"1": right, "2": wrong})
    assert all(row["majority"] == 0 and not row["passes"] for row in tied)
    assert len(rows) == 8 and "| mood | +0.45 | 16 of 16 | 16 of 16 | 0 of 16 | 16 of 16 | yes |" in tool.table(rows)


def test_reproduce_comparison_finds_nothing_in_a_copy_and_finds_a_changed_number(tmp_path):
    tool = load_tool("compare_reproduce")
    copy = tmp_path / "copy"
    shutil.copytree(SHIPPED, copy)
    assert "600 numbers compared, 600 equal, 0 different" in tool.report(copy)
    summary = read("calibration.json")
    summary["axes"]["mood"]["test_auc"] = 0.5
    (copy / "calibration.json").write_text(json.dumps(summary), encoding="utf-8")
    changed = tool.report(copy)
    assert "599 equal, 1 different" in changed and "`/axes/mood/test_auc` | 0.953 | 0.5 |" in changed
    assert tool.numeric_leaves({"a": [1, True, "x", None, {"b": 2.5}]}) == {"/a/0": 1.0, "/a/4/b": 2.5}


def test_a_dose_is_clear_only_when_the_interval_excludes_zero_and_points_the_intended_way():
    tool = load_tool("paired_check")
    assert tool.describe(0.3, [0.1, 0.2, 0.15, 0.12])["clear"] is True
    assert tool.describe(-0.3, [0.1, 0.2, 0.15, 0.12])["clear"] is False
    assert tool.describe(0.3, [0.4, -0.5, 0.3, -0.2])["clear"] is False
    row = tool.describe(-0.3, [-0.1, -0.2, 0.05, -0.12])
    assert row["intended"] == 3 and row["openings"] == 4 and row["change"] == -0.0925


class StubEngine:
    """Knob x puts the text in quotes. Both knobs up together breaks the text."""

    def __init__(self):
        self.knobs = {}
        self.batches = 0

    def story_prompt(self, story, controls=""):
        return story

    def set_knobs(self, axes, values):
        self.knobs = dict(values)

    def generate_batch(self, prompts, *_settings):
        self.batches += 1
        quoted = self.knobs.get("x", 0) > 0
        broken = quoted and self.knobs.get("y", 0) > 0
        body = "one two three four five six" + (" broken" if broken else "")
        return [f'"{body} {index}"' if quoted else f"{body} {index}" for index in range(len(prompts))]

    def continuation_nll(self, prompt, text):
        return 1.6 if "broken" in text else 1.0


def test_two_knobs_together_rules(tmp_path):
    tool = load_tool("pair_knobs")
    stories = [f"story {index}" for index in range(4)]
    plain = [f"one two three four five six {index}" for index in range(4)]
    quoted = [f'"{text}"' for text in plain]
    summary = {
        "settings": {"max_new_tokens": 8, "temperature": 0.8, "top_p": 0.95, "seed": 1, "max_perplexity_ratio": 1.5, "min_distinct_ratio": 0.85},
        "axes": {"x": {"usable": [-0.3, 0.3]}, "y": {"usable": [-0.15, 0.15]}},
    }
    samples = {
        "x": {"steering": {"0.0": plain, "-0.3": plain, "0.3": quoted}},
        "y": {"steering": {"0.0": plain, "-0.15": plain, "0.15": plain}},
    }
    axes_data = {"x": {"metric": "quoted_share"}, "y": {"metric": "quoted_share"}}
    engine = StubEngine()
    texts = tool.generate_corners(engine, summary, {}, stories, tmp_path / "texts.json", log=lambda _line: None)
    assert engine.batches == 4 and len(texts) == 4
    tool.generate_corners(engine, summary, {}, stories, tmp_path / "texts.json", log=lambda _line: None)
    assert engine.batches == 4, "corners already on disk are not generated again"

    pair = tool.analyse(engine, summary, samples, axes_data, stories, texts)["pairs"][0]
    fluent = {(corner["doses"]["x"], corner["doses"]["y"]): corner["fluent"] for corner in pair["corners"]}
    assert fluent == {(-0.3, -0.15): True, (-0.3, 0.15): True, (0.3, -0.15): True, (0.3, 0.15): False}
    assert pair["fluent_together"] is False
    works = {row["knob"]: row["still_works"] for row in pair["effects"]}
    assert works == {"x": True, "y": False}
    # The quoted share leaves out the two quotation marks: 29 of 31 characters.
    assert pair["effects"][0]["alone"]["difference"] == round(29 / 31, 4)
    assert "| x + y | no | 1.82 |" in tool.tables({"pairs": [pair]})


def test_the_tables_in_docs_are_what_the_shipped_files_give():
    docs = lambda name: (ROOT / "docs" / name).read_text(encoding="utf-8").strip()
    summary = read("calibration.json")
    rating = load_tool("blind_rating")
    answers = {
        path.stem.removeprefix("answers_"): json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((ROOT / "docs" / "rating").glob("answers_*.json"))
    }
    key = json.loads((ROOT / "docs" / "rating" / "key.json").read_text(encoding="utf-8"))
    assert shipped_pairs(rating)[1] == key, "the key in docs/rating is the one the shipped texts give"
    assert docs("rating_table.md") == rating.table(rating.score(key, answers))
    assert docs("paired_table.md") == load_tool("paired_check").paired_table(read("paired.json"), summary)
    assert docs("pairs_table.md") == load_tool("pair_knobs").tables(read("pairs.json"))
    assert docs("reproduce_table.md") == load_tool("compare_reproduce").report(ROOT / "docs" / "second_machine")


def test_shipped_follow_up_files_cover_every_knob_and_every_pair():
    extras = calibration.load_extras(SHIPPED)
    summary = read("calibration.json")
    assert set(extras["paired"]) == set(summary["axes"])
    assert all(len(rows) == len(summary["axes"][name]["steering"]) - 1 for name, rows in extras["paired"].items())
    assert len(extras["pairs"]["pairs"]) == 6
    assert all(len(pair["corners"]) == 4 and len(pair["effects"]) == 2 for pair in extras["pairs"]["pairs"])
