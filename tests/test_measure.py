import math

import pytest
import torch

from kemudi import measure
from kemudi.measure import control_text, judge_probabilities, summarise, validate_judge

class JudgeTokenizer:
    def apply_chat_template(self, messages, **_ignored):
        return messages[0]["content"]


class JudgeEngine:
    """Finds a text one nat per token likelier under the instruction that matches it.

    Texts with DANGER match `tense`, the others match `calm`. `lean` makes every text look
    more `tense` by that many nats, the way a real model prefers one of the two words.
    """

    tokenizer = JudgeTokenizer()

    def __init__(self, lean=0.0):
        self.lean = lean
        self.prompts = []

    def continuation_nll(self, prompt, text):
        self.prompts.append(prompt)
        if not text:
            return float("nan")
        told_tense = prompt.endswith("is tense.")
        fits = told_tense == ("DANGER" in text)
        return 3.0 - (1.0 if fits else 0.0) - (self.lean if told_tense else 0.0)


def test_judge_scores_a_text_by_which_instruction_explains_it_better():
    engine = JudgeEngine()
    assert measure.judge_scores(engine, ["a quiet pond", "DANGER ahead"], "calm", "tense") == [-1.0, 1.0]
    assert engine.prompts[0] == "You are a novelist. Everything you write is calm."
    assert engine.prompts[1] == "You are a novelist. Everything you write is tense."


def test_judge_scale_puts_typical_labelled_sentences_at_27_and_73_percent():
    calm, tense = judge_probabilities(JudgeEngine(), ["a quiet pond", "DANGER ahead"], "calm", "tense", centre=0.0, scale=1.0)
    assert calm == pytest.approx(1 / (1 + math.e)) and tense == pytest.approx(1 / (1 + 1 / math.e))
    wide = judge_probabilities(JudgeEngine(), ["DANGER ahead"], "calm", "tense", centre=0.0, scale=4.0)
    assert 0.5 < wide[0] < tense


def test_an_empty_text_counts_as_the_midpoint():
    assert judge_probabilities(JudgeEngine(), [""], "calm", "tense", centre=0.3, scale=0.1) == [0.5]


def test_validate_judge_reports_auc_zero_point_unit_and_accuracy():
    axis = {"metric": "judge", "low": "calm", "high": "tense",
            "low_examples": ["pond", "tea", "DANGER nap"], "high_examples": ["DANGER one", "DANGER two", "DANGER three"]}
    # even-numbered: calm scores -1 and +1 (mean 0), tense scores +1 and +1 (mean 1)
    assert validate_judge(JudgeEngine(), axis) == {"auc": 0.833, "centre": 0.5, "scale": 0.5, "held_out_accuracy": 1.0}
    backwards = dict(axis, low_examples=axis["high_examples"], high_examples=axis["low_examples"])
    assert validate_judge(JudgeEngine(), backwards)["auc"] < 0.5


def test_centre_removes_a_judge_that_leans_toward_one_word():
    """A judge that scores everything 3 nats too `tense` still splits the texts once it is centred."""
    leaning = JudgeEngine(lean=3.0)
    assert min(judge_probabilities(leaning, ["a quiet pond", "DANGER ahead"], "calm", "tense")) > 0.5
    axis = {"low": "calm", "high": "tense", "low_examples": ["pond", "tea"], "high_examples": ["DANGER one", "DANGER two"]}
    judge = validate_judge(leaning, axis)
    assert judge["centre"] == 3.0 and judge["held_out_accuracy"] == 1.0
    calm, tense = judge_probabilities(leaning, ["a quiet pond", "DANGER ahead"], "calm", "tense", **measure.judge_units(judge))
    assert calm == pytest.approx(1 - tense) and calm < 0.5 < tense


def test_control_text_names_axis_strength_and_pole():
    axis = {"low": "bleak", "high": "hopeful"}
    assert control_text("mood", axis, -3).endswith("CONTROLS\nmood: strongly toward bleak")
    assert control_text("mood", axis, 1).endswith("CONTROLS\nmood: slightly toward hopeful")


def test_summarise_compares_with_the_baseline_and_skips_empty_texts():
    baseline = [{"effect": 0.2, "nll": 2.0, "distinct_2": 1.0}, {"effect": 0.4, "nll": 2.0, "distinct_2": 0.8}]
    rows = [{"effect": 0.6, "nll": 2.0 + math.log(2), "distinct_2": 0.45}, {"effect": 0.8, "nll": float("nan"), "distinct_2": 0.45}]
    result = summarise(rows, baseline)
    assert result["effect"] == 0.7 and result["samples"] == 2
    assert result["perplexity_ratio"] == 2.0
    assert result["distinct_ratio"] == 0.5
    assert result["effect_ci"][0] <= 0.7 <= result["effect_ci"][1]
    assert summarise(baseline, baseline)["perplexity_ratio"] == 1.0


def test_settings_keep_zero_in_the_doses_and_not_in_the_prompt_levels():
    assert 0.0 in measure.SETTINGS["doses"] and 0 not in measure.SETTINGS["prompt_levels"]
    assert measure.SETTINGS["doses"] == sorted(measure.SETTINGS["doses"])


class DoseEngine:
    """Quotes more words the higher the knob on the real direction, and babbles past a dose of 0.5."""

    WORDS = "one two three four five six seven eight nine ten".split()

    def __init__(self, real_unit):
        self.real_unit = real_unit
        self.dose = 0.0
        self.on_real_direction = False
        self.controls_seen = []

    def set_knobs(self, axes, values):
        self.dose = sum(values.values())
        self.on_real_direction = any(torch.equal(axes[name].unit, self.real_unit) for name in values)

    def story_prompt(self, story, controls=""):
        if controls:
            self.controls_seen.append(controls)
        return f"{story}|{controls}"

    def generate_batch(self, prompts, *_sampling):
        if abs(self.dose) > 0.5:
            return ["la la la la la la la la la la"] * len(prompts)
        quoted = round(self.dose * 10) if self.on_real_direction and self.dose > 0 else 0
        if prompts[0].endswith("strongly toward dialogue"):
            quoted = 3
        text = '"' + " ".join(self.WORDS[:quoted]) + '" ' + " ".join(self.WORDS[quoted:]) if quoted else " ".join(self.WORDS)
        return [text] * len(prompts)

    def continuation_nll(self, _prompt, text):
        return 4.0 if text.startswith("la") else 2.0

    def unsteered(self):
        engine = self

        class Clean:
            def __enter__(self):
                self.saved = engine.dose
                engine.dose = 0.0

            def __exit__(self, *_error):
                engine.dose = self.saved

        return Clean()


def test_measure_finds_the_usable_range_and_runs_both_controls():
    from kemudi.engine import Axis

    unit = torch.zeros(8)
    unit[0] = 1.0
    axes = {"speech": Axis("speech", "narration", "dialogue", layer=1, unit=unit, scale=3.0)}
    data = {
        "evaluation_openers": ["Story one", "Story two"],
        "axes": {"speech": {"metric": "quoted_share", "low": "narration", "high": "dialogue"}},
    }
    summary = {"axes": {"speech": {}}}
    engine = DoseEngine(unit)
    samples = measure.measure(engine, summary, axes, data, log=lambda _line: None)
    entry = summary["axes"]["speech"]

    assert entry["usable"] == [-0.45, 0.45]
    by_dose = {point["dose"]: point for point in entry["steering"]}
    assert by_dose[0.0]["effect"] == 0 and by_dose[0.15]["effect"] < by_dose[0.45]["effect"]
    assert by_dose[0.6]["perplexity_ratio"] == pytest.approx(math.exp(2.0), abs=1e-3)
    assert by_dose[0.6]["distinct_ratio"] < 0.85
    assert entry["spearman_in_usable"] > 0.5

    assert [point["dose"] for point in entry["random_direction"]] == [-0.45, 0.45]
    assert all(point["effect"] == 0 for point in entry["random_direction"])

    assert [point["level"] for point in entry["prompt"]] == measure.SETTINGS["prompt_levels"]
    assert entry["prompt"][-1]["effect"] > 0 and entry["prompt"][0]["effect"] == 0
    assert len(engine.controls_seen) == 2 * len(measure.SETTINGS["prompt_levels"])
    assert engine.dose == 0.0, "the knobs must be back at zero when the measurement ends"
    assert set(samples["speech"]) == {"steering", "prompt", "random_direction"}
    assert len(samples["speech"]["steering"]) == 9 and len(samples["speech"]["steering"]["0.45"]) == 2
    assert set(samples["speech"]["random_direction"]) == {"-0.45", "0.45"}
    assert "judge" not in entry and entry["measurable"] is True


class MoodEngine(DoseEngine):
    """The same fake, with a judge: a text reads as hopeful in proportion to its quoted words."""

    tokenizer = JudgeTokenizer()

    def continuation_nll(self, prompt, text):
        if prompt.startswith("You are a novelist"):
            hopeful = text.count('"') / 2 + (1.0 if "sunny" in text else 0.0)
            return 3.0 - hopeful if prompt.endswith("is hopeful.") else 3.0
        return super().continuation_nll(prompt, text)


def test_measure_validates_the_judge_before_it_trusts_it():
    from kemudi.engine import Axis

    unit = torch.zeros(8)
    unit[0] = 1.0
    axes = {"mood": Axis("mood", "bleak", "hopeful", layer=1, unit=unit, scale=3.0)}
    axis = {"metric": "judge", "low": "bleak", "high": "hopeful",
            "low_examples": ["rain", "mud", "fog", "ash"], "high_examples": ["sunny day", "sunny hill", "sunny sea", "sunny room"]}
    data = {"evaluation_openers": ["Story one", "Story two"], "axes": {"mood": axis}}
    summary = {"axes": {"mood": {}}}
    measure.measure(MoodEngine(unit), summary, axes, data, log=lambda _line: None)
    entry = summary["axes"]["mood"]
    assert entry["judge"] == {"auc": 1.0, "centre": 0.5, "scale": 0.5, "held_out_accuracy": 1.0}
    assert entry["measurable"] is True
    by_dose = {point["dose"]: point["effect"] for point in entry["steering"]}
    assert by_dose[0.0] < by_dose[0.45] and by_dose[0.0] < 0.5 < by_dose[0.45]

    useless = dict(axis, high_examples=["rain again", "more mud", "thick fog", "cold ash"])
    summary = {"axes": {"mood": {}}}
    measure.measure(MoodEngine(unit), summary, axes, {**data, "axes": {"mood": useless}}, log=lambda _line: None)
    assert summary["axes"]["mood"]["measurable"] is False
