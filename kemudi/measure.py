"""Measure what each knob does: effect on the text, and what it costs in fluency.

For every dose the same story openings are continued with the same random seed, so the only
thing that differs between two doses is the vector. Three numbers come out per dose:

- effect: how far the text moved toward the axis's high pole. For `speech` this is the share of
  characters inside quotation marks, counted directly. For the other axes the unsteered model
  is the judge: it says how much better the text fits a writer told to be the high pole than
  one told to be the low pole. That judge is first tested on labelled sentences: half of them
  set its zero point and unit, all of them give its AUC, and an axis whose judge fails the test
  is reported as not validated.
- perplexity ratio: how much less likely the unsteered model finds the steered text, against dose 0.
- distinct-2 ratio: whether the text started to repeat itself, against dose 0.

Two controls are measured the same way. Prompt control: the style is requested in the system
prompt and no vector is added. Random direction: a random vector of the same size is added at
the same block, to show how much of the effect comes from the direction and not from the push.
"""

from __future__ import annotations

import dataclasses
import math

import torch

from .engine import Axis, Engine
from .stats import auc, bootstrap_ci, distinct_2, mean, quoted_share, spearman, usable_range

SETTINGS = {
    "doses": [-0.6, -0.45, -0.3, -0.15, 0.0, 0.15, 0.3, 0.45, 0.6],
    "prompt_levels": [-3, -2, -1, 1, 2, 3],
    "max_new_tokens": 48,
    "temperature": 0.8,
    "top_p": 0.95,
    "seed": 1234,
    "max_perplexity_ratio": 1.5,
    "min_distinct_ratio": 0.85,
    "min_judge_auc": 0.9,
}
STRENGTHS = {1: "slightly", 2: "moderately", 3: "strongly"}


JUDGE_SYSTEM = "You are a novelist. Everything you write is {pole}."
JUDGE_REQUEST = "Write a short passage."


def judge_prompt(engine: Engine, pole: str) -> str:
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM.format(pole=pole)},
        {"role": "user", "content": JUDGE_REQUEST},
    ]
    return engine.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def judge_scores(engine: Engine, texts: list[str], low: str, high: str) -> list[float]:
    """How much better each text fits a writer told to be `high` than one told to be `low`.

    The score is a likelihood ratio in nats per token: the text's mean negative log-likelihood
    after the `low` instruction minus the same after the `high` instruction. Both instructions
    are the same sentence with one word swapped, so everything else about the text cancels out.
    """
    low_prompt, high_prompt = judge_prompt(engine, low), judge_prompt(engine, high)
    return [engine.continuation_nll(low_prompt, text) - engine.continuation_nll(high_prompt, text) for text in texts]


def judge_probabilities(
    engine: Engine, texts: list[str], low: str, high: str, centre: float = 0.0, scale: float = 1.0
) -> list[float]:
    """Each text on a 0 to 1 scale from `low` to `high`.

    `centre` is the judge's zero point and `scale` its unit, both set on labelled sentences, so
    0.27 is a typical labelled `low` sentence, 0.5 is the midpoint and 0.73 a typical `high` one.
    An empty text has no likelihood and counts as the midpoint.
    """
    scores = judge_scores(engine, texts, low, high)
    return [0.5 if math.isnan(score) else 1 / (1 + math.exp((centre - score) / scale)) for score in scores]


def effects_of(engine: Engine, axis_data: dict, texts: list[str], centre: float = 0.0, scale: float = 1.0) -> list[float]:
    if axis_data["metric"] == "quoted_share":
        return [quoted_share(text) for text in texts]
    return judge_probabilities(engine, texts, axis_data["low"], axis_data["high"], centre, scale)


def validate_judge(engine: Engine, axis_data: dict) -> dict:
    """Test the judge on the labelled sentences and find its zero point and unit.

    The AUC uses every sentence and depends on neither. The zero point is the midpoint between
    the two classes on the even-numbered sentences and the unit is half the gap between them;
    the accuracy is counted on the odd-numbered ones, which had no part in setting either.
    """
    low = judge_scores(engine, axis_data["low_examples"], axis_data["low"], axis_data["high"])
    high = judge_scores(engine, axis_data["high_examples"], axis_data["low"], axis_data["high"])
    centre = (mean(low[0::2]) + mean(high[0::2])) / 2
    half_gap = (mean(high[0::2]) - mean(low[0::2])) / 2
    held_low, held_high = low[1::2], high[1::2]
    correct = sum(value < centre for value in held_low) + sum(value > centre for value in held_high)
    return {
        "auc": round(auc(low, high), 3),
        "centre": round(centre, 4),
        "scale": round(half_gap, 4) if half_gap > 0 else 1.0,
        "held_out_accuracy": round(correct / (len(held_low) + len(held_high)), 3),
    }


def control_text(axis_name: str, axis_data: dict, level: int) -> str:
    """System-prompt control in the style of prompt-steered editors: a named axis with a strength."""
    pole = axis_data["high"] if level > 0 else axis_data["low"]
    return f"Obey this control from the first sentence.\nCONTROLS\n{axis_name}: {STRENGTHS[abs(level)]} toward {pole}"


def run_condition(engine: Engine, axis_data: dict, stories: list[str], controls: str = "", judge: dict | None = None) -> list[dict]:
    """Continue every story under the current knobs and score each continuation without steering."""
    prompts = [engine.story_prompt(story, controls) for story in stories]
    texts = engine.generate_batch(
        prompts, SETTINGS["max_new_tokens"], SETTINGS["temperature"], SETTINGS["top_p"], SETTINGS["seed"]
    )
    plain_prompts = [engine.story_prompt(story) for story in stories]
    with engine.unsteered():
        effects = effects_of(engine, axis_data, texts, **judge_units(judge))
        return [
            {"text": text, "effect": effect, "nll": engine.continuation_nll(prompt, text), "distinct_2": distinct_2(text)}
            for prompt, text, effect in zip(plain_prompts, texts, effects)
        ]


def judge_units(judge: dict | None) -> dict:
    return {"centre": judge["centre"], "scale": judge["scale"]} if judge else {}


def random_direction(axis: Axis, seed: int) -> Axis:
    """The same knob pointing somewhere else: a random unit vector at right angles to the real one."""
    noise = torch.randn(axis.unit.shape, generator=torch.Generator().manual_seed(seed))
    noise = noise - (noise @ axis.unit) * axis.unit
    return dataclasses.replace(axis, unit=noise / noise.norm())


def summarise(rows: list[dict], baseline: list[dict]) -> dict:
    effects = [row["effect"] for row in rows]
    low, high = bootstrap_ci(effects)
    nll = mean([row["nll"] for row in rows if not math.isnan(row["nll"])])
    baseline_nll = mean([row["nll"] for row in baseline if not math.isnan(row["nll"])])
    return {
        "effect": round(mean(effects), 4),
        "effect_ci": [round(low, 4), round(high, 4)],
        "perplexity_ratio": round(math.exp(nll - baseline_nll), 3),
        "distinct_ratio": round(mean([row["distinct_2"] for row in rows]) / mean([row["distinct_2"] for row in baseline]), 3),
        "samples": len(rows),
    }


def measure(engine: Engine, summary: dict, axes: dict[str, Axis], data: dict, log=print, only: list[str] | None = None) -> dict:
    """Fill `summary` with the dose-response of every axis and return the sample texts."""
    stories = data["evaluation_openers"]
    summary["settings"] = SETTINGS
    samples: dict[str, dict] = {}

    for name in only or list(axes):
        axis_data = data["axes"][name]
        entry = summary["axes"][name]
        samples[name] = {"steering": {}, "prompt": {}, "random_direction": {}}

        if axis_data["metric"] == "judge":
            entry["judge"] = validate_judge(engine, axis_data)
            entry["measurable"] = entry["judge"]["auc"] >= SETTINGS["min_judge_auc"]
            log(f"{name}: judge {entry['judge']}")
        else:
            entry.pop("judge", None)
            entry["measurable"] = True
        judge = entry.get("judge")
        entry["metric"] = axis_data["metric"]

        engine.set_knobs(axes, {})
        baseline = run_condition(engine, axis_data, stories, judge=judge)

        steering, pairs = [], []
        for dose in SETTINGS["doses"]:
            engine.set_knobs(axes, {name: dose})
            rows = baseline if dose == 0 else run_condition(engine, axis_data, stories, judge=judge)
            steering.append({"dose": dose, **summarise(rows, baseline)})
            pairs.append((dose, rows))
            samples[name]["steering"][str(dose)] = [row["text"] for row in rows]
            log(f"{name} dose {dose:+.2f}: {steering[-1]}")
        engine.set_knobs(axes, {})

        prompt = []
        for level in SETTINGS["prompt_levels"]:
            rows = run_condition(engine, axis_data, stories, control_text(name, axis_data, level), judge)
            prompt.append({"level": level, **summarise(rows, baseline)})
            samples[name]["prompt"][str(level)] = [row["text"] for row in rows]
            log(f"{name} prompt level {level:+d}: {prompt[-1]}")

        doses = [point["dose"] for point in steering]
        low, high = usable_range(
            doses,
            [point["perplexity_ratio"] for point in steering],
            [point["distinct_ratio"] for point in steering],
            SETTINGS["max_perplexity_ratio"],
            SETTINGS["min_distinct_ratio"],
        )
        inside = [(dose, row["effect"]) for dose, rows in pairs if low <= dose <= high for row in rows]

        control = []
        placebo = {name: random_direction(axes[name], SETTINGS["seed"])}
        for dose in sorted({low, high} - {0.0}):
            engine.set_knobs(placebo, {name: dose})
            rows = run_condition(engine, axis_data, stories, judge=judge)
            control.append({"dose": dose, **summarise(rows, baseline)})
            samples[name]["random_direction"][str(dose)] = [row["text"] for row in rows]
            log(f"{name} random direction at dose {dose:+.2f}: {control[-1]}")
        engine.set_knobs(axes, {})
        entry["steering"] = steering
        entry["prompt"] = prompt
        entry["random_direction"] = control
        entry["usable"] = [low, high]
        entry["spearman_in_usable"] = round(spearman([d for d, _ in inside], [e for _, e in inside]), 3) if low < high else None
        log(f"{name}: usable {low:+.2f} to {high:+.2f}, rank correlation {entry['spearman_in_usable']}")
    return samples
