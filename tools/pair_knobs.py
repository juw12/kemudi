"""Two knobs together: does the text stay fluent, and does each knob still do its job?

Six pairs of knobs. For each pair, four corners: every combination of the two ends of each
knob's own usable range. The 16 openings, the seed and the sampling settings are those of the
main measurement, so every text is paired with the stored one-knob and dose-0 texts.

Rules, written before the first corner was generated:
- A corner is fluent when its perplexity ratio is at most 1.5 and its distinct-2 ratio at least
  0.85, both against dose 0. These are the limits of the main measurement.
- A knob still works next to another knob when, at each end of the other knob, the paired
  difference between its own high end and its own low end has a 95% interval above zero.
  The same difference with the other knob at zero is shown next to it.

Needs the model. Run from the repository root: python tools/pair_knobs.py
Texts are saved after every corner, so a stopped run continues where it was.
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kemudi import calibration  # noqa: E402
from kemudi.cli import resolve_model  # noqa: E402
from kemudi.engine import Engine  # noqa: E402
from kemudi.fetch import DEFAULT_MODEL  # noqa: E402
from kemudi.measure import effects_of, judge_units  # noqa: E402
from kemudi.stats import bootstrap_ci, distinct_2, mean  # noqa: E402


def corner_key(first: str, first_dose: float, second: str, second_dose: float) -> str:
    return f"{first}={first_dose:+.2f} {second}={second_dose:+.2f}"


def generate_corners(engine, summary: dict, axes: dict, stories: list[str], texts_path: Path, log=print) -> dict:
    """Write the 16 continuations of every corner; corners already in the file are kept."""
    texts = json.loads(texts_path.read_text(encoding="utf-8")) if texts_path.is_file() else {}
    settings = summary["settings"]
    prompts = [engine.story_prompt(story) for story in stories]
    for first, second in itertools.combinations(summary["axes"], 2):
        for first_dose, second_dose in itertools.product(summary["axes"][first]["usable"], summary["axes"][second]["usable"]):
            key = corner_key(first, first_dose, second, second_dose)
            if key in texts:
                continue
            engine.set_knobs(axes, {first: first_dose, second: second_dose})
            texts[key] = engine.generate_batch(
                prompts, settings["max_new_tokens"], settings["temperature"], settings["top_p"], settings["seed"]
            )
            engine.set_knobs(axes, {})
            texts_path.write_text(json.dumps(texts, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
            log(f"generated {key}")
    return texts


def contrast(high: list[float], low: list[float]) -> dict:
    changes = [a - b for a, b in zip(high, low)]
    lower, upper = bootstrap_ci(changes)
    return {"difference": round(mean(changes), 4), "interval": [round(lower, 4), round(upper, 4)], "above_zero": bool(lower > 0)}


def analyse(engine, summary: dict, samples: dict, axes_data: dict, stories: list[str], texts: dict) -> dict:
    """Score every corner without steering and apply the two rules."""
    settings = summary["settings"]
    prompts = [engine.story_prompt(story) for story in stories]
    engine.set_knobs({}, {})

    def fluency(batch: list[str]) -> tuple[float, float]:
        nll = [engine.continuation_nll(prompt, text) for prompt, text in zip(prompts, batch)]
        return mean([value for value in nll if not math.isnan(value)]), mean([distinct_2(text) for text in batch])

    def effects(name: str, batch: list[str]) -> list[float]:
        return effects_of(engine, axes_data[name], batch, **judge_units(summary["axes"][name].get("judge")))

    base_nll, base_distinct = fluency(samples[next(iter(samples))]["steering"]["0.0"])
    pairs = []
    for first, second in itertools.combinations(summary["axes"], 2):
        ends = {name: summary["axes"][name]["usable"] for name in (first, second)}
        corners, scored = [], {}
        for first_dose, second_dose in itertools.product(ends[first], ends[second]):
            batch = texts[corner_key(first, first_dose, second, second_dose)]
            nll, distinct = fluency(batch)
            ratio, distinct_ratio = math.exp(nll - base_nll), distinct / base_distinct
            scored[first_dose, second_dose] = {first: effects(first, batch), second: effects(second, batch)}
            corners.append(
                {
                    "doses": {first: first_dose, second: second_dose},
                    "perplexity_ratio": round(ratio, 3),
                    "distinct_ratio": round(distinct_ratio, 3),
                    "fluent": bool(ratio <= settings["max_perplexity_ratio"] and distinct_ratio >= settings["min_distinct_ratio"]),
                    "effect": {name: round(mean(values), 4) for name, values in scored[first_dose, second_dose].items()},
                }
            )

        def still_works(name: str, other: str) -> dict:
            low_end, high_end = ends[name]
            order = (lambda own, far: (own, far)) if name == first else (lambda own, far: (far, own))
            alone = samples[name]["steering"]
            next_to = [
                {"other_dose": far, **contrast(scored[order(high_end, far)][name], scored[order(low_end, far)][name])}
                for far in ends[other]
            ]
            return {
                "knob": name,
                "next_to": other,
                "alone": contrast(effects(name, alone[str(high_end)]), effects(name, alone[str(low_end)])),
                "with_other": next_to,
                "still_works": all(row["above_zero"] for row in next_to),
            }

        pairs.append(
            {
                "knobs": [first, second],
                "corners": corners,
                "fluent_together": all(corner["fluent"] for corner in corners),
                "effects": [still_works(first, second), still_works(second, first)],
            }
        )
    return {"openings": len(stories), "seed": settings["seed"], "pairs": pairs}


def tables(result: dict) -> str:
    lines = [
        "| Knobs | Fluent in all four corners | Highest perplexity ratio | Lowest distinct-2 ratio |",
        "|---|---|---|---|",
    ]
    for pair in result["pairs"]:
        lines.append(
            f"| {pair['knobs'][0]} + {pair['knobs'][1]} | {'yes' if pair['fluent_together'] else 'no'} "
            f"| {max(corner['perplexity_ratio'] for corner in pair['corners']):.2f} "
            f"| {min(corner['distinct_ratio'] for corner in pair['corners']):.2f} |"
        )
    show = lambda row: f"{row['difference']:+.3f} ({row['interval'][0]:+.3f} to {row['interval'][1]:+.3f})"
    lines += [
        "",
        "| Knob | Next to | High end minus low end, alone | Other knob at its low end | Other knob at its high end | Still works |",
        "|---|---|---|---|---|---|",
    ]
    for pair in result["pairs"]:
        for row in pair["effects"]:
            lines.append(
                f"| {row['knob']} | {row['next_to']} | {show(row['alone'])} | {show(row['with_other'][0])} "
                f"| {show(row['with_other'][1])} | {'yes' if row['still_works'] else 'no'} |"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    model_folder, model_name = resolve_model(DEFAULT_MODEL)
    folder = calibration.calibration_dir(model_name)
    summary, axes = calibration.load(folder)
    samples = json.loads((folder / "samples.json").read_text(encoding="utf-8"))
    data = calibration.load_axes_data()
    engine = Engine.load(model_folder)
    texts = generate_corners(engine, summary, axes, data["evaluation_openers"], folder / "pairs_samples.json")
    result = analyse(engine, summary, samples, data["axes"], data["evaluation_openers"], texts)
    (folder / "pairs.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    table = tables(result)
    (ROOT / "docs" / "pairs_table.md").write_text(table + "\n", encoding="utf-8", newline="\n")
    print(table)
