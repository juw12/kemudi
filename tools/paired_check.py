"""Paired check, added after the measurement: per opening, how far does each usable end move the effect from dose 0?

The stored texts are scored again with the same judge and the same units. Every dose used the
same 16 openings and the same seed, so the 16 differences to dose 0 are paired; their mean gets a
95% bootstrap interval. Needs the model. Run from the repository root: python tools/paired_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kemudi import calibration  # noqa: E402
from kemudi.cli import resolve_model  # noqa: E402
from kemudi.engine import Engine  # noqa: E402
from kemudi.fetch import DEFAULT_MODEL  # noqa: E402
from kemudi.measure import effects_of, judge_units  # noqa: E402
from kemudi.stats import bootstrap_ci, mean  # noqa: E402


def paired_table(engine, summary: dict, samples: dict, axes_data: dict) -> str:
    lines = [
        "| Axis | Dose | Mean change from dose 0 | 95% interval of the change | Openings that moved the intended way |",
        "|---|---|---|---|---|",
    ]
    for name, entry in summary["axes"].items():
        texts = samples[name]["steering"]
        units = judge_units(entry.get("judge"))

        def score(dose: float) -> list[float]:
            effects = effects_of(engine, axes_data[name], texts[str(dose)], **units)
            stored = next(point["effect"] for point in entry["steering"] if point["dose"] == dose)
            assert abs(mean(effects) - stored) < 0.001, f"{name} {dose}: rescored {mean(effects):.4f}, stored {stored}"
            return effects

        zero = score(0.0)
        for dose in entry["usable"]:
            if dose == 0:
                continue
            changes = [value - base for value, base in zip(score(dose), zero)]
            low, high = bootstrap_ci(changes)
            intended = sum(change * dose > 0 for change in changes)
            lines.append(f"| {name} | {dose:+.2f} | {mean(changes):+.3f} | {low:+.3f} to {high:+.3f} | {intended} of {len(changes)} |")
    return "\n".join(lines)


if __name__ == "__main__":
    model_folder, model_name = resolve_model(DEFAULT_MODEL)
    folder = calibration.calibration_dir(model_name)
    summary = json.loads((folder / "calibration.json").read_text(encoding="utf-8"))
    samples = json.loads((folder / "samples.json").read_text(encoding="utf-8"))
    table = paired_table(Engine.load(model_folder), summary, samples, calibration.load_axes_data()["axes"])
    (ROOT / "docs" / "paired_table.md").write_text(table + "\n", encoding="utf-8", newline="\n")
    print(table)
