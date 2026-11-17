"""Compare a fresh `kemudi reproduce` run with the numbers shipped in this repository.

Usage, from the repository root: python tools/compare_reproduce.py FOLDER
FOLDER is the --calibration folder of the fresh run. Prints a Markdown report: how close the
vectors are, how many of the generated texts are the same, and how far every published number
moved. Exit code 0 always: this is a report, not a gate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SHIPPED = ROOT / "kemudi" / "calibrations" / "qwen2.5-0.5b-instruct"


def numeric_leaves(value, path: str = "") -> dict[str, float]:
    """Every number in a nested JSON value, keyed by its path."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return {}
    if isinstance(value, (int, float)):
        return {path: float(value)}
    items = value.items() if isinstance(value, dict) else enumerate(value)
    leaves: dict[str, float] = {}
    for key, item in items:
        leaves.update(numeric_leaves(item, f"{path}/{key}"))
    return leaves


def compare_vectors(shipped, fresh) -> list[str]:
    lines = ["| Vector | Cosine with the shipped one | Largest difference in one number |", "|---|---|---|"]
    for name in shipped.files:
        a, b = shipped[name].astype(np.float64), fresh[name].astype(np.float64)
        cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        lines.append(f"| {name} | {cosine:.6f} | {np.abs(a - b).max():.2e} |")
    return lines


def compare_texts(shipped: dict, fresh: dict) -> list[str]:
    lines = ["| Axis | Texts | Same as shipped, to the letter |", "|---|---|---|"]
    for axis, groups in shipped.items():
        pairs = [
            (text, other)
            for kind, by_dose in groups.items()
            for dose, texts in by_dose.items()
            for text, other in zip(texts, fresh.get(axis, {}).get(kind, {}).get(dose, [None] * len(texts)))
        ]
        lines.append(f"| {axis} | {len(pairs)} | {sum(text == other for text, other in pairs)} |")
    return lines


def compare_numbers(shipped: dict, fresh: dict) -> list[str]:
    old, new = numeric_leaves(shipped), numeric_leaves(fresh)
    shared = sorted(set(old) & set(new))
    moved = sorted(((abs(old[path] - new[path]), path) for path in shared if old[path] != new[path]), reverse=True)
    lines = [
        f"{len(shared)} numbers compared, {len(shared) - len(moved)} equal, {len(moved)} different, "
        f"{len(set(old) ^ set(new))} present on one side only.",
        "",
    ]
    for axis, entry in shipped["axes"].items():
        other = fresh["axes"][axis]
        effects = [abs(a["effect"] - b["effect"]) for a, b in zip(entry["steering"], other["steering"])]
        lines.append(
            f"- {axis}: usable range {entry['usable']} shipped, {other['usable']} fresh; "
            f"held-out AUC {entry['test_auc']} and {other['test_auc']}; "
            f"largest move of a dose effect {max(effects):.4f}"
        )
    if moved:
        lines += ["", "| Largest differences | Shipped | Fresh |", "|---|---|---|"]
        lines += [f"| `{path}` | {old[path]:g} | {new[path]:g} |" for _gap, path in moved[:12]]
    return lines


def report(fresh_folder: Path) -> str:
    load = lambda folder, name: json.loads((folder / name).read_text(encoding="utf-8"))
    parts = [
        "#### Vectors",
        *compare_vectors(np.load(SHIPPED / "vectors.npz"), np.load(fresh_folder / "vectors.npz")),
        "",
        "#### Generated texts",
        *compare_texts(load(SHIPPED, "samples.json"), load(fresh_folder, "samples.json")),
        "",
        "#### Published numbers",
        *compare_numbers(load(SHIPPED, "calibration.json"), load(fresh_folder, "calibration.json")),
    ]
    return "\n".join(parts)


if __name__ == "__main__":
    print(report(Path(sys.argv[1])))
