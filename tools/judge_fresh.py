"""Check the judge on sentences it has never seen.

The judge's zero point and unit were set on half of the labelled sentences in axes.json. Here the
same judge, with those numbers frozen, scores tools/fresh_sentences.json: 48 new sentences per
axis, written by three other language models after the judge was fixed.
Rule, written before the run: the judge passes on an axis when its AUC on the fresh sentences is
at least 0.90, the limit the main measurement already uses. Accuracy is reported, not gated.
Needs the model. Run from the repository root: python tools/judge_fresh.py
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
from kemudi.measure import SETTINGS, judge_scores  # noqa: E402
from kemudi.stats import auc  # noqa: E402


def check(engine, summary: dict, axes_data: dict, fresh: dict) -> list[dict]:
    rows = []
    for name, poles in fresh.items():
        judge = summary["axes"][name]["judge"]
        words = axes_data[name]["low"], axes_data[name]["high"]
        low = judge_scores(engine, [row["text"] for row in poles["low"]], *words)
        high = judge_scores(engine, [row["text"] for row in poles["high"]], *words)
        correct = sum(score < judge["centre"] for score in low) + sum(score > judge["centre"] for score in high)
        fresh_auc = round(auc(low, high), 3)
        rows.append(
            {
                "axis": name,
                "sentences": len(low) + len(high),
                "auc": fresh_auc,
                "accuracy": round(correct / (len(low) + len(high)), 3),
                "labelled_auc": judge["auc"],
                "labelled_accuracy": judge["held_out_accuracy"],
                "passes": bool(fresh_auc >= SETTINGS["min_judge_auc"]),
            }
        )
    return rows


def table(rows: list[dict]) -> str:
    lines = [
        "| Axis | Fresh sentences | AUC on fresh sentences | Accuracy with the frozen zero point | AUC on the first labelled set | Passes (AUC at least 0.90) |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['axis']} | {row['sentences']} | {row['auc']:.3f} | {row['accuracy']:.3f} "
            f"| {row['labelled_auc']:.3f} | {'yes' if row['passes'] else 'no'} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    model_folder, model_name = resolve_model(DEFAULT_MODEL)
    summary = json.loads((calibration.calibration_dir(model_name) / "calibration.json").read_text(encoding="utf-8"))
    fresh = json.loads((ROOT / "tools" / "fresh_sentences.json").read_text(encoding="utf-8"))["axes"]
    rows = check(Engine.load(model_folder), summary, calibration.load_axes_data()["axes"], fresh)
    (ROOT / "docs" / "judge_fresh_table.md").write_text(table(rows) + "\n", encoding="utf-8", newline="\n")
    print(table(rows))
