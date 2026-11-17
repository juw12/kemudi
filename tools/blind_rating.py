"""Blind rating: can a reader who is not told the dose tell which text was steered?

For every knob and each end of its usable range there are 16 pairs: the dose-0 text and the
steered text for the same opening and seed, in an order shuffled by a fixed seed. The rater sees
the opening, two texts and one question ("Which text is more hopeful?") and answers A or B.

  python tools/blind_rating.py make        writes docs/rating/pairs.json (what a rater sees) and key.json
  python tools/blind_rating.py ask NAME    a person answers in the terminal, saved to docs/rating/answers_NAME.json
  python tools/blind_rating.py score       prints the table from every docs/rating/answers_*.json

Rule, written before any rating: with the positive end the steered text is the intended pick, with
the negative end the dose-0 text is. A knob end passes when the majority of the raters picks the
intended text in at least 12 of 16 pairs (chance gives that about 4 times in 100). A tie counts
as a miss. No model is needed.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = ROOT / "kemudi" / "calibrations" / "qwen2.5-0.5b-instruct"
RATING = ROOT / "docs" / "rating"
QUESTIONS = {"speech": "Which text contains more spoken dialogue?"}
PASS_MARK = 12
SEED = 2026


def make_pairs(summary: dict, samples: dict, axes_data: dict, openings: list[str]) -> tuple[list[dict], dict]:
    """Return what the rater sees and the answer key, both in one shuffled order."""
    rng = random.Random(SEED)
    items = []
    for name, entry in summary["axes"].items():
        question = QUESTIONS.get(name, f"Which text is more {axes_data[name]['high']}?")
        for dose in entry["usable"]:
            for opening, plain, steered in zip(openings, samples[name]["steering"]["0.0"], samples[name]["steering"][str(dose)]):
                steered_is_a = rng.random() < 0.5
                intended_is_steered = dose > 0
                items.append(
                    {
                        "question": question,
                        "opening": opening,
                        "A": steered if steered_is_a else plain,
                        "B": plain if steered_is_a else steered,
                        "axis": name,
                        "dose": dose,
                        "intended": "A" if steered_is_a == intended_is_steered else "B",
                    }
                )
    rng.shuffle(items)
    shown, key = [], {}
    for number, item in enumerate(items, start=1):
        pair_id = f"p{number:03d}"
        shown.append({"id": pair_id, **{field: item[field] for field in ("question", "opening", "A", "B")}})
        key[pair_id] = {field: item[field] for field in ("axis", "dose", "intended")}
    return shown, key


def score(key: dict, answers: dict[str, dict[str, str]]) -> list[dict]:
    """One row per knob end: intended picks per rater and by majority."""
    cells: dict[tuple[str, float], list[str]] = {}
    for pair_id, entry in key.items():
        cells.setdefault((entry["axis"], entry["dose"]), []).append(pair_id)
    rows = []
    for (axis, dose), pair_ids in sorted(cells.items()):
        per_rater = {
            rater: sum(given.get(pair_id) == key[pair_id]["intended"] for pair_id in pair_ids)
            for rater, given in answers.items()
        }
        majority = sum(
            2 * sum(given.get(pair_id) == key[pair_id]["intended"] for given in answers.values()) > len(answers)
            for pair_id in pair_ids
        )
        rows.append(
            {"axis": axis, "dose": dose, "pairs": len(pair_ids), "raters": per_rater, "majority": majority,
             "passes": bool(majority >= PASS_MARK)}
        )
    return rows


def table(rows: list[dict]) -> str:
    raters = list(rows[0]["raters"]) if rows else []
    lines = [
        "| Axis | Dose | " + " | ".join(f"Rater {name}" for name in raters) + " | Majority | Passes (12 of 16) |",
        "|---|---|" + "---|" * (len(raters) + 2),
    ]
    for row in rows:
        picks = " | ".join(f"{row['raters'][name]} of {row['pairs']}" for name in raters)
        lines.append(
            f"| {row['axis']} | {row['dose']:+.2f} | {picks} | {row['majority']} of {row['pairs']} | {'yes' if row['passes'] else 'no'} |"
        )
    return "\n".join(lines)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def ask(name: str) -> None:
    """A person rates in the terminal. Answers are saved after every pair, so the session can be resumed."""
    path = RATING / f"answers_{name}.json"
    answers = load(path) if path.is_file() else {}
    pairs = load(RATING / "pairs.json")
    for count, pair in enumerate(pairs, start=1):
        if pair["id"] in answers:
            continue
        print(f"\n[{count} of {len(pairs)}] Opening: {pair['opening']}\n\n  A: {pair['A']}\n\n  B: {pair['B']}\n")
        reply = ""
        while reply not in ("A", "B"):
            reply = input(f"{pair['question']} (A or B): ").strip().upper()
        answers[pair["id"]] = reply
        save(path, answers)
    print(f"Done. Saved to {path}. Now run: python tools/blind_rating.py score")


def read_answers() -> dict[str, dict[str, str]]:
    return {path.stem.removeprefix("answers_"): load(path) for path in sorted(RATING.glob("answers_*.json"))}


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "make":
        sys.path.insert(0, str(ROOT))
        from kemudi import calibration

        data = calibration.load_axes_data()
        shown, key = make_pairs(
            load(CALIBRATION / "calibration.json"), load(CALIBRATION / "samples.json"), data["axes"], data["evaluation_openers"]
        )
        save(RATING / "pairs.json", shown)
        save(RATING / "key.json", key)
        print(f"{len(shown)} pairs written to {RATING}")
    elif command == "ask" and len(sys.argv) == 3:
        ask(sys.argv[2])
    elif command == "score":
        result = table(score(load(RATING / "key.json"), read_answers()))
        (ROOT / "docs" / "rating_table.md").write_text(result + "\n", encoding="utf-8", newline="\n")
        print(result)
    else:
        sys.exit(__doc__)
