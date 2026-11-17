"""Turn a calibration into charts (SVG) and a results table (Markdown). Standard library only."""

from __future__ import annotations

import json
import re
from pathlib import Path

INK, SOFT, GRID, GOOD, BAD, BAND = "#1c1b19", "#6b675f", "#d9d3c5", "#0d6b63", "#a8321f", "#cfe6e1"
WIDTH, HEIGHT, TOP, BOTTOM = 760, 300, 46, 62
PANELS = {"steering": (56, 440), "prompt": (500, 740)}


def is_fluent(point: dict, settings: dict) -> bool:
    return (
        point["perplexity_ratio"] <= settings["max_perplexity_ratio"]
        and point["distinct_ratio"] >= settings["min_distinct_ratio"]
    )


def effect_label(entry: dict) -> str:
    if entry["metric"] == "quoted_share":
        return "share of text inside quotes"
    return f"judge scale, {entry['low']} 0 to {entry['high']} 1"


def axis_chart(name: str, entry: dict, settings: dict) -> str:
    """Two panels with one shared effect scale: vector doses on the left, prompt control on the right."""
    points = {"steering": entry["steering"], "prompt": entry["prompt"]}
    keys = {"steering": "dose", "prompt": "level"}
    ceiling = max(0.2, max(point["effect_ci"][1] for series in points.values() for point in series))
    ceiling = min(1.0, round(ceiling + 0.05, 1))

    def y(value: float) -> float:
        return TOP + (1 - min(value, ceiling) / ceiling) * (HEIGHT - TOP - BOTTOM)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" font-family="Consolas, monospace" font-size="11">',
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="#fffdf8"/>',
        f'<text x="16" y="22" font-size="15" font-weight="bold" fill="{INK}">{name}: {entry["low"]} to {entry["high"]}'
        f'{"" if entry.get("measurable", True) else " (judge failed its check)"}</text>',
        f'<text x="{WIDTH - 16}" y="22" text-anchor="end" fill="{SOFT}">y: {effect_label(entry)}, 95% interval</text>',
    ]
    for kind, (left, right) in PANELS.items():
        series = points[kind]
        key = keys[kind]
        values = [point[key] for point in series]
        span = max(abs(value) for value in values)

        def x(value: float, left=left, right=right, span=span) -> float:
            return left + (value + span) / (2 * span) * (right - left)

        if kind == "steering":
            low, high = entry["usable"]
            parts.append(f'<rect x="{x(low):.1f}" y="{TOP}" width="{x(high) - x(low):.1f}" height="{HEIGHT - TOP - BOTTOM}" fill="{BAND}"/>')
        for fraction in (0, 0.5, 1):
            level = fraction * ceiling
            parts.append(f'<line x1="{left}" x2="{right}" y1="{y(level):.1f}" y2="{y(level):.1f}" stroke="{GRID}"/>')
            if kind == "steering":
                parts.append(f'<text x="{left - 8}" y="{y(level) + 4:.1f}" text-anchor="end" fill="{SOFT}">{level:.2f}</text>')
        ordered = sorted(series, key=lambda point: point[key])
        parts.append(
            f'<polyline fill="none" stroke="{INK}" stroke-width="1.6" points="'
            + " ".join(f"{x(point[key]):.1f},{y(point['effect']):.1f}" for point in ordered)
            + '"/>'
        )
        for point in ordered:
            px = x(point[key])
            fluent = point[key] == 0 or is_fluent(point, settings)
            colour = GOOD if fluent else BAD
            parts.append(f'<line x1="{px:.1f}" x2="{px:.1f}" y1="{y(point["effect_ci"][0]):.1f}" y2="{y(point["effect_ci"][1]):.1f}" stroke="{colour}" stroke-width="1.4"/>')
            fill = colour if fluent else "#fffdf8"
            parts.append(f'<circle cx="{px:.1f}" cy="{y(point["effect"]):.1f}" r="4" fill="{fill}" stroke="{colour}" stroke-width="1.6"/>')
            tick = f"{point[key]:+.2f}" if kind == "steering" else f"{point[key]:+d}"
            parts.append(f'<text x="{px:.1f}" y="{HEIGHT - BOTTOM + 16}" text-anchor="middle" fill="{SOFT}">{tick}</text>')
            parts.append(f'<text x="{px:.1f}" y="{HEIGHT - BOTTOM + 31}" text-anchor="middle" fill="{colour}">{point["perplexity_ratio"]:.2f}</text>')
        if kind == "steering":
            for point in entry.get("random_direction", []):
                parts.append(
                    f'<rect x="{x(point["dose"]) - 4:.1f}" y="{y(point["effect"]) - 4:.1f}" width="8" height="8" '
                    f'fill="none" stroke="{SOFT}" stroke-width="1.6"/>'
                )
        title = "vector dose (fraction of hidden-state size)" if kind == "steering" else "prompt control (3 = strongly)"
        parts.append(f'<text x="{(left + right) / 2}" y="{HEIGHT - 10}" text-anchor="middle" fill="{SOFT}">{title}</text>')
    parts.append(f'<text x="16" y="{HEIGHT - BOTTOM + 31}" fill="{SOFT}">ppl x</text>')
    parts.append(f'<text x="16" y="38" fill="{SOFT}">filled = fluent, hollow = past the limit, band = usable range, square = random direction</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def best_fluent(series: list[dict], key: str, settings: dict, baseline: float) -> dict:
    """The fluent point that moves the effect furthest from the baseline, in each direction."""
    fluent = [point for point in series if point[key] != 0 and is_fluent(point, settings)]
    down = min((point for point in fluent if point[key] < 0), key=lambda point: point["effect"], default=None)
    up = max((point for point in fluent if point[key] > 0), key=lambda point: point["effect"], default=None)
    return {"down": down, "up": up, "baseline": baseline}


def results_table(summary: dict) -> str:
    settings = summary["settings"]
    lines = [
        "| Axis | Block | Held-out AUC | Judge AUC | Usable doses | Effect at dose 0 | Vector, usable range | Random direction, same doses | Prompt, fluent levels | Rank corr. |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, entry in summary["axes"].items():
        baseline = next(point["effect"] for point in entry["steering"] if point["dose"] == 0)
        low, high = entry["usable"]
        inside = [point for point in entry["steering"] if low <= point["dose"] <= high]
        vector = best_fluent(inside, "dose", settings, baseline)
        random = {
            "down": next((point for point in entry.get("random_direction", []) if point["dose"] < 0), None),
            "up": next((point for point in entry.get("random_direction", []) if point["dose"] > 0), None),
        }
        prompt = best_fluent(entry["prompt"], "level", settings, baseline)

        def span(best: dict) -> str:
            if not best["down"] and not best["up"]:
                return "none"
            low = f"{best['down']['effect']:.2f}" if best["down"] else "none"
            high = f"{best['up']['effect']:.2f}" if best["up"] else "none"
            return f"{low} to {high}"

        judge = f"{entry['judge']['auc']:.3f}" if "judge" in entry else "counted"
        if not entry.get("measurable", True):
            judge += f" (below {settings['min_judge_auc']}: effect not validated)"
        lines.append(
            f"| {name} ({entry['low']} to {entry['high']}) | {entry['layer']} | {entry['test_auc']:.2f} | {judge} "
            f"| {low:+.2f} to {high:+.2f} | {baseline:.2f} | {span(vector)} | {span(random)} | {span(prompt)} | {entry['spearman_in_usable']} |"
        )
    return "\n".join(lines)


def sample_block(summary: dict, samples: dict, every_dose: bool = False, limit: int = 240) -> str:
    """One story opening continued with the same seed: dose 0 and both ends of each usable range.

    With `every_dose` all measured doses are listed, including the ones where the text breaks.
    """
    lines = []
    for name, entry in summary["axes"].items():
        if name not in samples:
            continue
        low, high = entry["usable"]
        doses = [point["dose"] for point in entry["steering"]] if every_dose else sorted({low, 0.0, high})
        lines += [f"**{name}** ({entry['low']} to {entry['high']}), usable {low:+.2f} to {high:+.2f}", ""]
        for dose in doses:
            text = " ".join(samples[name]["steering"][str(dose)][0].split())
            lines.append(f"- `{dose:+.2f}` {text[:limit]}")
        lines.append("")
    return "\n".join(lines).rstrip()


def repeats_the_control(name: str, text: str) -> bool:
    """True when a text hands the control instruction back (`mood: strongly toward bleak`) and tells no story."""
    return "CONTROLS" in text or re.search(rf"\b{re.escape(name)}\s*:", text) is not None


def echo_table(summary: dict, samples: dict) -> str:
    """How many texts per kind of control repeat the control instruction instead of continuing the story."""
    lines = ["| Axis | Vector, all doses | Random direction | Prompt control, all levels |", "|---|---|---|---|"]
    for name in summary["axes"]:
        if name not in samples:
            continue
        cells = []
        for kind in ("steering", "random_direction", "prompt"):
            texts = [text for group in samples[name][kind].values() for text in group]
            cells.append(f"{sum(repeats_the_control(name, text) for text in texts)} of {len(texts)}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(calibration_folder: Path, out: Path) -> list[Path]:
    summary = json.loads((calibration_folder / "calibration.json").read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, entry in summary["axes"].items():
        if "steering" not in entry:
            continue
        path = out / f"dose_{name}.svg"
        path.write_text(axis_chart(name, entry, summary["settings"]) + "\n", encoding="utf-8", newline="\n")
        written.append(path)
    table = out / "results_table.md"
    table.write_text(results_table(summary) + "\n", encoding="utf-8", newline="\n")
    written.append(table)
    samples_path = calibration_folder / "samples.json"
    if samples_path.is_file():
        samples = json.loads(samples_path.read_text(encoding="utf-8"))
        listing = out / "samples.md"
        listing.write_text(sample_block(summary, samples, every_dose=True) + "\n", encoding="utf-8", newline="\n")
        echoes = out / "echo_table.md"
        echoes.write_text(echo_table(summary, samples) + "\n", encoding="utf-8", newline="\n")
        written += [listing, echoes]
    return written
