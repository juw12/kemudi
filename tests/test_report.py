import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from kemudi import calibration
from kemudi.report import best_fluent, dose_table, echo_table, is_fluent, repeats_the_control, results_table, sample_block, write_report

SETTINGS = {"max_perplexity_ratio": 1.5, "min_distinct_ratio": 0.85, "min_judge_auc": 0.9}


def point(key, value, effect, perplexity=1.0, distinct=1.0):
    return {key: value, "effect": effect, "effect_ci": [effect - 0.05, effect + 0.05], "perplexity_ratio": perplexity, "distinct_ratio": distinct, "samples": 16}


def test_fluent_needs_both_limits():
    assert is_fluent(point("dose", 0.1, 0.5, 1.5, 0.85), SETTINGS)
    assert not is_fluent(point("dose", 0.1, 0.5, 1.51, 1.0), SETTINGS)
    assert not is_fluent(point("dose", 0.1, 0.5, 1.0, 0.84), SETTINGS)


def test_best_fluent_ignores_points_past_the_limit():
    series = [point("dose", -0.2, 0.1, 3.0), point("dose", -0.1, 0.3), point("dose", 0.0, 0.5), point("dose", 0.1, 0.7), point("dose", 0.2, 0.9, 2.0)]
    best = best_fluent(series, "dose", SETTINGS, 0.5)
    assert best["down"]["dose"] == -0.1 and best["up"]["dose"] == 0.1
    assert best_fluent([point("dose", 0.2, 0.9, 2.0)], "dose", SETTINGS, 0.5)["up"] is None


def test_report_writes_valid_charts_and_a_table(tmp_path):
    summary = {
        "settings": SETTINGS,
        "axes": {
            "mood": {
                "low": "bleak", "high": "hopeful", "layer": 9, "test_auc": 0.97, "metric": "judge", "judge": {"auc": 0.95, "centre": 0.0, "scale": 0.1, "held_out_accuracy": 0.9},
                "usable": [-0.1, 0.1], "spearman_in_usable": 0.61,
                "steering": [point("dose", -0.2, 0.1, 3.0), point("dose", -0.1, 0.3), point("dose", 0.0, 0.5), point("dose", 0.1, 0.7), point("dose", 0.2, 0.9, 2.0)],
                "prompt": [point("level", -3, 0.2), point("level", 3, 0.8)],
                "random_direction": [point("dose", -0.1, 0.48), point("dose", 0.1, 0.52)],
            }
        },
    }
    (tmp_path / "calibration.json").write_text(json.dumps(summary), encoding="utf-8")
    chart, table = write_report(tmp_path, tmp_path / "docs")[:2]
    root = ET.parse(chart).getroot()
    circles = [element for element in root.iter() if element.tag.endswith("circle")]
    assert len(circles) == 7
    hollow = [circle for circle in circles if circle.get("fill") == "#fffdf8"]
    assert len(hollow) == 2
    assert len([element for element in root.iter() if element.tag.endswith("rect") and element.get("fill") == "none"]) == 2
    row = table.read_text(encoding="utf-8").splitlines()[2]
    assert row == "| mood (bleak to hopeful) | 9 | 0.97 | 0.950 | -0.10 to +0.10 | 0.50 | 0.30 to 0.70 | 0.48 to 0.52 | 0.20 to 0.80 | 0.61 |"


def test_bundled_calibrations_are_complete_and_consistent():
    """Every calibration shipped in the package must load and agree with its own settings."""
    folders = [folder for folder in calibration.BUNDLED.iterdir() if (folder / "calibration.json").is_file()]
    assert folders, "no bundled calibration"
    data = calibration.load_axes_data()
    for folder in folders:
        summary, axes = calibration.load(folder)
        assert set(axes) == set(data["axes"])
        for name, axis in axes.items():
            entry = summary["axes"][name]
            assert axis.unit.shape == (summary["width"],)
            assert float(axis.unit.norm()) == pytest.approx(1.0, abs=1e-4)
            assert 0 <= axis.layer < summary["blocks"]
            assert axis.scale > 0 and axis.gauge_half > 0
            assert [p["dose"] for p in entry["steering"]] == summary["settings"]["doses"]
            assert [p["level"] for p in entry["prompt"]] == summary["settings"]["prompt_levels"]
            low, high = entry["usable"]
            assert low <= 0 <= high
            assert [p["dose"] for p in entry["random_direction"]] == sorted({low, high} - {0.0})
            assert entry["test_auc"] >= 0.8, f"{name}: the vector does not separate held-out sentences"
            zero = next(p for p in entry["steering"] if p["dose"] == 0)
            assert zero["perplexity_ratio"] == 1.0 and zero["distinct_ratio"] == 1.0


def test_table_does_not_count_a_fluent_dose_beyond_the_usable_range(tmp_path):
    entry = {
        "low": "calm", "high": "tense", "layer": 12, "test_auc": 0.9, "metric": "quoted_share",
        "usable": [0.0, 0.2], "spearman_in_usable": 0.4,
        "steering": [point("dose", 0.0, 0.1), point("dose", 0.2, 0.3), point("dose", 0.4, 0.6, 2.0), point("dose", 0.6, 0.9)],
        "prompt": [point("level", 1, 0.2)],
        "random_direction": [point("dose", 0.2, 0.11)],
    }
    (tmp_path / "calibration.json").write_text(json.dumps({"settings": SETTINGS, "axes": {"tension": entry}}), encoding="utf-8")
    table = write_report(tmp_path, tmp_path / "docs")[1]
    row = table.read_text(encoding="utf-8").splitlines()[2]
    assert row == "| tension (calm to tense) | 12 | 0.90 | counted | +0.00 to +0.20 | 0.10 | none to 0.30 | none to 0.11 | none to 0.20 | 0.4 |"


def test_dose_table_gives_every_dose_its_interval_and_stars_the_ones_past_the_limit():
    summary = {"settings": SETTINGS, "axes": {
        "mood": {"steering": [point("dose", -0.2, 0.1, 3.0), point("dose", 0.0, 0.5, 9.0), point("dose", 0.2, 0.7)]},
        "speech": {"steering": [point("dose", 0.0, 0.2), point("dose", 0.2, 0.4, 1.0, 0.5)]},
        "unmeasured": {},
    }}
    assert dose_table(summary).splitlines() == [
        "| Dose | mood | speech |",
        "|---|---|---|",
        "| -0.20 | 0.10 (0.05 to 0.15) * | not measured |",
        "| +0.00 | 0.50 (0.45 to 0.55) | 0.20 (0.15 to 0.25) |",
        "| +0.20 | 0.70 (0.65 to 0.75) | 0.40 (0.35 to 0.45) * |",
    ]


ROOT = Path(__file__).parents[1]


def block(text: str, name: str) -> str:
    return text.split(f"<!-- {name}:start -->")[1].split(f"<!-- {name}:end -->")[0].strip()


def test_sample_block_lists_zero_and_the_usable_ends_or_every_dose():
    summary = {"axes": {"mood": {
        "low": "bleak", "high": "hopeful", "usable": [-0.2, 0.2],
        "steering": [{"dose": dose} for dose in (-0.4, -0.2, 0.0, 0.2, 0.4)],
    }}}
    samples = {"mood": {"steering": {str(dose): [f"text  at\n{dose}", "other"] for dose in (-0.4, -0.2, 0.0, 0.2, 0.4)}}}
    assert sample_block(summary, samples).splitlines()[2:] == ["- `-0.20` text at -0.2", "- `+0.00` text at 0.0", "- `+0.20` text at 0.2"]
    assert len(sample_block(summary, samples, every_dose=True).splitlines()) == 7


def test_readme_numbers_and_samples_are_the_measured_ones():
    """The README's results table and sample texts must be exactly what the shipped calibration produces."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for folder in calibration.BUNDLED.iterdir():
        summary = json.loads((folder / "calibration.json").read_text(encoding="utf-8"))
        samples = json.loads((folder / "samples.json").read_text(encoding="utf-8"))
        assert block(readme, "results") == results_table(summary)
        assert block(readme, "doses") == dose_table(summary)
        assert block(readme, "paired") == (ROOT / "docs" / "paired_table.md").read_text(encoding="utf-8").strip()
        assert block(readme, "samples") == sample_block(summary, samples)
        assert block(readme, "echoes") == echo_table(summary, samples)
    for name in ("judge_fresh", "rating", "pairs", "reproduce"):
        assert block(readme, name) == (ROOT / "docs" / f"{name}_table.md").read_text(encoding="utf-8").strip()


def test_echo_table_counts_texts_that_hand_the_control_back():
    assert repeats_the_control("mood", "mood: strongly toward bleak")
    assert repeats_the_control("mood", "Obey this control from the first sentence. CONTROLS")
    assert not repeats_the_control("mood", "Her mood lifted as the rain stopped.")
    assert not repeats_the_control("mood", 'He said: "the mood is fine."')
    samples = {"mood": {
        "steering": {"0.0": ["A story.", "Another story."], "0.2": ["More story.", "Still a story."]},
        "random_direction": {"0.2": ["A story.", "A story."]},
        "prompt": {"1": ["mood: slightly toward hopeful", "A story."], "-1": ["A story.", "CONTROLS mood"]},
    }}
    table = echo_table({"axes": {"mood": {}, "speech": {}}}, samples)
    assert table.splitlines()[2:] == ["| mood | 0 of 4 | 0 of 2 | 2 of 4 |"]


def test_table_flags_an_axis_whose_judge_failed_its_check(tmp_path):
    entry = {
        "low": "bleak", "high": "hopeful", "layer": 14, "test_auc": 0.95, "metric": "judge",
        "judge": {"auc": 0.898, "centre": 6.2, "scale": 1.0, "held_out_accuracy": 0.69}, "measurable": False,
        "usable": [0.0, 0.0], "spearman_in_usable": None,
        "steering": [point("dose", 0.0, 0.4), point("dose", 0.2, 0.5, 2.0)], "prompt": [point("level", 1, 0.45)], "random_direction": [],
    }
    (tmp_path / "calibration.json").write_text(json.dumps({"settings": SETTINGS, "axes": {"mood": entry}}), encoding="utf-8")
    chart, table = write_report(tmp_path, tmp_path / "docs")[:2]
    assert "| 0.898 (below 0.9: effect not validated) |" in table.read_text(encoding="utf-8")
    assert table.read_text(encoding="utf-8").splitlines()[2].endswith("| +0.00 to +0.00 | 0.40 | none | none | none to 0.45 | None |")
    assert "(judge failed its check)" in chart.read_text(encoding="utf-8")
