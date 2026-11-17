"""Read and write a calibration: the steering vectors of one model and what was measured about them."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .engine import Axis

PACKAGE = Path(__file__).parent
BUNDLED = PACKAGE / "calibrations"


def load_axes_data() -> dict:
    return json.loads((PACKAGE / "data" / "axes.json").read_text(encoding="utf-8"))


def calibration_dir(model_name: str) -> Path:
    return BUNDLED / model_name.lower()


def save(folder: Path, summary: dict, units: dict[str, torch.Tensor]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(folder / "vectors.npz", **{name: unit.numpy().astype(np.float32) for name, unit in units.items()})
    save_summary(folder, summary)


def save_summary(folder: Path, summary: dict) -> None:
    text = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    (folder / "calibration.json").write_text(text, encoding="utf-8", newline="\n")


def load(folder: Path) -> tuple[dict, dict[str, Axis]]:
    """Return the summary and one ready-to-use Axis per calibrated direction."""
    summary = json.loads((folder / "calibration.json").read_text(encoding="utf-8"))
    vectors = np.load(folder / "vectors.npz")
    axes = {
        name: Axis(
            name=name,
            low=entry["low"],
            high=entry["high"],
            layer=entry["layer"],
            unit=torch.from_numpy(vectors[name]),
            scale=entry["scale"],
            gauge_mid=entry["gauge_mid"],
            gauge_half=entry["gauge_half"],
        )
        for name, entry in summary["axes"].items()
    }
    return summary, axes
