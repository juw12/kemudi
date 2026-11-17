import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from kemudi import cli

ROOT = Path(__file__).parents[1]


def test_reproduce_runs_extract_measure_and_report_in_that_order(monkeypatch):
    calls = []
    for name in ("extract", "measure", "report"):
        monkeypatch.setattr(cli, f"command_{name}", lambda args, name=name: calls.append((name, args.axis, args.out)))
    cli.main(["reproduce", "--out", "charts"])
    assert calls == [("extract", None, "charts"), ("measure", None, "charts"), ("report", None, "charts")]


def test_knobs_are_parsed_as_numbers():
    assert cli.parse_knobs(["mood=0.3", "speech=-0.15"]) == {"mood": 0.3, "speech": -0.15}


def test_both_launchers_install_once_and_then_open_the_app():
    windows = (ROOT / "Start Kemudi.cmd").read_text(encoding="utf-8")
    unix = (ROOT / "start-kemudi.sh").read_text(encoding="utf-8")
    for script in (windows, unix):
        assert "-m venv .venv" in script
        assert "https://download.pytorch.org/whl/cpu" in script
        assert "-m kemudi serve --open" in script
    assert unix.startswith("#!/bin/sh\n") and "\r" not in unix


@pytest.mark.skipif(sys.platform == "win32" or not shutil.which("sh"), reason="needs a POSIX shell")
def test_unix_launcher_is_valid_shell():
    subprocess.run(["sh", "-n", str(ROOT / "start-kemudi.sh")], check=True)
