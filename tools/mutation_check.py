"""Mutation check: break the code in one place at a time and confirm that the tests notice.

Works on a temporary copy of the repository, so the real files are never touched.
Run from the repository root: python tools/mutation_check.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MUTATIONS = [
    ("the hook subtracts the vector", "kemudi/engine.py", "steered = hidden + delta", "steered = hidden - delta"),
    (
        "the usable range ignores repetition",
        "kemudi/stats.py",
        "return perplexity_ratio[index] <= max_perplexity_ratio and distinct_ratio[index] >= min_distinct_ratio",
        "return perplexity_ratio[index] <= max_perplexity_ratio",
    ),
    ("the server accepts any Host header", "kemudi/server.py", "if host in LOCAL_HOSTS:", "if True:"),
    (
        "the judge scores the wrong way round",
        "kemudi/measure.py",
        "engine.continuation_nll(low_prompt, text) - engine.continuation_nll(high_prompt, text)",
        "engine.continuation_nll(high_prompt, text) - engine.continuation_nll(low_prompt, text)",
    ),
    (
        "the block choice ignores held-out separation",
        "kemudi/extract.py",
        "close = [layer for layer, value in dev_auc.items() if value >= best - AUC_TIE]",
        "close = list(dev_auc)",
    ),
    (
        "the random control keeps the real direction",
        "kemudi/measure.py",
        "noise = noise - (noise @ axis.unit) * axis.unit",
        "noise = noise + 50 * axis.unit",
    ),
]


def tests_pass(repo: Path) -> bool:
    command = [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"]
    return subprocess.run(command, cwd=repo, capture_output=True).returncode == 0


def main() -> int:
    with tempfile.TemporaryDirectory() as scratch:
        repo = Path(scratch) / "kemudi"
        shutil.copytree(ROOT, repo, ignore=shutil.ignore_patterns(".git", ".venv", "__pycache__", ".pytest_cache", "*.egg-info"))
        if not tests_pass(repo):
            print("The tests fail before any mutation. Fix that first.")
            return 1
        missed = 0
        for label, relative, old, new in MUTATIONS:
            path = repo / relative
            source = path.read_text(encoding="utf-8")
            if source.count(old) != 1:
                print(f"STALE   {label}: the line to mutate was not found exactly once")
                missed += 1
                continue
            path.write_text(source.replace(old, new), encoding="utf-8", newline="\n")
            caught = not tests_pass(repo)
            path.write_text(source, encoding="utf-8", newline="\n")
            print(f"{'caught' if caught else 'MISSED'}  {label}")
            missed += not caught
        print(f"{len(MUTATIONS) - missed} of {len(MUTATIONS)} mutations caught")
        return 1 if missed else 0


if __name__ == "__main__":
    sys.exit(main())
