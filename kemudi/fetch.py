"""Download model files with resume. Tries Hugging Face first, then ModelScope."""

from __future__ import annotations

import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

MODEL_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "model.safetensors",
]

MIRRORS = [
    "https://huggingface.co/{repo}/resolve/main/{name}",
    "https://modelscope.cn/models/{repo}/resolve/master/{name}",
]

CHUNK_BYTES = 1 << 20
ATTEMPTS_PER_MIRROR = 8


def home() -> Path:
    """Folder for downloaded models. Override with the KEMUDI_HOME environment variable."""
    return Path(os.environ.get("KEMUDI_HOME", Path.home() / ".cache" / "kemudi"))


def model_dir(repo: str = DEFAULT_MODEL) -> Path:
    return home() / "models" / repo.split("/")[-1]


def is_complete(folder: Path) -> bool:
    return all((folder / name).is_file() for name in MODEL_FILES)


def fetch_model(repo: str = DEFAULT_MODEL, log=print) -> Path:
    """Download every model file that is missing and return the model folder."""
    folder = model_dir(repo)
    folder.mkdir(parents=True, exist_ok=True)
    for name in MODEL_FILES:
        target = folder / name
        if target.is_file():
            continue
        _fetch_file(repo, name, target, log)
    return folder


def _fetch_file(repo: str, name: str, target: Path, log) -> None:
    partial = target.with_name(target.name + ".part")
    errors = []
    for mirror in MIRRORS:
        url = mirror.format(repo=repo, name=name)
        for attempt in range(ATTEMPTS_PER_MIRROR):
            try:
                _download(url, partial, log)
                partial.replace(target)
                return
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as error:
                errors.append(f"{url}: {error}")
                if isinstance(error, urllib.error.HTTPError) and error.code in (401, 403, 404):
                    break
                time.sleep(min(2 + 2 * attempt, 10))
    raise RuntimeError(f"could not download {name}:\n  " + "\n  ".join(errors[-4:]))


def _download(url: str, partial: Path, log) -> None:
    """Append to `partial` from where it stopped. Returns only when the server's full length is on disk."""
    done = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={"User-Agent": "kemudi", "Range": f"bytes={done}-"})
    with urllib.request.urlopen(request, timeout=30) as response:
        resumed = response.status == 206
        if not resumed:
            done = 0
        remaining = int(response.headers.get("Content-Length") or 0)
        total = done + remaining
        last_report = time.monotonic()
        with open(partial, "ab" if resumed else "wb") as handle:
            while True:
                chunk = response.read(CHUNK_BYTES)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
                if time.monotonic() - last_report > 15:
                    last_report = time.monotonic()
                    log(f"  {partial.stem}: {done / 1e6:.0f} of {total / 1e6:.0f} MB")
    if total and done < total:
        raise ConnectionError(f"stream ended at {done} of {total} bytes")


if __name__ == "__main__":
    print(fetch_model(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL))
