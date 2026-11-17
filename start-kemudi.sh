#!/bin/sh
# One-step start on macOS and Linux: makes a private Python environment the first time, then opens the app.
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "First start: installing Kemudi. This downloads about 300 MB once."
  python3 -m venv .venv
  .venv/bin/python -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
  .venv/bin/python -m pip install --quiet -e .
fi
exec .venv/bin/python -m kemudi serve --open
