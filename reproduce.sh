#!/bin/sh
# Rebuild the two final submissions. PY310 / PY313 point to Python 3.10 and 3.13 interpreters with
# requirements-py310.txt and requirements.txt installed (defaults: .venv310 and .venv313 in the repo root).
set -e
cd "$(dirname "$0")"
PY310=${PY310:-.venv310/bin/python}
PY313=${PY313:-.venv313/bin/python}
"$PY310" src/reproduce_stage1.py "$@"
"$PY313" src/reproduce_stage2.py --check "$@"
