#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

VENV_DIR="$ROOT_DIR/.venv"
if [[ -x "$VENV_DIR/bin/python" && -x "$VENV_DIR/bin/pip" ]]; then
  PYTHON="$VENV_DIR/bin/python"
  PIP="$VENV_DIR/bin/pip"
else
  PYTHON="python3"
  PIP="python3 -m pip"
fi

SKIP_INSTALL="${SKIP_INSTALL:-0}"

echo "Installing minimal dependencies for tests..."
if [[ "$PIP" == "python3 -m pip" ]]; then
  python3 -m pip install -U pip
else
  "$PIP" install -U pip
fi
"$PIP" install -r requirements-tests.txt

# Normalization test needs numpy.
# On some local Python versions, installing numpy pinned in this repo may fail to build;
# the normalization test itself will skip if numpy is unavailable.
if [[ "$SKIP_INSTALL" != "1" ]]; then
  "$PYTHON" -c "import numpy" >/dev/null 2>&1 || "$PIP" install -r dependencies/requirements_0.txt || true
fi

if [[ "$SKIP_INSTALL" != "1" ]]; then
  # Needed to import `domino.testing` and execute `piece_dry_run`.
  "$PIP" install -U "domino-py[cli]"
else
  # If domino isn't installed, fall back to installing it.
  "$PYTHON" -c "import domino" >/dev/null 2>&1 || "$PIP" install -U "domino-py[cli]"
fi

echo "Running all tests..."
"$PYTHON" -m pytest --cov=pieces --cov-report=xml --cov-report=term-missing "$@"
