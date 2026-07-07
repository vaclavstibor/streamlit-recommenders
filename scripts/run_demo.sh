#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3.11}"

if [[ ! -d .venv ]]; then
  echo "Creating .venv with $PYTHON ..."
  "$PYTHON" -m venv .venv
  .venv/bin/pip install -e ".[dev]"
fi

DEMO="${1:-examples/3_models_comparison_rows.py}"
shift || true
exec .venv/bin/streamlit run "$DEMO" "$@"
