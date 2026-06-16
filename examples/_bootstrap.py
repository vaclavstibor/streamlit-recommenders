"""Ensure streamlit_recommenders is importable; show setup hint otherwise."""

from __future__ import annotations

import sys
from pathlib import Path


def import_sr():
    try:
        import streamlit_recommenders as sr

        return sr
    except ModuleNotFoundError:
        root = Path(__file__).resolve().parents[1]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        try:
            import streamlit_recommenders as sr

            return sr
        except ModuleNotFoundError:
            raise SystemExit(
                "\nstreamlit_recommenders is not installed in this Python environment.\n\n"
                "From the project root, run:\n"
                "  python3.11 -m venv .venv\n"
                "  .venv/bin/pip install -e \".[dev]\"\n"
                "  .venv/bin/streamlit run examples/showcase_demo.py\n\n"
                "Or use:  ./scripts/run_demo.sh\n"
            ) from None
