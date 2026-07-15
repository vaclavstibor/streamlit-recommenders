"""CLI: python -m streamlit_recommenders.data.prepare --dataset ml-32m [--with-posters]."""

from __future__ import annotations

import argparse
from pathlib import Path

from streamlit_recommenders.data.prepare.goodbooks import prepare_goodbooks
from streamlit_recommenders.data.prepare.movielens import MOVIELENS_URLS, prepare_movielens


def main() -> None:
    """Parse CLI arguments and dispatch to the goodbooks or MovieLens preparer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="ml-32m",
        help=f"Dataset name: {', '.join(sorted(MOVIELENS_URLS))}, or 'goodbooks'.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output directory.")
    parser.add_argument(
        "--with-posters",
        action="store_true",
        help="Fetch TMDB posters/descriptions (needs TMDB_API_KEY). MovieLens only.",
    )
    parser.add_argument(
        "--poster-limit",
        type=int,
        default=0,
        help="Max posters to download (0 = all matched items).",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild even if already complete.")
    args = parser.parse_args()

    if args.dataset in {"goodbooks", "goodbooks-10k"}:
        prepare_goodbooks(args.output, force=args.force)
        return

    prepare_movielens(
        args.dataset,
        args.output,
        with_posters=args.with_posters,
        poster_limit=args.poster_limit,
        force=args.force,
    )


if __name__ == "__main__":
    main()
