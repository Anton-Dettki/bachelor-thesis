#!/usr/bin/env python3
"""CLI wrapper for grouped next-event prediction on the CASAS2 ADL dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from CASAS2.queries import EXAMPLE_QUERIES  # noqa: E402
from fpm.grouped import run_grouped_evaluation  # noqa: E402
from shared.ltl_filter import parse_ltl_query  # noqa: E402

DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "grouped"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CASAS2 group-based next-event prediction")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--include-errors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-analog", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--n-clusters", default="auto", help="Number of clusters or 'auto'")
    parser.add_argument(
        "--ltl",
        type=str,
        default="",
        help="LTL query to filter clients before grouping (atoms as M07_ON)",
    )
    parser.add_argument(
        "--example-query",
        type=str,
        default=None,
        choices=sorted(EXAMPLE_QUERIES),
        help="Use a predefined example LTL query from CASAS2/queries.py",
    )
    parser.add_argument(
        "--min-match-traces",
        type=int,
        default=1,
        help="Minimum LTL-matching training traces required to keep a client",
    )
    parser.add_argument("--no-discovery-baseline", action="store_true")
    parser.add_argument("--no-workflow-graphs", action="store_true")
    parser.add_argument("--no-per-client-baseline", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        ltl_query = parse_ltl_query(args.ltl, args.example_query, EXAMPLE_QUERIES)
        result = run_grouped_evaluation(
            args.data_dir,
            args.output_dir,
            ltl=ltl_query,
            n_clusters=args.n_clusters,
            eval_protocol="casas2",
            train_fraction=args.train_fraction,
            include_errors=args.include_errors,
            skip_analog=args.skip_analog,
            min_matching_traces=args.min_match_traces,
            include_markov_baselines=not args.no_discovery_baseline,
            include_per_client_baseline=not args.no_per_client_baseline,
            write_workflow_graphs=not args.no_workflow_graphs,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print(pd.DataFrame(result["comparison"]).to_string(index=False))


if __name__ == "__main__":
    main()
