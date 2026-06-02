#!/usr/bin/env python3
"""Run federated social process mining: aggregate filtered traces and discover SOW model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pm4py

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fpm.aggregator import (  # noqa: E402
    Aggregator,
    aggregator_metrics,
    contribution_summary,
)
from fpm.event_log import DEFAULT_EVENT_LOG_DIR  # noqa: E402
from fpm.loader import SUBJECT_IDS  # noqa: E402
from fpm.ltl import PatternQuery  # noqa: E402
from fpm.queries import SCENARIO_QUERIES, query_slug  # noqa: E402
from fpm.social import SocialProcessMiner  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate filtered traces from phones and discover a SOW model "
            "(SOWCompact pipeline step 4)."
        )
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="LTL query string (ASCII operators). Required unless --scenario is set.",
    )
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIO_QUERIES),
        default=None,
        help="Use one of the predefined SOWCompact validation scenarios.",
    )
    parser.add_argument(
        "--event-log-dir",
        type=Path,
        default=DEFAULT_EVENT_LOG_DIR,
        help="Directory containing generated event logs",
    )
    parser.add_argument(
        "--subject",
        type=int,
        choices=SUBJECT_IDS,
        default=None,
        help="Include only this subject (1-7). Default: all subjects.",
    )
    parser.add_argument(
        "--min-traces",
        type=int,
        default=1,
        help="Minimum matching traces required for a phone to contribute",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "sow",
        help="Base directory for integrated logs and SOW models",
    )
    parser.add_argument(
        "--skip-discovery",
        action="store_true",
        help="Only aggregate traces; do not run Heuristic Miner",
    )
    return parser.parse_args()


def resolve_query(args: argparse.Namespace) -> PatternQuery:
    if args.query and args.scenario:
        raise SystemExit("Specify either --query or --scenario, not both.")
    if args.scenario:
        return PatternQuery.parse(SCENARIO_QUERIES[args.scenario])
    if args.query:
        return PatternQuery.parse(args.query)
    raise SystemExit("Provide --query or --scenario.")


def print_summary(result, metrics: dict) -> None:
    print(f"Query: {result.query}")
    header = (
        f"{'Subject':<10} {'Contrib':>8} {'Matching':>9} {'Total':>7} {'SizeKB':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in metrics["contributions"]:
        print(
            f"{row['subject_label']:<10} "
            f"{str(row['meets_pattern']):>8} "
            f"{row['matching_traces']:>9} "
            f"{row['total_traces']:>7} "
            f"{row['size_kb']:>8.3f}"
        )
    print()
    print(f"Contributors: {metrics['contributor_count']}")
    print(f"Integrated traces: {metrics['integrated_traces']}")
    print(f"Integrated events: {metrics['integrated_events']}")
    print(f"Merge time: {metrics['merge_time_s']:.6f}s")
    if "discovery_time_s" in metrics:
        print(f"Discovery time: {metrics['discovery_time_s']:.6f}s")


def main() -> None:
    args = parse_args()
    pattern = resolve_query(args)
    subject_ids = [args.subject] if args.subject is not None else list(SUBJECT_IDS)

    print(f"Running social mining for {pattern.text!r} ...")
    aggregator = Aggregator.from_subject_ids(
        subject_ids,
        event_log_dir=args.event_log_dir,
    )
    result = aggregator.run(pattern, min_traces=args.min_traces)

    output_dir = args.output_dir / query_slug(result.query)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = aggregator_metrics(result)
    metrics["contributions"] = [
        contribution_summary(c) for c in result.contributions
    ]

    query_payload = {
        "query": result.query,
        "contributing_subjects": result.contributing_subjects,
        "contributions": metrics["contributions"],
    }

    if result.integrated_log.empty:
        query_path = output_dir / "query.json"
        metrics_path = output_dir / "metrics.json"
        query_path.write_text(json.dumps(query_payload, indent=2), encoding="utf-8")
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print()
        print_summary(result, metrics)
        print("\nNo subjects matched the query; wrote summary only.")
        print(f"  {query_path}")
        print(f"  {metrics_path}")
        return

    integrated_xes = output_dir / "integrated_log.xes"
    integrated_csv = output_dir / "integrated_log.csv"
    pm4py.write_xes(result.integrated_log, str(integrated_xes))
    result.integrated_log.to_csv(integrated_csv, index=False)
    print(f"Wrote {integrated_xes}")
    print(f"Wrote {integrated_csv}")

    if not args.skip_discovery:
        miner = SocialProcessMiner()
        discovery = miner.discover_with_stats(result.integrated_log)
        artifacts = miner.write_artifacts(
            discovery.net,
            discovery.initial_marking,
            discovery.final_marking,
            output_dir,
        )
        metrics["model"] = discovery.stats
        metrics["discovery_time_s"] = discovery.stats["discovery_time_s"]
        print(f"Wrote {artifacts['pnml']}")
        print(f"Wrote {artifacts['png']}")

    query_path = output_dir / "query.json"
    metrics_path = output_dir / "metrics.json"
    query_path.write_text(json.dumps(query_payload, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print()
    print_summary(result, metrics)
    print(f"\nWrote {query_path}")
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
