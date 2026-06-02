#!/usr/bin/env python3
"""Federated social mining over HTTP: broadcast query, aggregate, discover SOW model."""

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
from fpm.loader import SUBJECT_IDS  # noqa: E402
from fpm.ltl import PatternQuery  # noqa: E402
from fpm.queries import SCENARIO_QUERIES, query_slug  # noqa: E402
from fpm.social import SocialProcessMiner  # noqa: E402


def default_phone_urls(subject_ids: list[int] | None = None) -> list[str]:
    ids = subject_ids if subject_ids is not None else list(SUBJECT_IDS)
    return [f"http://127.0.0.1:{8000 + sid}" for sid in ids]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run federated social mining over HTTP phone APIs "
            "(SOWCompact pipeline Phase D)."
        )
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="LTL query string. Required unless --scenario is set.",
    )
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIO_QUERIES),
        default=None,
        help="Predefined SOWCompact validation scenario.",
    )
    parser.add_argument(
        "--phones",
        nargs="+",
        default=None,
        help="Phone base URLs (default: http://127.0.0.1:8001 .. 8007)",
    )
    parser.add_argument(
        "--subject",
        type=int,
        choices=SUBJECT_IDS,
        default=None,
        help="Use only this subject's phone (must be running).",
    )
    parser.add_argument(
        "--min-traces",
        type=int,
        default=1,
        help="Minimum matching traces required for a phone to contribute",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout per phone (seconds)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "sow_federated",
        help="Output directory for integrated log and SOW model",
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
        f"{'Subject':<10} {'Contrib':>8} {'Matching':>9} {'Bytes':>8} {'TimeS':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in metrics["contributions"]:
        err = f" ERR:{row['error'][:30]}" if row.get("error") else ""
        print(
            f"{row['subject_label']:<10} "
            f"{str(row['meets_pattern']):>8} "
            f"{row['matching_traces']:>9} "
            f"{row.get('bytes_transferred', 0):>8} "
            f"{row.get('request_time_s', 0.0):>8.4f}"
            f"{err}"
        )
    print()
    print(f"Contributors: {metrics['contributor_count']}")
    print(f"Integrated traces: {metrics['integrated_traces']}")
    print(f"Integrated events: {metrics['integrated_events']}")
    print(f"Merge time: {metrics['merge_time_s']:.6f}s")
    if "total_bytes_received" in metrics:
        print(f"Network bytes received: {metrics['total_bytes_received']}")
        print(f"Total request time: {metrics['total_request_time_s']:.6f}s")
    if metrics.get("phone_errors"):
        print(f"Phone errors: {metrics['phone_errors']}")
    if "discovery_time_s" in metrics:
        print(f"Discovery time: {metrics['discovery_time_s']:.6f}s")


def main() -> None:
    args = parse_args()
    pattern = resolve_query(args)

    if args.phones:
        urls = args.phones
    elif args.subject is not None:
        urls = default_phone_urls([args.subject])
    else:
        urls = default_phone_urls()

    print(f"Federated mining for {pattern.text!r}")
    print(f"Phones: {', '.join(urls)}")

    aggregator = Aggregator.from_endpoints(urls, timeout=args.timeout)
    result = aggregator.run(pattern, min_traces=args.min_traces)

    output_dir = args.output_dir / query_slug(result.query)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = aggregator_metrics(result)
    metrics["contributions"] = [
        contribution_summary(c) for c in result.contributions
    ]
    metrics["phone_urls"] = urls

    query_payload = {
        "query": result.query,
        "contributing_subjects": result.contributing_subjects,
        "phone_urls": urls,
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
