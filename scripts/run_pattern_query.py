#!/usr/bin/env python3
"""Run an LTL pattern query against generated per-subject event logs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pm4py

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fpm.event_log import DEFAULT_EVENT_LOG_DIR  # noqa: E402
from fpm.loader import SUBJECT_IDS  # noqa: E402
from fpm.ltl import PatternQuery  # noqa: E402
from fpm.phone import Phone  # noqa: E402
from fpm.queries import SCENARIO_QUERIES, query_slug  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an LTL pattern query on per-subject event logs "
            "(SOWCompact pipeline step 3)."
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
        help="Evaluate only this subject (1-7). Default: all subjects.",
    )
    parser.add_argument(
        "--min-traces",
        type=int,
        default=1,
        help="Minimum matching traces required for a phone to 'meet' the pattern",
    )
    parser.add_argument(
        "--write-filtered",
        action="store_true",
        help="Write filtered event logs for matching subjects",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "filtered",
        help="Base directory for filtered logs when --write-filtered is set",
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


def evaluate_subject(
    subject_id: int,
    pattern: PatternQuery,
    *,
    event_log_dir: Path,
    min_traces: int,
    write_filtered: bool,
    output_dir: Path,
) -> dict:
    phone = Phone(subject_id, event_log_dir=event_log_dir)
    sequences = phone.trace_sequences()
    matching = phone.select_matching_traces(pattern)
    filtered = phone.filtered_log(pattern)
    meets = len(matching) >= min_traces

    result = {
        "subject_id": subject_id,
        "subject_label": phone.subject_label,
        "meets_pattern": meets,
        "matching_traces": len(matching),
        "total_traces": len(sequences),
        "matching_case_ids": matching,
        "filtered_events": len(filtered),
    }

    if write_filtered and meets:
        query_dir = output_dir / query_slug(pattern.text) / phone.subject_label
        query_dir.mkdir(parents=True, exist_ok=True)
        xes_path = query_dir / "filtered_log.xes"
        csv_path = query_dir / "filtered_log.csv"
        pm4py.write_xes(filtered, str(xes_path))
        filtered.to_csv(csv_path, index=False)
        result["filtered_xes"] = str(xes_path)
        result["filtered_csv"] = str(csv_path)

    return result


def print_summary(rows: list[dict], query_text: str) -> None:
    print(f"Query: {query_text}")
    header = (
        f"{'Subject':<10} {'Meets':>6} {'Matching':>9} {'Total':>7} {'FilteredEv':>11}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['subject_label']:<10} "
            f"{str(row['meets_pattern']):>6} "
            f"{row['matching_traces']:>9} "
            f"{row['total_traces']:>7} "
            f"{row['filtered_events']:>11}"
        )


def main() -> None:
    args = parse_args()
    pattern = resolve_query(args)
    subject_ids = [args.subject] if args.subject is not None else list(SUBJECT_IDS)

    rows: list[dict] = []
    for subject_id in subject_ids:
        print(f"Evaluating {pattern.text!r} on subject{subject_id} ...")
        row = evaluate_subject(
            subject_id,
            pattern,
            event_log_dir=args.event_log_dir,
            min_traces=args.min_traces,
            write_filtered=args.write_filtered,
            output_dir=args.output_dir,
        )
        rows.append(row)
        if row.get("filtered_xes"):
            print(f"  Wrote {row['filtered_xes']}")

    print()
    print_summary(rows, pattern.text)

    if args.write_filtered:
        summary_path = args.output_dir / query_slug(pattern.text) / "query.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps({"query": pattern.text, "results": rows}, indent=2),
            encoding="utf-8",
        )
        print(f"\nWrote summary to {summary_path}")


if __name__ == "__main__":
    main()
