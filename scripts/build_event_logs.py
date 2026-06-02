#!/usr/bin/env python3
"""Step 1: build per-subject event logs from raw activity.csv files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fpm.event_log import (  # noqa: E402
    DEFAULT_EVENT_LOG_DIR,
    build_subject_event_log,
    write_subject_event_log,
)
from fpm.loader import CASE_ID, DEFAULT_DATASET_ROOT, SUBJECT_IDS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate per-subject XES event logs from dailylog2016_dataset "
            "(SOWCompact pipeline step 1)."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root directory containing subject1..subject7 folders",
    )
    parser.add_argument(
        "--subject",
        type=int,
        choices=SUBJECT_IDS,
        default=None,
        help="Process only this subject (1-7). Default: all subjects.",
    )
    parser.add_argument(
        "--no-collapse-repeats",
        action="store_true",
        help="Keep consecutive duplicate activities in the event log",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_EVENT_LOG_DIR,
        help="Directory for generated event log files",
    )
    return parser.parse_args()


def print_summary(rows: list[dict]) -> None:
    header = f"{'Subject':<10} {'Traces':>7} {'Events':>7} {'Activities':>11}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['subject_label']:<10} "
            f"{row['traces']:>7} "
            f"{row['events']:>7} "
            f"{len(row['activities']):>11}"
        )


def main() -> None:
    args = parse_args()
    subject_ids = [args.subject] if args.subject is not None else list(SUBJECT_IDS)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    for subject_id in subject_ids:
        print(f"Building event log for subject{subject_id} ...")
        log = build_subject_event_log(
            args.dataset_root,
            subject_id,
            collapse_repeats=not args.no_collapse_repeats,
        )
        paths = write_subject_event_log(log, args.output_dir, subject_id)
        summary_rows.append(
            {
                "subject_label": f"subject{subject_id}",
                "traces": log[CASE_ID].nunique(),
                "events": len(log),
                "activities": sorted(log["concept:name"].unique()),
            }
        )
        print(f"  Wrote {paths['xes']}")
        print(f"  Wrote {paths['csv']}")
        print(f"  Wrote {paths['stats']}")

    print()
    print_summary(summary_rows)


if __name__ == "__main__":
    main()
