#!/usr/bin/env python3
"""Build temporal train/validation splits for next-activity prediction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fpm.event_log import DEFAULT_EVENT_LOG_DIR, load_event_log, subject_event_log_xes_path  # noqa: E402
from fpm.loader import SUBJECT_IDS  # noqa: E402
from fpm.split import (  # noqa: E402
    DEFAULT_SPLIT_DIR,
    build_global_split,
    global_split_dir,
    subject_split,
    subject_split_dir,
    validate_split,
    write_split,
)

EXPECTED_VAL_TRACES = {
    1: 4,  # 14 traces * 0.25 -> round(3.5) = 4
    2: 4,  # 16 * 0.25 = 4
    3: 2,  # 10 * 0.25 = 2.5 -> round = 2
    4: 3,  # 12 * 0.25 = 3
    5: 1,  # 2 * 0.25 = 0.5 -> round = 0, min_val_traces = 1
    6: 2,  # 9 * 0.25 = 2.25 -> round = 2
    7: 3,  # 11 * 0.25 = 2.75 -> round = 3
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate temporal 75/25 train/validation splits per subject and globally "
            "(predictive process monitoring pipeline step)."
        )
    )
    parser.add_argument(
        "--event-log-dir",
        type=Path,
        default=DEFAULT_EVENT_LOG_DIR,
        help="Directory containing per-subject event logs from build_event_logs.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_SPLIT_DIR,
        help="Directory for generated split artifacts",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.25,
        help="Fraction of traces held out for validation (default: 0.25)",
    )
    parser.add_argument(
        "--subject",
        type=int,
        choices=SUBJECT_IDS,
        default=None,
        help="Process only this subject (1-7). Default: all subjects.",
    )
    return parser.parse_args()


def print_summary(rows: list[dict]) -> None:
    header = (
        f"{'Subject':<10} {'Total':>7} {'Train':>7} {'Val':>7} "
        f"{'TrainEv':>9} {'ValEv':>7}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['subject_label']:<10} "
            f"{row['total_traces']:>7} "
            f"{row['train_traces']:>7} "
            f"{row['val_traces']:>7} "
            f"{row['train_events']:>9} "
            f"{row['val_events']:>7}"
        )


def validate_expected_counts(subject_id: int, val_traces: int) -> None:
    expected = EXPECTED_VAL_TRACES.get(subject_id)
    if expected is not None and val_traces != expected:
        raise ValueError(
            f"subject{subject_id}: expected {expected} validation traces, got {val_traces}"
        )


def main() -> None:
    args = parse_args()
    subject_ids = [args.subject] if args.subject is not None else list(SUBJECT_IDS)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    subject_results = {}

    for subject_id in subject_ids:
        print(f"Splitting subject{subject_id} ...")
        source_log = load_event_log(
            subject_event_log_xes_path(args.event_log_dir, subject_id)
        )
        result = subject_split(
            subject_id,
            event_log_dir=args.event_log_dir,
            val_fraction=args.val_fraction,
        )
        validate_split(result, log=source_log)
        validate_expected_counts(subject_id, len(result.val_case_ids))

        out_dir = subject_split_dir(args.output_dir, subject_id)
        paths = write_split(
            result,
            out_dir,
            subject_id=subject_id,
            subject_label=f"subject{subject_id}",
            source_log=source_log,
        )
        subject_results[subject_id] = result

        summary_rows.append(
            {
                "subject_label": f"subject{subject_id}",
                "total_traces": result.total_traces,
                "train_traces": len(result.train_case_ids),
                "val_traces": len(result.val_case_ids),
                "train_events": len(result.train_log),
                "val_events": len(result.val_log),
            }
        )
        print(f"  Wrote {paths['train_xes']}")
        print(f"  Wrote {paths['val_xes']}")
        print(f"  Wrote {paths['manifest']}")

    print()
    print("Per-subject splits:")
    print_summary(summary_rows)

    if len(subject_ids) > 1:
        print()
        print("Building global pooled split ...")
        global_result = build_global_split(subject_results)
        validate_split(global_result)
        global_paths = write_split(
            global_result,
            global_split_dir(args.output_dir),
            subject_label="global",
        )
        print(f"  Wrote {global_paths['train_xes']}")
        print(f"  Wrote {global_paths['val_xes']}")
        print(f"  Wrote {global_paths['manifest']}")
        print()
        print(
            f"Global: {global_result.total_traces} traces -> "
            f"{len(global_result.train_case_ids)} train / "
            f"{len(global_result.val_case_ids)} val"
        )


if __name__ == "__main__":
    main()
