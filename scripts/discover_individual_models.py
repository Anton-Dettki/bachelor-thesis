#!/usr/bin/env python3
"""Step 2: discover individual Alpha+ models from generated event log files."""

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
from fpm.phone import Phone  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run individual Alpha+ process discovery from generated event logs "
            "(SOWCompact pipeline step 2)."
        )
    )
    parser.add_argument(
        "--event-log-dir",
        type=Path,
        default=DEFAULT_EVENT_LOG_DIR,
        help="Directory containing generated event logs from build_event_logs.py",
    )
    parser.add_argument(
        "--subject",
        type=int,
        choices=SUBJECT_IDS,
        default=None,
        help="Process only this subject (1-7). Default: all subjects.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "individual",
        help="Directory for per-subject model artifacts",
    )
    return parser.parse_args()


def discover_for_subject(phone: Phone, output_dir: Path) -> dict:
    subject_dir = output_dir / phone.subject_label
    subject_dir.mkdir(parents=True, exist_ok=True)

    net, initial_marking, final_marking = phone.discover_model()
    stats = phone.model_stats()

    pnml_path = subject_dir / "model.pnml"
    png_path = subject_dir / "model.png"
    stats_path = subject_dir / "stats.json"

    pm4py.write_pnml(net, initial_marking, final_marking, str(pnml_path))
    pm4py.save_vis_petri_net(
        net,
        initial_marking,
        final_marking,
        str(png_path),
    )
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    return stats


def print_summary(rows: list[dict]) -> None:
    header = (
        f"{'Subject':<10} {'Traces':>7} {'Events':>7} "
        f"{'Activities':>11} {'Places':>7} {'Trans.':>7} {'Arcs':>6}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['subject_label']:<10} "
            f"{row['traces']:>7} "
            f"{row['events']:>7} "
            f"{len(row['activities_in_log']):>11} "
            f"{row['places']:>7} "
            f"{row['transitions']:>7} "
            f"{row['arcs']:>6}"
        )


def main() -> None:
    args = parse_args()
    subject_ids = [args.subject] if args.subject is not None else list(SUBJECT_IDS)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    for subject_id in subject_ids:
        print(f"Discovering Alpha+ model for subject{subject_id} ...")
        phone = Phone(subject_id, event_log_dir=args.event_log_dir)
        stats = discover_for_subject(phone, args.output_dir)
        summary_rows.append(stats)

        subject_out = args.output_dir / phone.subject_label
        print(f"  Read  {phone.event_log_path}")
        print(f"  Wrote {subject_out / 'model.pnml'}")
        print(f"  Wrote {subject_out / 'model.png'}")
        print(f"  Wrote {subject_out / 'stats.json'}")

    print()
    print_summary(summary_rows)


if __name__ == "__main__":
    main()
