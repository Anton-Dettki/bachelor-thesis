#!/usr/bin/env python3
"""Build a directly-follows graph (DFG) from the Chinook ADL sensor dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pm4py

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chinook_loader import (  # noqa: E402
    ACTIVITY,
    CASE_ID,
    TIMESTAMP,
    load_event_log_dataframe,
    resolve_dataset_zip,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover and visualize a DFG from Chinook ADL sensor logs."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=ROOT / "dataset",
        help="Directory containing adl_noerror.zip and adl_error.zip",
    )
    parser.add_argument(
        "--variant",
        choices=("adl_noerror", "adl_error"),
        default="adl_noerror",
        help="Which dataset split to use",
    )
    parser.add_argument(
        "--task",
        type=int,
        choices=range(1, 6),
        default=None,
        help="Optional task filter (1-5). If omitted, all tasks are included.",
    )
    parser.add_argument(
        "--activity-coverage",
        type=float,
        default=0.8,
        help=(
            "Keep the most frequent activities until this fraction of all events "
            "is covered (0-1). Lower values produce sparser, more readable DFGs."
        ),
    )
    parser.add_argument(
        "--path-coverage",
        type=float,
        default=0.2,
        help=(
            "Keep the most frequent directly-follows paths until this fraction "
            "of path occurrences is covered (0-1). Applied after DFG discovery."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "dfg",
        help="Directory for generated artifacts",
    )
    return parser.parse_args()


def filter_by_activity_coverage(event_log, coverage: float):
    if not 0 < coverage <= 1:
        raise ValueError("activity-coverage must be in (0, 1].")

    activities = pm4py.get_event_attribute_values(event_log, ACTIVITY)
    total_events = sum(activities.values())
    target = total_events * coverage

    kept: list[str] = []
    running = 0
    for activity, count in sorted(activities.items(), key=lambda item: -item[1]):
        kept.append(activity)
        running += count
        if running >= target:
            break

    return pm4py.filter_event_attribute_values(
        event_log,
        ACTIVITY,
        kept,
        level="event",
        retain=True,
    )


def main() -> None:
    args = parse_args()
    zip_path = resolve_dataset_zip(args.dataset_dir, args.variant)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading traces from {zip_path} ...")
    raw_df = load_event_log_dataframe(zip_path, task_filter=args.task)
    event_log = pm4py.format_dataframe(
        raw_df,
        case_id=CASE_ID,
        activity_key=ACTIVITY,
        timestamp_key=TIMESTAMP,
    )

    n_cases = raw_df[CASE_ID].nunique()
    n_events = len(raw_df)
    n_activities = raw_df[ACTIVITY].nunique()
    print(f"Loaded {n_events:,} events across {n_cases} traces ({n_activities} activities).")

    filtered_log = filter_by_activity_coverage(event_log, args.activity_coverage)
    kept_activities = pm4py.get_event_attribute_values(filtered_log, ACTIVITY)
    print(
        f"Filtered to {len(kept_activities)} activities "
        f"({args.activity_coverage:.0%} event coverage)."
    )

    dfg, start_activities, end_activities = pm4py.discover_dfg(filtered_log)
    print(
        f"Discovered DFG with {len(dfg)} directly-follows relations, "
        f"{len(start_activities)} start activities, "
        f"{len(end_activities)} end activities."
    )

    vis_dfg, vis_start, vis_end = pm4py.filter_dfg_paths_percentage(
        dfg,
        start_activities,
        end_activities,
        percentage=args.path_coverage,
    )
    print(
        f"Visualization DFG reduced to {len(vis_dfg)} paths "
        f"({args.path_coverage:.0%} path coverage)."
    )

    stem = args.variant if args.task is None else f"{args.variant}_t{args.task}"
    png_path = output_dir / f"{stem}.png"
    json_path = output_dir / f"{stem}.json"

    pm4py.save_vis_dfg(
        vis_dfg,
        vis_start,
        vis_end,
        str(png_path),
    )

    payload = {
        "variant": args.variant,
        "task": args.task,
        "activity_coverage": args.activity_coverage,
        "path_coverage": args.path_coverage,
        "cases": n_cases,
        "events": n_events,
        "activities_before_filter": n_activities,
        "activities_after_filter": len(kept_activities),
        "dfg_edges": len(dfg),
        "visualization_edges": len(vis_dfg),
        "start_activities": start_activities,
        "end_activities": end_activities,
        "directly_follows": {
            f"{source} -> {target}": frequency
            for (source, target), frequency in sorted(
                dfg.items(), key=lambda item: -item[1]
            )
        },
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = output_dir / f"{stem}_event_log.csv"
    filtered_log.to_csv(csv_path, index=False)

    print(f"Wrote DFG visualization to {png_path}")
    print(f"Wrote DFG statistics to {json_path}")
    print(f"Wrote filtered event log to {csv_path}")


if __name__ == "__main__":
    main()
