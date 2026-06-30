#!/usr/bin/env python3
"""Build LTL-group prefix -> next-activity datasets from train/validation splits."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fpm.aggregator import namespace_filtered_log  # noqa: E402
from fpm.event_log import load_event_log  # noqa: E402
from fpm.loader import SUBJECT_IDS  # noqa: E402
from fpm.ltl import PatternQuery  # noqa: E402
from fpm.phone import select_matching_case_ids  # noqa: E402
from fpm.prefix import (  # noqa: E402
    Vocabulary,
    build_prefix_frame,
    encode_frame,
    filter_trainable_target_classes,
    prefix_manifest,
    validate_prefix_frame,
)
from fpm.queries import SCENARIO_QUERIES, query_slug  # noqa: E402
from fpm.split import DEFAULT_SPLIT_DIR, _filter_log_by_cases, subject_split_dir  # noqa: E402

DEFAULT_GROUP_PREFIX_DIR = ROOT / "output" / "prefix" / "group"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build LTL-group prefix datasets: filter train/val splits by scenario "
            "query per subject, pool matching days, and emit prefix CSVs."
        )
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help=(
            "Scenario key from SCENARIO_QUERIES (e.g. scenario2_no_sport). "
            "Default: all scenarios with enough matching train traces."
        ),
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Raw LTL query text (alternative to --scenario; slug derived via query_slug).",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=DEFAULT_SPLIT_DIR,
        help="Directory containing split artifacts from build_splits.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_GROUP_PREFIX_DIR,
        help="Root directory for group prefix datasets",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=3,
        help="Prefix window size (default: 3)",
    )
    parser.add_argument(
        "--min-train-traces",
        type=int,
        default=5,
        help="Skip scenarios with fewer matching train traces (default: 5)",
    )
    return parser.parse_args()


def resolve_scenarios(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.scenario and args.query:
        raise ValueError("Specify only one of --scenario or --query")

    if args.query:
        slug = query_slug(args.query)
        return [(slug, args.query)]

    if args.scenario:
        if args.scenario not in SCENARIO_QUERIES:
            known = ", ".join(sorted(SCENARIO_QUERIES))
            raise ValueError(f"Unknown scenario {args.scenario!r}; choose from {known}")
        return [(args.scenario, SCENARIO_QUERIES[args.scenario])]

    return list(SCENARIO_QUERIES.items())


def print_summary(rows: list[dict]) -> None:
    header = (
        f"{'Scenario':<34} {'TrainTr':>7} {'ValTr':>6} "
        f"{'TrainPx':>8} {'ValPx':>7} {'Subjects':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['scenario']:<34} "
            f"{row['train_traces']:>7} "
            f"{row['val_traces']:>6} "
            f"{row['train_samples']:>8} "
            f"{row['val_samples']:>7} "
            f"{row['contributing_subjects']:>9}"
        )


def build_group(
    scenario: str,
    query_text: str,
    *,
    split_dir: Path,
    output_dir: Path,
    window: int,
    min_train_traces: int,
) -> dict | None:
    pattern = PatternQuery.parse(query_text)
    del pattern  # parsed for validation only; select_matching_case_ids re-parses

    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    membership: dict[str, dict[str, list[str]]] = {}
    train_trace_total = 0
    val_trace_total = 0
    contributing_subjects = 0

    for subject_id in SUBJECT_IDS:
        subject_label = f"subject{subject_id}"
        subject_dir = subject_split_dir(split_dir, subject_id)
        train_log = load_event_log(subject_dir / "train.xes")
        val_log = load_event_log(subject_dir / "val.xes")

        train_matching = select_matching_case_ids(train_log, query_text)
        val_matching = select_matching_case_ids(val_log, query_text)
        membership[subject_label] = {
            "train_case_ids": train_matching,
            "val_case_ids": val_matching,
        }

        if train_matching:
            contributing_subjects += 1
            train_trace_total += len(train_matching)
            filtered_train = namespace_filtered_log(
                _filter_log_by_cases(train_log, train_matching),
                subject_label,
            )
            train_parts.append(filtered_train)

        if val_matching:
            val_trace_total += len(val_matching)
            filtered_val = namespace_filtered_log(
                _filter_log_by_cases(val_log, val_matching),
                subject_label,
            )
            val_parts.append(filtered_val)

    if train_trace_total < min_train_traces:
        warnings.warn(
            f"Skipping {scenario}: only {train_trace_total} matching train traces "
            f"(min {min_train_traces}).",
            stacklevel=2,
        )
        return None

    train_log = (
        pd.concat(train_parts, ignore_index=True)
        if train_parts
        else pd.DataFrame()
    )
    val_log = (
        pd.concat(val_parts, ignore_index=True)
        if val_parts
        else pd.DataFrame()
    )

    train_frame = build_prefix_frame(
        train_log,
        window=window,
    )
    val_frame = build_prefix_frame(
        val_log,
        window=window,
    )
    if not train_log.empty:
        validate_prefix_frame(train_frame, train_log, window=window)
    if not val_log.empty:
        validate_prefix_frame(val_frame, val_log, window=window)
    train_frame, val_frame, class_filter = filter_trainable_target_classes(
        train_frame,
        val_frame,
    )

    vocab = Vocabulary.canonical()
    unknown = vocab.covers(train_log) | vocab.covers(val_log)
    if unknown:
        raise ValueError(
            f"{scenario}: activities outside ACTIVITY_TAXONOMY: {sorted(unknown)}. "
            "Update fpm.loader.ACTIVITY_TAXONOMY."
        )

    train_encoded = encode_frame(
        train_frame,
        vocab,
        window=window,
    )
    val_encoded = encode_frame(
        val_frame,
        vocab,
        window=window,
    )

    scenario_dir = output_dir / scenario
    scenario_dir.mkdir(parents=True, exist_ok=True)
    train_csv = scenario_dir / "train.csv"
    val_csv = scenario_dir / "val.csv"
    vocab_json = scenario_dir / "vocab.json"
    manifest_path = scenario_dir / "prefix_manifest.json"
    membership_path = scenario_dir / "membership.json"

    train_encoded.to_csv(train_csv, index=False)
    val_encoded.to_csv(val_csv, index=False)
    vocab.write_json(vocab_json)
    manifest_path.write_text(
        json.dumps(
            {
                **prefix_manifest(
                    scope=scenario,
                    window=window,
                    train_samples=len(train_encoded),
                    val_samples=len(val_encoded),
                    n_activities=vocab.size,
                    class_filter=class_filter,
                ),
                "query": query_text,
                "train_traces": train_trace_total,
                "val_traces": val_trace_total,
                "contributing_subjects": contributing_subjects,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    membership_path.write_text(
        json.dumps(
            {
                "scenario": scenario,
                "query": query_text,
                "train_traces": train_trace_total,
                "val_traces": val_trace_total,
                "contributing_subjects": contributing_subjects,
                "subjects": membership,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "scenario": scenario,
        "train_traces": train_trace_total,
        "val_traces": val_trace_total,
        "train_samples": len(train_encoded),
        "val_samples": len(val_encoded),
        "contributing_subjects": contributing_subjects,
        "paths": {
            "train": train_csv,
            "val": val_csv,
            "vocab": vocab_json,
            "manifest": manifest_path,
            "membership": membership_path,
        },
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = resolve_scenarios(args)
    summary_rows: list[dict] = []

    for scenario, query_text in scenarios:
        print(f"Building group prefix dataset for {scenario} ...")
        result = build_group(
            scenario,
            query_text,
            split_dir=args.split_dir,
            output_dir=args.output_dir,
            window=args.window,
            min_train_traces=args.min_train_traces,
        )
        if result is None:
            continue
        summary_rows.append(result)
        print(f"  Wrote {result['paths']['train']}")
        print(f"  Wrote {result['paths']['val']}")
        print(f"  Wrote {result['paths']['membership']}")

    if not summary_rows:
        raise SystemExit("No group prefix datasets were built.")

    print()
    print_summary(summary_rows)


if __name__ == "__main__":
    main()
