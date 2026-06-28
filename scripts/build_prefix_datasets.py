#!/usr/bin/env python3
"""Build prefix -> next-activity datasets from train/validation splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fpm.event_log import load_event_log  # noqa: E402
from fpm.loader import SUBJECT_IDS  # noqa: E402
from fpm.prefix import (  # noqa: E402
    DEFAULT_PREFIX_DIR,
    Vocabulary,
    build_prefix_frame,
    encode_frame,
    prefix_manifest,
    validate_prefix_frame,
)
from fpm.split import DEFAULT_SPLIT_DIR, global_split_dir, subject_split_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build prefix -> next-activity datasets from train/validation splits "
            "(predictive process monitoring pipeline step)."
        )
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
        default=DEFAULT_PREFIX_DIR,
        help="Directory for generated prefix datasets",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=3,
        help="Prefix window size (default: 3)",
    )
    parser.add_argument(
        "--subject",
        type=int,
        choices=SUBJECT_IDS,
        default=None,
        help="Process only this subject (1-7). Default: all subjects + global.",
    )
    return parser.parse_args()


def print_summary(rows: list[dict]) -> None:
    header = f"{'Scope':<12} {'Train':>7} {'Val':>7} {'Activities':>11}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['scope']:<12} "
            f"{row['train_samples']:>7} "
            f"{row['val_samples']:>7} "
            f"{row['n_activities']:>11}"
        )


def build_scope(
    scope: str,
    split_path: Path,
    output_path: Path,
    *,
    window: int,
    subject_id: int | None = None,
) -> dict:
    train_log = load_event_log(split_path / "train.xes")
    val_log = load_event_log(split_path / "val.xes")

    train_frame = build_prefix_frame(
        train_log,
        window=window,
        subject_id=subject_id,
    )
    val_frame = build_prefix_frame(
        val_log,
        window=window,
        subject_id=subject_id,
    )
    validate_prefix_frame(train_frame, train_log, window=window)
    validate_prefix_frame(val_frame, val_log, window=window)

    # Use the declared activity taxonomy (split-independent) rather than a
    # vocabulary derived from this scope's train+val logs. Deriving from
    # train+val would leak validation-only activities into the label space and
    # the Markov smoothing denominator, and would give each subject a different
    # integer id space (breaking federated count aggregation).
    vocab = Vocabulary.canonical()
    unknown = vocab.covers(train_log) | vocab.covers(val_log)
    if unknown:
        raise ValueError(
            f"{scope}: activities outside ACTIVITY_TAXONOMY: {sorted(unknown)}. "
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

    output_path.mkdir(parents=True, exist_ok=True)
    train_csv = output_path / "train.csv"
    val_csv = output_path / "val.csv"
    vocab_json = output_path / "vocab.json"
    manifest_path = output_path / "prefix_manifest.json"

    train_encoded.to_csv(train_csv, index=False)
    val_encoded.to_csv(val_csv, index=False)
    vocab.write_json(vocab_json)
    manifest_path.write_text(
        json.dumps(
            prefix_manifest(
                scope=scope,
                window=window,
                train_samples=len(train_encoded),
                val_samples=len(val_encoded),
                n_activities=vocab.size,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "scope": scope,
        "train_samples": len(train_encoded),
        "val_samples": len(val_encoded),
        "n_activities": vocab.size,
        "paths": {
            "train": train_csv,
            "val": val_csv,
            "vocab": vocab_json,
            "manifest": manifest_path,
        },
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    subject_ids = [args.subject] if args.subject is not None else list(SUBJECT_IDS)
    summary_rows: list[dict] = []

    for subject_id in subject_ids:
        scope = f"subject{subject_id}"
        print(f"Building prefix dataset for {scope} ...")
        result = build_scope(
            scope,
            subject_split_dir(args.split_dir, subject_id),
            args.output_dir / scope,
            window=args.window,
            subject_id=subject_id,
        )
        summary_rows.append(result)
        print(f"  Wrote {result['paths']['train']}")
        print(f"  Wrote {result['paths']['val']}")
        print(f"  Wrote {result['paths']['vocab']}")

    if args.subject is None:
        print()
        print("Building global prefix dataset ...")
        global_result = build_scope(
            "global",
            global_split_dir(args.split_dir),
            args.output_dir / "global",
            window=args.window,
        )
        summary_rows.append(global_result)
        print(f"  Wrote {global_result['paths']['train']}")
        print(f"  Wrote {global_result['paths']['val']}")
        print(f"  Wrote {global_result['paths']['vocab']}")

    print()
    print_summary(summary_rows)


if __name__ == "__main__":
    main()
