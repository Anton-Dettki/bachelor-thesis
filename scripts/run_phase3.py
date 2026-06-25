#!/usr/bin/env python3
"""Run the full Phase 3 next-activity prediction pipeline end-to-end."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fpm.loader import DEFAULT_DATASET_ROOT  # noqa: E402
from fpm.prefix import (  # noqa: E402
    DEFAULT_PREFIX_DIR,
    FEATURE_SET_BASIC,
    VALID_FEATURE_SETS,
    resolve_feature_set,
)
from fpm.predict import DEFAULT_FEDERATED_MODEL_DIR, DEFAULT_MODEL_DIR  # noqa: E402
from fpm.queries import SCENARIO_QUERIES  # noqa: E402
from fpm.settings import VALID_TIMESTAMP_SOURCES  # noqa: E402

DEFAULT_GROUP_PREFIX_DIR = ROOT / "output" / "prefix" / "group"
DEFAULT_GROUP_MODEL_DIR = ROOT / "output" / "models" / "group"
SCRIPTS = ROOT / "scripts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full Phase 3 predictive pipeline: event logs (CSV, keep repeats), "
            "splits, prefix datasets, local + federated + group prediction."
        )
    )
    parser.add_argument(
        "--timestamp-source",
        choices=VALID_TIMESTAMP_SOURCES,
        default="csv",
        help="Event log timestamp source (default: csv for real temporal order)",
    )
    parser.add_argument(
        "--collapse-repeats",
        action="store_true",
        help="Collapse consecutive duplicate activities (default: keep repeats)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=3,
        help="Prefix window size (default: 3)",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default=None,
        help=(
            "Comma-separated scenario keys for group prediction "
            f"(default: all keys in SCENARIO_QUERIES: {', '.join(SCENARIO_QUERIES)})"
        ),
    )
    parser.add_argument(
        "--min-train-traces",
        type=int,
        default=5,
        help="Skip group scenarios with fewer matching train traces (default: 5)",
    )
    parser.add_argument(
        "--skip-event-logs",
        action="store_true",
        help="Skip build_event_logs.py",
    )
    parser.add_argument(
        "--skip-splits",
        action="store_true",
        help="Skip build_splits.py",
    )
    parser.add_argument(
        "--skip-prefix",
        action="store_true",
        help="Skip build_prefix_datasets.py",
    )
    parser.add_argument(
        "--skip-local-models",
        action="store_true",
        help="Skip train_local_models.py",
    )
    parser.add_argument(
        "--skip-federated",
        action="store_true",
        help="Skip run_federated_prediction.py",
    )
    parser.add_argument(
        "--skip-group-prefix",
        action="store_true",
        help="Skip build_group_prefix_datasets.py",
    )
    parser.add_argument(
        "--skip-group-prediction",
        action="store_true",
        help="Skip run_group_prediction.py for all scenarios",
    )
    parser.add_argument(
        "--feature-set",
        choices=VALID_FEATURE_SETS,
        default=FEATURE_SET_BASIC,
        help=(
            "Additional prefix features to emit: basic, temporal, or enhanced "
            "(default: basic)."
        ),
    )
    parser.add_argument(
        "--time-features",
        action="store_true",
        help=(
            "Deprecated alias for --feature-set temporal when --feature-set is basic."
        ),
    )
    return parser.parse_args()


def parse_scenarios(raw: str | None) -> list[str]:
    if raw is None:
        return list(SCENARIO_QUERIES)
    names = [name.strip() for name in raw.split(",") if name.strip()]
    if not names:
        raise ValueError("At least one scenario must be specified")
    unknown = [name for name in names if name not in SCENARIO_QUERIES]
    if unknown:
        known = ", ".join(sorted(SCENARIO_QUERIES))
        raise ValueError(f"Unknown scenarios: {', '.join(unknown)}; choose from {known}")
    return names


def run_step(name: str, cmd: list[str]) -> None:
    print()
    print("=" * 72)
    print(name)
    print("=" * 72)
    print("$", " ".join(cmd))
    result = subprocess.run(cmd, cwd=ROOT, check=False)
    if result.returncode != 0:
        raise SystemExit(f"Step failed ({name}): exit code {result.returncode}")


def python_script(script: str, *args: str) -> list[str]:
    return [sys.executable, str(SCRIPTS / script), *args]


def event_log_args(args: argparse.Namespace) -> list[str]:
    cmd_args = ["--timestamp-source", args.timestamp_source]
    if not args.collapse_repeats:
        cmd_args.append("--no-collapse-repeats")
    return cmd_args


def prefix_args(args: argparse.Namespace) -> list[str]:
    feature_set = resolve_feature_set(
        args.feature_set,
        include_time_features=args.time_features,
    )
    cmd_args = ["--window", str(args.window), "--feature-set", feature_set]
    return cmd_args


def built_group_scenarios(group_prefix_dir: Path, scenarios: list[str]) -> list[str]:
    built: list[str] = []
    for scenario in scenarios:
        train_csv = group_prefix_dir / scenario / "train.csv"
        if train_csv.exists():
            built.append(scenario)
    return built


def print_sowcompact_warning(timestamp_source: str) -> None:
    if timestamp_source == "xes":
        return
    print(
        "Note: Phase 3 uses CSV event logs under output/event_logs/. "
        "SOWCompact Section 7 reproduction (compare_sowcompact.py) requires "
        "XES-mode logs — rebuild with: "
        "python scripts/build_event_logs.py --timestamp-source xes --no-collapse-repeats"
    )
    print()


def print_artifact_summary(
    *,
    local_comparison: Path,
    federated_comparison: Path,
    group_model_dir: Path,
    group_scenarios: list[str],
) -> None:
    print()
    print("=" * 72)
    print("Phase 3 complete — comparison artifacts")
    print("=" * 72)
    print(f"  Local:      {local_comparison}")
    print(f"  Federated:  {federated_comparison}")
    for scenario in group_scenarios:
        path = group_model_dir / scenario / "comparison.csv"
        print(f"  Group:      {path}")


def main() -> None:
    args = parse_args()
    scenarios = parse_scenarios(args.scenarios)

    if not DEFAULT_DATASET_ROOT.exists():
        raise SystemExit(
            f"Dataset not found at {DEFAULT_DATASET_ROOT}. "
            "Place dailylog2016_dataset/ with subjectN/data/activity.xes before running."
        )

    print_sowcompact_warning(args.timestamp_source)

    if not args.skip_event_logs:
        run_step(
            "Step 1/7 — build event logs",
            python_script("build_event_logs.py", *event_log_args(args)),
        )

    if not args.skip_splits:
        run_step("Step 2/7 — build train/val splits", python_script("build_splits.py"))

    if not args.skip_prefix:
        run_step(
            "Step 3/7 — build prefix datasets",
            python_script("build_prefix_datasets.py", *prefix_args(args)),
        )

    if not args.skip_local_models:
        run_step(
            "Step 4/7 — train local models",
            python_script("train_local_models.py"),
        )

    if not args.skip_federated:
        run_step(
            "Step 5/7 — run federated prediction",
            python_script("run_federated_prediction.py"),
        )

    if not args.skip_group_prefix:
        if args.scenarios is None:
            run_step(
                "Step 6/7 — build group prefix datasets (all scenarios)",
                python_script(
                    "build_group_prefix_datasets.py",
                    *prefix_args(args),
                    "--min-train-traces",
                    str(args.min_train_traces),
                ),
            )
        else:
            for index, scenario in enumerate(scenarios, start=1):
                run_step(
                    f"Step 6/7 — build group prefix for {scenario} ({index}/{len(scenarios)})",
                    python_script(
                        "build_group_prefix_datasets.py",
                        "--scenario",
                        scenario,
                        *prefix_args(args),
                        "--min-train-traces",
                        str(args.min_train_traces),
                    ),
                )

    group_scenarios: list[str] = []
    if not args.skip_group_prediction:
        group_scenarios = built_group_scenarios(DEFAULT_GROUP_PREFIX_DIR, scenarios)
        if not group_scenarios:
            print()
            print(
                "Warning: no group prefix datasets found; skipping group prediction. "
                "Run build_group_prefix_datasets.py or lower --min-train-traces."
            )
        for index, scenario in enumerate(group_scenarios, start=1):
            run_step(
                f"Step 7/7 — group prediction for {scenario} ({index}/{len(group_scenarios)})",
                python_script("run_group_prediction.py", "--scenario", scenario),
            )

    print_artifact_summary(
        local_comparison=DEFAULT_MODEL_DIR / "comparison.csv",
        federated_comparison=DEFAULT_FEDERATED_MODEL_DIR / "comparison.csv",
        group_model_dir=DEFAULT_GROUP_MODEL_DIR,
        group_scenarios=group_scenarios,
    )


if __name__ == "__main__":
    main()
