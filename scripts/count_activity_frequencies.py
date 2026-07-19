#!/usr/bin/env python3
"""Count how often each sensor/activity appears in the CASAS Chinook dataset.

Loads every trial CSV without applying the project's sensor filter, then reports:
  1. Raw row counts per sensor ID (one CSV row = one event)
  2. Sensor-level activity counts (states collapsed, consecutive duplicates removed)

Use the percentages to justify excluding rare sensors (see shared/sensor_filter.py).
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.event_abstraction import abstract_trace  # noqa: E402
from shared.sensor_filter import EXCLUDED_SENSORS, normalize_sensor_id  # noqa: E402

_FILE_RE = re.compile(r"^(p\d+)\.t(\d+)\.csv$")
DEFAULT_DATA_DIR = ROOT / "data"
RARE_THRESHOLD_PCT = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count sensor/activity frequencies across the full dataset"
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--include-errors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include adl_error trials (default: yes)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path to write the frequency table as CSV",
    )
    return parser.parse_args()


def iter_csv_paths(data_dir: Path, *, include_errors: bool) -> list[Path]:
    folders = ["adl_noerror"]
    if include_errors:
        folders.append("adl_error")

    paths: list[Path] = []
    for folder in folders:
        folder_path = data_dir / folder
        if not folder_path.exists():
            continue
        paths.extend(sorted(folder_path.glob("p*.t*.csv")))
    return paths


def load_unfiltered_traces(data_dir: Path, *, include_errors: bool) -> list[list[str]]:
    """Return one list of sensor IDs per trial (no filtering)."""
    import pandas as pd

    traces: list[list[str]] = []
    for csv_path in iter_csv_paths(data_dir, include_errors=include_errors):
        if not _FILE_RE.match(csv_path.name):
            continue
        frame = pd.read_csv(csv_path)
        frame = frame.assign(
            timestamp=pd.to_datetime(frame["date"] + " " + frame["time"], format="mixed")
        )
        frame = frame.sort_values("timestamp", kind="stable")
        sensors = [normalize_sensor_id(sensor) for sensor in frame["sensor"]]
        traces.append(sensors)
    if not traces:
        raise FileNotFoundError(f"No trial CSVs found under {data_dir}")
    return traces


def sensor_status(
    sensor: str,
    share_pct: float,
    *,
    excluded: frozenset[str],
    apply_rare_threshold: bool,
) -> str:
    if sensor in excluded:
        return "EXCLUDED (project filter)"
    if sensor.startswith("AD1"):
        return "EXCLUDED (analog, default)"
    if apply_rare_threshold and share_pct < RARE_THRESHOLD_PCT:
        return f"rare (<{RARE_THRESHOLD_PCT}%)"
    return "kept"


def format_table(
    counts: Counter[str],
    *,
    title: str,
    excluded: frozenset[str],
    apply_rare_threshold: bool,
) -> str:
    total = sum(counts.values())
    lines = [
        title,
        f"Total events: {total:,}  |  Unique sensors/activities: {len(counts)}",
        "",
        f"{'Sensor':<12} {'Count':>8} {'Share %':>10} {'Status':<28}",
        "-" * 62,
    ]
    for sensor, count in counts.most_common():
        share = 100.0 * count / total if total else 0.0
        status = sensor_status(
            sensor,
            share,
            excluded=excluded,
            apply_rare_threshold=apply_rare_threshold,
        )
        lines.append(f"{sensor:<12} {count:>8,} {share:>9.2f}% {status:<28}")

    excluded_count = sum(
        count
        for sensor, count in counts.items()
        if sensor in excluded or sensor.startswith("AD1")
    )
    kept_count = total - excluded_count
    lines.extend(
        [
            "-" * 62,
            f"{'Kept':<12} {kept_count:>8,} {100.0 * kept_count / total if total else 0:>9.2f}%",
            f"{'Excluded':<12} {excluded_count:>8,} {100.0 * excluded_count / total if total else 0:>9.2f}%",
            "",
        ]
    )
    return "\n".join(lines)


def to_rows(
    counts: Counter[str],
    *,
    view: str,
    excluded: frozenset[str],
    apply_rare_threshold: bool,
) -> list[dict[str, object]]:
    total = sum(counts.values())
    rows: list[dict[str, object]] = []
    for sensor, count in counts.most_common():
        share = 100.0 * count / total if total else 0.0
        status = sensor_status(
            sensor,
            share,
            excluded=excluded,
            apply_rare_threshold=apply_rare_threshold,
        )
        rows.append(
            {
                "view": view,
                "sensor": sensor,
                "count": count,
                "share_pct": round(share, 4),
                "status": status,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    traces = load_unfiltered_traces(args.data_dir, include_errors=args.include_errors)

    raw_counts: Counter[str] = Counter()
    sensor_counts: Counter[str] = Counter()
    for sensors in traces:
        raw_counts.update(sensors)
        # Treat each sensor ID as the activity label (same as the project's sensor view).
        sensor_counts.update(abstract_trace(sensors, collapse_consecutive=True))

    n_files = len(traces)
    print(
        f"Dataset: {args.data_dir}\n"
        f"Trials loaded: {n_files}\n"
        f"Include adl_error: {args.include_errors}\n"
        f"Rare threshold used in docs: < {RARE_THRESHOLD_PCT}% of sensor-level labels\n"
        f"Project EXCLUDED_SENSORS: {sorted(EXCLUDED_SENSORS)}\n"
    )
    print(
        format_table(
            raw_counts,
            title="=== Raw CSV rows per sensor (unfiltered) ===",
            excluded=EXCLUDED_SENSORS,
            apply_rare_threshold=False,
        )
    )
    print(
        format_table(
            sensor_counts,
            title="=== Sensor-level activities (collapsed visits, unfiltered) ===",
            excluded=EXCLUDED_SENSORS,
            apply_rare_threshold=True,
        )
    )

    rare = [
        (sensor, count, 100.0 * count / sum(sensor_counts.values()))
        for sensor, count in sensor_counts.most_common()
        if (100.0 * count / sum(sensor_counts.values())) < RARE_THRESHOLD_PCT
        and not sensor.startswith("AD1")
    ]
    if rare:
        print(
            f"Sensors below {RARE_THRESHOLD_PCT}% at sensor level "
            "(justification for exclusion if listed in EXCLUDED_SENSORS):"
        )
        for sensor, count, share in rare:
            marker = " ← excluded" if sensor in EXCLUDED_SENSORS else ""
            print(f"  {sensor}: {count:,} ({share:.3f}%){marker}")

    if args.csv is not None:
        import pandas as pd

        rows = to_rows(
            raw_counts,
            view="raw",
            excluded=EXCLUDED_SENSORS,
            apply_rare_threshold=False,
        )
        rows.extend(
            to_rows(
                sensor_counts,
                view="sensor",
                excluded=EXCLUDED_SENSORS,
                apply_rare_threshold=True,
            )
        )
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.csv, index=False)
        print(f"\nWrote frequency table to {args.csv}")


if __name__ == "__main__":
    main()
