"""Build, persist, and load per-subject event logs for the FPM pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pm4py

from fpm.loader import (
    ACTIVITY,
    CASE_ID,
    DEFAULT_DATASET_ROOT,
    SUBJECT_IDS,
    TIMESTAMP,
    collapse_consecutive_activities,
    load_subject_csv_log,
    load_subject_log,
    subject_csv_path,
    subject_xes_path,
)
from fpm.settings import TIMESTAMP_SOURCE, VALID_TIMESTAMP_SOURCES

DEFAULT_EVENT_LOG_DIR = Path(__file__).resolve().parents[1] / "output" / "event_logs"


def subject_event_log_dir(event_log_dir: Path, subject_id: int) -> Path:
    return event_log_dir / f"subject{subject_id}"


def subject_event_log_xes_path(event_log_dir: Path, subject_id: int) -> Path:
    return subject_event_log_dir(event_log_dir, subject_id) / "event_log.xes"


def subject_event_log_csv_path(event_log_dir: Path, subject_id: int) -> Path:
    return subject_event_log_dir(event_log_dir, subject_id) / "event_log.csv"


def build_subject_event_log(
    dataset_root: Path,
    subject_id: int,
    *,
    collapse_repeats: bool = True,
    timestamp_source: str = TIMESTAMP_SOURCE,
) -> pd.DataFrame:
    """Build a pm4py event log for one subject.

    ``timestamp_source`` selects where event times come from:

    - ``"xes"`` (default): synthetic order timestamps from ``activity.xes``,
      with ``caseN`` case ids. This is the SOWCompact reproduction path.
    - ``"csv"``: real ``attr_endtime`` timestamps from ``activity.csv``, with
      ``dayN`` case ids. The CSV files contain a few records that are not
      present in the XES logs, so this path is *not* metric-compatible with the
      SOWCompact Section 7 reference values — use it for predictive/temporal
      work only.
    """
    if subject_id not in SUBJECT_IDS:
        raise ValueError(f"subject_id must be one of {SUBJECT_IDS}, got {subject_id!r}")
    if timestamp_source not in VALID_TIMESTAMP_SOURCES:
        raise ValueError(
            f"timestamp_source must be one of {VALID_TIMESTAMP_SOURCES}, "
            f"got {timestamp_source!r}"
        )

    if timestamp_source == "csv":
        log = load_subject_csv_log(subject_csv_path(dataset_root, subject_id))
    else:
        log = load_subject_log(subject_xes_path(dataset_root, subject_id))

    if collapse_repeats:
        log = collapse_consecutive_activities(log)
    return log


def event_log_stats(log: pd.DataFrame, subject_id: int) -> dict[str, Any]:
    activities = pm4py.get_event_attribute_values(log, ACTIVITY)
    return {
        "subject_id": subject_id,
        "subject_label": f"subject{subject_id}",
        "traces": len(pm4py.get_event_attribute_values(log, CASE_ID)),
        "events": len(log),
        "activities": sorted(activities),
        "activity_counts": dict(sorted(activities.items())),
    }


def write_subject_event_log(
    log: pd.DataFrame,
    event_log_dir: Path,
    subject_id: int,
) -> dict[str, Path]:
    """Persist one subject's event log as XES and CSV."""
    subject_dir = subject_event_log_dir(event_log_dir, subject_id)
    subject_dir.mkdir(parents=True, exist_ok=True)

    xes_path = subject_event_log_xes_path(event_log_dir, subject_id)
    csv_path = subject_event_log_csv_path(event_log_dir, subject_id)
    stats_path = subject_dir / "log_stats.json"

    pm4py.write_xes(log, str(xes_path))
    log.to_csv(csv_path, index=False)
    stats_path.write_text(
        json.dumps(event_log_stats(log, subject_id), indent=2),
        encoding="utf-8",
    )

    return {"xes": xes_path, "csv": csv_path, "stats": stats_path}


def load_event_log(xes_path: Path):
    """Load a previously generated event log file."""
    if not xes_path.exists():
        raise FileNotFoundError(
            f"Event log not found: {xes_path}. "
            "Run scripts/build_event_logs.py first."
        )

    log = pm4py.read_xes(str(xes_path))
    if not isinstance(log, pd.DataFrame):
        log = pm4py.convert_to_dataframe(log)

    log[TIMESTAMP] = pd.to_datetime(log[TIMESTAMP], errors="coerce")
    if log[TIMESTAMP].isna().any():
        raise ValueError(f"Event log {xes_path} is missing valid timestamps.")

    return pm4py.format_dataframe(
        log,
        case_id=CASE_ID,
        activity_key=ACTIVITY,
        timestamp_key=TIMESTAMP,
    )
