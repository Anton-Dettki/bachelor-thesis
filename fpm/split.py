"""Temporal trace-level train/validation splits for predictive process monitoring."""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pm4py

from fpm.aggregator import namespace_filtered_log
from fpm.event_log import (
    DEFAULT_EVENT_LOG_DIR,
    load_event_log,
    subject_event_log_xes_path,
)
from fpm.loader import ACTIVITY, CASE_ID, SUBJECT_IDS, TIMESTAMP

logger = logging.getLogger(__name__)

DEFAULT_SPLIT_DIR = Path(__file__).resolve().parents[1] / "output" / "splits"


@dataclass
class SplitResult:
    train_log: pd.DataFrame
    val_log: pd.DataFrame
    train_case_ids: list[str]
    val_case_ids: list[str]
    val_fraction: float
    total_traces: int


CASE_INDEX = "@@case_index"


def _case_sort_key(case_id: str) -> tuple[int, str]:
    """Natural sort key for case ids like case0, case10, day3."""
    text = str(case_id)
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        return (int(digits), text)
    return (0, text)


def _ordered_case_ids(log: pd.DataFrame) -> list[str]:
    """Return case ids in temporal trace order.

    Trace ordering depends on whether the log carries real timestamps:

    - **Real timestamps** (CSV source): traces start at distinct wall-clock
      times, so they are ordered by their first-event timestamp (with a natural
      numeric case-id tiebreak). This is detected by finding more than one
      distinct first-event timestamp across traces.
    - **Synthetic timestamps** (XES source): every trace starts at the same
      fixed epoch, so ``@@case_index`` (pm4py XES import order, which reflects
      chronological day order) is used instead.
    """
    if log.empty:
        return []

    first_events = (
        log.sort_values([CASE_ID, TIMESTAMP])
        .groupby(CASE_ID, sort=False)
        .first()
        .reset_index()
    )

    distinct_first_timestamps = first_events[TIMESTAMP].nunique(dropna=True)
    if distinct_first_timestamps > 1:
        first_events["_case_sort"] = first_events[CASE_ID].astype(str).map(_case_sort_key)
        ordered = first_events.sort_values(
            [TIMESTAMP, "_case_sort", CASE_ID], kind="stable"
        )
        return ordered[CASE_ID].astype(str).tolist()

    if CASE_INDEX in log.columns:
        trace_order = (
            log.groupby(CASE_ID, sort=False)[CASE_INDEX]
            .min()
            .reset_index()
            .sort_values([CASE_INDEX, CASE_ID], kind="stable")
        )
        return trace_order[CASE_ID].astype(str).tolist()

    first_events["_case_sort"] = first_events[CASE_ID].astype(str).map(_case_sort_key)
    ordered = first_events.sort_values([TIMESTAMP, "_case_sort", CASE_ID], kind="stable")
    return ordered[CASE_ID].astype(str).tolist()


def _filter_log_by_cases(log: pd.DataFrame, case_ids: list[str]) -> pd.DataFrame:
    if not case_ids:
        return log.iloc[0:0].copy()
    allowed = set(case_ids)
    filtered = log[log[CASE_ID].astype(str).isin(allowed)].copy()
    return filtered.sort_values([CASE_ID, TIMESTAMP]).reset_index(drop=True)


def _validation_trace_count(
    n: int,
    *,
    val_fraction: float,
    min_val_traces: int,
) -> int:
    if n <= 1:
        return 0

    val_count = round(n * val_fraction)
    val_count = max(val_count, min_val_traces)
    return min(val_count, n - 1)


def temporal_trace_split(
    log: pd.DataFrame,
    *,
    val_fraction: float = 0.25,
    min_val_traces: int = 1,
) -> SplitResult:
    """Split an event log into train/validation by temporal trace order.

    Traces are ordered by the timestamp of their first event. The most recent
    ``val_fraction`` of traces (at least ``min_val_traces`` when possible) are
    held out for validation; the remainder is used for training.

    Subjects with a single trace are assigned entirely to training and produce
    an empty validation log.
    """
    if val_fraction <= 0 or val_fraction >= 1:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction!r}")

    ordered_cases = _ordered_case_ids(log)
    n = len(ordered_cases)
    val_count = _validation_trace_count(
        n,
        val_fraction=val_fraction,
        min_val_traces=min_val_traces,
    )

    if n == 1 and val_count == 0:
        warnings.warn(
            f"Only one trace in log; assigning all data to train with empty validation.",
            stacklevel=2,
        )

    val_case_ids = ordered_cases[n - val_count :] if val_count else []
    train_case_ids = ordered_cases[: n - val_count]

    train_log = _filter_log_by_cases(log, train_case_ids)
    val_log = _filter_log_by_cases(log, val_case_ids)

    return SplitResult(
        train_log=train_log,
        val_log=val_log,
        train_case_ids=train_case_ids,
        val_case_ids=val_case_ids,
        val_fraction=val_fraction,
        total_traces=n,
    )


def _trace_time_bounds(log: pd.DataFrame, case_ids: list[str]) -> dict[str, Any]:
    if not case_ids:
        return {"first_timestamp": None, "last_timestamp": None}

    subset = _filter_log_by_cases(log, case_ids)
    timestamps = pd.to_datetime(subset[TIMESTAMP], errors="coerce")
    return {
        "first_timestamp": timestamps.min().isoformat() if not timestamps.empty else None,
        "last_timestamp": timestamps.max().isoformat() if not timestamps.empty else None,
    }


def split_manifest(
    result: SplitResult,
    *,
    subject_id: int | None = None,
    subject_label: str | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "val_fraction": result.val_fraction,
        "total_traces": result.total_traces,
        "train_traces": len(result.train_case_ids),
        "val_traces": len(result.val_case_ids),
        "train_events": len(result.train_log),
        "val_events": len(result.val_log),
        "train_case_ids": result.train_case_ids,
        "val_case_ids": result.val_case_ids,
        "train_time_bounds": _trace_time_bounds(result.train_log, result.train_case_ids),
        "val_time_bounds": _trace_time_bounds(result.val_log, result.val_case_ids),
    }
    if subject_id is not None:
        manifest["subject_id"] = subject_id
    if subject_label is not None:
        manifest["subject_label"] = subject_label
    return manifest


def subject_split(
    subject_id: int,
    *,
    event_log_dir: Path = DEFAULT_EVENT_LOG_DIR,
    val_fraction: float = 0.25,
    min_val_traces: int = 1,
) -> SplitResult:
    """Load one subject's event log and apply a temporal train/validation split."""
    if subject_id not in SUBJECT_IDS:
        raise ValueError(f"subject_id must be one of {SUBJECT_IDS}, got {subject_id!r}")

    log = load_event_log(subject_event_log_xes_path(event_log_dir, subject_id))
    return temporal_trace_split(
        log,
        val_fraction=val_fraction,
        min_val_traces=min_val_traces,
    )


def build_global_split(
    subject_results: dict[int, SplitResult],
) -> SplitResult:
    """Pool per-subject train/validation logs into global train/validation logs."""
    if not subject_results:
        raise ValueError("subject_results must contain at least one subject split.")

    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    train_case_ids: list[str] = []
    val_case_ids: list[str] = []
    total_traces = 0
    val_fraction: float | None = None

    for subject_id in sorted(subject_results):
        result = subject_results[subject_id]
        subject_label = f"subject{subject_id}"
        total_traces += result.total_traces
        if val_fraction is None:
            val_fraction = result.val_fraction
        elif val_fraction != result.val_fraction:
            raise ValueError("All subject splits must use the same val_fraction.")

        if not result.train_log.empty:
            namespaced_train = namespace_filtered_log(result.train_log, subject_label)
            train_parts.append(namespaced_train)
            train_case_ids.extend(namespaced_train[CASE_ID].astype(str).unique().tolist())

        if not result.val_log.empty:
            namespaced_val = namespace_filtered_log(result.val_log, subject_label)
            val_parts.append(namespaced_val)
            val_case_ids.extend(namespaced_val[CASE_ID].astype(str).unique().tolist())

    train_log = (
        pd.concat(train_parts, ignore_index=True)
        if train_parts
        else pm4py.format_dataframe(
            pd.DataFrame(columns=[CASE_ID, ACTIVITY, TIMESTAMP]),
            case_id=CASE_ID,
            activity_key=ACTIVITY,
            timestamp_key=TIMESTAMP,
        )
    )
    val_log = (
        pd.concat(val_parts, ignore_index=True)
        if val_parts
        else pm4py.format_dataframe(
            pd.DataFrame(columns=[CASE_ID, ACTIVITY, TIMESTAMP]),
            case_id=CASE_ID,
            activity_key=ACTIVITY,
            timestamp_key=TIMESTAMP,
        )
    )

    train_log = pm4py.format_dataframe(
        train_log,
        case_id=CASE_ID,
        activity_key=ACTIVITY,
        timestamp_key=TIMESTAMP,
    )
    val_log = pm4py.format_dataframe(
        val_log,
        case_id=CASE_ID,
        activity_key=ACTIVITY,
        timestamp_key=TIMESTAMP,
    )

    return SplitResult(
        train_log=train_log,
        val_log=val_log,
        train_case_ids=train_case_ids,
        val_case_ids=val_case_ids,
        val_fraction=val_fraction if val_fraction is not None else 0.25,
        total_traces=total_traces,
    )


def validate_split(result: SplitResult, *, log: pd.DataFrame | None = None) -> None:
    """Assert split integrity: disjoint, exhaustive, and temporally ordered."""
    train_set = set(result.train_case_ids)
    val_set = set(result.val_case_ids)

    if train_set & val_set:
        raise ValueError(
            f"Train and validation case ids overlap: {sorted(train_set & val_set)}"
        )

    if log is not None:
        all_cases = set(_ordered_case_ids(log))
        union = train_set | val_set
        if union != all_cases:
            missing = sorted(all_cases - union)
            extra = sorted(union - all_cases)
            raise ValueError(
                f"Split case ids do not match full log "
                f"(missing={missing}, extra={extra})"
            )

        if result.train_case_ids and result.val_case_ids:
            ordered = _ordered_case_ids(log)
            position = {case_id: index for index, case_id in enumerate(ordered)}
            train_positions = [position[case_id] for case_id in result.train_case_ids]
            val_positions = [position[case_id] for case_id in result.val_case_ids]
            if max(train_positions) >= min(val_positions):
                raise ValueError(
                    "Validation traces must come after training traces in temporal order "
                    f"(max_train_pos={max(train_positions)}, min_val_pos={min(val_positions)})"
                )


def subject_split_dir(split_dir: Path, subject_id: int) -> Path:
    return split_dir / f"subject{subject_id}"


def global_split_dir(split_dir: Path) -> Path:
    return split_dir / "global"


def write_split(
    result: SplitResult,
    output_dir: Path,
    *,
    subject_id: int | None = None,
    subject_label: str | None = None,
    source_log: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Persist train/validation splits as XES, CSV, and a manifest JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    validate_split(result, log=source_log)

    train_xes = output_dir / "train.xes"
    val_xes = output_dir / "val.xes"
    train_csv = output_dir / "train.csv"
    val_csv = output_dir / "val.csv"
    manifest_path = output_dir / "split_manifest.json"

    pm4py.write_xes(result.train_log, str(train_xes))
    pm4py.write_xes(result.val_log, str(val_xes))
    result.train_log.to_csv(train_csv, index=False)
    result.val_log.to_csv(val_csv, index=False)
    manifest_path.write_text(
        json.dumps(
            split_manifest(
                result,
                subject_id=subject_id,
                subject_label=subject_label,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "train_xes": train_xes,
        "val_xes": val_xes,
        "train_csv": train_csv,
        "val_csv": val_csv,
        "manifest": manifest_path,
    }
