"""Build prefix -> next-activity datasets from event logs."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from fpm.loader import ACTIVITY, ACTIVITY_TAXONOMY, CASE_ID, START_TIMESTAMP, TIMESTAMP

PAD_TOKEN = "<PAD>"
EVENT_INDEX = "@@index"

DEFAULT_PREFIX_DIR = Path(__file__).resolve().parents[1] / "output" / "prefix"

FEATURE_SET_BASIC = "basic"
FEATURE_SET_TEMPORAL = "temporal"
FEATURE_SET_ENHANCED = "enhanced"
VALID_FEATURE_SETS = (
    FEATURE_SET_BASIC,
    FEATURE_SET_TEMPORAL,
    FEATURE_SET_ENHANCED,
)

LEGACY_TIME_FEATURE_COLUMNS = [
    "hour",
    "hour_bin",
    "day_of_week",
    "minutes_since_day_start",
    "minutes_since_prev_event",
]
CYCLIC_TIME_FEATURE_COLUMNS = [
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "is_weekend",
    "minutes_since_midnight",
    "log_minutes_since_prev_event",
]
TIME_FEATURE_COLUMNS = [
    *LEGACY_TIME_FEATURE_COLUMNS,
    *CYCLIC_TIME_FEATURE_COLUMNS,
]
DURATION_FEATURE_COLUMNS = [
    "activity_duration_minutes",
    "log_activity_duration_minutes",
    "gap_since_prev_event_minutes",
    "cumulative_activity_duration_minutes",
    "mean_activity_duration_minutes_so_far",
]
HISTORY_FEATURE_COLUMNS = [
    "current_activity_count_so_far",
    "current_activity_seen_before",
    "unique_activities_so_far",
    "current_activity_run_length",
    "prefix_length_ratio",
]


def resolve_feature_set(
    feature_set: str = FEATURE_SET_BASIC,
    *,
    include_time_features: bool = False,
) -> str:
    """Resolve the feature-set name, preserving the legacy time-features flag."""
    if feature_set not in VALID_FEATURE_SETS:
        raise ValueError(
            f"feature_set must be one of {VALID_FEATURE_SETS}, got {feature_set!r}"
        )
    if include_time_features and feature_set == FEATURE_SET_BASIC:
        return FEATURE_SET_TEMPORAL
    return feature_set


def feature_columns_for_set(feature_set: str) -> list[str]:
    """Return non-prefix feature columns emitted by a feature set."""
    if feature_set == FEATURE_SET_BASIC:
        return []
    if feature_set == FEATURE_SET_TEMPORAL:
        return list(TIME_FEATURE_COLUMNS)
    if feature_set == FEATURE_SET_ENHANCED:
        return [
            *TIME_FEATURE_COLUMNS,
            *DURATION_FEATURE_COLUMNS,
            *HISTORY_FEATURE_COLUMNS,
        ]
    raise ValueError(
        f"feature_set must be one of {VALID_FEATURE_SETS}, got {feature_set!r}"
    )


def hour_bin(hour: int) -> int:
    """Map clock hour to a coarse time-of-day bucket."""
    if hour < 6:
        return 0  # night
    if hour < 12:
        return 1  # morning
    if hour < 18:
        return 2  # afternoon
    return 3  # evening


def iter_traces(log: pd.DataFrame) -> Iterator[tuple[str, list[str]]]:
    """Yield (case_id, activity_sequence) for each trace in event order."""
    if log.empty:
        return

    sort_cols = [CASE_ID, TIMESTAMP]
    if EVENT_INDEX in log.columns:
        sort_cols.append(EVENT_INDEX)

    ordered = log.sort_values(sort_cols, kind="stable")
    for case_id, group in ordered.groupby(CASE_ID, sort=False):
        activities = group[ACTIVITY].astype(str).tolist()
        if activities:
            yield str(case_id), activities


def iter_traces_with_timestamps(
    log: pd.DataFrame,
) -> Iterator[tuple[str, list[str], list[pd.Timestamp]]]:
    """Yield (case_id, activities, timestamps) for each trace in event order."""
    if log.empty:
        return
    if TIMESTAMP not in log.columns:
        raise ValueError(f"Event log is missing timestamp column {TIMESTAMP!r}")

    sort_cols = [CASE_ID, TIMESTAMP]
    if EVENT_INDEX in log.columns:
        sort_cols.append(EVENT_INDEX)

    ordered = log.sort_values(sort_cols, kind="stable")
    for case_id, group in ordered.groupby(CASE_ID, sort=False):
        activities = group[ACTIVITY].astype(str).tolist()
        timestamps = pd.to_datetime(group[TIMESTAMP], errors="coerce").tolist()
        if activities:
            if any(pd.isna(ts) for ts in timestamps):
                raise ValueError(f"Trace {case_id!r} contains invalid timestamps")
            yield str(case_id), activities, timestamps


def iter_traces_with_event_times(
    log: pd.DataFrame,
) -> Iterator[tuple[str, list[str], list[pd.Timestamp], list[pd.Timestamp]]]:
    """Yield traces with end timestamps and optional start timestamps."""
    if log.empty:
        return
    if TIMESTAMP not in log.columns:
        raise ValueError(f"Event log is missing timestamp column {TIMESTAMP!r}")

    sort_cols = [CASE_ID, TIMESTAMP]
    if EVENT_INDEX in log.columns:
        sort_cols.append(EVENT_INDEX)

    ordered = log.sort_values(sort_cols, kind="stable")
    for case_id, group in ordered.groupby(CASE_ID, sort=False):
        activities = group[ACTIVITY].astype(str).tolist()
        timestamps = pd.to_datetime(group[TIMESTAMP], errors="coerce").tolist()
        if START_TIMESTAMP in group.columns:
            start_timestamps = pd.to_datetime(
                group[START_TIMESTAMP],
                errors="coerce",
            ).tolist()
        else:
            start_timestamps = [pd.NaT] * len(timestamps)
        if activities:
            if any(pd.isna(ts) for ts in timestamps):
                raise ValueError(f"Trace {case_id!r} contains invalid timestamps")
            yield str(case_id), activities, timestamps, start_timestamps


def _prefix_at(activities: list[str], position: int, window: int) -> list[str]:
    start = max(0, position - window + 1)
    prefix = activities[start : position + 1]
    if len(prefix) < window:
        prefix = [PAD_TOKEN] * (window - len(prefix)) + prefix
    return prefix


def _time_features_for_position(
    timestamps: list[pd.Timestamp],
    position: int,
) -> dict[str, int | float]:
    """Derive leakage-free timestamp features from the current prefix event.

    ``minutes_since_day_start`` keeps the legacy meaning: elapsed minutes since
    the first event in the trace, not since midnight.
    """
    current = timestamps[position]
    day_start = timestamps[0]
    hour = int(current.hour)
    day_of_week = int(current.dayofweek)
    minutes_since_day_start = (current - day_start).total_seconds() / 60.0
    if position == 0:
        minutes_since_prev_event = 0.0
    else:
        previous = timestamps[position - 1]
        minutes_since_prev_event = (current - previous).total_seconds() / 60.0
    minutes_since_prev_event = max(minutes_since_prev_event, 0.0)
    minutes_since_midnight = (
        current.hour * 60.0
        + current.minute
        + current.second / 60.0
        + current.microsecond / 60_000_000.0
    )
    hour_angle = 2.0 * math.pi * minutes_since_midnight / 1440.0
    day_angle = 2.0 * math.pi * day_of_week / 7.0

    return {
        "hour": hour,
        "hour_bin": hour_bin(hour),
        "day_of_week": day_of_week,
        "minutes_since_day_start": minutes_since_day_start,
        "minutes_since_prev_event": minutes_since_prev_event,
        "hour_sin": math.sin(hour_angle),
        "hour_cos": math.cos(hour_angle),
        "day_of_week_sin": math.sin(day_angle),
        "day_of_week_cos": math.cos(day_angle),
        "is_weekend": int(day_of_week >= 5),
        "minutes_since_midnight": minutes_since_midnight,
        "log_minutes_since_prev_event": math.log1p(minutes_since_prev_event),
    }


def _minutes_between(start: pd.Timestamp, end: pd.Timestamp) -> float:
    if pd.isna(start) or pd.isna(end):
        return 0.0
    return max((end - start).total_seconds() / 60.0, 0.0)


def _duration_features_for_position(
    timestamps: list[pd.Timestamp],
    start_timestamps: list[pd.Timestamp],
    durations: list[float],
    position: int,
) -> dict[str, float]:
    duration = durations[position]
    cumulative = sum(durations[: position + 1])
    if position == 0:
        gap = 0.0
    else:
        gap = _minutes_between(timestamps[position - 1], start_timestamps[position])

    return {
        "activity_duration_minutes": duration,
        "log_activity_duration_minutes": math.log1p(duration),
        "gap_since_prev_event_minutes": gap,
        "cumulative_activity_duration_minutes": cumulative,
        "mean_activity_duration_minutes_so_far": cumulative / (position + 1),
    }


def _current_activity_run_length(activities: list[str], position: int) -> int:
    current = activities[position]
    length = 0
    for index in range(position, -1, -1):
        if activities[index] != current:
            break
        length += 1
    return length


def _history_features_for_position(
    activities: list[str],
    position: int,
    *,
    window: int,
) -> dict[str, int | float]:
    observed = activities[: position + 1]
    current = activities[position]
    current_count = observed.count(current)

    return {
        "current_activity_count_so_far": current_count,
        "current_activity_seen_before": int(current_count > 1),
        "unique_activities_so_far": len(set(observed)),
        "current_activity_run_length": _current_activity_run_length(activities, position),
        "prefix_length_ratio": min(position + 1, window) / window,
    }


def _empty_prefix_columns(*, window: int, feature_set: str) -> list[str]:
    prefix_cols = [f"e{i}" for i in range(window)]
    columns = ["case_id", "position", *prefix_cols, "next_activity"]
    columns.extend(feature_columns_for_set(feature_set))
    return columns


def build_prefix_frame(
    log: pd.DataFrame,
    *,
    window: int = 3,
    include_time_features: bool = False,
    feature_set: str = FEATURE_SET_BASIC,
) -> pd.DataFrame:
    """Extract prefix -> next-activity rows from an event log.

    Each row contains ``case_id``, ``position``, ``e0``..``e{window-1}`` (strings),
    and ``next_activity``. A trace of length L yields L-1 samples.

    ``feature_set`` controls additional leakage-free feature columns. The
    legacy ``include_time_features`` flag maps ``basic`` to ``temporal``.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window!r}")
    resolved_feature_set = resolve_feature_set(
        feature_set,
        include_time_features=include_time_features,
    )

    rows: list[dict[str, Any]] = []
    prefix_cols = [f"e{i}" for i in range(window)]
    if resolved_feature_set == FEATURE_SET_BASIC:
        trace_iter = (
            (case_id, activities, None, None)
            for case_id, activities in iter_traces(log)
        )
    else:
        trace_iter = iter_traces_with_event_times(log)

    for case_id, activities, timestamps, start_timestamps in trace_iter:
        durations = (
            [
                _minutes_between(start_timestamps[index], timestamps[index])
                for index in range(len(activities))
            ]
            if resolved_feature_set == FEATURE_SET_ENHANCED
            else []
        )
        for position in range(len(activities) - 1):
            prefix = _prefix_at(activities, position, window)
            row: dict[str, Any] = {
                "case_id": case_id,
                "position": position,
            }
            for col, activity in zip(prefix_cols, prefix, strict=True):
                row[col] = activity
            row["next_activity"] = activities[position + 1]
            if resolved_feature_set in (FEATURE_SET_TEMPORAL, FEATURE_SET_ENHANCED):
                assert timestamps is not None
                row.update(_time_features_for_position(timestamps, position))
            if resolved_feature_set == FEATURE_SET_ENHANCED:
                assert timestamps is not None
                assert start_timestamps is not None
                row.update(
                    _duration_features_for_position(
                        timestamps,
                        start_timestamps,
                        durations,
                        position,
                    )
                )
                row.update(
                    _history_features_for_position(
                        activities,
                        position,
                        window=window,
                    )
                )
            rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=_empty_prefix_columns(
                window=window,
                feature_set=resolved_feature_set,
            )
        )

    return pd.DataFrame(rows)


def expected_sample_count(log: pd.DataFrame) -> int:
    """Return the number of prefix samples a log should produce."""
    return sum(max(len(activities) - 1, 0) for _, activities in iter_traces(log))


def validate_prefix_frame(frame: pd.DataFrame, log: pd.DataFrame, *, window: int) -> None:
    """Light sanity checks on a prefix frame."""
    expected = expected_sample_count(log)
    if len(frame) != expected:
        raise ValueError(
            f"Expected {expected} prefix samples, got {len(frame)}"
        )

    prefix_cols = [f"e{i}" for i in range(window)]
    for col in prefix_cols:
        if col not in frame.columns:
            raise ValueError(f"Missing prefix column {col!r}")


@dataclass
class Vocabulary:
    """Activity name <-> integer id mapping. Index 0 is always PAD_TOKEN."""

    activities: list[str]

    def __post_init__(self) -> None:
        if not self.activities or self.activities[0] != PAD_TOKEN:
            raise ValueError(f"Vocabulary must start with {PAD_TOKEN!r}")

    @classmethod
    def from_logs(cls, *logs: pd.DataFrame) -> Vocabulary:
        names: set[str] = set()
        for log in logs:
            if log.empty:
                continue
            names.update(log[ACTIVITY].astype(str).unique())
        names.discard(PAD_TOKEN)
        return cls([PAD_TOKEN, *sorted(names)])

    @classmethod
    def canonical(cls) -> Vocabulary:
        """Build the shared vocabulary from the declared activity taxonomy.

        Unlike :meth:`from_logs`, this does not depend on the contents of any
        train/validation split, so it cannot leak validation-only activities
        into the encoding and yields the same integer ids for every scope.
        """
        return cls([PAD_TOKEN, *ACTIVITY_TAXONOMY])

    def covers(self, log: pd.DataFrame) -> set[str]:
        """Return activities in ``log`` that are absent from this vocabulary."""
        if log.empty:
            return set()
        names = set(log[ACTIVITY].astype(str).unique())
        names.discard(PAD_TOKEN)
        return names - set(self.activities)

    @property
    def size(self) -> int:
        return len(self.activities)

    def encode(self, name: str) -> int:
        try:
            return self.activities.index(name)
        except ValueError as exc:
            raise KeyError(f"Unknown activity {name!r}") from exc

    def decode(self, index: int) -> str:
        return self.activities[index]

    def to_dict(self) -> dict[str, Any]:
        return {"activities": self.activities}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Vocabulary:
        return cls(list(data["activities"]))

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def read_json(cls, path: Path) -> Vocabulary:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def encode_frame(
    frame: pd.DataFrame,
    vocab: Vocabulary,
    *,
    window: int = 3,
    include_time_features: bool = False,
    feature_set: str = FEATURE_SET_BASIC,
) -> pd.DataFrame:
    """Label-encode prefix and target columns using ``vocab``."""
    resolved_feature_set = resolve_feature_set(
        feature_set,
        include_time_features=include_time_features,
    )
    if frame.empty:
        return pd.DataFrame(
            columns=_empty_prefix_columns(
                window=window,
                feature_set=resolved_feature_set,
            )
        )

    prefix_cols = [f"e{i}" for i in range(window)]
    encoded = frame.copy()
    for col in [*prefix_cols, "next_activity"]:
        encoded[col] = encoded[col].map(vocab.encode)
    return encoded


def prefix_manifest(
    *,
    scope: str,
    window: int,
    train_samples: int,
    val_samples: int,
    n_activities: int,
    time_features: bool = False,
    feature_set: str = FEATURE_SET_BASIC,
) -> dict[str, Any]:
    resolved_feature_set = resolve_feature_set(
        feature_set,
        include_time_features=time_features,
    )
    feature_columns = feature_columns_for_set(resolved_feature_set)
    temporal_columns = (
        list(TIME_FEATURE_COLUMNS)
        if resolved_feature_set in (FEATURE_SET_TEMPORAL, FEATURE_SET_ENHANCED)
        else []
    )
    duration_columns = (
        list(DURATION_FEATURE_COLUMNS)
        if resolved_feature_set == FEATURE_SET_ENHANCED
        else []
    )
    history_columns = (
        list(HISTORY_FEATURE_COLUMNS)
        if resolved_feature_set == FEATURE_SET_ENHANCED
        else []
    )
    payload: dict[str, Any] = {
        "scope": scope,
        "window": window,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "n_activities": n_activities,
        "feature_set": resolved_feature_set,
        "feature_columns": feature_columns,
        "temporal_feature_columns": temporal_columns,
        "duration_feature_columns": duration_columns,
        "history_feature_columns": history_columns,
    }
    if resolved_feature_set in (FEATURE_SET_TEMPORAL, FEATURE_SET_ENHANCED):
        payload["time_features"] = True
        payload["time_feature_columns"] = list(TIME_FEATURE_COLUMNS)
    return payload
