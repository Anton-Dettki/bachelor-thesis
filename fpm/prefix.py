"""Build prefix -> next-activity datasets from event logs."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from fpm.loader import ACTIVITY, ACTIVITY_TAXONOMY, CASE_ID, START_TIMESTAMP, TIMESTAMP

PAD_TOKEN = "<PAD>"
EVENT_INDEX = "@@index"
RESOURCE = "org:resource"
SUBJECT_ID_RE = re.compile(r"subject(\d+)")
CASE_ID_NUMBER_RE = re.compile(r"(?:day|case)(\d+)")

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
    "log_minutes_since_day_start",
    "log_minutes_since_prev_event",
    "month",
    "day_of_month",
    "week_of_year",
    "month_sin",
    "month_cos",
    "day_of_month_sin",
    "day_of_month_cos",
    "is_month_start",
    "is_month_end",
    "trace_start_hour",
    "trace_start_hour_sin",
    "trace_start_hour_cos",
    "trace_start_minutes_since_midnight",
]
TIME_FEATURE_COLUMNS = [
    *LEGACY_TIME_FEATURE_COLUMNS,
    *CYCLIC_TIME_FEATURE_COLUMNS,
]
DURATION_FEATURE_COLUMNS = [
    "activity_duration_minutes",
    "log_activity_duration_minutes",
    "previous_activity_duration_minutes",
    "log_previous_activity_duration_minutes",
    "gap_since_prev_event_minutes",
    "cumulative_activity_duration_minutes",
    "mean_activity_duration_minutes_so_far",
]
HISTORY_FEATURE_COLUMNS = [
    "events_seen_so_far",
    "log_events_seen_so_far",
    "current_activity_count_so_far",
    "current_activity_seen_before",
    "current_activity_frequency_so_far",
    "unique_activities_so_far",
    "unique_activity_ratio_so_far",
    "dominant_activity_count_so_far",
    "dominant_activity_ratio_so_far",
    "activity_repetition_count_so_far",
    "current_activity_run_length",
    "prefix_length_ratio",
]
RECENCY_FEATURE_COLUMNS = [
    "events_since_last_same_activity",
    "minutes_since_last_same_activity",
    "log_minutes_since_last_same_activity",
]
TRANSITION_FEATURE_COLUMNS = [
    "same_as_previous_activity",
    "activity_switch_count_so_far",
    "activity_switch_ratio_so_far",
    "window_unique_activities",
    "window_switch_count",
    "window_switch_ratio",
    "window_repetition_count",
    "window_repetition_ratio",
]
CONTEXT_FEATURE_COLUMNS = [
    "subject_id",
    "case_id_number",
    "log_case_id_number",
]

DEFAULT_FEATURE_SET = FEATURE_SET_ENHANCED


def resolve_feature_set(
    feature_set: str = FEATURE_SET_BASIC,
    *,
    include_time_features: bool = False,
) -> str:
    """Resolve the internal feature group, preserving legacy API behavior."""
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
            *RECENCY_FEATURE_COLUMNS,
            *TRANSITION_FEATURE_COLUMNS,
            *CONTEXT_FEATURE_COLUMNS,
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
) -> Iterator[tuple[str, int, list[str], list[pd.Timestamp], list[pd.Timestamp]]]:
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
            yield (
                str(case_id),
                _subject_id_for_trace(case_id, group),
                activities,
                timestamps,
                start_timestamps,
            )


def _parse_subject_id(value: Any) -> int:
    match = SUBJECT_ID_RE.search(str(value))
    return int(match.group(1)) if match else 0


def _parse_case_id_number(value: Any) -> int:
    match = CASE_ID_NUMBER_RE.search(str(value))
    return int(match.group(1)) if match else 0


def _subject_id_for_trace(case_id: Any, group: pd.DataFrame) -> int:
    if RESOURCE in group.columns:
        resources = group[RESOURCE].dropna().astype(str)
        for resource in resources:
            subject_id = _parse_subject_id(resource)
            if subject_id:
                return subject_id
    return _parse_subject_id(case_id)


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
    trace_start = timestamps[0]
    day_start = timestamps[0]
    hour = int(current.hour)
    day_of_week = int(current.dayofweek)
    month = int(current.month)
    day_of_month = int(current.day)
    week_of_year = int(current.isocalendar().week)
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
    month_angle = 2.0 * math.pi * (month - 1) / 12.0
    day_of_month_angle = 2.0 * math.pi * (day_of_month - 1) / 31.0
    trace_start_minutes_since_midnight = (
        trace_start.hour * 60.0
        + trace_start.minute
        + trace_start.second / 60.0
        + trace_start.microsecond / 60_000_000.0
    )
    trace_start_angle = 2.0 * math.pi * trace_start_minutes_since_midnight / 1440.0

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
        "log_minutes_since_day_start": math.log1p(max(minutes_since_day_start, 0.0)),
        "log_minutes_since_prev_event": math.log1p(minutes_since_prev_event),
        "month": month,
        "day_of_month": day_of_month,
        "week_of_year": week_of_year,
        "month_sin": math.sin(month_angle),
        "month_cos": math.cos(month_angle),
        "day_of_month_sin": math.sin(day_of_month_angle),
        "day_of_month_cos": math.cos(day_of_month_angle),
        "is_month_start": int(current.is_month_start),
        "is_month_end": int(current.is_month_end),
        "trace_start_hour": int(trace_start.hour),
        "trace_start_hour_sin": math.sin(trace_start_angle),
        "trace_start_hour_cos": math.cos(trace_start_angle),
        "trace_start_minutes_since_midnight": trace_start_minutes_since_midnight,
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
    previous_duration = durations[position - 1] if position > 0 else 0.0
    cumulative = sum(durations[: position + 1])
    if position == 0:
        gap = 0.0
    else:
        gap = _minutes_between(timestamps[position - 1], start_timestamps[position])

    return {
        "activity_duration_minutes": duration,
        "log_activity_duration_minutes": math.log1p(duration),
        "previous_activity_duration_minutes": previous_duration,
        "log_previous_activity_duration_minutes": math.log1p(previous_duration),
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
    events_seen = position + 1
    unique_count = len(set(observed))
    dominant_count = max(observed.count(activity) for activity in set(observed))

    return {
        "events_seen_so_far": events_seen,
        "log_events_seen_so_far": math.log1p(events_seen),
        "current_activity_count_so_far": current_count,
        "current_activity_seen_before": int(current_count > 1),
        "current_activity_frequency_so_far": current_count / events_seen,
        "unique_activities_so_far": unique_count,
        "unique_activity_ratio_so_far": unique_count / events_seen,
        "dominant_activity_count_so_far": dominant_count,
        "dominant_activity_ratio_so_far": dominant_count / events_seen,
        "activity_repetition_count_so_far": events_seen - unique_count,
        "current_activity_run_length": _current_activity_run_length(activities, position),
        "prefix_length_ratio": min(position + 1, window) / window,
    }


def _recency_features_for_position(
    activities: list[str],
    timestamps: list[pd.Timestamp],
    position: int,
) -> dict[str, int | float]:
    current = activities[position]
    previous_same_index: int | None = None
    for index in range(position - 1, -1, -1):
        if activities[index] == current:
            previous_same_index = index
            break

    if previous_same_index is None:
        events_since = 0
        minutes_since = 0.0
    else:
        events_since = position - previous_same_index
        minutes_since = _minutes_between(
            timestamps[previous_same_index],
            timestamps[position],
        )

    return {
        "events_since_last_same_activity": events_since,
        "minutes_since_last_same_activity": minutes_since,
        "log_minutes_since_last_same_activity": math.log1p(minutes_since),
    }


def _switch_count(values: list[str]) -> int:
    return sum(
        left != right
        for left, right in zip(values, values[1:], strict=False)
    )


def _transition_features_for_position(
    activities: list[str],
    position: int,
    *,
    window: int,
) -> dict[str, int | float]:
    observed = activities[: position + 1]
    recent = activities[max(0, position - window + 1) : position + 1]
    switch_count = _switch_count(observed)
    window_switch_count = _switch_count(recent)
    window_denominator = max(len(recent) - 1, 1)
    window_unique = len(set(recent))

    return {
        "same_as_previous_activity": int(
            position > 0 and activities[position] == activities[position - 1]
        ),
        "activity_switch_count_so_far": switch_count,
        "activity_switch_ratio_so_far": switch_count / position if position > 0 else 0.0,
        "window_unique_activities": window_unique,
        "window_switch_count": window_switch_count,
        "window_switch_ratio": window_switch_count / window_denominator,
        "window_repetition_count": len(recent) - window_unique,
        "window_repetition_ratio": (len(recent) - window_unique) / len(recent),
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
    feature_set: str = DEFAULT_FEATURE_SET,
    subject_id: int | None = None,
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
            (case_id, 0, activities, None, None)
            for case_id, activities in iter_traces(log)
        )
    else:
        trace_iter = iter_traces_with_event_times(log)

    for case_id, trace_subject_id, activities, timestamps, start_timestamps in trace_iter:
        effective_subject_id = subject_id if subject_id is not None else trace_subject_id
        case_id_number = _parse_case_id_number(case_id)
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
                row.update(
                    _recency_features_for_position(
                        activities,
                        timestamps,
                        position,
                    )
                )
                row.update(
                    _transition_features_for_position(
                        activities,
                        position,
                        window=window,
                    )
                )
                row["subject_id"] = effective_subject_id
                row["case_id_number"] = case_id_number
                row["log_case_id_number"] = math.log1p(case_id_number)
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
    feature_set: str = DEFAULT_FEATURE_SET,
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
    feature_set: str = DEFAULT_FEATURE_SET,
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
    recency_columns = (
        list(RECENCY_FEATURE_COLUMNS)
        if resolved_feature_set == FEATURE_SET_ENHANCED
        else []
    )
    transition_columns = (
        list(TRANSITION_FEATURE_COLUMNS)
        if resolved_feature_set == FEATURE_SET_ENHANCED
        else []
    )
    context_columns = (
        list(CONTEXT_FEATURE_COLUMNS)
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
        "recency_feature_columns": recency_columns,
        "transition_feature_columns": transition_columns,
        "context_feature_columns": context_columns,
    }
    if resolved_feature_set in (FEATURE_SET_TEMPORAL, FEATURE_SET_ENHANCED):
        payload["time_features"] = True
        payload["time_feature_columns"] = list(TIME_FEATURE_COLUMNS)
    return payload
