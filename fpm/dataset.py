"""Dataset loading utilities for the Chinook smart-home sensor CSVs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from fpm.ltl import PatternQuery
from shared.event_abstraction import AbstractionLevel, normalize_trace

DEFAULT_DATA_DIR = Path("data")
EVAL_TRIAL = 5
_FILE_RE = re.compile(r"^(p\d+)\.t(\d+)\.csv$")
_TOKEN_RE = re.compile(r"[^A-Za-z0-9_]+")


@dataclass(frozen=True)
class SensorTrace:
    participant: str
    trial: int
    source: str
    path: Path
    events: tuple[str, ...]

    @property
    def event_count(self) -> int:
        return len(self.events)


def sensor_token(sensor: object, message: object) -> str:
    """Convert one raw sensor event into an LTL/model-friendly token."""
    raw = f"{sensor}_{message}".strip().upper()
    return _TOKEN_RE.sub("_", raw).strip("_")


def participant_ids(data_dir: Path | str = DEFAULT_DATA_DIR) -> list[str]:
    ids: set[str] = set()
    for csv_path in Path(data_dir).glob("adl_*/*.csv"):
        match = _FILE_RE.match(csv_path.name)
        if match:
            ids.add(match.group(1))
    return sorted(ids)


def load_trace(path: Path) -> SensorTrace:
    match = _FILE_RE.match(path.name)
    if not match:
        raise ValueError(f"Unexpected trace filename: {path.name}")

    participant, trial_text = match.groups()
    frame = pd.read_csv(path)
    required = {"date", "time", "sensor", "message"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    frame = frame.assign(
        timestamp=pd.to_datetime(frame["date"] + " " + frame["time"], format="mixed")
    )
    frame = frame.sort_values("timestamp", kind="stable")
    events = tuple(sensor_token(row.sensor, row.message) for row in frame.itertuples())

    return SensorTrace(
        participant=participant,
        trial=int(trial_text),
        source=path.parent.name,
        path=path,
        events=events,
    )


def load_participant(
    participant: str,
    data_dir: Path | str = DEFAULT_DATA_DIR,
) -> list[SensorTrace]:
    data_path = Path(data_dir)
    traces: list[SensorTrace] = []
    for csv_path in sorted(data_path.glob(f"adl_*/*{participant}.t*.csv")):
        traces.append(load_trace(csv_path))
    if not traces:
        raise ValueError(f"No traces found for participant {participant!r} in {data_path}")
    return sorted(traces, key=lambda trace: trace.trial)


def load_all(data_dir: Path | str = DEFAULT_DATA_DIR) -> dict[str, list[SensorTrace]]:
    return {
        participant: load_participant(participant, data_dir)
        for participant in participant_ids(data_dir)
    }


def filter_traces(
    traces: Sequence[SensorTrace],
    query_text: str = "",
) -> tuple[list[SensorTrace], float]:
    stripped = query_text.strip()
    if not stripped:
        return list(traces), 1.0

    query = PatternQuery.parse(stripped)
    matched = [trace for trace in traces if query.satisfied_by(trace.events)]
    fraction = len(matched) / len(traces) if traces else 0.0
    return matched, fraction


def training_traces(
    traces: Sequence[SensorTrace],
    eval_trial: int = EVAL_TRIAL,
) -> list[SensorTrace]:
    """Return participant traces used for model fitting (all trials except holdout)."""
    return [trace for trace in traces if trace.trial != eval_trial]


def _within_trial_split(
    traces: Sequence[SensorTrace],
    *,
    train_fraction: float = 0.8,
    abstraction: AbstractionLevel = "sensor",
) -> tuple[list[tuple[str, ...]], list[tuple[str, ...]]]:
    train: list[tuple[str, ...]] = []
    eval_: list[tuple[str, ...]] = []
    for trace in traces:
        events = list(trace.events)
        normalized = normalize_trace(events, abstraction)
        if len(normalized) < 2:
            continue
        split_idx = max(1, int(len(normalized) * train_fraction))
        if split_idx >= len(normalized):
            split_idx = len(normalized) - 1
        train.append(tuple(normalized[:split_idx]))
        eval_.append(tuple(normalized[split_idx:]))
    return train, eval_


def split_traces(
    traces: Sequence[SensorTrace],
    fallback_eval: Sequence[SensorTrace] | None = None,
    eval_trial: int = EVAL_TRIAL,
    *,
    abstraction: AbstractionLevel = "sensor",
) -> tuple[list[tuple[str, ...]], list[tuple[str, ...]]]:
    train = [
        tuple(normalize_trace(trace.events, abstraction))
        for trace in traces
        if trace.trial != eval_trial
    ]
    eval_ = [
        tuple(normalize_trace(trace.events, abstraction))
        for trace in traces
        if trace.trial == eval_trial
    ]

    if not train:
        train = [tuple(normalize_trace(trace.events, abstraction)) for trace in traces]
    if not eval_ and fallback_eval is not None:
        eval_ = [
            tuple(normalize_trace(trace.events, abstraction))
            for trace in fallback_eval
            if trace.trial == eval_trial
        ]
    if not eval_:
        eval_ = [tuple(normalize_trace(trace.events, abstraction)) for trace in traces[-1:]]

    return train, eval_


def split_traces_protocol(
    traces: Sequence[SensorTrace],
    *,
    protocol: str = "federated",
    train_fraction: float = 0.8,
    eval_trial: int = EVAL_TRIAL,
    abstraction: AbstractionLevel = "sensor",
) -> tuple[list[tuple[str, ...]], list[tuple[str, ...]]]:
    """Split traces for local client evaluation.

    ``casas2`` uses an 80/20 chronological split inside each trace.
    ``federated`` trains on all non-holdout trials and evaluates on the holdout
    trial, without borrowing an unmatched holdout trace from outside ``traces``.
    """
    normalized = protocol.strip().lower()
    if normalized == "casas2":
        return _within_trial_split(
            traces,
            train_fraction=train_fraction,
            abstraction=abstraction,
        )
    if normalized != "federated":
        raise ValueError("protocol must be 'casas2' or 'federated'")

    train = [
        tuple(normalize_trace(trace.events, abstraction))
        for trace in traces
        if trace.trial != eval_trial
    ]
    eval_ = [
        tuple(normalize_trace(trace.events, abstraction))
        for trace in traces
        if trace.trial == eval_trial
    ]
    if not eval_ and train:
        return _within_trial_split(
            [trace for trace in traces if trace.trial != eval_trial],
            train_fraction=train_fraction,
            abstraction=abstraction,
        )
    return train, eval_


def event_count(traces: Iterable[Sequence[str]]) -> int:
    return sum(len(trace) for trace in traces)
