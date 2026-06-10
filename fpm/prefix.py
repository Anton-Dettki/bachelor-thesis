"""Build prefix -> next-activity datasets from event logs."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from fpm.loader import ACTIVITY, ACTIVITY_TAXONOMY, CASE_ID, TIMESTAMP

PAD_TOKEN = "<PAD>"
EVENT_INDEX = "@@index"

DEFAULT_PREFIX_DIR = Path(__file__).resolve().parents[1] / "output" / "prefix"


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


def _prefix_at(activities: list[str], position: int, window: int) -> list[str]:
    start = max(0, position - window + 1)
    prefix = activities[start : position + 1]
    if len(prefix) < window:
        prefix = [PAD_TOKEN] * (window - len(prefix)) + prefix
    return prefix


def build_prefix_frame(log: pd.DataFrame, *, window: int = 3) -> pd.DataFrame:
    """Extract prefix -> next-activity rows from an event log.

    Each row contains ``case_id``, ``position``, ``e0``..``e{window-1}`` (strings),
    and ``next_activity``. A trace of length L yields L-1 samples.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window!r}")

    rows: list[dict[str, Any]] = []
    prefix_cols = [f"e{i}" for i in range(window)]

    for case_id, activities in iter_traces(log):
        for position in range(len(activities) - 1):
            prefix = _prefix_at(activities, position, window)
            row: dict[str, Any] = {
                "case_id": case_id,
                "position": position,
            }
            for col, activity in zip(prefix_cols, prefix, strict=True):
                row[col] = activity
            row["next_activity"] = activities[position + 1]
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["case_id", "position", *prefix_cols, "next_activity"])

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


def encode_frame(frame: pd.DataFrame, vocab: Vocabulary, *, window: int = 3) -> pd.DataFrame:
    """Label-encode prefix and target columns using ``vocab``."""
    if frame.empty:
        prefix_cols = [f"e{i}" for i in range(window)]
        return pd.DataFrame(columns=["case_id", "position", *prefix_cols, "next_activity"])

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
) -> dict[str, Any]:
    return {
        "scope": scope,
        "window": window,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "n_activities": n_activities,
    }
