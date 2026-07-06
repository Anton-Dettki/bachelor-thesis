#!/usr/bin/env python3
"""Global next-event prediction baseline for the CASAS2 smart-home ADL dataset."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from shared.event_abstraction import AbstractionLevel, normalize_trace

WINDOW = 3
PAD = "<PAD>"
_FILE_RE = re.compile(r"^(p\d+)\.t(\d+)\.csv$")
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


@dataclass(frozen=True)
class Sample:
    client_id: str
    case_id: str
    task: int
    position: int
    prefix: tuple[str, ...]
    label: str
    split: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CASAS2 global next-event prediction baseline")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--include-errors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-analog", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-depth", type=int, default=25)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    return parser.parse_args()


def event_label(sensor: object, message: object) -> str:
    return f"{sensor}={message}"


def load_events(
    data_dir: Path,
    *,
    include_errors: bool = True,
    skip_analog: bool = True,
) -> pd.DataFrame:
    """Load CASAS2 trial CSVs into a normalized event table."""
    folders = ["adl_noerror"]
    if include_errors:
        folders.append("adl_error")

    rows: list[dict[str, object]] = []
    for folder in folders:
        folder_path = data_dir / folder
        if not folder_path.exists():
            continue
        for csv_path in sorted(folder_path.glob("p*.t*.csv")):
            match = _FILE_RE.match(csv_path.name)
            if not match:
                continue
            participant, task_text = match.groups()
            frame = pd.read_csv(csv_path)
            frame = frame.assign(
                timestamp=pd.to_datetime(frame["date"] + " " + frame["time"], format="mixed")
            )
            frame = frame.sort_values("timestamp", kind="stable")
            for row in frame.itertuples(index=False):
                sensor = str(row.sensor)
                if skip_analog and sensor.upper().startswith("AD1"):
                    continue
                rows.append(
                    {
                        "timestamp": row.timestamp,
                        "sensor": sensor,
                        "message": str(row.message),
                        "event": event_label(sensor, row.message),
                        "participant": participant,
                        "task": int(task_text),
                        "source": folder,
                        "case_id": f"{participant}.t{task_text}",
                    }
                )

    if not rows:
        raise FileNotFoundError(f"No CASAS2 CSV files found under {data_dir}")
    return pd.DataFrame(rows)


def _prefix_tokens(events: list[str], position: int, window: int = WINDOW) -> tuple[str, ...]:
    start = max(0, position - window)
    prefix = events[start:position]
    while len(prefix) < window:
        prefix.insert(0, PAD)
    return tuple(prefix[-window:])


def build_samples(
    events_df: pd.DataFrame,
    *,
    train_fraction: float = 0.8,
    abstraction: AbstractionLevel = "sensor",
) -> list[Sample]:
    """Build prefix -> next-event samples with per-trial chronological split."""
    samples: list[Sample] = []
    for case_id, case_df in events_df.groupby("case_id", sort=True):
        case_df = case_df.sort_values("timestamp", kind="stable")
        events = normalize_trace(case_df["event"].tolist(), abstraction)
        client_id = str(case_df["participant"].iloc[0])
        task = int(case_df["task"].iloc[0])
        if len(events) < 2:
            continue

        split_idx = max(1, int(len(events) * train_fraction))
        if split_idx >= len(events):
            split_idx = len(events) - 1

        for position in range(1, len(events)):
            split = "train" if position < split_idx else "test"
            samples.append(
                Sample(
                    client_id=client_id,
                    case_id=case_id,
                    task=task,
                    position=position,
                    prefix=_prefix_tokens(events, position),
                    label=events[position],
                    split=split,
                )
            )
    return samples


def build_vocabs(samples: Sequence[Sample]) -> tuple[dict[str, int], dict[str, int]]:
    """Build global event and client vocabularies from all sample labels."""
    train_samples = [sample for sample in samples if sample.split == "train"]
    events = sorted(
        {sample.label for sample in samples}
        | {token for sample in samples for token in sample.prefix if token != PAD}
    )
    clients = sorted({sample.client_id for sample in train_samples})
    event_map = {event: index for index, event in enumerate(events)}
    client_map = {client: index for index, client in enumerate(clients)}
    return event_map, client_map


def vectorize(
    samples: Sequence[Sample],
    event_map: dict[str, int],
    *,
    include_client: bool = True,
    client_map: dict[str, int] | None = None,
) -> tuple[list[dict[str, int | str]], np.ndarray]:
    """Convert samples to dict features and encoded labels."""
    x_dicts: list[dict[str, int | str]] = []
    labels: list[int] = []
    for sample in samples:
        features: dict[str, int | str] = {
            f"e{index}": event_map.get(token, -1) for index, token in enumerate(sample.prefix)
        }
        if include_client and client_map is not None:
            features["participant_id"] = client_map[sample.client_id]
        x_dicts.append(features)
        labels.append(event_map[sample.label])
    return x_dicts, np.asarray(labels, dtype=int)


def train_global_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    max_depth: int = 25,
    min_samples_leaf: int = 5,
) -> DecisionTreeClassifier:
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        criterion="entropy",
        random_state=0,
    )
    model.fit(x_train, y_train)
    return model


def evaluate_model(
    model: DecisionTreeClassifier,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    y_pred = model.predict(x_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "n_train": int(len(y_test)),  # placeholder overwritten by caller
    }


def main() -> None:
    args = parse_args()
    events_df = load_events(
        args.data_dir,
        include_errors=args.include_errors,
        skip_analog=args.skip_analog,
    )
    samples = build_samples(events_df, train_fraction=args.train_fraction)
    event_map, client_map = build_vocabs(samples)
    inv_event_map = {index: event for event, index in event_map.items()}

    train_samples = [sample for sample in samples if sample.split == "train"]
    test_samples = [sample for sample in samples if sample.split == "test"]

    x_train_dicts, y_train = vectorize(train_samples, event_map, include_client=True, client_map=client_map)
    x_test_dicts, y_test = vectorize(test_samples, event_map, include_client=True, client_map=client_map)

    vectorizer = DictVectorizer(sparse=False)
    x_train = vectorizer.fit_transform(x_train_dicts)
    x_test = vectorizer.transform(x_test_dicts)

    model = train_global_model(
        x_train,
        y_train,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
    )
    y_pred = model.predict(x_test)

    print("CASAS2 global baseline")
    print(f"Participants: {len(client_map)}")
    print(f"Classes: {len(event_map)}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Test samples: {len(test_samples)}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"Macro F1: {f1_score(y_test, y_pred, average='macro', zero_division=0):.3f}")
    print(f"Weighted F1: {f1_score(y_test, y_pred, average='weighted', zero_division=0):.3f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "metric": ["accuracy", "macro_f1", "weighted_f1", "train_samples", "test_samples", "classes", "participants"],
            "value": [
                accuracy_score(y_test, y_pred),
                f1_score(y_test, y_pred, average="macro", zero_division=0),
                f1_score(y_test, y_pred, average="weighted", zero_division=0),
                len(train_samples),
                len(test_samples),
                len(event_map),
                len(client_map),
            ],
        }
    ).to_csv(args.output_dir / "global_metrics.csv", index=False)


if __name__ == "__main__":
    main()
