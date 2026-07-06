"""CASAS2-style local client models for Docker endpoints."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, f1_score

from CASAS2.main import PAD, Sample, train_global_model, vectorize
from shared.discovery_baseline import MarkovPredictor
from shared.event_abstraction import AbstractionLevel, normalize_trace
from shared.ltl_filter import event_to_ltl_token

EVAL_TRIAL = 5
TRAIN_FRACTION = 0.8
_FILE_RE = re.compile(r"^(p\d+)\.t(\d+)\.csv$")


@dataclass(frozen=True)
class LocalTrace:
    participant: str
    trial: int
    events: list[str]


def load_local_traces(
    data_dir: Path,
    participant: str,
    *,
    include_errors: bool = True,
    skip_analog: bool = True,
) -> list[LocalTrace]:
    folders = ["adl_noerror"]
    if include_errors:
        folders.append("adl_error")

    traces: list[LocalTrace] = []
    for folder in folders:
        for csv_path in sorted((data_dir / folder).glob(f"{participant}.t*.csv")):
            match = _FILE_RE.match(csv_path.name)
            if not match:
                continue
            client_id, trial_text = match.groups()
            frame = pd.read_csv(csv_path)
            frame = frame.assign(
                timestamp=pd.to_datetime(
                    frame["date"] + " " + frame["time"],
                    format="mixed",
                )
            )
            frame = frame.sort_values("timestamp", kind="stable")
            events: list[str] = []
            for row in frame.itertuples(index=False):
                sensor = str(row.sensor)
                if skip_analog and sensor.upper().startswith("AD1"):
                    continue
                events.append(f"{sensor}={row.message}")
            traces.append(LocalTrace(client_id, int(trial_text), events))
    return sorted(traces, key=lambda trace: trace.trial)


def matching_fraction(traces: Sequence[LocalTrace], ltl: str) -> tuple[list[LocalTrace], float]:
    stripped = ltl.strip()
    if not stripped:
        return list(traces), 1.0

    from fpm.ltl import PatternQuery

    query = PatternQuery.parse(stripped)
    matched = [
        trace
        for trace in traces
        if query.satisfied_by(tuple(event_to_ltl_token(event) for event in trace.events))
    ]
    return matched, (len(matched) / len(traces) if traces else 0.0)


def _prefix_tokens(events: Sequence[str], position: int, window: int = 3) -> tuple[str, ...]:
    start = max(0, position - window)
    prefix = list(events[start:position])
    while len(prefix) < window:
        prefix.insert(0, PAD)
    return tuple(prefix[-window:])


def _samples_from_events(
    participant: str,
    trial: int,
    events: Sequence[str],
    split: str,
) -> list[Sample]:
    return [
        Sample(
            client_id=participant,
            case_id=f"{participant}.t{trial}",
            task=trial,
            position=position,
            prefix=_prefix_tokens(events, position),
            label=events[position],
            split=split,
        )
        for position in range(1, len(events))
    ]


def _samples_by_split_index(
    participant: str,
    trial: int,
    events: Sequence[str],
    split_idx: int,
) -> tuple[list[Sample], list[Sample]]:
    train: list[Sample] = []
    test: list[Sample] = []
    for position in range(1, len(events)):
        sample = Sample(
            client_id=participant,
            case_id=f"{participant}.t{trial}",
            task=trial,
            position=position,
            prefix=_prefix_tokens(events, position),
            label=events[position],
            split="train" if position < split_idx else "test",
        )
        if position < split_idx:
            train.append(sample)
        else:
            test.append(sample)
    return train, test


def _build_samples(
    traces: Sequence[LocalTrace],
    *,
    protocol: str,
    abstraction: AbstractionLevel = "sensor",
) -> tuple[list[Sample], list[Sample], list[list[str]]]:
    train_samples: list[Sample] = []
    test_samples: list[Sample] = []
    train_traces: list[list[str]] = []

    if protocol == "casas2":
        for trace in traces:
            events = normalize_trace(trace.events, abstraction)
            if len(events) < 2:
                continue
            split_idx = max(1, int(len(events) * TRAIN_FRACTION))
            if split_idx >= len(events):
                split_idx = len(events) - 1
            train_traces.append(events[:split_idx])
            trace_train, trace_test = _samples_by_split_index(
                trace.participant,
                trace.trial,
                events,
                split_idx,
            )
            train_samples.extend(trace_train)
            test_samples.extend(trace_test)
        return train_samples, test_samples, train_traces

    if protocol != "federated":
        raise ValueError("protocol must be 'casas2' or 'federated'")

    for trace in traces:
        events = normalize_trace(trace.events, abstraction)
        if len(events) < 2:
            continue
        if trace.trial == EVAL_TRIAL:
            test_samples.extend(_samples_from_events(trace.participant, trace.trial, events, "test"))
        else:
            train_traces.append(events)
            train_samples.extend(_samples_from_events(trace.participant, trace.trial, events, "train"))
    return train_samples, test_samples, train_traces


def _event_map(train_samples: Sequence[Sample], test_samples: Sequence[Sample]) -> dict[str, int]:
    events = sorted(
        {sample.label for sample in train_samples + test_samples}
        | {
            token
            for sample in train_samples + test_samples
            for token in sample.prefix
            if token != PAD
        }
    )
    return {event: index for index, event in enumerate(events)}


def _encoded_labels(samples: Sequence[Sample], event_map: dict[str, int]) -> np.ndarray:
    return np.asarray([event_map.get(sample.label, -1) for sample in samples], dtype=int)


def _feature_dicts(samples: Sequence[Sample], event_map: dict[str, int]) -> list[dict[str, int]]:
    return [
        {
            f"e{index}": event_map.get(token, -1)
            for index, token in enumerate(sample.prefix)
        }
        for sample in samples
    ]


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if len(y_true) == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def train_and_evaluate_casas_client(
    data_dir: Path,
    participant: str,
    *,
    model_name: str,
    protocol: str,
    abstraction: AbstractionLevel = "sensor",
) -> dict[str, Any]:
    traces = load_local_traces(data_dir, participant)
    train_samples, test_samples, train_traces = _build_samples(
        traces,
        protocol=protocol,
        abstraction=abstraction,
    )
    event_map = _event_map(train_samples, test_samples)
    y_true = _encoded_labels(test_samples, event_map)

    if model_name == "casas_markov":
        model = MarkovPredictor.fit(train_traces, use_trigram=True)
        predictions = [
            model.predict_label(sample.prefix, label_encoder=event_map)
            for sample in test_samples
        ]
        y_pred = np.asarray(
            [prediction if prediction is not None else -1 for prediction in predictions],
            dtype=int,
        )
        metrics = _metrics(y_true, y_pred)
        return {
            "model": "casas_markov",
            **metrics,
            "correct": int((y_true == y_pred).sum()),
            "total": int(len(y_true)),
            "params": {
                "type": "casas_markov",
                "states": len(model.transitions),
                "bigram_contexts": len(model.bigram_context),
                "classes": len(event_map),
            },
        }

    if model_name != "casas_tree":
        raise ValueError("model_name must be 'casas_tree' or 'casas_markov'")

    if not train_samples or len({sample.label for sample in train_samples}) < 2:
        y_pred = np.full_like(y_true, fill_value=-1)
        metrics = _metrics(y_true, y_pred)
        return {
            "model": "casas_tree",
            **metrics,
            "correct": int((y_true == y_pred).sum()),
            "total": int(len(y_true)),
            "params": {
                "type": "casas_tree",
                "fitted": False,
                "fallback_reason": "not enough next-event classes",
            },
        }

    x_train_dicts, y_train = vectorize(train_samples, event_map, include_client=False)
    x_test_dicts = _feature_dicts(test_samples, event_map)
    vectorizer = DictVectorizer(sparse=False)
    x_train = vectorizer.fit_transform(x_train_dicts)
    x_test = vectorizer.transform(x_test_dicts)
    model = train_global_model(x_train, y_train, max_depth=25, min_samples_leaf=5)
    y_pred = model.predict(x_test)
    metrics = _metrics(y_true, y_pred)
    return {
        "model": "casas_tree",
        **metrics,
        "correct": int((y_true == y_pred).sum()),
        "total": int(len(y_true)),
        "params": {
            "type": "casas_tree",
            "fitted": True,
            "classes": len(event_map),
            "features": len(vectorizer.get_feature_names_out()),
            "n_nodes": int(model.tree_.node_count),
            "n_leaves": int(model.get_n_leaves()),
            "max_depth": 25,
            "min_samples_leaf": 5,
        },
    }


def train_and_evaluate_casas_client_views(
    data_dir: Path,
    participant: str,
    *,
    model_name: str,
    protocol: str,
) -> dict[str, dict[str, Any]]:
    """Evaluate one CASAS2 client model at both abstraction levels."""
    return {
        level: train_and_evaluate_casas_client(
            data_dir,
            participant,
            model_name=model_name,
            protocol=protocol,
            abstraction=level,
        )
        for level in ("sensor", "raw")
    }
