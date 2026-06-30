"""Local baseline predictors for next-activity prediction."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from fpm.prefix import (
    CONTEXT_FEATURE_COLUMNS,
    DURATION_FEATURE_COLUMNS,
    HISTORY_FEATURE_COLUMNS,
    RECENCY_FEATURE_COLUMNS,
    TIME_FEATURE_COLUMNS,
    TRANSITION_FEATURE_COLUMNS,
    Vocabulary,
)

PREFIX_COL_RE = re.compile(r"^e(\d+)$")
DEFAULT_PREFIX_COLS = ["e0", "e1", "e2"]
AUX_FEATURE_COLUMNS = [
    "position",
    *TIME_FEATURE_COLUMNS,
    *DURATION_FEATURE_COLUMNS,
    *HISTORY_FEATURE_COLUMNS,
    *RECENCY_FEATURE_COLUMNS,
    *TRANSITION_FEATURE_COLUMNS,
    *CONTEXT_FEATURE_COLUMNS,
]
TARGET = "next_activity"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "output" / "models"
DEFAULT_FEDERATED_MODEL_DIR = DEFAULT_MODEL_DIR / "federated"
PAD_ID = 0
LOGREG_MODEL = "logreg"
TREE_MODEL = "tree"
RANDOM_FOREST_MODEL = "random_forest"
DEFAULT_LOGREG_EPOCHS = 50
DEFAULT_LOGREG_LEARNING_RATE = 0.1
DEFAULT_LOGREG_BATCH_SIZE = 32
DEFAULT_LOGREG_L2 = 0.0001
DEFAULT_LOGREG_SEED = 0
DEFAULT_LOGREG_CLASS_WEIGHT = "balanced"
DEFAULT_LEARNING_CURVE_FRACTIONS = (0.1, 0.25, 0.5, 0.75, 1.0)
AUX_FEATURE_SCALES = {
    "position": 100.0,
    "hour": 23.0,
    "hour_bin": 3.0,
    "day_of_week": 6.0,
    "minutes_since_day_start": 1440.0,
    "minutes_since_prev_event": 1440.0,
    "minutes_since_midnight": 1440.0,
    "log_minutes_since_day_start": 8.0,
    "log_minutes_since_prev_event": 8.0,
    "month": 12.0,
    "day_of_month": 31.0,
    "week_of_year": 53.0,
    "trace_start_hour": 23.0,
    "trace_start_minutes_since_midnight": 1440.0,
    "activity_duration_minutes": 1440.0,
    "log_activity_duration_minutes": 8.0,
    "previous_activity_duration_minutes": 1440.0,
    "log_previous_activity_duration_minutes": 8.0,
    "gap_since_prev_event_minutes": 1440.0,
    "cumulative_activity_duration_minutes": 1440.0,
    "mean_activity_duration_minutes_so_far": 1440.0,
    "events_seen_so_far": 100.0,
    "log_events_seen_so_far": 8.0,
    "current_activity_count_so_far": 100.0,
    "current_activity_frequency_so_far": 1.0,
    "unique_activities_so_far": 12.0,
    "unique_activity_ratio_so_far": 1.0,
    "dominant_activity_count_so_far": 100.0,
    "dominant_activity_ratio_so_far": 1.0,
    "activity_repetition_count_so_far": 100.0,
    "current_activity_run_length": 100.0,
    "prefix_length_ratio": 1.0,
    "events_since_last_same_activity": 100.0,
    "minutes_since_last_same_activity": 1440.0,
    "log_minutes_since_last_same_activity": 8.0,
    "activity_switch_count_so_far": 100.0,
    "activity_switch_ratio_so_far": 1.0,
    "window_unique_activities": 12.0,
    "window_switch_count": 10.0,
    "window_switch_ratio": 1.0,
    "window_repetition_count": 10.0,
    "window_repetition_ratio": 1.0,
    "subject_id": 7.0,
    "case_id_number": 1000.0,
    "log_case_id_number": 8.0,
}


class BaselinePredictor(Protocol):
    def fit(self, train_df: pd.DataFrame, vocab: Vocabulary) -> None: ...

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray | None: ...


def prefix_columns(df: pd.DataFrame) -> list[str]:
    """Return prefix feature columns ordered by their numeric suffix."""
    columns: list[tuple[int, str]] = []
    for col in df.columns:
        match = PREFIX_COL_RE.fullmatch(str(col))
        if match:
            columns.append((int(match.group(1)), str(col)))

    if not columns:
        return DEFAULT_PREFIX_COLS
    return [col for _, col in sorted(columns)]


def aux_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric timestamp/progress columns present in a prefix frame."""
    return [col for col in AUX_FEATURE_COLUMNS if col in df.columns]


def tree_feature_matrix(
    X: pd.DataFrame,
    vocab: Vocabulary | int,
    *,
    prefix_cols: list[str] | None = None,
    aux_cols: list[str] | None = None,
) -> np.ndarray:
    """One-hot encode activity prefixes and append numeric auxiliary features."""
    cols = prefix_cols or prefix_columns(X)
    aux = aux_cols if aux_cols is not None else aux_feature_columns(X)
    activity_block = onehot_encode(X, vocab, prefix_cols=cols)
    if not aux:
        return activity_block
    aux_block = X[aux].astype(float).to_numpy()
    return np.hstack([activity_block, aux_block])


def _scaled_aux_matrix(X: pd.DataFrame, aux_cols: list[str]) -> np.ndarray:
    if not aux_cols:
        return np.zeros((len(X), 0), dtype=float)

    missing = [col for col in aux_cols if col not in X.columns]
    if missing:
        raise ValueError(f"Prediction data is missing auxiliary columns: {missing}")

    aux_block = X[aux_cols].astype(float).to_numpy(copy=True)
    for index, col in enumerate(aux_cols):
        scale = AUX_FEATURE_SCALES.get(col, 1.0)
        if scale != 0:
            aux_block[:, index] = aux_block[:, index] / scale
    return aux_block


def logreg_feature_matrix(
    X: pd.DataFrame,
    vocab: Vocabulary | int,
    *,
    prefix_cols: list[str] | None = None,
    aux_cols: list[str] | None = None,
) -> np.ndarray:
    """One-hot prefix features plus fixed-scale numeric auxiliary features."""
    cols = prefix_cols or prefix_columns(X)
    aux = aux_cols if aux_cols is not None else aux_feature_columns(X)
    activity_block = onehot_encode(X, vocab, prefix_cols=cols)
    if not aux:
        return activity_block
    return np.hstack([activity_block, _scaled_aux_matrix(X, aux)])


def split_xy(
    df: pd.DataFrame,
    *,
    prefix_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Split a prefix frame into feature columns and target array."""
    cols = prefix_cols or prefix_columns(df)
    if df.empty:
        return pd.DataFrame(columns=cols), np.array([], dtype=int)

    missing = [col for col in [*cols, TARGET] if col not in df.columns]
    if missing:
        raise ValueError(f"Prefix dataset is missing columns: {missing}")
    return df[cols], df[TARGET].to_numpy(dtype=int)


def load_scope(prefix_dir: Path, scope: str) -> tuple[pd.DataFrame, pd.DataFrame, Vocabulary]:
    """Load train/val CSVs and vocabulary for a scope (e.g. subject1, global)."""
    scope_dir = prefix_dir / scope
    train_df = pd.read_csv(scope_dir / "train.csv")
    val_df = pd.read_csv(scope_dir / "val.csv")
    vocab = Vocabulary.read_json(scope_dir / "vocab.json")
    return train_df, val_df, vocab


def onehot_encode(
    X: pd.DataFrame,
    vocab: Vocabulary | int,
    *,
    prefix_cols: list[str] | None = None,
) -> np.ndarray:
    """One-hot encode prefix columns over vocabulary size (deterministic order)."""
    cols = prefix_cols or prefix_columns(X)
    n = len(X)
    vocab_size = vocab.size if isinstance(vocab, Vocabulary) else int(vocab)
    if n == 0:
        return np.zeros((0, len(cols) * vocab_size), dtype=float)

    blocks: list[np.ndarray] = []
    for col in cols:
        ids = X[col].astype(int).to_numpy()
        block = np.zeros((n, vocab_size), dtype=float)
        block[np.arange(n), ids] = 1.0
        blocks.append(block)
    return np.hstack(blocks)


def _scatter_proba_excluding_pad(
    partial: np.ndarray,
    classes: np.ndarray,
    vocab_size: int,
) -> np.ndarray:
    """Map sklearn class probabilities into a full-vocab matrix with PAD=0."""
    n = partial.shape[0]
    full = np.zeros((n, vocab_size), dtype=float)
    for j, cls_id in enumerate(classes.astype(int)):
        full[:, cls_id] = partial[:, j]
    full[:, PAD_ID] = 0.0
    totals = full.sum(axis=1, keepdims=True)
    totals = np.where(totals == 0, 1.0, totals)
    return full / totals


def _counts_to_json(counts: dict[int, int]) -> dict[str, int]:
    return {str(k): v for k, v in counts.items()}


def _counts_from_json(data: dict[str, int]) -> dict[int, int]:
    return {int(k): v for k, v in data.items()}


def _transition_counts_to_json(counts: dict[str, dict[int, int]]) -> dict[str, dict[str, int]]:
    return {k: _counts_to_json(v) for k, v in counts.items()}


def _transition_counts_from_json(data: dict[str, dict[str, int]]) -> dict[str, dict[int, int]]:
    return {k: _counts_from_json(v) for k, v in data.items()}


def _serialize_context(values: tuple[int, ...]) -> str:
    return ",".join(str(v) for v in values)


def _row_context(row: pd.Series, cols: list[str]) -> tuple[int, ...]:
    return tuple(int(row[col]) for col in cols)


def _context_has_pad(context: tuple[int, ...]) -> bool:
    return PAD_ID in context


@dataclass
class FrequencyBaseline:
    """Predict the global majority next_activity; ignores prefix."""

    majority: int = 0
    counts: dict[int, int] = field(default_factory=dict)
    _vocab_size: int = 0

    def fit(self, train_df: pd.DataFrame, vocab: Vocabulary) -> None:
        self._vocab_size = vocab.size
        self.counts = {}
        if train_df.empty:
            self.majority = PAD_ID
            return

        counter = Counter(train_df[TARGET].astype(int).tolist())
        self.counts = dict(counter)
        self.majority = counter.most_common(1)[0][0]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        n = len(X)
        if n == 0:
            return np.array([], dtype=int)
        return np.full(n, self.majority, dtype=int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        n = len(X)
        if n == 0:
            return np.zeros((0, self._vocab_size), dtype=float)
        return np.tile(self._marginal_proba(), (n, 1))

    def _marginal_proba(self) -> np.ndarray:
        proba = np.zeros(self._vocab_size, dtype=float)
        if not self.counts:
            return proba
        total = sum(self.counts.values())
        for cls_id, count in self.counts.items():
            proba[cls_id] = count / total
        return proba

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "frequency",
            "majority": self.majority,
            "counts": _counts_to_json(self.counts),
            "vocab_size": self._vocab_size,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FrequencyBaseline:
        majority = int(data["majority"])
        counts = _counts_from_json(data["counts"])
        vocab_size = int(
            data.get(
                "vocab_size",
                max([majority, *counts.keys()], default=PAD_ID) + 1,
            )
        )
        return cls(
            majority=majority,
            counts=counts,
            _vocab_size=vocab_size,
        )


@dataclass
class MarkovBaseline:
    """Count-based Markov baseline with Laplace smoothing.

    Order 1 estimates ``P(next | e2)``; order 3 estimates
    ``P(next | e0, e1, e2)`` with suffix backoff to order 2, order 1,
    then the marginal next-activity distribution.
    """

    order: int = 1
    alpha: float = 1.0
    transitions: dict[str, dict[int, int]] = field(default_factory=dict)
    backoff_transitions: dict[int, dict[str, dict[int, int]]] = field(
        default_factory=dict
    )
    marginal: dict[int, int] = field(default_factory=dict)
    context_cols: list[str] = field(default_factory=list)
    _vocab_size: int = 0

    @property
    def context_col(self) -> str | None:
        """Backward-compatible single context column for order-1 artifacts."""
        return self.context_cols[-1] if self.context_cols else None

    def _resolve_context_cols(self, df: pd.DataFrame) -> list[str]:
        cols = prefix_columns(df)
        if self.order == 1:
            return [cols[-1]]
        if len(cols) < self.order:
            raise ValueError(
                f"Markov order {self.order} requires at least {self.order} prefix "
                f"columns, got {len(cols)}"
            )
        return cols[-self.order :]

    def fit(self, train_df: pd.DataFrame, vocab: Vocabulary) -> None:
        self._vocab_size = vocab.size
        self.transitions = {}
        self.backoff_transitions = {}
        self.marginal = {}
        if train_df.empty:
            self.context_cols = self._resolve_context_cols(train_df)
            self._ensure_backoff_tables()
            return

        self.context_cols = self._resolve_context_cols(train_df)
        for _, row in train_df.iterrows():
            nxt = int(row[TARGET])
            self.marginal[nxt] = self.marginal.get(nxt, 0) + 1
            for level in range(1, self.order + 1):
                context_cols = self.context_cols[-level:]
                context = _serialize_context(_row_context(row, context_cols))
                table = self.backoff_transitions.setdefault(level, {})
                bucket = table.setdefault(context, {})
                bucket[nxt] = bucket.get(nxt, 0) + 1
        self._ensure_backoff_tables()
        self.transitions = dict(self.backoff_transitions.get(self.order, {}))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if len(X) == 0:
            return np.array([], dtype=int)
        proba = self.predict_proba(X)
        return proba.argmax(axis=1)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        n = len(X)
        if n == 0:
            return np.zeros((0, self._vocab_size), dtype=float)

        rows = np.zeros((n, self._vocab_size), dtype=float)
        context_cols = self.context_cols or self._resolve_context_cols(X)
        missing = [col for col in context_cols if col not in X.columns]
        if missing:
            raise ValueError(f"Prediction data is missing context columns: {missing}")

        for i, (_, row) in enumerate(X.iterrows()):
            context_values = _row_context(row, context_cols)
            rows[i], _level = self._smoothed_backoff_context(context_values)
        return rows

    def _ensure_backoff_tables(self) -> None:
        """Populate suffix tables for old artifacts or empty models."""
        if self.backoff_transitions:
            for level in range(1, self.order + 1):
                self.backoff_transitions.setdefault(level, {})
            return

        self.backoff_transitions = {level: {} for level in range(1, self.order + 1)}
        if not self.transitions:
            return

        # Legacy artifacts only stored the full-order table. Reconstruct lower
        # suffix counts by aggregating keys such as "0,1,2" -> "1,2" -> "2".
        for context_key, counts in self.transitions.items():
            values = tuple(int(value) for value in str(context_key).split(","))
            if len(values) == 1 and self.order == 1:
                levels = [1]
            else:
                levels = range(1, min(self.order, len(values)) + 1)
            for level in levels:
                suffix_key = _serialize_context(values[-level:])
                table = self.backoff_transitions.setdefault(level, {})
                bucket = table.setdefault(suffix_key, {})
                for cls_id, count in counts.items():
                    bucket[int(cls_id)] = bucket.get(int(cls_id), 0) + int(count)

    def _smoothed_backoff_context(
        self,
        context_values: tuple[int, ...],
    ) -> tuple[np.ndarray, int]:
        self._ensure_backoff_tables()
        max_level = min(self.order, len(context_values))
        for level in range(max_level, 0, -1):
            suffix = context_values[-level:]
            context_key = _serialize_context(suffix)
            if _context_has_pad(suffix):
                continue
            if context_key in self.backoff_transitions.get(level, {}):
                return self._smoothed_context(level, context_key), level
        return self._smoothed_marginal(), 0

    def _smoothed_marginal(self) -> np.ndarray:
        proba = np.full(self._vocab_size, self.alpha, dtype=float)
        for cls_id, count in self.marginal.items():
            proba[cls_id] += count
        return self._normalize_excluding_pad(proba)

    def _smoothed_context(self, level: int, context_key: str) -> np.ndarray:
        proba = np.full(self._vocab_size, self.alpha, dtype=float)
        for cls_id, count in self.backoff_transitions.get(level, {}).get(context_key, {}).items():
            proba[cls_id] += count
        return self._normalize_excluding_pad(proba)

    def context_diagnostics(self, X: pd.DataFrame) -> dict[str, Any]:
        """Return context coverage and backoff use for a prediction frame."""
        context_cols = self.context_cols or self._resolve_context_cols(X)
        missing = [col for col in context_cols if col not in X.columns]
        if missing:
            raise ValueError(f"Prediction data is missing context columns: {missing}")

        self._ensure_backoff_tables()
        level_counts = {level: 0 for level in range(0, self.order + 1)}
        full_seen = 0
        full_pad = 0
        for _, row in X.iterrows():
            context_values = _row_context(row, context_cols)
            full_key = _serialize_context(context_values)
            if _context_has_pad(context_values):
                full_pad += 1
            elif full_key in self.backoff_transitions.get(self.order, {}):
                full_seen += 1
            _proba, level = self._smoothed_backoff_context(context_values)
            level_counts[level] += 1

        total = len(X)
        return {
            "n": int(total),
            "order": int(self.order),
            "context_cols": context_cols,
            "full_context_seen": int(full_seen),
            "full_context_seen_rate": float(full_seen / total) if total else 0.0,
            "full_context_pad": int(full_pad),
            "full_context_pad_rate": float(full_pad / total) if total else 0.0,
            "selected_level_counts": {
                ("marginal" if level == 0 else f"order{level}"): int(count)
                for level, count in sorted(level_counts.items())
            },
            "marginal_fallback_rate": (
                float(level_counts.get(0, 0) / total) if total else 0.0
            ),
        }

    @staticmethod
    def _normalize_excluding_pad(proba: np.ndarray) -> np.ndarray:
        # PAD is never a valid next activity, so it must not receive
        # probability mass (it would otherwise occupy top-k candidate slots).
        proba[PAD_ID] = 0.0
        total = proba.sum()
        if total == 0:
            return proba
        return proba / total

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": "markov_order3" if self.order == 3 else "markov",
            "order": self.order,
            "alpha": self.alpha,
            "transitions": _transition_counts_to_json(self.transitions),
            "backoff_transitions": {
                str(level): _transition_counts_to_json(table)
                for level, table in sorted(self.backoff_transitions.items())
            },
            "marginal": _counts_to_json(self.marginal),
            "context_cols": self.context_cols,
            "vocab_size": self._vocab_size,
        }
        if self.order == 1 and self.context_col is not None:
            payload["context_col"] = self.context_col
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MarkovBaseline:
        order = int(data.get("order", 1))
        context_cols = list(data.get("context_cols") or [])
        if not context_cols:
            legacy_col = data.get("context_col")
            if legacy_col:
                context_cols = [str(legacy_col)]
            elif order == 3:
                context_cols = DEFAULT_PREFIX_COLS.copy()
            else:
                context_cols = [DEFAULT_PREFIX_COLS[-1]]

        transitions = _transition_counts_from_json(data["transitions"])
        # Legacy order-1 artifacts stored integer context keys in JSON.
        if order == 1:
            transitions = {
                (k if "," in k else str(int(k))): v for k, v in transitions.items()
            }

        backoff_raw = data.get("backoff_transitions") or {}
        backoff_transitions = {
            int(level): _transition_counts_from_json(table)
            for level, table in backoff_raw.items()
        }

        model = cls(
            order=order,
            alpha=float(data.get("alpha", 1.0)),
            transitions=transitions,
            backoff_transitions=backoff_transitions,
            marginal=_counts_from_json(data["marginal"]),
            context_cols=context_cols,
        )
        model._vocab_size = int(data.get("vocab_size", 0))
        model._ensure_backoff_tables()
        model.transitions = dict(model.backoff_transitions.get(order, transitions))
        return model


@dataclass
class MarkovOrder1Baseline(MarkovBaseline):
    """Order-1 Markov: P(next | e2)."""

    order: int = 1


@dataclass
class MarkovOrder3Baseline(MarkovBaseline):
    """Order-3 Markov: P(next | e0, e1, e2)."""

    order: int = 3


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    totals = exp.sum(axis=1, keepdims=True)
    totals = np.where(totals == 0, 1.0, totals)
    return exp / totals


@dataclass
class LogisticRegressionModel:
    """Multiclass softmax regression for FedAvg next-activity prediction."""

    epochs: int = DEFAULT_LOGREG_EPOCHS
    learning_rate: float = DEFAULT_LOGREG_LEARNING_RATE
    batch_size: int = DEFAULT_LOGREG_BATCH_SIZE
    l2: float = DEFAULT_LOGREG_L2
    seed: int = DEFAULT_LOGREG_SEED
    class_weight: str | None = DEFAULT_LOGREG_CLASS_WEIGHT
    weights: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=float))
    intercept: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    classes_: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    class_weights_: np.ndarray = field(default_factory=lambda: np.ones(0, dtype=float))
    _vocab_size: int = 0
    _prefix_cols: list[str] = field(default_factory=list)
    _aux_feature_cols: list[str] = field(default_factory=list)
    training_metadata: dict[str, Any] = field(default_factory=dict)

    def fit(self, train_df: pd.DataFrame, vocab: Vocabulary) -> None:
        self.initialize(train_df, vocab)
        self.train_local(
            train_df,
            vocab,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            l2=self.l2,
            seed=self.seed,
        )

    def initialize(self, frame: pd.DataFrame, vocab: Vocabulary) -> None:
        self._vocab_size = vocab.size
        self._prefix_cols = prefix_columns(frame)
        self._aux_feature_cols = aux_feature_columns(frame)
        self.classes_ = np.arange(1, vocab.size, dtype=int)
        n_features = len(self._prefix_cols) * vocab.size + len(self._aux_feature_cols)
        self.weights = np.zeros((n_features, len(self.classes_)), dtype=float)
        self.intercept = np.zeros(len(self.classes_), dtype=float)
        self.class_weights_ = np.ones(len(self.classes_), dtype=float)
        self.training_metadata = {
            "epochs": 0,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "l2": self.l2,
            "seed": self.seed,
            "class_weight": self.class_weight,
            "n_train": 0,
        }

    def train_local(
        self,
        train_df: pd.DataFrame,
        vocab: Vocabulary,
        *,
        epochs: int,
        learning_rate: float,
        batch_size: int,
        l2: float,
        seed: int,
    ) -> None:
        if self._vocab_size == 0 or self.weights.size == 0:
            self.initialize(train_df, vocab)
        self._validate_shape()

        if train_df.empty or epochs <= 0:
            self.training_metadata = {
                "epochs": max(int(epochs), 0),
                "learning_rate": float(learning_rate),
                "batch_size": int(batch_size),
                "l2": float(l2),
                "seed": int(seed),
                "class_weight": self.class_weight,
                "n_train": int(len(train_df)),
            }
            return

        feature_cols = [*self._prefix_cols, *self._aux_feature_cols]
        missing = [col for col in [*feature_cols, TARGET] if col not in train_df.columns]
        if missing:
            raise ValueError(f"Training data is missing columns: {missing}")

        X_train = logreg_feature_matrix(
            train_df[feature_cols],
            self._vocab_size,
            prefix_cols=self._prefix_cols,
            aux_cols=self._aux_feature_cols,
        )
        y_train = train_df[TARGET].to_numpy(dtype=int)
        valid = (y_train > PAD_ID) & (y_train < self._vocab_size)
        if not np.all(valid):
            X_train = X_train[valid]
            y_train = y_train[valid]
        if len(y_train) == 0:
            self.training_metadata = {
                "epochs": int(epochs),
                "learning_rate": float(learning_rate),
                "batch_size": int(batch_size),
                "l2": float(l2),
                "seed": int(seed),
                "class_weight": self.class_weight,
                "n_train": 0,
            }
            return

        class_index = y_train - 1
        self.class_weights_ = self._compute_class_weights(y_train)
        rng = np.random.default_rng(seed)
        effective_batch = max(1, int(batch_size))

        for _epoch in range(int(epochs)):
            indices = np.arange(len(y_train))
            rng.shuffle(indices)
            for start in range(0, len(indices), effective_batch):
                batch_idx = indices[start : start + effective_batch]
                X_batch = X_train[batch_idx]
                y_batch = class_index[batch_idx]

                logits = X_batch @ self.weights + self.intercept
                grad_logits = _softmax(logits)
                grad_logits[np.arange(len(y_batch)), y_batch] -= 1.0
                sample_weights = self.class_weights_[y_batch]
                weight_total = float(sample_weights.sum())
                if weight_total <= 0:
                    weight_total = float(len(y_batch))
                    sample_weights = np.ones(len(y_batch), dtype=float)
                grad_logits *= sample_weights[:, np.newaxis]
                grad_logits /= weight_total

                grad_w = X_batch.T @ grad_logits + float(l2) * self.weights
                grad_b = grad_logits.sum(axis=0)
                self.weights -= float(learning_rate) * grad_w
                self.intercept -= float(learning_rate) * grad_b

        self.training_metadata = {
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "batch_size": int(batch_size),
            "l2": float(l2),
            "seed": int(seed),
            "class_weight": self.class_weight,
            "class_weights": self.class_weights_.astype(float).tolist(),
            "n_train": int(len(y_train)),
        }

    def _compute_class_weights(self, y_train: np.ndarray) -> np.ndarray:
        weights = np.ones(len(self.classes_), dtype=float)
        if self.class_weight != "balanced":
            return weights

        counts = Counter(y_train.astype(int).tolist())
        n_classes = len(self.classes_)
        n_samples = len(y_train)
        if n_classes == 0 or n_samples == 0:
            return weights
        for index, cls_id in enumerate(self.classes_.astype(int)):
            count = counts.get(int(cls_id), 0)
            weights[index] = n_samples / (n_classes * count) if count else 1.0
        return weights

    def _validate_shape(self) -> None:
        expected_features = len(self._prefix_cols) * self._vocab_size + len(
            self._aux_feature_cols
        )
        expected_classes = max(self._vocab_size - 1, 0)
        if self.weights.shape != (expected_features, expected_classes):
            raise ValueError(
                "Logistic regression weight shape does not match metadata: "
                f"expected {(expected_features, expected_classes)}, got {self.weights.shape}"
            )
        if self.intercept.shape != (expected_classes,):
            raise ValueError(
                "Logistic regression intercept shape does not match metadata: "
                f"expected {(expected_classes,)}, got {self.intercept.shape}"
            )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if len(X) == 0:
            return np.array([], dtype=int)
        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        n = len(X)
        if n == 0:
            return np.zeros((0, self._vocab_size), dtype=float)
        if self._vocab_size == 0 or self.weights.size == 0:
            return np.zeros((n, self._vocab_size), dtype=float)

        self._validate_shape()
        feature_cols = [*self._prefix_cols, *self._aux_feature_cols]
        missing = [col for col in feature_cols if col not in X.columns]
        if missing:
            raise ValueError(f"Prediction data is missing columns: {missing}")

        X_encoded = logreg_feature_matrix(
            X[feature_cols],
            self._vocab_size,
            prefix_cols=self._prefix_cols,
            aux_cols=self._aux_feature_cols,
        )
        partial = _softmax(X_encoded @ self.weights + self.intercept)
        full = np.zeros((n, self._vocab_size), dtype=float)
        full[:, self.classes_] = partial
        full[:, PAD_ID] = 0.0
        totals = full.sum(axis=1, keepdims=True)
        totals = np.where(totals == 0, 1.0, totals)
        return full / totals

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": LOGREG_MODEL,
            "vocab_size": self._vocab_size,
            "classes": self.classes_.astype(int).tolist(),
            "class_weight": self.class_weight,
            "class_weights": self.class_weights_.astype(float).tolist(),
            "prefix_cols": self._prefix_cols,
            "aux_feature_cols": self._aux_feature_cols,
            "weights": self.weights.astype(float).tolist(),
            "intercept": self.intercept.astype(float).tolist(),
            "training": dict(self.training_metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LogisticRegressionModel:
        model = cls()
        model._vocab_size = int(data.get("vocab_size", 0))
        model.classes_ = np.array(data.get("classes", []), dtype=int)
        model.class_weight = data.get("class_weight", DEFAULT_LOGREG_CLASS_WEIGHT)
        model.class_weights_ = np.array(data.get("class_weights", []), dtype=float)
        model._prefix_cols = list(data.get("prefix_cols") or DEFAULT_PREFIX_COLS)
        model._aux_feature_cols = list(data.get("aux_feature_cols") or [])
        model.weights = np.array(data.get("weights", []), dtype=float)
        model.intercept = np.array(data.get("intercept", []), dtype=float)
        model.training_metadata = dict(data.get("training") or {})
        if model._vocab_size and model.classes_.size == 0:
            model.classes_ = np.arange(1, model._vocab_size, dtype=int)
        if model.weights.size == 0 and model._vocab_size:
            n_features = len(model._prefix_cols) * model._vocab_size + len(
                model._aux_feature_cols
            )
            model.weights = np.zeros((n_features, len(model.classes_)), dtype=float)
        if model.intercept.size == 0 and model._vocab_size:
            model.intercept = np.zeros(len(model.classes_), dtype=float)
        if model.class_weights_.size == 0 and model._vocab_size:
            model.class_weights_ = np.ones(len(model.classes_), dtype=float)
        if model._vocab_size:
            model._validate_shape()
        return model


def merge_frequency(parts: list[dict[str, Any]]) -> FrequencyBaseline:
    """Sum next-activity counts from per-subject frequency params."""
    if not parts:
        return FrequencyBaseline()

    vocab_sizes = {int(p.get("vocab_size", 0)) for p in parts}
    if len(vocab_sizes) != 1:
        raise ValueError(f"Inconsistent vocab_size in frequency parts: {vocab_sizes}")
    vocab_size = vocab_sizes.pop()
    if vocab_size == 0:
        raise ValueError("frequency parts missing vocab_size")

    merged_counts: Counter[int] = Counter()
    for part in parts:
        merged_counts.update(_counts_from_json(part["counts"]))

    counts = dict(merged_counts)
    majority = max(counts, key=counts.get) if counts else PAD_ID
    return FrequencyBaseline(majority=majority, counts=counts, _vocab_size=vocab_size)


def merge_markov(parts: list[dict[str, Any]]) -> MarkovBaseline:
    """Sum transition and marginal counts from per-subject Markov params."""
    if not parts:
        return MarkovBaseline()

    vocab_sizes = {int(p.get("vocab_size", 0)) for p in parts}
    alphas = {float(p.get("alpha", 1.0)) for p in parts}
    orders = {int(p.get("order", 1)) for p in parts}
    context_cols = {tuple(p.get("context_cols") or [p.get("context_col", "e2")]) for p in parts}
    if len(vocab_sizes) != 1 or len(alphas) != 1 or len(orders) != 1 or len(context_cols) != 1:
        raise ValueError("Inconsistent Markov metadata across federated parts")

    merged_transitions: dict[str, dict[int, int]] = {}
    merged_backoff: dict[int, dict[str, dict[int, int]]] = {}
    merged_marginal: Counter[int] = Counter()
    for part in parts:
        model = MarkovBaseline.from_dict(part)
        for context, bucket in model.transitions.items():
            dst = merged_transitions.setdefault(context, {})
            for nxt, count in bucket.items():
                dst[nxt] = dst.get(nxt, 0) + count
        for level, table in model.backoff_transitions.items():
            merged_table = merged_backoff.setdefault(level, {})
            for context, bucket in table.items():
                dst = merged_table.setdefault(context, {})
                for nxt, count in bucket.items():
                    dst[nxt] = dst.get(nxt, 0) + count
        for nxt, count in model.marginal.items():
            merged_marginal[nxt] += count

    order = orders.pop()
    model_cls = MarkovOrder3Baseline if order == 3 else MarkovOrder1Baseline
    return model_cls(
        order=order,
        alpha=alphas.pop(),
        transitions=merged_transitions,
        backoff_transitions=merged_backoff,
        marginal=dict(merged_marginal),
        context_cols=list(context_cols.pop()),
        _vocab_size=vocab_sizes.pop(),
    )


ADDITIVE_FEDERATED_MODELS: dict[str, tuple[type, Any]] = {
    "frequency": (FrequencyBaseline, merge_frequency),
    "markov": (MarkovOrder1Baseline, merge_markov),
    "markov_order3": (MarkovOrder3Baseline, merge_markov),
}

FEDAVG_MODELS: dict[str, type] = {
    LOGREG_MODEL: LogisticRegressionModel,
}

# Backward-compatible alias: these models support one-shot additive parameter merge.
FEDERATED_MODELS = ADDITIVE_FEDERATED_MODELS


def federated_model_names() -> list[str]:
    """Return all model names accepted by federated prediction CLIs."""
    return sorted([*ADDITIVE_FEDERATED_MODELS, *FEDAVG_MODELS])


def is_additive_model(model_name: str) -> bool:
    return model_name in ADDITIVE_FEDERATED_MODELS


def is_fedavg_model(model_name: str) -> bool:
    return model_name in FEDAVG_MODELS


def fit_model(
    model_name: str,
    train_df: pd.DataFrame,
    vocab: Vocabulary,
    **kwargs: Any,
) -> Any:
    """Fit a local next-activity model by registered model name."""
    if model_name in ADDITIVE_FEDERATED_MODELS:
        cls, _ = ADDITIVE_FEDERATED_MODELS[model_name]
        model = cls()
        model.fit(train_df, vocab)
        return model
    if model_name in FEDAVG_MODELS:
        cls = FEDAVG_MODELS[model_name]
        model = cls(**kwargs)
        model.fit(train_df, vocab)
        return model
    raise ValueError(
        f"Unknown federated model {model_name!r}; "
        f"choose from {federated_model_names()}"
    )


def fit_params(model_name: str, train_df: pd.DataFrame, vocab: Vocabulary) -> dict[str, Any]:
    """Fit an additive federated model and return JSON-serializable parameters."""
    if model_name not in ADDITIVE_FEDERATED_MODELS:
        raise ValueError(
            f"Unknown additive federated model {model_name!r}; "
            f"choose from {sorted(ADDITIVE_FEDERATED_MODELS)}"
        )
    cls, _ = ADDITIVE_FEDERATED_MODELS[model_name]
    model = cls()
    model.fit(train_df, vocab)
    return model.to_dict()


def merge_params(model_name: str, parts: list[dict[str, Any]]) -> Any:
    """Merge per-subject parameter dicts into one global model instance."""
    if model_name not in ADDITIVE_FEDERATED_MODELS:
        raise ValueError(
            f"Unknown additive federated model {model_name!r}; "
            f"choose from {sorted(ADDITIVE_FEDERATED_MODELS)}"
        )
    _, merge_fn = ADDITIVE_FEDERATED_MODELS[model_name]
    return merge_fn(parts)


def initialize_fedavg_model(
    model_name: str,
    train_df: pd.DataFrame,
    vocab: Vocabulary,
) -> Any:
    """Create an initialized but untrained FedAvg model."""
    if model_name not in FEDAVG_MODELS:
        raise ValueError(
            f"Unknown FedAvg model {model_name!r}; choose from {sorted(FEDAVG_MODELS)}"
        )
    cls = FEDAVG_MODELS[model_name]
    model = cls()
    model.initialize(train_df, vocab)
    return model


def fedavg_update(
    model_name: str,
    state: dict[str, Any],
    train_df: pd.DataFrame,
    vocab: Vocabulary,
    *,
    local_epochs: int,
    learning_rate: float,
    batch_size: int,
    l2: float,
    seed: int,
) -> Any:
    """Train one local FedAvg update from the provided global model state."""
    if model_name != LOGREG_MODEL:
        raise ValueError(f"Unknown FedAvg model {model_name!r}")
    model = LogisticRegressionModel.from_dict(state)
    model.train_local(
        train_df,
        vocab,
        epochs=local_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        l2=l2,
        seed=seed,
    )
    return model


def average_fedavg_models(
    model_name: str,
    weighted_states: list[tuple[int, dict[str, Any]]],
    *,
    fallback_state: dict[str, Any],
) -> Any:
    """Weighted-average FedAvg model parameters by local training sample count."""
    if model_name != LOGREG_MODEL:
        raise ValueError(f"Unknown FedAvg model {model_name!r}")

    usable = [
        (int(n_train), LogisticRegressionModel.from_dict(state))
        for n_train, state in weighted_states
        if int(n_train) > 0
    ]
    if not usable:
        return LogisticRegressionModel.from_dict(fallback_state)

    total = sum(n_train for n_train, _ in usable)
    first = usable[0][1]
    weights = np.zeros_like(first.weights)
    intercept = np.zeros_like(first.intercept)
    for n_train, model in usable:
        if (
            model._vocab_size != first._vocab_size
            or model._prefix_cols != first._prefix_cols
            or model._aux_feature_cols != first._aux_feature_cols
            or not np.array_equal(model.classes_, first.classes_)
            or model.weights.shape != first.weights.shape
        ):
            raise ValueError("Inconsistent FedAvg logistic regression metadata")
        factor = n_train / total
        weights += factor * model.weights
        intercept += factor * model.intercept

    averaged = LogisticRegressionModel.from_dict(first.to_dict())
    averaged.weights = weights
    averaged.intercept = intercept
    averaged.training_metadata = {
        **averaged.training_metadata,
        "fedavg_weighted_n_train": int(total),
    }
    return averaged


def model_from_dict(model_name: str, data: dict[str, Any]) -> Any:
    """Restore a persisted model by registered name."""
    if model_name in ADDITIVE_FEDERATED_MODELS:
        cls, _ = ADDITIVE_FEDERATED_MODELS[model_name]
        return cls.from_dict(data)
    if model_name == LOGREG_MODEL:
        return LogisticRegressionModel.from_dict(data)
    raise ValueError(f"Unknown model {model_name!r}; choose from {federated_model_names()}")


def params_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Return whether two serialized model payloads are identical."""
    return left == right


def fit_subject_model(
    model_name: str,
    train_df: pd.DataFrame,
    vocab: Vocabulary,
    **kwargs: Any,
) -> Any:
    """Fit an independently-trained per-subject model using the shared vocabulary."""
    if model_name == TREE_MODEL:
        model = DecisionTreeModel()
        model.fit(train_df, vocab)
        return model
    if model_name == RANDOM_FOREST_MODEL:
        model = RandomForestModel()
        model.fit(train_df, vocab)
        return model
    if model_name in federated_model_names():
        return fit_model(model_name, train_df, vocab, **kwargs)
    raise ValueError(
        f"Unknown subject model {model_name!r}; "
        f"choose from {sorted([*federated_model_names(), TREE_MODEL, RANDOM_FOREST_MODEL])}"
    )


def predictor_feature_frame(model: Any, frame: pd.DataFrame) -> pd.DataFrame:
    """Return the feature columns expected by a fitted predictor."""
    if isinstance(model, (DecisionTreeModel, RandomForestModel, LogisticRegressionModel)):
        aux_cols = model._aux_feature_cols or aux_feature_columns(frame)
        prefix_cols = model._prefix_cols or prefix_columns(frame)
        return frame[[*prefix_cols, *aux_cols]]
    X_val, _ = split_xy(frame)
    return X_val


def ensemble_soft_vote(proba_list: list[np.ndarray]) -> np.ndarray:
    """Average aligned probability matrices with equal weight per model."""
    if not proba_list:
        raise ValueError("At least one probability matrix is required")
    reference = proba_list[0]
    if reference.ndim != 2:
        raise ValueError("Probability matrices must be 2-dimensional")
    for index, proba in enumerate(proba_list[1:], start=1):
        if proba.shape != reference.shape:
            raise ValueError(
                f"Probability matrix {index} has shape {proba.shape}, "
                f"expected {reference.shape}"
            )
    stacked = np.stack(proba_list, axis=0)
    return stacked.mean(axis=0)


def build_subject_ensemble_models(
    model_name: str,
    subject_ids: list[int],
    prefix_dir: Path,
    vocab: Vocabulary,
    **fit_kwargs: Any,
) -> list[Any]:
    """Fit one local model per subject on its train split with a shared vocabulary."""
    models: list[Any] = []
    for subject_id in subject_ids:
        subj_train, _, _ = load_scope(prefix_dir, f"subject{subject_id}")
        models.append(fit_subject_model(model_name, subj_train, vocab, **fit_kwargs))
    return models


def evaluate_ensemble_soft_vote(
    models: list[Any],
    val_df: pd.DataFrame,
    vocab: Vocabulary,
) -> dict[str, float]:
    """Score an equal-weight soft-vote ensemble on a validation prefix frame."""
    y_val, y_pred, ensemble_proba = ensemble_predictions(models, val_df)
    return evaluate(y_val, y_pred, vocab=vocab, y_proba=ensemble_proba)


def predictor_predictions(
    model: BaselinePredictor,
    val_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return true labels, top-1 predictions, and probabilities for a predictor."""
    X_val = predictor_feature_frame(model, val_df)
    _, y_val = split_xy(val_df)
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)
    return y_val, y_pred, y_proba


def ensemble_predictions(
    models: list[Any],
    val_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return labels and equal-weight soft-vote predictions for ensemble members."""
    if not models:
        raise ValueError("At least one model is required for ensemble evaluation")
    _, y_val = split_xy(val_df)
    proba_parts: list[np.ndarray] = []
    for model in models:
        X_val = predictor_feature_frame(model, val_df)
        proba = model.predict_proba(X_val)
        if proba is None:
            raise ValueError("Ensemble member did not return probabilities")
        proba_parts.append(proba)
    ensemble_proba = ensemble_soft_vote(proba_parts)
    y_pred = ensemble_proba.argmax(axis=1)
    return y_val, y_pred, ensemble_proba


def confusion_matrix_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    vocab: Vocabulary,
    *,
    include_pad: bool = False,
) -> tuple[np.ndarray, list[int], list[str]]:
    """Build a confusion matrix over vocabulary ids.

    Rows are actual classes and columns are predicted classes.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    start = 0 if include_pad else 1
    class_ids = list(range(start, vocab.size))
    labels = [vocab.decode(class_id) for class_id in class_ids]
    index = {class_id: pos for pos, class_id in enumerate(class_ids)}
    matrix = np.zeros((len(class_ids), len(class_ids)), dtype=int)
    for true, pred in zip(y_true.astype(int), y_pred.astype(int), strict=True):
        if int(true) not in index or int(pred) not in index:
            continue
        matrix[index[int(true)], index[int(pred)]] += 1
    return matrix, class_ids, labels


def write_confusion_matrix_artifacts(
    output_dir: Path,
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    vocab: Vocabulary,
) -> dict[str, str]:
    """Write confusion matrix as CSV and PNG and return relative artifact paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix, _, labels = confusion_matrix_counts(y_true, y_pred, vocab)
    csv_path = output_dir / f"{name}.csv"
    png_path = output_dir / f"{name}.png"
    norm_csv_path = output_dir / f"{name}_normalized.csv"
    norm_png_path = output_dir / f"{name}_normalized.png"

    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(csv_path)
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0,
    )
    pd.DataFrame(normalized, index=labels, columns=labels).to_csv(norm_csv_path)

    fig_width = max(8.0, len(labels) * 0.7)
    fig, ax = plt.subplots(figsize=(fig_width, fig_width))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    ax.set_title(f"Confusion matrix: {name}")
    ax.set_xlabel("Predicted activity")
    ax.set_ylabel("True activity")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    max_count = int(matrix.max()) if matrix.size else 0
    threshold = max_count / 2 if max_count else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            count = int(matrix[row, col])
            if count:
                color = "white" if count > threshold else "black"
                ax.text(col, row, str(count), ha="center", va="center", color=color)

    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(fig_width, fig_width))
    image = ax.imshow(normalized, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    ax.set_title(f"Normalized confusion matrix: {name}")
    ax.set_xlabel("Predicted activity")
    ax.set_ylabel("True activity")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    for row in range(normalized.shape[0]):
        for col in range(normalized.shape[1]):
            value = float(normalized[row, col])
            if value:
                color = "white" if value > 0.5 else "black"
                ax.text(
                    col,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color=color,
                )

    fig.tight_layout()
    fig.savefig(norm_png_path, dpi=160)
    plt.close(fig)
    return {
        "csv": str(csv_path),
        "png": str(png_path),
        "normalized_csv": str(norm_csv_path),
        "normalized_png": str(norm_png_path),
    }


def _label_for_id(vocab: Vocabulary, class_id: int) -> str:
    if 0 <= int(class_id) < vocab.size:
        return vocab.decode(int(class_id))
    return str(class_id)


def prediction_diagnostics(
    model: BaselinePredictor,
    val_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    vocab: Vocabulary,
) -> dict[str, Any]:
    """Summarize prediction collapse and model-specific context behavior."""
    n = len(y_true)
    true_counts = Counter(int(value) for value in y_true.astype(int))
    pred_counts = Counter(int(value) for value in y_pred.astype(int))
    top_pred_id, top_pred_count = pred_counts.most_common(1)[0] if pred_counts else (PAD_ID, 0)

    diagnostics: dict[str, Any] = {
        "n": int(n),
        "true_distribution": {
            _label_for_id(vocab, cls_id): {
                "count": int(count),
                "share": float(count / n) if n else 0.0,
            }
            for cls_id, count in sorted(true_counts.items())
        },
        "predicted_distribution": {
            _label_for_id(vocab, cls_id): {
                "count": int(count),
                "share": float(count / n) if n else 0.0,
            }
            for cls_id, count in sorted(pred_counts.items())
        },
        "nonzero_predicted_classes": int(sum(count > 0 for count in pred_counts.values())),
        "majority_predicted_label": _label_for_id(vocab, int(top_pred_id)),
        "majority_predicted_count": int(top_pred_count),
        "majority_predicted_share": float(top_pred_count / n) if n else 0.0,
    }
    if isinstance(model, MarkovBaseline):
        X_val = predictor_feature_frame(model, val_df)
        diagnostics["markov_context"] = model.context_diagnostics(X_val)
    return diagnostics


def write_prediction_diagnostics_artifacts(
    output_dir: Path,
    name: str,
    diagnostics: dict[str, Any],
) -> dict[str, str]:
    """Write prediction diagnostics as JSON and compact distribution CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{name}.json"
    csv_path = output_dir / f"{name}.csv"
    write_json(json_path, diagnostics)

    rows: list[dict[str, Any]] = []
    for kind in ("true_distribution", "predicted_distribution"):
        for label, values in diagnostics.get(kind, {}).items():
            rows.append(
                {
                    "kind": kind.replace("_distribution", ""),
                    "label": label,
                    "count": values["count"],
                    "share": values["share"],
                }
            )
    markov_context = diagnostics.get("markov_context")
    if markov_context:
        for level, count in markov_context.get("selected_level_counts", {}).items():
            rows.append(
                {
                    "kind": "markov_selected_level",
                    "label": level,
                    "count": count,
                    "share": count / markov_context["n"] if markov_context["n"] else 0.0,
                }
            )
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return {"json": str(json_path), "csv": str(csv_path)}


def learning_curve_rows(
    model_factory: Any,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    vocab: Vocabulary,
    *,
    fractions: tuple[float, ...] = DEFAULT_LEARNING_CURVE_FRACTIONS,
) -> list[dict[str, float | int]]:
    """Fit a model on increasing train fractions and return validation metrics."""
    rows: list[dict[str, float | int]] = []
    n_train_total = len(train_df)
    if n_train_total == 0:
        return rows

    case_ids = (
        train_df["case_id"].drop_duplicates().astype(str).tolist()
        if "case_id" in train_df.columns
        else []
    )
    n_cases_total = len(case_ids)

    seen_sizes: set[int] = set()
    for fraction in fractions:
        if fraction <= 0 or fraction > 1:
            raise ValueError("Learning curve fractions must be in (0, 1]")
        if n_cases_total:
            n_train_cases = max(1, int(math.ceil(n_cases_total * fraction)))
            selected_cases = set(case_ids[:n_train_cases])
            subset = train_df[
                train_df["case_id"].astype(str).isin(selected_cases)
            ].copy()
        else:
            n_train_cases = 0
            n_train = max(1, int(math.ceil(n_train_total * fraction)))
            subset = train_df.iloc[:n_train].copy()

        n_train = len(subset)
        if n_train in seen_sizes:
            continue
        seen_sizes.add(n_train)

        model = model_factory()
        model.fit(subset, vocab)
        metrics = evaluate_predictor(model, val_df, vocab)
        row: dict[str, float | int] = {
            "train_fraction": float(fraction),
            "n_train_cases": int(n_train_cases),
            "n_train": int(n_train),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
        }
        if "top3_accuracy" in metrics:
            row["top3_accuracy"] = metrics["top3_accuracy"]
        rows.append(row)
    return rows


def write_learning_curve_artifacts(
    output_dir: Path,
    name: str,
    rows: list[dict[str, float | int]],
) -> dict[str, str]:
    """Write learning curve rows as CSV and PNG."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{name}.csv"
    png_path = output_dir / f"{name}.png"
    frame = pd.DataFrame(rows)
    frame.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    if not frame.empty:
        ax.plot(frame["n_train"], frame["accuracy"], marker="o", label="accuracy")
        ax.plot(frame["n_train"], frame["macro_f1"], marker="o", label="macro-F1")
        if "top3_accuracy" in frame:
            ax.plot(frame["n_train"], frame["top3_accuracy"], marker="o", label="top-3")
    ax.set_title(f"Learning curve: {name}")
    ax.set_xlabel("Training samples")
    ax.set_ylabel("Validation score")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    return {"csv": str(csv_path), "png": str(png_path)}


def evaluate_predictor(
    model: BaselinePredictor,
    val_df: pd.DataFrame,
    vocab: Vocabulary,
) -> dict[str, float]:
    """Score a fitted predictor on a validation prefix frame."""
    y_val, y_pred, y_proba = predictor_predictions(model, val_df)
    return evaluate(y_val, y_pred, vocab=vocab, y_proba=y_proba)


@dataclass
class DecisionTreeModel:
    """sklearn decision tree on one-hot encoded prefix features.

    Unlike Markov counts, a fitted tree is not additive and is not intended
    for federated sum-merge in this step.
    """

    max_depth: int | None = 4
    min_samples_leaf: int = 3
    class_weight: str | None = "balanced"
    _tree: DecisionTreeClassifier | None = None
    classes_: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    _vocab_size: int = 0
    _prefix_cols: list[str] = field(default_factory=list)
    _aux_feature_cols: list[str] = field(default_factory=list)

    def fit(self, train_df: pd.DataFrame, vocab: Vocabulary) -> None:
        self._vocab_size = vocab.size
        self._prefix_cols = prefix_columns(train_df)
        self._aux_feature_cols = aux_feature_columns(train_df)
        self._tree = None
        self.classes_ = np.array([], dtype=int)
        if train_df.empty:
            return

        feature_cols = [*self._prefix_cols, *self._aux_feature_cols]
        X_train = train_df[feature_cols]
        y_train = train_df[TARGET].to_numpy(dtype=int)
        X_encoded = tree_feature_matrix(
            X_train,
            vocab,
            prefix_cols=self._prefix_cols,
            aux_cols=self._aux_feature_cols,
        )
        tree = DecisionTreeClassifier(
            random_state=0,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=self.class_weight,
        )
        tree.fit(X_encoded, y_train)
        self._tree = tree
        self.classes_ = tree.classes_.astype(int)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if len(X) == 0:
            return np.array([], dtype=int)
        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        n = len(X)
        if n == 0:
            return np.zeros((0, self._vocab_size), dtype=float)
        if self._tree is None:
            return np.zeros((n, self._vocab_size), dtype=float)

        cols = self._prefix_cols or prefix_columns(X)
        aux_cols = self._aux_feature_cols or aux_feature_columns(X)
        feature_df = X[[*cols, *aux_cols]] if aux_cols else X[cols]
        X_encoded = tree_feature_matrix(
            feature_df,
            self._vocab_size,
            prefix_cols=cols,
            aux_cols=aux_cols,
        )
        partial = self._tree.predict_proba(X_encoded)
        return _scatter_proba_excluding_pad(partial, self.classes_, self._vocab_size)

    def to_dict(self) -> dict[str, Any]:
        tree = self._tree
        return {
            "type": "tree",
            "params": tree.get_params() if tree is not None else {},
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "class_weight": self.class_weight,
            "n_features": int(tree.n_features_in_) if tree is not None else 0,
            "classes": self.classes_.astype(int).tolist(),
            "feature_importances": (
                tree.feature_importances_.astype(float).tolist()
                if tree is not None
                else []
            ),
            "vocab_size": self._vocab_size,
            "prefix_cols": self._prefix_cols,
            "aux_feature_cols": self._aux_feature_cols,
        }


@dataclass
class RandomForestModel:
    """Balanced random forest on one-hot encoded prefix and auxiliary features."""

    n_estimators: int = 300
    max_depth: int | None = 5
    min_samples_leaf: int = 2
    class_weight: str | None = "balanced_subsample"
    random_state: int = 0
    n_jobs: int | None = -1
    _forest: RandomForestClassifier | None = None
    classes_: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    _vocab_size: int = 0
    _prefix_cols: list[str] = field(default_factory=list)
    _aux_feature_cols: list[str] = field(default_factory=list)

    def fit(self, train_df: pd.DataFrame, vocab: Vocabulary) -> None:
        self._vocab_size = vocab.size
        self._prefix_cols = prefix_columns(train_df)
        self._aux_feature_cols = aux_feature_columns(train_df)
        self._forest = None
        self.classes_ = np.array([], dtype=int)
        if train_df.empty:
            return

        feature_cols = [*self._prefix_cols, *self._aux_feature_cols]
        X_train = train_df[feature_cols]
        y_train = train_df[TARGET].to_numpy(dtype=int)
        X_encoded = tree_feature_matrix(
            X_train,
            vocab,
            prefix_cols=self._prefix_cols,
            aux_cols=self._aux_feature_cols,
        )
        forest = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        forest.fit(X_encoded, y_train)
        self._forest = forest
        self.classes_ = forest.classes_.astype(int)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if len(X) == 0:
            return np.array([], dtype=int)
        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        n = len(X)
        if n == 0:
            return np.zeros((0, self._vocab_size), dtype=float)
        if self._forest is None:
            return np.zeros((n, self._vocab_size), dtype=float)

        cols = self._prefix_cols or prefix_columns(X)
        aux_cols = self._aux_feature_cols or aux_feature_columns(X)
        feature_df = X[[*cols, *aux_cols]] if aux_cols else X[cols]
        X_encoded = tree_feature_matrix(
            feature_df,
            self._vocab_size,
            prefix_cols=cols,
            aux_cols=aux_cols,
        )
        partial = self._forest.predict_proba(X_encoded)
        return _scatter_proba_excluding_pad(partial, self.classes_, self._vocab_size)

    def to_dict(self) -> dict[str, Any]:
        forest = self._forest
        return {
            "type": RANDOM_FOREST_MODEL,
            "params": forest.get_params() if forest is not None else {},
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "n_features": int(forest.n_features_in_) if forest is not None else 0,
            "classes": self.classes_.astype(int).tolist(),
            "feature_importances": (
                forest.feature_importances_.astype(float).tolist()
                if forest is not None
                else []
            ),
            "vocab_size": self._vocab_size,
            "prefix_cols": self._prefix_cols,
            "aux_feature_cols": self._aux_feature_cols,
        }


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-averaged F1 over classes present in y_true (zero_division=0)."""
    if len(y_true) == 0:
        return 0.0

    classes = np.unique(y_true)
    f1_scores: list[float] = []
    for cls in classes:
        tp = int(np.sum((y_true == cls) & (y_pred == cls)))
        fp = int(np.sum((y_true != cls) & (y_pred == cls)))
        fn = int(np.sum((y_true == cls) & (y_pred != cls)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))
    return float(np.mean(f1_scores))


def top_k_accuracy(y_true: np.ndarray, y_proba: np.ndarray, *, k: int = 3) -> float:
    if len(y_true) == 0:
        return 0.0
    top_k = np.argsort(-y_proba, axis=1)[:, :k]
    hits = [true in row for true, row in zip(y_true, top_k, strict=True)]
    return float(np.mean(hits))


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    vocab: Vocabulary,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute validation metrics for a baseline predictor."""
    del vocab  # reserved for future human-readable diagnostics
    metrics: dict[str, float] = {
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
    }
    if y_proba is not None:
        metrics["top3_accuracy"] = top_k_accuracy(y_true, y_proba, k=3)
    return metrics


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
