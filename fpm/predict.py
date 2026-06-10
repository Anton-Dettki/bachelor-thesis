"""Local baseline predictors for next-activity prediction."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from fpm.prefix import Vocabulary

PREFIX_COL_RE = re.compile(r"^e(\d+)$")
DEFAULT_PREFIX_COLS = ["e0", "e1", "e2"]
TARGET = "next_activity"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "output" / "models"
PAD_ID = 0


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


def _counts_to_json(counts: dict[int, int]) -> dict[str, int]:
    return {str(k): v for k, v in counts.items()}


def _counts_from_json(data: dict[str, int]) -> dict[int, int]:
    return {int(k): v for k, v in data.items()}


def _nested_counts_to_json(counts: dict[int, dict[int, int]]) -> dict[str, dict[str, int]]:
    return {str(k): _counts_to_json(v) for k, v in counts.items()}


def _nested_counts_from_json(data: dict[str, dict[str, int]]) -> dict[int, dict[int, int]]:
    return {int(k): _counts_from_json(v) for k, v in data.items()}


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
    """Order-1 Markov baseline: P(next | e2) with Laplace smoothing.

    When e2 is PAD (id=0) or the context was unseen in training, falls back
    to the marginal next_activity distribution. Order-3 P(next | e0,e1,e2) is
    not implemented here but could be added as a separate variant.
    """

    order: int = 1
    alpha: float = 1.0
    transitions: dict[int, dict[int, int]] = field(default_factory=dict)
    marginal: dict[int, int] = field(default_factory=dict)
    context_col: str | None = None
    _vocab_size: int = 0

    def fit(self, train_df: pd.DataFrame, vocab: Vocabulary) -> None:
        self._vocab_size = vocab.size
        self.transitions = {}
        self.marginal = {}
        self.context_col = prefix_columns(train_df)[-1]
        if train_df.empty:
            return

        for _, row in train_df.iterrows():
            context = int(row[self.context_col])
            nxt = int(row[TARGET])
            self.marginal[nxt] = self.marginal.get(nxt, 0) + 1
            bucket = self.transitions.setdefault(context, {})
            bucket[nxt] = bucket.get(nxt, 0) + 1

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if len(X) == 0:
            return np.array([], dtype=int)
        proba = self.predict_proba(X)
        return proba.argmax(axis=1)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        n = len(X)
        if n == 0:
            return np.zeros((0, self._vocab_size), dtype=float)

        marginal = self._smoothed_marginal()
        rows = np.zeros((n, self._vocab_size), dtype=float)
        context_col = self.context_col or prefix_columns(X)[-1]
        if context_col not in X.columns:
            raise ValueError(f"Prediction data is missing context column {context_col!r}")

        for i, context in enumerate(X[context_col].astype(int).tolist()):
            if context == PAD_ID or context not in self.transitions:
                rows[i] = marginal
            else:
                rows[i] = self._smoothed_context(context)
        return rows

    def _smoothed_marginal(self) -> np.ndarray:
        proba = np.full(self._vocab_size, self.alpha, dtype=float)
        for cls_id, count in self.marginal.items():
            proba[cls_id] += count
        return self._normalize_excluding_pad(proba)

    def _smoothed_context(self, context: int) -> np.ndarray:
        proba = np.full(self._vocab_size, self.alpha, dtype=float)
        for cls_id, count in self.transitions.get(context, {}).items():
            proba[cls_id] += count
        return self._normalize_excluding_pad(proba)

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
        return {
            "type": "markov",
            "order": self.order,
            "alpha": self.alpha,
            "transitions": _nested_counts_to_json(self.transitions),
            "marginal": _counts_to_json(self.marginal),
            "context_col": self.context_col,
            "vocab_size": self._vocab_size,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MarkovBaseline:
        model = cls(
            order=int(data.get("order", 1)),
            alpha=float(data.get("alpha", 1.0)),
            transitions=_nested_counts_from_json(data["transitions"]),
            marginal=_counts_from_json(data["marginal"]),
            context_col=data.get("context_col", "e2"),
        )
        model._vocab_size = int(data.get("vocab_size", 0))
        return model


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
