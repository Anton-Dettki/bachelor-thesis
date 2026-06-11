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
from sklearn.tree import DecisionTreeClassifier

from fpm.prefix import Vocabulary

PREFIX_COL_RE = re.compile(r"^e(\d+)$")
DEFAULT_PREFIX_COLS = ["e0", "e1", "e2"]
TARGET = "next_activity"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "output" / "models"
DEFAULT_FEDERATED_MODEL_DIR = DEFAULT_MODEL_DIR / "federated"
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
    ``P(next | e0, e1, e2)``. Contexts containing PAD (id=0) or unseen in
    training fall back to the marginal next-activity distribution.
    """

    order: int = 1
    alpha: float = 1.0
    transitions: dict[str, dict[int, int]] = field(default_factory=dict)
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
        self.marginal = {}
        if train_df.empty:
            self.context_cols = self._resolve_context_cols(train_df)
            return

        self.context_cols = self._resolve_context_cols(train_df)
        for _, row in train_df.iterrows():
            context = _serialize_context(_row_context(row, self.context_cols))
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
        context_cols = self.context_cols or self._resolve_context_cols(X)
        missing = [col for col in context_cols if col not in X.columns]
        if missing:
            raise ValueError(f"Prediction data is missing context columns: {missing}")

        for i, (_, row) in enumerate(X.iterrows()):
            context_values = _row_context(row, context_cols)
            context_key = _serialize_context(context_values)
            if _context_has_pad(context_values) or context_key not in self.transitions:
                rows[i] = marginal
            else:
                rows[i] = self._smoothed_context(context_key)
        return rows

    def _smoothed_marginal(self) -> np.ndarray:
        proba = np.full(self._vocab_size, self.alpha, dtype=float)
        for cls_id, count in self.marginal.items():
            proba[cls_id] += count
        return self._normalize_excluding_pad(proba)

    def _smoothed_context(self, context_key: str) -> np.ndarray:
        proba = np.full(self._vocab_size, self.alpha, dtype=float)
        for cls_id, count in self.transitions.get(context_key, {}).items():
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
        payload: dict[str, Any] = {
            "type": "markov_order3" if self.order == 3 else "markov",
            "order": self.order,
            "alpha": self.alpha,
            "transitions": _transition_counts_to_json(self.transitions),
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

        model = cls(
            order=order,
            alpha=float(data.get("alpha", 1.0)),
            transitions=transitions,
            marginal=_counts_from_json(data["marginal"]),
            context_cols=context_cols,
        )
        model._vocab_size = int(data.get("vocab_size", 0))
        return model


@dataclass
class MarkovOrder1Baseline(MarkovBaseline):
    """Order-1 Markov: P(next | e2)."""

    order: int = 1


@dataclass
class MarkovOrder3Baseline(MarkovBaseline):
    """Order-3 Markov: P(next | e0, e1, e2)."""

    order: int = 3


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
    merged_marginal: Counter[int] = Counter()
    for part in parts:
        model = MarkovBaseline.from_dict(part)
        for context, bucket in model.transitions.items():
            dst = merged_transitions.setdefault(context, {})
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
        marginal=dict(merged_marginal),
        context_cols=list(context_cols.pop()),
        _vocab_size=vocab_sizes.pop(),
    )


FEDERATED_MODELS: dict[str, tuple[type, Any]] = {
    "frequency": (FrequencyBaseline, merge_frequency),
    "markov": (MarkovOrder1Baseline, merge_markov),
    "markov_order3": (MarkovOrder3Baseline, merge_markov),
}


def fit_model(model_name: str, train_df: pd.DataFrame, vocab: Vocabulary) -> Any:
    """Fit an additive federated model on local training data."""
    if model_name not in FEDERATED_MODELS:
        raise ValueError(
            f"Unknown federated model {model_name!r}; "
            f"choose from {sorted(FEDERATED_MODELS)}"
        )
    cls, _ = FEDERATED_MODELS[model_name]
    model = cls()
    model.fit(train_df, vocab)
    return model


def fit_params(model_name: str, train_df: pd.DataFrame, vocab: Vocabulary) -> dict[str, Any]:
    """Fit a federated model and return JSON-serializable parameters."""
    return fit_model(model_name, train_df, vocab).to_dict()


def merge_params(model_name: str, parts: list[dict[str, Any]]) -> Any:
    """Merge per-subject parameter dicts into one global model instance."""
    if model_name not in FEDERATED_MODELS:
        raise ValueError(
            f"Unknown federated model {model_name!r}; "
            f"choose from {sorted(FEDERATED_MODELS)}"
        )
    _, merge_fn = FEDERATED_MODELS[model_name]
    return merge_fn(parts)


def params_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Return whether two serialized model payloads are identical."""
    return left == right


def evaluate_predictor(
    model: BaselinePredictor,
    val_df: pd.DataFrame,
    vocab: Vocabulary,
) -> dict[str, float]:
    """Score a fitted predictor on a validation prefix frame."""
    X_val, y_val = split_xy(val_df)
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)
    return evaluate(y_val, y_pred, vocab=vocab, y_proba=y_proba)


@dataclass
class DecisionTreeModel:
    """sklearn decision tree on one-hot encoded prefix features.

    Unlike Markov counts, a fitted tree is not additive and is not intended
    for federated sum-merge in this step.
    """

    _tree: DecisionTreeClassifier | None = None
    classes_: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    _vocab_size: int = 0
    _prefix_cols: list[str] = field(default_factory=list)

    def fit(self, train_df: pd.DataFrame, vocab: Vocabulary) -> None:
        self._vocab_size = vocab.size
        self._prefix_cols = prefix_columns(train_df)
        self._tree = None
        self.classes_ = np.array([], dtype=int)
        if train_df.empty:
            return

        X_train, y_train = split_xy(train_df, prefix_cols=self._prefix_cols)
        X_encoded = onehot_encode(X_train, vocab, prefix_cols=self._prefix_cols)
        tree = DecisionTreeClassifier(random_state=0)
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
        X_encoded = onehot_encode(X, self._vocab_size, prefix_cols=cols)
        partial = self._tree.predict_proba(X_encoded)
        return _scatter_proba_excluding_pad(partial, self.classes_, self._vocab_size)

    def to_dict(self) -> dict[str, Any]:
        tree = self._tree
        return {
            "type": "tree",
            "params": tree.get_params() if tree is not None else {},
            "n_features": int(tree.n_features_in_) if tree is not None else 0,
            "classes": self.classes_.astype(int).tolist(),
            "feature_importances": (
                tree.feature_importances_.astype(float).tolist()
                if tree is not None
                else []
            ),
            "vocab_size": self._vocab_size,
            "prefix_cols": self._prefix_cols,
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
