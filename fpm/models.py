"""Local next-event prediction models used by each federated client."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Protocol, Sequence

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text

Trace = Sequence[str]
DEFAULT_WINDOW = 3
DEFAULT_TREE_MAX_DEPTH = 8


def _prefix_features(prefix: Trace, window: int = DEFAULT_WINDOW) -> dict[str, str | int]:
    events = list(prefix[-window:])
    features: dict[str, str | int] = {"prefix_len": len(prefix)}
    for pos in range(window):
        event = events[-(pos + 1)] if pos < len(events) else "<START>"
        features[f"prev_{pos + 1}"] = event
    return features


def _export_tree_nodes(
    tree: DecisionTreeClassifier,
    feature_names: list[str],
) -> list[dict[str, Any]]:
    tree_ = tree.tree_
    nodes: list[dict[str, Any]] = []
    for node_id in range(tree_.node_count):
        if tree_.feature[node_id] >= 0:
            feature = feature_names[tree_.feature[node_id]]
            threshold = float(tree_.threshold[node_id])
            split = f"{feature} <= {threshold:.4f}"
        else:
            feature = None
            threshold = None
            split = "leaf"
        nodes.append(
            {
                "id": node_id,
                "split": split,
                "feature": feature,
                "threshold": threshold,
                "left": int(tree_.children_left[node_id]),
                "right": int(tree_.children_right[node_id]),
                "n_samples": int(tree_.n_node_samples[node_id]),
                "class_distribution": {
                    str(label): int(count)
                    for label, count in zip(tree.classes_, tree_.value[node_id][0])
                },
            }
        )
    return nodes


class NextEventModel(Protocol):
    name: str

    def fit(self, traces: Sequence[Trace]) -> None:
        ...

    def predict_next(self, prefix: Trace) -> str | None:
        ...

    def params(self) -> dict[str, Any]:
        ...


def _pairs(traces: Iterable[Trace]) -> Iterable[tuple[tuple[str, ...], str]]:
    for trace in traces:
        for idx in range(1, len(trace)):
            yield tuple(trace[:idx]), trace[idx]


def _event_totals(traces: Iterable[Trace]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for trace in traces:
        counts.update(trace)
    return counts


class FrequencyModel:
    name = "frequency"

    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self.default: str | None = None

    def fit(self, traces: Sequence[Trace]) -> None:
        self.counts = _event_totals(traces)
        self.default = self.counts.most_common(1)[0][0] if self.counts else None

    def predict_next(self, prefix: Trace) -> str | None:
        return self.default

    def params(self) -> dict[str, Any]:
        return {
            "type": self.name,
            "event_counts": dict(self.counts),
            "default": self.default,
        }


class MarkovModel:
    name = "markov"

    def __init__(self) -> None:
        self.transitions: dict[str, Counter[str]] = defaultdict(Counter)
        self.fallback = FrequencyModel()

    def fit(self, traces: Sequence[Trace]) -> None:
        self.transitions = defaultdict(Counter)
        self.fallback.fit(traces)
        for trace in traces:
            for current, nxt in zip(trace, trace[1:]):
                self.transitions[current][nxt] += 1

    def predict_next(self, prefix: Trace) -> str | None:
        if prefix:
            counts = self.transitions.get(prefix[-1])
            if counts:
                return counts.most_common(1)[0][0]
        return self.fallback.predict_next(prefix)

    def params(self) -> dict[str, Any]:
        return {
            "type": self.name,
            "transitions": {
                source: dict(targets) for source, targets in self.transitions.items()
            },
            "fallback": self.fallback.params(),
        }


class PrefixWindowModel:
    """Shared prefix-feature training flow for sklearn classifiers."""

    name = "prefix_model"

    def __init__(
        self,
        model: Any,
        window: int = DEFAULT_WINDOW,
        min_examples: int = 2,
        min_classes: int = 2,
        fallback_reason: str = "not enough data for model",
    ) -> None:
        self.window = window
        self.min_examples = min_examples
        self.min_classes = min_classes
        self.fallback_reason = fallback_reason
        self.vectorizer = DictVectorizer(sparse=False)
        self.model = model
        self.fallback = MarkovModel()
        self.is_fitted = False
        self.fit_error: str | None = None

    def fit(self, traces: Sequence[Trace]) -> None:
        self.fallback.fit(traces)
        examples = list(_pairs(traces))
        labels = sorted({label for _, label in examples})
        if len(examples) < self.min_examples or len(labels) < self.min_classes:
            self.is_fitted = False
            self.fit_error = self.fallback_reason
            return

        x_dicts = [_prefix_features(prefix, self.window) for prefix, _ in examples]
        y = np.array([label for _, label in examples])
        x = self.vectorizer.fit_transform(x_dicts)
        self.model.fit(x, y)
        self.is_fitted = True
        self.fit_error = None

    def predict_next(self, prefix: Trace) -> str | None:
        if not self.is_fitted:
            return self.fallback.predict_next(prefix)
        x = self.vectorizer.transform([_prefix_features(prefix, self.window)])
        return str(self.model.predict(x)[0])

    def feature_names(self) -> list[str]:
        return self.vectorizer.get_feature_names_out().tolist()


class DecisionTreeModel(PrefixWindowModel):
    name = "tree"

    def __init__(self, window: int = DEFAULT_WINDOW, max_depth: int = DEFAULT_TREE_MAX_DEPTH) -> None:
        self.max_depth = max_depth
        super().__init__(
            model=DecisionTreeClassifier(max_depth=max_depth, random_state=0),
            window=window,
            fallback_reason="not enough next-event classes for decision tree",
        )

    def params(self) -> dict[str, Any]:
        if not self.is_fitted:
            return {
                "type": self.name,
                "fitted": False,
                "fallback_reason": self.fit_error,
                "fallback": self.fallback.params(),
            }

        feature_names = self.feature_names()
        return {
            "type": self.name,
            "fitted": True,
            "window": self.window,
            "max_depth": self.max_depth,
            "classes": self.model.classes_.tolist(),
            "feature_names": feature_names,
            "n_nodes": int(self.model.tree_.node_count),
            "n_leaves": int(self.model.get_n_leaves()),
            "rules": export_text(
                self.model,
                feature_names=feature_names,
                max_depth=6,
            ),
            "nodes": _export_tree_nodes(self.model, feature_names),
        }


class LogisticRegressionModel(PrefixWindowModel):
    name = "logreg"

    def __init__(self, window: int = DEFAULT_WINDOW) -> None:
        super().__init__(
            model=LogisticRegression(max_iter=500, solver="lbfgs"),
            window=window,
            fallback_reason="not enough next-event classes for logistic regression",
        )

    def params(self) -> dict[str, Any]:
        if not self.is_fitted:
            return {
                "type": self.name,
                "fitted": False,
                "fallback_reason": self.fit_error,
                "fallback": self.fallback.params(),
            }

        return {
            "type": self.name,
            "fitted": True,
            "window": self.window,
            "classes": self.model.classes_.tolist(),
            "feature_names": self.feature_names(),
            "coef": np.round(self.model.coef_, 6).tolist(),
            "intercept": np.round(self.model.intercept_, 6).tolist(),
        }


def create_model(name: str) -> NextEventModel:
    normalized = name.strip().lower()
    if normalized == "tree":
        return DecisionTreeModel()
    if normalized == "frequency":
        return FrequencyModel()
    if normalized == "markov":
        return MarkovModel()
    if normalized == "logreg":
        return LogisticRegressionModel()
    raise ValueError(
        f"Unknown model {name!r}; expected tree, frequency, markov, or logreg"
    )


@dataclass(frozen=True)
class Evaluation:
    accuracy: float
    correct: int
    total: int


def evaluate(model: NextEventModel, traces: Sequence[Trace]) -> Evaluation:
    correct = 0
    total = 0
    for prefix, label in _pairs(traces):
        prediction = model.predict_next(prefix)
        correct += int(prediction == label)
        total += 1
    return Evaluation(
        accuracy=(correct / total if total else 0.0),
        correct=correct,
        total=total,
    )


def train_and_evaluate(
    model_name: str,
    train_traces: Sequence[Trace],
    eval_traces: Sequence[Trace],
) -> dict[str, Any]:
    model = create_model(model_name)
    model.fit(train_traces)
    evaluation = evaluate(model, eval_traces)
    return {
        "model": model.name,
        "accuracy": evaluation.accuracy,
        "correct": evaluation.correct,
        "total": evaluation.total,
        "params": model.params(),
    }
