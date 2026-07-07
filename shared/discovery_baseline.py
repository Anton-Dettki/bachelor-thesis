"""Discovery-style Markov baselines for next-activity prediction."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

Trace = Sequence[str]


@dataclass
class MarkovPredictor:
    """Variable-order Markov chain with backoff to shorter contexts."""

    transitions: dict[str, Counter[str]]
    bigram_context: dict[tuple[str, ...], Counter[str]]
    fallback_counts: Counter[str]
    default_label: int | str | None = None
    max_order: int = 3

    @classmethod
    def fit(
        cls,
        traces: Iterable[Trace],
        *,
        use_trigram: bool = False,
        max_order: int | None = None,
    ) -> "MarkovPredictor":
        order = max_order if max_order is not None else (3 if use_trigram else 1)
        order = max(1, order)
        transitions: dict[str, Counter[str]] = defaultdict(Counter)
        ngram_context: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        fallback_counts: Counter[str] = Counter()

        for trace in traces:
            fallback_counts.update(trace)
            for index in range(len(trace) - 1):
                current = trace[index]
                nxt = trace[index + 1]
                transitions[current][nxt] += 1
                for context_size in range(2, order + 1):
                    if index >= context_size - 1:
                        context = tuple(trace[index - context_size + 1 : index + 1])
                        ngram_context[context][nxt] += 1

        default = fallback_counts.most_common(1)[0][0] if fallback_counts else None
        return cls(
            transitions=dict(transitions),
            bigram_context=dict(ngram_context),
            fallback_counts=fallback_counts,
            default_label=default,
            max_order=order,
        )

    def has_context(self, prefix: Trace) -> bool:
        """Return True when this model has seen the prefix context before."""
        for order in range(min(self.max_order, len(prefix)), 1, -1):
            if self.bigram_context.get(tuple(prefix[-order:])):
                return True
        return bool(prefix and self.transitions.get(prefix[-1]))

    def predict_label(self, prefix: Trace, label_encoder: Mapping[str, int] | None = None) -> int | str | None:
        """Predict the next label from a prefix sequence."""
        for order in range(min(self.max_order, len(prefix)), 1, -1):
            context = tuple(prefix[-order:])
            counts = self.bigram_context.get(context)
            if counts:
                prediction = counts.most_common(1)[0][0]
                return label_encoder[prediction] if label_encoder else prediction

        if prefix:
            counts = self.transitions.get(prefix[-1])
            if counts:
                prediction = counts.most_common(1)[0][0]
                return label_encoder[prediction] if label_encoder else prediction

        if self.default_label is None:
            return None
        if label_encoder and self.default_label in label_encoder:
            return label_encoder[self.default_label]
        return self.default_label


def predict_samples(
    predictor: MarkovPredictor,
    prefixes: Sequence[Sequence[str]],
    *,
    label_encoder: Mapping[str, int] | None = None,
) -> np.ndarray:
    """Predict encoded labels for a list of prefixes."""
    predictions: list[int | str] = []
    for prefix in prefixes:
        prediction = predictor.predict_label(prefix, label_encoder=label_encoder)
        if prediction is None:
            prediction = predictor.default_label if predictor.default_label is not None else 0
        predictions.append(prediction)
    return np.asarray(predictions)


def fit_group_markov_models(
    traces_by_client: Mapping[str, list[Trace]],
    assignments: Mapping[str, int],
    *,
    use_trigram: bool = True,
    global_traces: Iterable[Trace] | None = None,
) -> dict[int, MarkovPredictor]:
    """Train one Markov model per cluster from pooled client traces."""
    traces_by_group: dict[int, list[Trace]] = defaultdict(list)
    for client_id, traces in traces_by_client.items():
        cluster_id = assignments[client_id]
        traces_by_group[cluster_id].extend(traces)
    global_pool = list(global_traces) if global_traces is not None else []
    models: dict[int, MarkovPredictor] = {}
    for group_id, traces in traces_by_group.items():
        training_pool = traces + global_pool if global_pool else traces
        models[group_id] = MarkovPredictor.fit(training_pool, use_trigram=True, max_order=4)
    return models


def predict_routed_markov(
    prefixes: Sequence[Sequence[str]],
    client_ids: Sequence[str],
    group_models: Mapping[int, MarkovPredictor],
    global_model: MarkovPredictor,
    assignments: Mapping[str, int],
    *,
    label_encoder: Mapping[str, int] | None = None,
) -> np.ndarray:
    """Route each prefix to its group's Markov model."""
    predictions: list[int | str] = []
    for prefix, client_id in zip(prefixes, client_ids):
        cluster_id = assignments.get(client_id)
        model = group_models.get(cluster_id, global_model) if cluster_id is not None else global_model
        if model is not global_model and not model.has_context(prefix):
            prediction = global_model.predict_label(prefix, label_encoder=label_encoder)
        else:
            prediction = model.predict_label(prefix, label_encoder=label_encoder)
        if prediction is None:
            prediction = global_model.predict_label(prefix, label_encoder=label_encoder)
        if prediction is None:
            prediction = global_model.default_label if global_model.default_label is not None else 0
        predictions.append(prediction)
    return np.asarray(predictions)
