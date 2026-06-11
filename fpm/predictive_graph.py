"""Convert order-1 Markov baselines into in-memory predictive workflow graphs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fpm.predict import PAD_ID, MarkovBaseline
from fpm.prefix import Vocabulary


@dataclass(frozen=True)
class PredictiveGraphEdge:
    source_id: int
    source: str
    target_id: int
    target: str
    count: int
    probability: float


@dataclass(frozen=True)
class PredictiveGraphStats:
    nodes: int
    edges_total: int
    edges_after_filter: int
    min_probability: float
    sum_counts_after_filter: int
    sum_probability_after_filter: float


@dataclass(frozen=True)
class PredictiveGraph:
    nodes: list[str]
    edges: list[PredictiveGraphEdge]
    stats: PredictiveGraphStats


def _vocab_size(model: MarkovBaseline, vocab: Vocabulary) -> int:
    if model._vocab_size > 0:
        return model._vocab_size
    return vocab.size


def _smoothed_context_probs(
    model: MarkovBaseline,
    context_key: str,
    vocab_size: int,
) -> np.ndarray:
    """Match ``MarkovBaseline._smoothed_context`` without mutating the model."""
    proba = np.full(vocab_size, model.alpha, dtype=float)
    for cls_id, count in model.transitions.get(context_key, {}).items():
        proba[cls_id] += count
    return MarkovBaseline._normalize_excluding_pad(proba)


def markov_to_predictive_graph(
    model: MarkovBaseline,
    vocab: Vocabulary,
    *,
    min_probability: float = 0.0,
) -> PredictiveGraph:
    """Convert one order-1 Markov model into a compact predictive workflow graph."""
    if model.order != 1:
        raise ValueError(
            "Predictive graph conversion currently supports only order-1 Markov models"
        )
    if min_probability < 0.0 or min_probability > 1.0:
        raise ValueError(
            f"min_probability must be between 0 and 1, got {min_probability}"
        )

    vocab_size = _vocab_size(model, vocab)
    candidates: list[PredictiveGraphEdge] = []

    for source_key, target_counts in model.transitions.items():
        source_id = int(source_key)
        if source_id == PAD_ID:
            continue

        proba = _smoothed_context_probs(model, source_key, vocab_size)
        source_label = vocab.decode(source_id)

        for target_id, count in target_counts.items():
            if target_id == PAD_ID or count <= 0:
                continue

            candidates.append(
                PredictiveGraphEdge(
                    source_id=source_id,
                    source=source_label,
                    target_id=target_id,
                    target=vocab.decode(target_id),
                    count=count,
                    probability=float(proba[target_id]),
                )
            )

    candidates.sort(key=lambda edge: (edge.source_id, edge.target_id))
    filtered = [
        edge for edge in candidates if edge.probability >= min_probability
    ]

    node_ids = sorted(
        {edge.source_id for edge in filtered} | {edge.target_id for edge in filtered}
    )
    nodes = [vocab.decode(node_id) for node_id in node_ids]

    stats = PredictiveGraphStats(
        nodes=len(nodes),
        edges_total=len(candidates),
        edges_after_filter=len(filtered),
        min_probability=min_probability,
        sum_counts_after_filter=sum(edge.count for edge in filtered),
        sum_probability_after_filter=sum(edge.probability for edge in filtered),
    )
    return PredictiveGraph(nodes=nodes, edges=filtered, stats=stats)
