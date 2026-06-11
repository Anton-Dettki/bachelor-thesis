"""Tests for order-1 Markov-to-predictive-graph conversion."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fpm.predict import MarkovOrder1Baseline, MarkovOrder3Baseline  # noqa: E402
from fpm.predictive_graph import markov_to_predictive_graph  # noqa: E402
from fpm.prefix import Vocabulary  # noqa: E402


def _vocab(size: int = 4) -> Vocabulary:
    names = ["<PAD>"] + [f"A{i}" for i in range(1, size)]
    return Vocabulary(activities=names)


def _order1_model(
    *,
    transitions: dict[str, dict[int, int]],
    alpha: float = 1.0,
    vocab_size: int = 4,
) -> MarkovOrder1Baseline:
    model = MarkovOrder1Baseline(
        order=1,
        alpha=alpha,
        transitions=transitions,
        marginal={},
        context_cols=["e2"],
    )
    model._vocab_size = vocab_size
    return model


class PredictiveGraphTests(unittest.TestCase):
    def test_order1_counts_become_smoothed_probabilities(self) -> None:
        vocab = _vocab()
        model = _order1_model(
            transitions={"1": {2: 3, 3: 1}},
            alpha=1.0,
            vocab_size=vocab.size,
        )

        graph = markov_to_predictive_graph(model, vocab)

        self.assertEqual(len(graph.edges), 2)
        by_target = {edge.target_id: edge for edge in graph.edges}
        self.assertAlmostEqual(by_target[2].probability, 4.0 / 7.0)
        self.assertAlmostEqual(by_target[3].probability, 2.0 / 7.0)
        self.assertEqual(by_target[2].count, 3)
        self.assertEqual(by_target[3].count, 1)

    def test_pad_source_and_target_are_not_emitted(self) -> None:
        vocab = _vocab()
        model = _order1_model(
            transitions={
                "0": {1: 5},
                "1": {0: 2, 2: 4},
            },
            vocab_size=vocab.size,
        )

        graph = markov_to_predictive_graph(model, vocab)

        self.assertEqual(len(graph.edges), 1)
        edge = graph.edges[0]
        self.assertEqual(edge.source_id, 1)
        self.assertEqual(edge.target_id, 2)
        self.assertNotIn("<PAD>", graph.nodes)
        self.assertTrue(all(edge.source_id != 0 for edge in graph.edges))
        self.assertTrue(all(edge.target_id != 0 for edge in graph.edges))

    def test_min_probability_filters_edges_and_preserves_candidate_stats(self) -> None:
        vocab = _vocab(size=5)
        model = _order1_model(
            transitions={"1": {2: 100, 3: 1}},
            alpha=1.0,
            vocab_size=vocab.size,
        )

        graph = markov_to_predictive_graph(model, vocab, min_probability=0.5)

        self.assertEqual(graph.stats.edges_total, 2)
        self.assertEqual(graph.stats.edges_after_filter, 1)
        self.assertEqual(len(graph.edges), 1)
        self.assertEqual(graph.edges[0].target_id, 2)
        self.assertEqual(graph.stats.sum_counts_after_filter, 100)
        self.assertAlmostEqual(
            graph.stats.sum_probability_after_filter,
            graph.edges[0].probability,
        )

    def test_edges_and_nodes_are_deterministically_ordered(self) -> None:
        vocab = _vocab(size=5)
        model = _order1_model(
            transitions={
                "3": {4: 1},
                "1": {2: 2},
                "2": {3: 1},
            },
            vocab_size=vocab.size,
        )

        graph = markov_to_predictive_graph(model, vocab)

        self.assertEqual(
            [(edge.source_id, edge.target_id) for edge in graph.edges],
            [(1, 2), (2, 3), (3, 4)],
        )
        self.assertEqual(graph.nodes, ["A1", "A2", "A3", "A4"])

    def test_order3_is_rejected_explicitly(self) -> None:
        vocab = _vocab()
        model = MarkovOrder3Baseline(
            order=3,
            transitions={"0,0,1": {2: 1}},
            marginal={},
            context_cols=["e0", "e1", "e2"],
        )
        model._vocab_size = vocab.size

        with self.assertRaisesRegex(ValueError, "order-1"):
            markov_to_predictive_graph(model, vocab)

    def test_invalid_min_probability_is_rejected(self) -> None:
        vocab = _vocab()
        model = _order1_model(transitions={"1": {2: 1}}, vocab_size=vocab.size)

        with self.assertRaises(ValueError):
            markov_to_predictive_graph(model, vocab, min_probability=-0.1)
        with self.assertRaises(ValueError):
            markov_to_predictive_graph(model, vocab, min_probability=1.1)


if __name__ == "__main__":
    unittest.main()
