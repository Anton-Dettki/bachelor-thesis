"""Tests for the single-model predictive graph builder CLI."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_predictive_graphs import (  # noqa: E402
    build_graph_artifacts,
    main,
)


def _tiny_model_payload() -> dict:
    return {
        "type": "markov",
        "order": 1,
        "alpha": 1.0,
        "transitions": {
            "3": {"4": 1},
            "1": {"2": 2},
            "2": {"3": 1},
        },
        "marginal": {},
        "context_cols": ["e2"],
        "vocab_size": 5,
    }


def _tiny_vocab_payload() -> dict:
    return {"activities": ["<PAD>", "A1", "A2", "A3", "A4"]}


def _expected_graph_payload() -> dict:
    return {
        "nodes": ["A1", "A2", "A3", "A4"],
        "edges": [
            {
                "source_id": 1,
                "source": "A1",
                "target_id": 2,
                "target": "A2",
                "count": 2,
                "probability": 0.5,
            },
            {
                "source_id": 2,
                "source": "A2",
                "target_id": 3,
                "target": "A3",
                "count": 1,
                "probability": 0.4,
            },
            {
                "source_id": 3,
                "source": "A3",
                "target_id": 4,
                "target": "A4",
                "count": 1,
                "probability": 0.4,
            },
        ],
    }


class BuildPredictiveGraphsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.model_path = self.root / "markov.json"
        self.vocab_path = self.root / "vocab.json"
        self.output_dir = self.root / "out"

        self.model_path.write_text(
            json.dumps(_tiny_model_payload(), indent=2),
            encoding="utf-8",
        )
        self.vocab_path.write_text(
            json.dumps(_tiny_vocab_payload(), indent=2),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_writes_deterministic_graph_and_stats_json(self) -> None:
        artifacts = build_graph_artifacts(
            self.model_path,
            self.vocab_path,
            self.output_dir,
            min_probability=0.0,
            write_png=False,
        )

        graph_payload = json.loads(artifacts["graph_json"].read_text(encoding="utf-8"))
        stats_payload = json.loads(artifacts["stats_json"].read_text(encoding="utf-8"))

        self.assertEqual(graph_payload, _expected_graph_payload())
        self.assertEqual(stats_payload["nodes"], 4)
        self.assertEqual(stats_payload["edges_total"], 3)
        self.assertEqual(stats_payload["edges_after_filter"], 3)
        self.assertEqual(stats_payload["min_probability"], 0.0)
        self.assertEqual(stats_payload["sum_counts_after_filter"], 4)
        self.assertAlmostEqual(stats_payload["sum_probability_after_filter"], 1.3)
        self.assertEqual(stats_payload["model_path"], str(self.model_path))
        self.assertEqual(stats_payload["vocab_path"], str(self.vocab_path))
        self.assertEqual(stats_payload["model_type"], "markov")
        self.assertEqual(stats_payload["order"], 1)
        self.assertEqual(stats_payload["alpha"], 1.0)

    def test_min_probability_filters_written_graph(self) -> None:
        build_graph_artifacts(
            self.model_path,
            self.vocab_path,
            self.output_dir,
            min_probability=0.5,
            write_png=False,
        )

        graph_payload = json.loads(
            (self.output_dir / "graph.json").read_text(encoding="utf-8")
        )
        stats_payload = json.loads(
            (self.output_dir / "stats.json").read_text(encoding="utf-8")
        )

        self.assertEqual(len(graph_payload["edges"]), 1)
        self.assertEqual(graph_payload["edges"][0]["target_id"], 2)
        self.assertEqual(stats_payload["edges_total"], 3)
        self.assertEqual(stats_payload["edges_after_filter"], 1)
        self.assertEqual(stats_payload["min_probability"], 0.5)

    def test_missing_model_path_fails_clearly(self) -> None:
        missing = self.root / "missing-markov.json"
        with self.assertRaisesRegex(FileNotFoundError, "Model file not found"):
            build_graph_artifacts(
                missing,
                self.vocab_path,
                self.output_dir,
                write_png=False,
            )

    def test_missing_vocab_path_fails_clearly(self) -> None:
        missing = self.root / "missing-vocab.json"
        with self.assertRaisesRegex(FileNotFoundError, "Vocabulary file not found"):
            build_graph_artifacts(
                self.model_path,
                missing,
                self.output_dir,
                write_png=False,
            )

    def test_cli_writes_png_when_enabled(self) -> None:
        main(
            [
                "--model-path",
                str(self.model_path),
                "--vocab-path",
                str(self.vocab_path),
                "--output-dir",
                str(self.output_dir),
            ]
        )

        png_path = self.output_dir / "graph.png"
        self.assertTrue(png_path.exists())
        self.assertGreater(png_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
