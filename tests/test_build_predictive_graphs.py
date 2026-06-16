"""Tests for the predictive graph builder CLI."""

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
    discover_group_scenarios,
    main,
    resolve_build_targets,
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


class BatchLayoutFixture:
    """Minimal Phase 3 layout for batch graph builder tests."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.models_dir = root / "models"
        self.prefix_dir = root / "prefix"
        self.group_models_dir = self.models_dir / "group"
        self.group_prefix_dir = self.prefix_dir / "group"
        self.graphs_dir = root / "graphs"

        self._write_model(self.models_dir / "global" / "markov.json")
        self._write_vocab(self.prefix_dir / "global" / "vocab.json")
        self._write_model(self.models_dir / "federated" / "markov.json")
        self._write_group("scenario1_shopping_mealprep")
        self._write_group("scenario2_no_sport")

    def _write_model(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_tiny_model_payload(), indent=2), encoding="utf-8")

    def _write_vocab(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_tiny_vocab_payload(), indent=2), encoding="utf-8")

    def _write_group(self, scenario: str) -> None:
        self._write_model(self.group_models_dir / scenario / "markov.json")
        self._write_vocab(self.group_prefix_dir / scenario / "vocab.json")


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

    def test_single_model_mode_does_not_write_manifest(self) -> None:
        main(
            [
                "--model-path",
                str(self.model_path),
                "--vocab-path",
                str(self.vocab_path),
                "--output-dir",
                str(self.output_dir),
                "--no-png",
            ]
        )

        self.assertTrue((self.output_dir / "graph.json").exists())
        self.assertFalse((self.root / "graphs" / "manifest.json").exists())


class BatchBuildPredictiveGraphsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.layout = BatchLayoutFixture(self.root)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _batch_args(self, *extra: str) -> list[str]:
        return [
            *extra,
            "--models-dir",
            str(self.layout.models_dir),
            "--prefix-dir",
            str(self.layout.prefix_dir),
            "--group-models-dir",
            str(self.layout.group_models_dir),
            "--group-prefix-dir",
            str(self.layout.group_prefix_dir),
            "--graphs-dir",
            str(self.layout.graphs_dir),
            "--no-png",
        ]

    def _manifest(self) -> dict:
        manifest_path = self.layout.graphs_dir / "manifest.json"
        self.assertTrue(manifest_path.exists())
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def test_scope_global_writes_graph_and_manifest(self) -> None:
        main(self._batch_args("--scope", "global"))

        graph_path = self.layout.graphs_dir / "global" / "markov" / "graph.json"
        self.assertTrue(graph_path.exists())
        graph_payload = json.loads(graph_path.read_text(encoding="utf-8"))
        self.assertEqual(graph_payload, _expected_graph_payload())

        manifest = self._manifest()
        self.assertEqual(len(manifest["graphs"]), 1)
        entry = manifest["graphs"][0]
        self.assertEqual(entry["scope"], "global")
        self.assertEqual(entry["model"], "markov")
        self.assertNotIn("scenario", entry)

    def test_scope_federated_uses_global_vocab(self) -> None:
        main(self._batch_args("--scope", "federated"))

        graph_path = self.layout.graphs_dir / "federated" / "markov" / "graph.json"
        stats_path = self.layout.graphs_dir / "federated" / "markov" / "stats.json"
        self.assertTrue(graph_path.exists())

        stats_payload = json.loads(stats_path.read_text(encoding="utf-8"))
        self.assertEqual(
            stats_payload["vocab_path"],
            str(self.layout.prefix_dir / "global" / "vocab.json"),
        )

        manifest = self._manifest()
        self.assertEqual(manifest["graphs"][0]["scope"], "federated")

    def test_scenario_writes_group_graph_with_scenario_in_manifest(self) -> None:
        main(self._batch_args("--scenario", "scenario2_no_sport"))

        graph_path = (
            self.layout.graphs_dir / "group" / "scenario2_no_sport" / "markov" / "graph.json"
        )
        self.assertTrue(graph_path.exists())

        manifest = self._manifest()
        entry = manifest["graphs"][0]
        self.assertEqual(entry["scope"], "group")
        self.assertEqual(entry["scenario"], "scenario2_no_sport")
        self.assertEqual(
            entry["output_dir"],
            str(self.layout.graphs_dir / "group" / "scenario2_no_sport" / "markov"),
        )

    def test_all_groups_discovers_scenarios_in_sorted_order(self) -> None:
        main(self._batch_args("--all-groups"))

        manifest = self._manifest()
        scenarios = [entry["scenario"] for entry in manifest["graphs"]]
        self.assertEqual(
            scenarios,
            ["scenario1_shopping_mealprep", "scenario2_no_sport"],
        )

    def test_discover_group_scenarios_requires_matching_vocab(self) -> None:
        orphan = self.layout.group_models_dir / "orphan_scenario"
        orphan.mkdir(parents=True, exist_ok=True)
        orphan.joinpath("markov.json").write_text(
            json.dumps(_tiny_model_payload(), indent=2),
            encoding="utf-8",
        )

        discovered = discover_group_scenarios(
            self.layout.group_models_dir,
            self.layout.group_prefix_dir,
        )
        self.assertEqual(
            discovered,
            ["scenario1_shopping_mealprep", "scenario2_no_sport"],
        )

    def test_batch_deduplicates_overlapping_selectors(self) -> None:
        import argparse

        args = argparse.Namespace(
            scopes=[],
            scenarios=["scenario2_no_sport"],
            all_groups=True,
            models_dir=self.layout.models_dir,
            prefix_dir=self.layout.prefix_dir,
            group_models_dir=self.layout.group_models_dir,
            group_prefix_dir=self.layout.group_prefix_dir,
            graphs_dir=self.layout.graphs_dir,
        )
        targets = resolve_build_targets(args)
        scenario_targets = [target for target in targets if target.scope == "group"]
        self.assertEqual(len(scenario_targets), 2)
        output_dirs = [str(target.output_dir) for target in targets]
        self.assertEqual(len(output_dirs), len(set(output_dirs)))


if __name__ == "__main__":
    unittest.main()
