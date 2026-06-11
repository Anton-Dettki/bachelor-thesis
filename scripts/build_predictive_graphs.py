#!/usr/bin/env python3
"""Build Phase 4 predictive graph artifacts for one order-1 Markov model."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fpm.predict import MarkovBaseline, write_json  # noqa: E402
from fpm.predictive_graph import PredictiveGraph, markov_to_predictive_graph  # noqa: E402
from fpm.prefix import Vocabulary  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert one persisted order-1 Markov model into Phase 4 graph "
            "artifacts (graph.json, stats.json, and optionally graph.png)."
        )
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to a persisted Markov model JSON (e.g. markov.json)",
    )
    parser.add_argument(
        "--vocab-path",
        type=Path,
        required=True,
        help="Path to vocabulary JSON (e.g. vocab.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for graph.json, stats.json, and graph.png",
    )
    parser.add_argument(
        "--min-probability",
        type=float,
        default=0.0,
        help="Drop edges whose smoothed probability is below this threshold (default: 0.0)",
    )
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="Skip writing graph.png",
    )
    return parser.parse_args(argv)


def require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def graph_to_dict(graph: PredictiveGraph) -> dict[str, Any]:
    return {
        "nodes": list(graph.nodes),
        "edges": [asdict(edge) for edge in graph.edges],
    }


def stats_to_dict(
    graph: PredictiveGraph,
    *,
    model_path: Path,
    vocab_path: Path,
    model_payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        **asdict(graph.stats),
        "model_path": str(model_path),
        "vocab_path": str(vocab_path),
        "model_type": model_payload.get("type", "markov"),
        "order": int(model_payload.get("order", 1)),
        "alpha": float(model_payload.get("alpha", 1.0)),
    }


def graph_to_pm4py_dfg(
    graph: PredictiveGraph,
) -> tuple[dict[tuple[str, str], int], dict[str, int], dict[str, int]]:
    """Convert a predictive graph into pm4py DFG inputs.

    pm4py expects integer edge frequencies, so PNG weights use raw transition
    counts while ``graph.json`` keeps the smoothed probabilities.
    """
    dfg = {(edge.source, edge.target): edge.count for edge in graph.edges}

    source_nodes = {edge.source for edge in graph.edges}
    target_nodes = {edge.target for edge in graph.edges}
    start_nodes = source_nodes - target_nodes
    end_nodes = target_nodes - source_nodes
    if not start_nodes:
        start_nodes = source_nodes
    if not end_nodes:
        end_nodes = target_nodes

    start_activities = {node: 1 for node in sorted(start_nodes)}
    end_activities = {node: 1 for node in sorted(end_nodes)}
    return dfg, start_activities, end_activities


def write_graph_png(graph: PredictiveGraph, path: Path) -> None:
    import pm4py

    path.parent.mkdir(parents=True, exist_ok=True)
    if not graph.edges:
        return

    dfg, start_activities, end_activities = graph_to_pm4py_dfg(graph)
    pm4py.save_vis_dfg(dfg, start_activities, end_activities, str(path))


def build_graph_artifacts(
    model_path: Path,
    vocab_path: Path,
    output_dir: Path,
    *,
    min_probability: float = 0.0,
    write_png: bool = True,
) -> dict[str, Path]:
    """Load one model, convert it, and write graph artifacts."""
    require_file(model_path, "Model file")
    require_file(vocab_path, "Vocabulary file")

    model_payload = json.loads(model_path.read_text(encoding="utf-8"))
    model = MarkovBaseline.from_dict(model_payload)
    vocab = Vocabulary.read_json(vocab_path)
    graph = markov_to_predictive_graph(model, vocab, min_probability=min_probability)

    output_dir.mkdir(parents=True, exist_ok=True)
    graph_path = output_dir / "graph.json"
    stats_path = output_dir / "stats.json"
    png_path = output_dir / "graph.png"

    write_json(graph_path, graph_to_dict(graph))
    write_json(
        stats_path,
        stats_to_dict(
            graph,
            model_path=model_path,
            vocab_path=vocab_path,
            model_payload=model_payload,
        ),
    )

    artifacts = {"graph_json": graph_path, "stats_json": stats_path}
    if write_png:
        write_graph_png(graph, png_path)
        artifacts["graph_png"] = png_path
    return artifacts


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    artifacts = build_graph_artifacts(
        args.model_path,
        args.vocab_path,
        args.output_dir,
        min_probability=args.min_probability,
        write_png=not args.no_png,
    )
    print(f"Wrote {artifacts['graph_json']}")
    print(f"Wrote {artifacts['stats_json']}")
    if "graph_png" in artifacts:
        print(f"Wrote {artifacts['graph_png']}")


if __name__ == "__main__":
    main()
