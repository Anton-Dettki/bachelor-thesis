#!/usr/bin/env python3
"""Build Phase 4 predictive graph artifacts for Markov models."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fpm.predict import (  # noqa: E402
    DEFAULT_MODEL_DIR,
    MarkovBaseline,
    write_json,
)
from fpm.predictive_graph import PredictiveGraph, markov_to_predictive_graph  # noqa: E402
from fpm.prefix import DEFAULT_PREFIX_DIR  # noqa: E402
from fpm.prefix import Vocabulary  # noqa: E402
from fpm.queries import SCENARIO_QUERIES  # noqa: E402

DEFAULT_GRAPHS_DIR = ROOT / "output" / "graphs"
DEFAULT_GROUP_MODEL_DIR = DEFAULT_MODEL_DIR / "group"
DEFAULT_GROUP_PREFIX_DIR = DEFAULT_PREFIX_DIR / "group"
DEFAULT_MODEL_NAME = "markov"


@dataclass(frozen=True)
class GraphBuildTarget:
    scope: str
    model: str
    model_path: Path
    vocab_path: Path
    output_dir: Path
    scenario: str | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert persisted order-1 Markov models into Phase 4 graph "
            "artifacts (graph.json, stats.json, and optionally graph.png)."
        )
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to a persisted Markov model JSON (single-model mode)",
    )
    parser.add_argument(
        "--vocab-path",
        type=Path,
        default=None,
        help="Path to vocabulary JSON (single-model mode)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for graph.json, stats.json, and graph.png (single-model mode)",
    )
    parser.add_argument(
        "--scope",
        dest="scopes",
        action="append",
        choices=("global", "federated"),
        help="Batch scope to build (repeatable: global, federated)",
    )
    parser.add_argument(
        "--scenario",
        dest="scenarios",
        action="append",
        default=None,
        help="Group scenario key (repeatable, e.g. scenario2_no_sport)",
    )
    parser.add_argument(
        "--all-groups",
        action="store_true",
        help="Build graphs for every group scenario with model and vocab artifacts",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Root directory for local/global model artifacts",
    )
    parser.add_argument(
        "--prefix-dir",
        type=Path,
        default=DEFAULT_PREFIX_DIR,
        help="Root directory for prefix datasets",
    )
    parser.add_argument(
        "--group-models-dir",
        type=Path,
        default=DEFAULT_GROUP_MODEL_DIR,
        help="Directory containing group model artifacts",
    )
    parser.add_argument(
        "--group-prefix-dir",
        type=Path,
        default=DEFAULT_GROUP_PREFIX_DIR,
        help="Directory containing group prefix datasets",
    )
    parser.add_argument(
        "--graphs-dir",
        type=Path,
        default=DEFAULT_GRAPHS_DIR,
        help="Root directory for graph artifacts and manifest.json",
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


def global_target(
    *,
    models_dir: Path,
    prefix_dir: Path,
    graphs_dir: Path,
    model_name: str = DEFAULT_MODEL_NAME,
) -> GraphBuildTarget:
    return GraphBuildTarget(
        scope="global",
        model=model_name,
        model_path=models_dir / "global" / f"{model_name}.json",
        vocab_path=prefix_dir / "global" / "vocab.json",
        output_dir=graphs_dir / "global" / model_name,
    )


def federated_target(
    *,
    federated_dir: Path,
    prefix_dir: Path,
    graphs_dir: Path,
    model_name: str = DEFAULT_MODEL_NAME,
) -> GraphBuildTarget:
    return GraphBuildTarget(
        scope="federated",
        model=model_name,
        model_path=federated_dir / f"{model_name}.json",
        vocab_path=prefix_dir / "global" / "vocab.json",
        output_dir=graphs_dir / "federated" / model_name,
    )


def group_target(
    scenario: str,
    *,
    group_models_dir: Path,
    group_prefix_dir: Path,
    graphs_dir: Path,
    model_name: str = DEFAULT_MODEL_NAME,
) -> GraphBuildTarget:
    if scenario not in SCENARIO_QUERIES:
        known = ", ".join(sorted(SCENARIO_QUERIES))
        raise ValueError(f"Unknown scenario {scenario!r}; choose from {known}")
    return GraphBuildTarget(
        scope="group",
        model=model_name,
        scenario=scenario,
        model_path=group_models_dir / scenario / f"{model_name}.json",
        vocab_path=group_prefix_dir / scenario / "vocab.json",
        output_dir=graphs_dir / "group" / scenario / model_name,
    )


def discover_group_scenarios(
    group_models_dir: Path,
    group_prefix_dir: Path,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
) -> list[str]:
    if not group_models_dir.exists():
        return []
    scenarios: list[str] = []
    for scenario_dir in sorted(group_models_dir.iterdir()):
        if not scenario_dir.is_dir():
            continue
        model_path = scenario_dir / f"{model_name}.json"
        vocab_path = group_prefix_dir / scenario_dir.name / "vocab.json"
        if model_path.exists() and vocab_path.exists():
            scenarios.append(scenario_dir.name)
    return scenarios


def resolve_build_targets(args: argparse.Namespace) -> list[GraphBuildTarget]:
    targets: list[GraphBuildTarget] = []
    federated_dir = args.models_dir / "federated"

    for scope in args.scopes or []:
        if scope == "global":
            targets.append(
                global_target(
                    models_dir=args.models_dir,
                    prefix_dir=args.prefix_dir,
                    graphs_dir=args.graphs_dir,
                )
            )
        elif scope == "federated":
            targets.append(
                federated_target(
                    federated_dir=federated_dir,
                    prefix_dir=args.prefix_dir,
                    graphs_dir=args.graphs_dir,
                )
            )

    for scenario in args.scenarios or []:
        targets.append(
            group_target(
                scenario,
                group_models_dir=args.group_models_dir,
                group_prefix_dir=args.group_prefix_dir,
                graphs_dir=args.graphs_dir,
            )
        )

    if args.all_groups:
        for scenario in discover_group_scenarios(
            args.group_models_dir,
            args.group_prefix_dir,
        ):
            targets.append(
                group_target(
                    scenario,
                    group_models_dir=args.group_models_dir,
                    group_prefix_dir=args.group_prefix_dir,
                    graphs_dir=args.graphs_dir,
                )
            )

    deduped: dict[Path, GraphBuildTarget] = {}
    for target in targets:
        deduped[target.output_dir.resolve()] = target
    return sorted(deduped.values(), key=lambda target: str(target.output_dir))


def manifest_entry(
    target: GraphBuildTarget,
    artifacts: dict[str, Path],
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "scope": target.scope,
        "model": target.model,
        "model_path": str(target.model_path),
        "vocab_path": str(target.vocab_path),
        "output_dir": str(target.output_dir),
        "artifacts": {
            "graph_json": str(artifacts["graph_json"]),
            "stats_json": str(artifacts["stats_json"]),
        },
    }
    if target.scenario is not None:
        entry["scenario"] = target.scenario
    if "graph_png" in artifacts:
        entry["artifacts"]["graph_png"] = str(artifacts["graph_png"])
    return entry


def build_batch(
    targets: list[GraphBuildTarget],
    *,
    graphs_dir: Path,
    min_probability: float,
    write_png: bool,
) -> Path:
    if not targets:
        raise ValueError("No graph build targets resolved")

    graphs_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for target in targets:
        artifacts = build_graph_artifacts(
            target.model_path,
            target.vocab_path,
            target.output_dir,
            min_probability=min_probability,
            write_png=write_png,
        )
        entries.append(manifest_entry(target, artifacts))
        print(f"Wrote {artifacts['graph_json']}")
        print(f"Wrote {artifacts['stats_json']}")
        if "graph_png" in artifacts:
            print(f"Wrote {artifacts['graph_png']}")

    manifest_path = graphs_dir / "manifest.json"
    write_json(manifest_path, {"graphs": entries})
    print(f"Wrote {manifest_path}")
    return manifest_path


def resolve_mode(args: argparse.Namespace) -> str:
    single_args = (args.model_path, args.vocab_path, args.output_dir)
    if any(single_args):
        if not all(single_args):
            raise SystemExit(
                "Single-model mode requires --model-path, --vocab-path, "
                "and --output-dir together."
            )
        return "single"

    if args.scopes or args.scenarios or args.all_groups:
        return "batch"

    raise SystemExit(
        "Provide either single-model paths "
        "(--model-path, --vocab-path, --output-dir) "
        "or batch selectors (--scope, --scenario, --all-groups)."
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    mode = resolve_mode(args)
    write_png = not args.no_png

    if mode == "single":
        artifacts = build_graph_artifacts(
            args.model_path,
            args.vocab_path,
            args.output_dir,
            min_probability=args.min_probability,
            write_png=write_png,
        )
        print(f"Wrote {artifacts['graph_json']}")
        print(f"Wrote {artifacts['stats_json']}")
        if "graph_png" in artifacts:
            print(f"Wrote {artifacts['graph_png']}")
        return

    targets = resolve_build_targets(args)
    build_batch(
        targets,
        graphs_dir=args.graphs_dir,
        min_probability=args.min_probability,
        write_png=write_png,
    )


if __name__ == "__main__":
    main()
