"""Compact probabilistic workflow graphs from next-event predictions."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

Trace = Sequence[str]
PAD_TOKEN = "<PAD>"
DEFAULT_MIN_PROBABILITY = 0.05


@dataclass(frozen=True)
class WorkflowGraph:
    nodes: list[str]
    edges: list[dict[str, float | str]]
    density: float
    source: str = "predictions"

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return len(self.edges)


def count_transitions(traces: Iterable[Trace]) -> dict[str, Counter[str]]:
    """Count direct transitions between consecutive events in observed traces."""
    transitions: dict[str, Counter[str]] = defaultdict(Counter)
    for trace in traces:
        for left, right in zip(trace, trace[1:]):
            transitions[left][right] += 1
    return transitions


def _current_event(prefix: Sequence[str], *, pad_token: str = PAD_TOKEN) -> str | None:
    """Return the last observed event token in a prediction prefix."""
    for token in reversed(prefix):
        if token != pad_token:
            return token
    return None


def count_predicted_transitions(
    prefixes: Sequence[Trace],
    predictions: Sequence[int],
    inverse_event_map: Mapping[int, str],
    *,
    pad_token: str = PAD_TOKEN,
) -> dict[str, Counter[str]]:
    """Count transitions implied by next-event predictions on partial traces."""
    transitions: dict[str, Counter[str]] = defaultdict(Counter)
    for prefix, prediction in zip(prefixes, predictions):
        source = _current_event(prefix, pad_token=pad_token)
        if source is None:
            continue
        target = inverse_event_map.get(int(prediction))
        if target is None:
            continue
        transitions[source][target] += 1
    return transitions


def _workflow_graph_from_transition_counts(
    transitions: Mapping[str, Counter[str]],
    *,
    min_probability: float = DEFAULT_MIN_PROBABILITY,
    top_n_per_source: int | None = None,
    source: str = "predictions",
) -> WorkflowGraph:
    nodes = sorted(
        transitions.keys() | {target for counts in transitions.values() for target in counts}
    )
    edges: list[dict[str, float | str]] = []

    for source_node, targets in sorted(transitions.items()):
        total = sum(targets.values())
        if total <= 0:
            continue
        ranked = sorted(
            ((target, count / total) for target, count in targets.items()),
            key=lambda item: item[1],
            reverse=True,
        )
        if top_n_per_source is not None:
            ranked = ranked[:top_n_per_source]
        for target, probability in ranked:
            if probability >= min_probability:
                edges.append(
                    {
                        "from": source_node,
                        "to": target,
                        "probability": float(probability),
                    }
                )

    n_nodes = len(nodes)
    max_edges = n_nodes * (n_nodes - 1) if n_nodes > 1 else 1
    density = len(edges) / max_edges
    return WorkflowGraph(nodes=nodes, edges=edges, density=density, source=source)


def build_workflow_graph(
    traces: Iterable[Trace],
    *,
    min_probability: float = DEFAULT_MIN_PROBABILITY,
    top_n_per_source: int | None = None,
) -> WorkflowGraph:
    """Build a compact graph from observed trace transitions (discovery baseline)."""
    transitions = count_transitions(traces)
    return _workflow_graph_from_transition_counts(
        transitions,
        min_probability=min_probability,
        top_n_per_source=top_n_per_source,
        source="traces",
    )


def build_workflow_graph_from_predictions(
    prefixes: Sequence[Trace],
    predictions: Sequence[int],
    inverse_event_map: Mapping[int, str],
    *,
    pad_token: str = PAD_TOKEN,
    min_probability: float = DEFAULT_MIN_PROBABILITY,
    top_n_per_source: int | None = None,
) -> WorkflowGraph:
    """Build a compact graph from next-event predictions on partial traces."""
    transitions = count_predicted_transitions(
        prefixes,
        predictions,
        inverse_event_map,
        pad_token=pad_token,
    )
    return _workflow_graph_from_transition_counts(
        transitions,
        min_probability=min_probability,
        top_n_per_source=top_n_per_source,
        source="predictions",
    )


def save_workflow_graph(
    graph: WorkflowGraph,
    output_dir: Path,
    name: str,
    *,
    write_png: bool = True,
) -> None:
    """Export workflow graph as JSON and optional PNG."""
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": graph.source,
        "nodes": graph.nodes,
        "edges": graph.edges,
        "density": graph.density,
        "n_nodes": graph.n_nodes,
        "n_edges": graph.n_edges,
    }
    json_path = output_dir / f"{name}_workflow.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if not write_png or not graph.edges:
        return

    try:
        import matplotlib.pyplot as plt
        import networkx as nx

        directed = nx.DiGraph()
        directed.add_nodes_from(graph.nodes)
        for edge in graph.edges:
            directed.add_edge(
                str(edge["from"]),
                str(edge["to"]),
                weight=float(edge["probability"]),
            )

        fig, axis = plt.subplots(figsize=(10, 8))
        layout = nx.spring_layout(directed, seed=0)
        widths = [max(0.5, directed[u][v]["weight"] * 4) for u, v in directed.edges()]
        nx.draw_networkx_nodes(directed, layout, ax=axis, node_size=500, node_color="#6c9bd2")
        nx.draw_networkx_labels(directed, layout, ax=axis, font_size=7)
        nx.draw_networkx_edges(
            directed,
            layout,
            ax=axis,
            width=widths,
            arrows=True,
            arrowsize=10,
            edge_color="#444444",
        )
        axis.set_title(name.replace("_", " "))
        axis.axis("off")
        fig.tight_layout()
        fig.savefig(output_dir / f"{name}_workflow.png", dpi=150)
        plt.close(fig)
    except ImportError:
        pass


def graphs_from_group_predictions(
    prefixes_by_group: Mapping[int, Sequence[Trace]],
    predictions_by_group: Mapping[int, Sequence[int]],
    inverse_event_map: Mapping[int, str],
    output_dir: Path,
    *,
    min_probability: float = DEFAULT_MIN_PROBABILITY,
    top_n_per_source: int | None = None,
    write_png: bool = True,
) -> dict[int, WorkflowGraph]:
    """Build and save one predictive workflow graph per cluster group."""
    graphs: dict[int, WorkflowGraph] = {}
    for group_id in sorted(prefixes_by_group):
        prefixes = prefixes_by_group[group_id]
        predictions = predictions_by_group[group_id]
        graph = build_workflow_graph_from_predictions(
            prefixes,
            predictions,
            inverse_event_map,
            min_probability=min_probability,
            top_n_per_source=top_n_per_source,
        )
        graphs[group_id] = graph
        save_workflow_graph(
            graph,
            output_dir,
            f"group_{group_id}",
            write_png=write_png,
        )
    return graphs


def split_predictions_by_group(
    samples: Sequence,
    predictions: Sequence[int],
    assignments: Mapping[str, int],
    *,
    matched_clients: Iterable[str] | None = None,
) -> tuple[dict[int, list[Trace]], dict[int, list[int]]]:
    """Partition test prefixes and predictions by behavioral cluster."""
    matched = set(matched_clients) if matched_clients is not None else None
    prefixes_by_group: dict[int, list[Trace]] = defaultdict(list)
    predictions_by_group: dict[int, list[int]] = defaultdict(list)

    for sample, prediction in zip(samples, predictions):
        if matched is not None and sample.client_id not in matched:
            continue
        group_id = assignments.get(sample.client_id)
        if group_id is None:
            continue
        prefixes_by_group[group_id].append(sample.prefix)
        predictions_by_group[group_id].append(int(prediction))

    return dict(prefixes_by_group), dict(predictions_by_group)


def workflow_graph_summary(
    graph: WorkflowGraph,
    *,
    artifact_stem: str,
    group_id: int | None = None,
    clients: Sequence[str] | None = None,
    top_transitions: int = 8,
) -> dict[str, object]:
    """Serialize a workflow graph for dashboard display."""
    ranked_edges = sorted(
        graph.edges,
        key=lambda edge: float(edge["probability"]),
        reverse=True,
    )
    return {
        "id": artifact_stem,
        "group_id": group_id,
        "title": artifact_stem.replace("_", " "),
        "source": graph.source,
        "n_nodes": graph.n_nodes,
        "n_edges": graph.n_edges,
        "density": round(graph.density, 4),
        "image": f"{artifact_stem}_workflow.png",
        "json": f"{artifact_stem}_workflow.json",
        "client_count": len(clients) if clients is not None else None,
        "clients": sorted(clients) if clients is not None else [],
        "top_transitions": [
            {
                "from": edge["from"],
                "to": edge["to"],
                "probability": round(float(edge["probability"]), 4),
            }
            for edge in ranked_edges[:top_transitions]
        ],
    }


def workflow_graphs_payload(
    global_graph: WorkflowGraph,
    grouped_graphs: Mapping[int, WorkflowGraph],
    assignments: Mapping[str, int],
    *,
    min_probability: float = DEFAULT_MIN_PROBABILITY,
) -> dict[str, object]:
    """Build structured workflow metadata for the dashboard API."""
    clients_by_group: dict[int, list[str]] = defaultdict(list)
    for client_id, group_id in assignments.items():
        clients_by_group[group_id].append(client_id)

    return {
        "min_probability": min_probability,
        "global": workflow_graph_summary(global_graph, artifact_stem="global"),
        "groups": [
            workflow_graph_summary(
                graph,
                artifact_stem=f"group_{group_id}",
                group_id=group_id,
                clients=clients_by_group.get(group_id, []),
            )
            for group_id, graph in sorted(grouped_graphs.items())
        ],
    }
