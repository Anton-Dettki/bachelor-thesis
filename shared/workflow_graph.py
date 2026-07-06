"""Probabilistic workflow graphs from event traces."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

Trace = Sequence[str]


@dataclass(frozen=True)
class WorkflowGraph:
    nodes: list[str]
    edges: list[dict[str, float | str]]
    density: float

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return len(self.edges)


def count_transitions(traces: Iterable[Trace]) -> dict[str, Counter[str]]:
    """Count direct transitions between consecutive events."""
    transitions: dict[str, Counter[str]] = defaultdict(Counter)
    for trace in traces:
        for left, right in zip(trace, trace[1:]):
            transitions[left][right] += 1
    return transitions


def build_workflow_graph(
    traces: Iterable[Trace],
    *,
    min_probability: float = 0.05,
    top_n_per_source: int | None = None,
) -> WorkflowGraph:
    """Build a compact probabilistic transition graph from traces."""
    transitions = count_transitions(traces)
    nodes = sorted(transitions.keys() | {target for counts in transitions.values() for target in counts})
    edges: list[dict[str, float | str]] = []

    for source, targets in sorted(transitions.items()):
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
                edges.append({"from": source, "to": target, "probability": float(probability)})

    n_nodes = len(nodes)
    max_edges = n_nodes * (n_nodes - 1) if n_nodes > 1 else 1
    density = len(edges) / max_edges
    return WorkflowGraph(nodes=nodes, edges=edges, density=density)


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


def graphs_from_group_traces(
    traces_by_group: Mapping[int, list[Trace]],
    output_dir: Path,
    *,
    min_probability: float = 0.05,
    write_png: bool = True,
) -> dict[int, WorkflowGraph]:
    """Build and save one workflow graph per cluster group."""
    graphs: dict[int, WorkflowGraph] = {}
    for group_id, traces in sorted(traces_by_group.items()):
        graph = build_workflow_graph(traces, min_probability=min_probability)
        graphs[group_id] = graph
        save_workflow_graph(
            graph,
            output_dir,
            f"group_{group_id}",
            write_png=write_png,
        )
    return graphs
