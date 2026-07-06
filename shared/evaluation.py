"""Evaluation metrics and comparison tables for grouped prediction."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


@dataclass
class ApproachResult:
    name: str
    y_true: np.ndarray
    y_pred: np.ndarray
    graph_nodes: int | None = None
    graph_edges: int | None = None
    train_seconds: float | None = None
    cluster_metrics: dict[int, float] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        if len(self.y_true) == 0:
            return 0.0
        return float(accuracy_score(self.y_true, self.y_pred))

    @property
    def macro_f1(self) -> float:
        if len(self.y_true) == 0:
            return 0.0
        return float(f1_score(self.y_true, self.y_pred, average="macro", zero_division=0))

    @property
    def weighted_f1(self) -> float:
        if len(self.y_true) == 0:
            return 0.0
        return float(f1_score(self.y_true, y_pred=self.y_pred, average="weighted", zero_division=0))


def per_cluster_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cluster_ids: Sequence[int],
) -> dict[int, float]:
    """Compute accuracy for each cluster on routed test samples."""
    metrics: dict[int, float] = {}
    cluster_ids_arr = np.asarray(cluster_ids)
    for cluster_id in sorted(set(cluster_ids_arr)):
        mask = cluster_ids_arr == cluster_id
        if not mask.any():
            continue
        metrics[int(cluster_id)] = float(accuracy_score(y_true[mask], y_pred[mask]))
    return metrics


def comparison_rows(results: Sequence[ApproachResult]) -> list[dict[str, object]]:
    """Build rows for grouped comparison tables."""
    rows: list[dict[str, object]] = []
    for result in results:
        rows.append(
            {
                "Approach": result.name,
                "Accuracy": round(result.accuracy, 4),
                "Macro_F1": round(result.macro_f1, 4),
                "Weighted_F1": round(result.weighted_f1, 4),
                "Graph_Nodes": result.graph_nodes if result.graph_nodes is not None else "N/A",
                "Graph_Edges": result.graph_edges if result.graph_edges is not None else "N/A",
                "Train_Time_s": round(result.train_seconds, 2) if result.train_seconds is not None else "N/A",
            }
        )
    return rows


def save_comparison(
    results: Sequence[ApproachResult],
    output_dir: Path,
    *,
    extra_lines: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Write unified comparison table to CSV and text."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(comparison_rows(results))
    frame.to_csv(output_dir / "grouped_comparison.csv", index=False)

    lines = ["Grouped prediction comparison", "=" * 72, frame.to_string(index=False), ""]
    for result in results:
        if result.cluster_metrics:
            parts = ", ".join(
                f"cluster {cluster_id}: {accuracy:.3f}"
                for cluster_id, accuracy in sorted(result.cluster_metrics.items())
            )
            lines.append(f"{result.name} per-cluster accuracy: {parts}")

    if extra_lines:
        lines.extend(extra_lines)

    (output_dir / "grouped_comparison.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return frame


def print_comparison(results: Sequence[ApproachResult]) -> None:
    """Print comparison table to stdout."""
    frame = pd.DataFrame(comparison_rows(results))
    print(frame.to_string(index=False))
