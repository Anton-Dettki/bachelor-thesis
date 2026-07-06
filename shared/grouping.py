"""Behavioral profiling and client clustering for group-based prediction."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score

DEFAULT_K_RANGE = (2, 3, 4, 5, 6)
MIN_CLUSTER_SIZE = 2


@dataclass(frozen=True)
class ClusterResult:
    assignments: dict[str, int]
    silhouette: float
    n_clusters: int
    profile_columns: list[str]


def _normalize(counter: Counter[str]) -> dict[str, float]:
    total = sum(counter.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in counter.items()}


def _bigram_counts(events: Sequence[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for left, right in zip(events, events[1:]):
        counts[f"{left}\t{right}"] += 1
    return counts


def build_transition_profile(
    events: Sequence[str],
    vocabulary: Sequence[str] | None = None,
) -> dict[str, float]:
    """Normalized direct transition counts for a single client."""
    bigrams = _normalize(_bigram_counts(events))
    if vocabulary is None:
        return bigrams
    return {f"{left}\t{right}": bigrams.get(f"{left}\t{right}", 0.0) for left in vocabulary for right in vocabulary}


def build_frequency_profile(
    events: Sequence[str],
    vocabulary: Sequence[str] | None = None,
) -> dict[str, float]:
    """Normalized event frequency distribution for a single client."""
    freqs = _normalize(Counter(events))
    if vocabulary is None:
        return freqs
    return {token: freqs.get(token, 0.0) for token in vocabulary}


def build_task_profiles(
    events_by_task: Mapping[int, Sequence[str]],
    vocabulary: Sequence[str],
    n_tasks: int = 5,
) -> dict[str, float]:
    """Optional per-task frequency breakdown as additional profile dimensions."""
    profile: dict[str, float] = {}
    for task in range(1, n_tasks + 1):
        freqs = build_frequency_profile(events_by_task.get(task, ()), vocabulary)
        for token in vocabulary:
            profile[f"task{task}:{token}"] = freqs.get(token, 0.0)
    return profile


def build_client_profile(
    events: Sequence[str],
    *,
    vocabulary: Sequence[str] | None = None,
    events_by_task: Mapping[int, Sequence[str]] | None = None,
    include_task_breakdown: bool = False,
    n_tasks: int = 5,
) -> dict[str, float]:
    """Combine transition and frequency profiles into one behavioral vector."""
    if vocabulary is None:
        vocabulary = sorted(set(events))

    profile: dict[str, float] = {}
    transitions = build_transition_profile(events, vocabulary)
    frequencies = build_frequency_profile(events, vocabulary)
    for key, value in transitions.items():
        profile[f"tr:{key}"] = value
    for key, value in frequencies.items():
        profile[f"fr:{key}"] = value

    if include_task_breakdown and events_by_task is not None:
        task_part = build_task_profiles(events_by_task, vocabulary, n_tasks=n_tasks)
        profile.update({f"tk:{key}": value for key, value in task_part.items()})
    return profile


def profiles_to_matrix(
    profiles: Mapping[str, Mapping[str, float]],
) -> tuple[np.ndarray, list[str], list[str]]:
    """Convert client profile dicts into a dense matrix."""
    client_ids = sorted(profiles)
    columns = sorted({key for profile in profiles.values() for key in profile})
    matrix = np.zeros((len(client_ids), len(columns)), dtype=float)
    for row, client_id in enumerate(client_ids):
        profile = profiles[client_id]
        for col, name in enumerate(columns):
            matrix[row, col] = profile.get(name, 0.0)

    row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0
    matrix = matrix / row_norms
    return matrix, client_ids, columns


def _merge_small_clusters(
    labels: np.ndarray,
    matrix: np.ndarray,
    min_size: int = MIN_CLUSTER_SIZE,
) -> np.ndarray:
    """Merge clusters with fewer than min_size clients into the nearest neighbor."""
    labels = labels.copy()
    while True:
        counts = Counter(labels)
        tiny = [cluster for cluster, count in counts.items() if count < min_size]
        if not tiny:
            break

        for cluster_id in tiny:
            members = np.where(labels == cluster_id)[0]
            if len(members) == 0:
                continue
            centroid = matrix[members].mean(axis=0, keepdims=True)
            other_clusters = [c for c in counts if c != cluster_id and counts[c] >= min_size]
            if not other_clusters:
                other_clusters = [c for c in counts if c != cluster_id]
            if not other_clusters:
                break

            best_cluster = min(
                other_clusters,
                key=lambda candidate: np.linalg.norm(
                    centroid - matrix[np.where(labels == candidate)[0]].mean(axis=0)
                ),
            )
            labels[members] = best_cluster
    return labels


def select_n_clusters(
    matrix: np.ndarray,
    k_range: Sequence[int] = DEFAULT_K_RANGE,
) -> int:
    """Pick K with the best silhouette score."""
    if len(matrix) <= 2:
        return 1

    best_k = k_range[0]
    best_score = -1.0
    for k in k_range:
        if k >= len(matrix):
            continue
        model = KMeans(n_clusters=k, random_state=0, n_init=10)
        labels = model.fit_predict(matrix)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(matrix, labels)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k


def cluster_clients(
    profiles: Mapping[str, Mapping[str, float]],
    *,
    n_clusters: int | str = "auto",
    k_range: Sequence[int] = DEFAULT_K_RANGE,
    method: str = "kmeans",
) -> ClusterResult:
    """Cluster clients by behavioral profile similarity."""
    matrix, client_ids, columns = profiles_to_matrix(profiles)
    if len(client_ids) <= 1:
        assignments = {client_ids[0]: 0} if client_ids else {}
        return ClusterResult(assignments, 0.0, 1, columns)

    if n_clusters == "auto":
        k = select_n_clusters(matrix, k_range=k_range)
    else:
        k = max(1, min(int(n_clusters), len(client_ids)))

    if k == 1:
        labels = np.zeros(len(client_ids), dtype=int)
    elif method == "agglomerative":
        model = AgglomerativeClustering(n_clusters=k)
        labels = model.fit_predict(matrix)
    else:
        model = KMeans(n_clusters=k, random_state=0, n_init=10)
        labels = model.fit_predict(matrix)

    labels = _merge_small_clusters(labels, matrix)
    unique = sorted(set(labels))
    remap = {old: new for new, old in enumerate(unique)}
    labels = np.array([remap[label] for label in labels])

    silhouette = 0.0
    if len(unique) > 1 and len(client_ids) > len(unique):
        silhouette = float(silhouette_score(matrix, labels))

    assignments = {client_id: int(label) for client_id, label in zip(client_ids, labels)}
    return ClusterResult(assignments, silhouette, len(unique), columns)


def save_cluster_outputs(
    result: ClusterResult,
    profiles: Mapping[str, Mapping[str, float]],
    output_dir: Path,
    *,
    write_dendrogram: bool = True,
) -> None:
    """Persist cluster assignments, summary, and optional dendrogram."""
    output_dir.mkdir(parents=True, exist_ok=True)

    assignments_path = output_dir / "cluster_assignments.json"
    assignments_path.write_text(
        json.dumps(result.assignments, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    rows = []
    for client_id, profile in sorted(profiles.items()):
        row = {"client_id": client_id, "cluster_id": result.assignments[client_id]}
        row.update(profile)
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "behavioral_profiles.csv", index=False)

    cluster_members: dict[int, list[str]] = defaultdict(list)
    for client_id, cluster_id in result.assignments.items():
        cluster_members[cluster_id].append(client_id)

    summary_lines = [
        f"Clusters: {result.n_clusters}",
        f"Silhouette score: {result.silhouette:.4f}",
        "",
    ]
    for cluster_id in sorted(cluster_members):
        members = ", ".join(sorted(cluster_members[cluster_id]))
        summary_lines.append(f"Cluster {cluster_id} ({len(cluster_members[cluster_id])}): {members}")
    (output_dir / "cluster_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    if write_dendrogram and len(profiles) > 2:
        try:
            import matplotlib.pyplot as plt
            from scipy.cluster.hierarchy import dendrogram, linkage

            matrix, client_ids, _ = profiles_to_matrix(profiles)
            linkage_matrix = linkage(matrix, method="ward")
            fig, axis = plt.subplots(figsize=(12, 6))
            dendrogram(linkage_matrix, labels=client_ids, ax=axis, leaf_rotation=90)
            axis.set_title("Client behavioral profile dendrogram")
            fig.tight_layout()
            fig.savefig(output_dir / "cluster_dendrogram.png", dpi=150)
            plt.close(fig)
        except ImportError:
            pass
