"""Reusable grouped prediction evaluation for the federated dashboard and CASAS2 CLI."""

from __future__ import annotations

import os
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

if not os.environ.get("LOKY_MAX_CPU_COUNT"):
    os.environ["LOKY_MAX_CPU_COUNT"] = "1"
if not os.environ.get("MPLBACKEND"):
    os.environ["MPLBACKEND"] = "Agg"
if not os.environ.get("MPLCONFIGDIR"):
    os.environ["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "matplotlib")

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import MaxAbsScaler
from sklearn.tree import DecisionTreeClassifier

from CASAS2.main import (
    PAD,
    Sample,
    build_samples,
    build_vocabs,
    load_events,
    train_global_model,
    vectorize,
)
from fpm.dataset import EVAL_TRIAL, load_all
from shared.discovery_baseline import (
    MarkovPredictor,
    fit_group_markov_models,
    predict_routed_markov,
    predict_samples,
)
from shared.evaluation import (
    ApproachResult,
    comparison_rows,
    per_cluster_accuracy,
    save_comparison,
)
from shared.grouping import ClusterResult, build_client_profile, cluster_clients, save_cluster_outputs
from shared.ltl_filter import (
    LTLFilterResult,
    events_by_task_from_traces,
    events_from_traces,
    filter_clients_by_ltl,
    save_ltl_filter_summary,
)
from shared.workflow_graph import (
    WorkflowGraph,
    build_workflow_graph_from_predictions,
    save_workflow_graph,
    split_predictions_by_group,
    workflow_graphs_payload,
)

DEFAULT_DATA_DIR = Path("data")
DEFAULT_OUTPUT_DIR = Path("fpm") / "outputs" / "grouped"
DEFAULT_TRAIN_FRACTION = 0.8


@dataclass(frozen=True)
class PreparedData:
    samples: list[Sample]
    train_samples: list[Sample]
    test_samples: list[Sample]
    event_map: dict[str, int]
    client_map: dict[str, int]
    train_traces_by_client: dict[str, list[list[str]]]
    case_ids_by_trace: dict[str, list[str]]
    task_by_case: dict[str, int]
    global_train_traces: list[list[str]]


def normalize_n_clusters(value: int | str) -> int | str:
    """Normalize API/CLI cluster input into ``auto`` or a positive integer."""
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped == "auto":
            return "auto"
        return max(1, int(stripped))
    return max(1, int(value))


def _prefix_tokens(events: Sequence[str], position: int, window: int = 3) -> tuple[str, ...]:
    start = max(0, position - window)
    prefix = list(events[start:position])
    while len(prefix) < window:
        prefix.insert(0, PAD)
    return tuple(prefix[-window:])


def _train_group_models(
    train_samples: Sequence[Sample],
    assignments: Mapping[str, int],
    event_map: dict[str, int],
    *,
    max_depth: int = 25,
    min_samples_leaf: int = 5,
) -> tuple[dict[int, DecisionTreeClassifier], dict[int, DictVectorizer], float]:
    grouped_samples: dict[int, list[Sample]] = defaultdict(list)
    for sample in train_samples:
        grouped_samples[assignments[sample.client_id]].append(sample)

    models: dict[int, DecisionTreeClassifier] = {}
    vectorizers: dict[int, DictVectorizer] = {}
    start = time.perf_counter()
    for group_id, group_samples in grouped_samples.items():
        x_dicts, y = vectorize(group_samples, event_map, include_client=False)
        vectorizer = DictVectorizer(sparse=False)
        x = vectorizer.fit_transform(x_dicts)
        models[group_id] = train_global_model(
            x,
            y,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
        )
        vectorizers[group_id] = vectorizer
    elapsed = time.perf_counter() - start
    return models, vectorizers, elapsed


def _predict_grouped(
    test_samples: Sequence[Sample],
    assignments: Mapping[str, int],
    models: Mapping[int, DecisionTreeClassifier],
    vectorizers: Mapping[int, DictVectorizer],
    event_map: dict[str, int],
    global_model: DecisionTreeClassifier,
    global_vectorizer: DictVectorizer,
) -> tuple[np.ndarray, np.ndarray]:
    x_dicts, y_true = vectorize(test_samples, event_map, include_client=False)
    predictions: list[int] = []
    for sample, features in zip(test_samples, x_dicts):
        cluster_id = assignments.get(sample.client_id)
        if cluster_id is None or cluster_id not in models:
            x = global_vectorizer.transform([features])
            predictions.append(int(global_model.predict(x)[0]))
            continue
        x = vectorizers[cluster_id].transform([features])
        predictions.append(int(models[cluster_id].predict(x)[0]))
    return np.asarray(predictions), y_true


def _train_per_client_models(
    train_samples: Sequence[Sample],
    event_map: dict[str, int],
) -> tuple[dict[str, DecisionTreeClassifier], dict[str, DictVectorizer]]:
    by_client: dict[str, list[Sample]] = defaultdict(list)
    for sample in train_samples:
        by_client[sample.client_id].append(sample)

    models: dict[str, DecisionTreeClassifier] = {}
    vectorizers: dict[str, DictVectorizer] = {}
    for client_id, client_samples in by_client.items():
        x_dicts, y = vectorize(client_samples, event_map, include_client=False)
        if len(set(y)) < 2:
            continue
        vectorizer = DictVectorizer(sparse=False)
        x = vectorizer.fit_transform(x_dicts)
        model = train_global_model(x, y, max_depth=25, min_samples_leaf=5)
        models[client_id] = model
        vectorizers[client_id] = vectorizer
    return models, vectorizers


def _predict_per_client(
    test_samples: Sequence[Sample],
    models: Mapping[str, DecisionTreeClassifier],
    vectorizers: Mapping[str, DictVectorizer],
    event_map: dict[str, int],
    global_model: DecisionTreeClassifier,
    global_vectorizer: DictVectorizer,
) -> np.ndarray:
    predictions: list[int] = []
    for sample in test_samples:
        x_dicts, _ = vectorize([sample], event_map, include_client=False)
        features = x_dicts[0]
        if sample.client_id in models:
            x = vectorizers[sample.client_id].transform([features])
            predictions.append(int(models[sample.client_id].predict(x)[0]))
        else:
            x = global_vectorizer.transform([features])
            predictions.append(int(global_model.predict(x)[0]))
    return np.asarray(predictions)


def _feature_dicts(
    samples: Sequence[Sample],
    event_map: dict[str, int],
) -> list[dict[str, int]]:
    return [
        {
            f"e{index}": event_map.get(token, -1)
            for index, token in enumerate(sample.prefix)
        }
        for sample in samples
    ]


def _predict_local_tree_ensemble(
    test_samples: Sequence[Sample],
    models: Mapping[str, DecisionTreeClassifier],
    vectorizers: Mapping[str, DictVectorizer],
    event_map: dict[str, int],
) -> np.ndarray:
    """Average predict_proba over all per-client tree models."""
    if not models:
        return np.zeros(len(test_samples), dtype=int)

    x_dicts = _feature_dicts(test_samples, event_map)
    classes = np.asarray(sorted(event_map.values()), dtype=int)
    class_to_column = {int(label): index for index, label in enumerate(classes)}
    probability_sum = np.zeros((len(test_samples), len(classes)), dtype=float)
    n_models = 0

    for client_id, model in models.items():
        vectorizer = vectorizers[client_id]
        probabilities = model.predict_proba(vectorizer.transform(x_dicts))
        for model_col, label in enumerate(model.classes_):
            target_col = class_to_column.get(int(label))
            if target_col is not None:
                probability_sum[:, target_col] += probabilities[:, model_col]
        n_models += 1

    if n_models == 0:
        return np.zeros(len(test_samples), dtype=int)
    return classes[np.argmax(probability_sum / n_models, axis=1)]


def _predict_fedavg_sgd(
    train_samples: Sequence[Sample],
    test_samples: Sequence[Sample],
    event_map: dict[str, int],
    *,
    rounds: int = 10,
) -> tuple[np.ndarray, float]:
    """Federated averaging simulation with local SGDClassifier updates."""
    start = time.perf_counter()
    if not train_samples:
        return np.zeros(len(test_samples), dtype=int), 0.0

    train_x_dicts = _feature_dicts(train_samples, event_map)
    train_y = np.asarray([event_map[sample.label] for sample in train_samples], dtype=int)
    test_x_dicts = _feature_dicts(test_samples, event_map)
    classes = np.asarray(sorted(set(train_y)), dtype=int)
    if len(classes) < 2:
        return np.full(len(test_samples), classes[0] if len(classes) else 0), 0.0

    vectorizer = DictVectorizer(sparse=True)
    x_train = vectorizer.fit_transform(train_x_dicts)
    x_test = vectorizer.transform(test_x_dicts)
    scaler = MaxAbsScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    indices_by_client: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(train_samples):
        indices_by_client[sample.client_id].append(index)

    coef: np.ndarray | None = None
    intercept: np.ndarray | None = None
    for round_index in range(rounds):
        local_coefs: list[np.ndarray] = []
        local_intercepts: list[np.ndarray] = []
        weights: list[int] = []
        for client_id, indices in sorted(indices_by_client.items()):
            if not indices:
                continue
            local = SGDClassifier(
                loss="log_loss",
                alpha=0.0001,
                max_iter=1,
                tol=None,
                random_state=round_index,
            )
            x_client = x_train[indices]
            y_client = train_y[indices]
            local.partial_fit(x_client[:1], y_client[:1], classes=classes)
            if coef is not None and intercept is not None:
                local.coef_ = coef.copy()
                local.intercept_ = intercept.copy()
            local.partial_fit(x_client, y_client)
            local_coefs.append(local.coef_.copy())
            local_intercepts.append(local.intercept_.copy())
            weights.append(len(indices))

        total_weight = sum(weights)
        if total_weight == 0:
            break
        coef = sum(weight * value for weight, value in zip(weights, local_coefs)) / total_weight
        intercept = (
            sum(weight * value for weight, value in zip(weights, local_intercepts))
            / total_weight
        )

    if coef is None or intercept is None:
        return np.zeros(len(test_samples), dtype=int), time.perf_counter() - start
    scores = x_test @ coef.T + intercept
    return classes[np.asarray(scores).argmax(axis=1)], time.perf_counter() - start


def _prepare_casas2_data(
    data_dir: Path,
    *,
    train_fraction: float,
    include_errors: bool,
    skip_analog: bool,
) -> PreparedData:
    events_df = load_events(
        data_dir,
        include_errors=include_errors,
        skip_analog=skip_analog,
    )
    samples = build_samples(events_df, train_fraction=train_fraction)
    event_map, client_map = build_vocabs(samples)
    train_samples = [sample for sample in samples if sample.split == "train"]
    test_samples = [sample for sample in samples if sample.split == "test"]

    train_traces_by_client: dict[str, list[list[str]]] = defaultdict(list)
    case_ids_by_trace: dict[str, list[str]] = defaultdict(list)
    task_by_case: dict[str, int] = {}
    global_train_traces: list[list[str]] = []

    for case_id, case_df in events_df.groupby("case_id", sort=True):
        case_df = case_df.sort_values("timestamp", kind="stable")
        events = case_df["event"].tolist()
        client_id = str(case_df["participant"].iloc[0])
        task = int(case_df["task"].iloc[0])
        task_by_case[str(case_id)] = task
        split_idx = max(1, int(len(events) * train_fraction))
        if split_idx >= len(events):
            split_idx = len(events) - 1
        train_events = events[:split_idx]
        if train_events:
            train_traces_by_client[client_id].append(train_events)
            case_ids_by_trace[client_id].append(str(case_id))
            if split_idx >= 2:
                global_train_traces.append(train_events)

    return PreparedData(
        samples=samples,
        train_samples=train_samples,
        test_samples=test_samples,
        event_map=event_map,
        client_map=client_map,
        train_traces_by_client=dict(train_traces_by_client),
        case_ids_by_trace=dict(case_ids_by_trace),
        task_by_case=task_by_case,
        global_train_traces=global_train_traces,
    )


def _prepare_federated_data(data_dir: Path) -> PreparedData:
    participants = load_all(data_dir)
    samples: list[Sample] = []
    train_traces_by_client: dict[str, list[list[str]]] = defaultdict(list)
    case_ids_by_trace: dict[str, list[str]] = defaultdict(list)
    task_by_case: dict[str, int] = {}
    global_train_traces: list[list[str]] = []

    for client_id, traces in participants.items():
        for trace in traces:
            events = list(trace.events)
            case_id = f"{client_id}.t{trace.trial}"
            task_by_case[case_id] = trace.trial
            if trace.trial != EVAL_TRIAL and events:
                train_traces_by_client[client_id].append(events)
                case_ids_by_trace[client_id].append(case_id)
                global_train_traces.append(events)
            if len(events) < 2:
                continue
            split = "test" if trace.trial == EVAL_TRIAL else "train"
            for position in range(1, len(events)):
                samples.append(
                    Sample(
                        client_id=client_id,
                        case_id=case_id,
                        task=trace.trial,
                        position=position,
                        prefix=_prefix_tokens(events, position),
                        label=events[position],
                        split=split,
                    )
                )

    train_samples = [sample for sample in samples if sample.split == "train"]
    test_samples = [sample for sample in samples if sample.split == "test"]
    event_map, client_map = _build_federated_vocabs(train_samples, test_samples)

    return PreparedData(
        samples=samples,
        train_samples=train_samples,
        test_samples=test_samples,
        event_map=event_map,
        client_map=client_map,
        train_traces_by_client=dict(train_traces_by_client),
        case_ids_by_trace=dict(case_ids_by_trace),
        task_by_case=task_by_case,
        global_train_traces=global_train_traces,
    )


def _build_federated_vocabs(
    train_samples: Sequence[Sample],
    test_samples: Sequence[Sample],
) -> tuple[dict[str, int], dict[str, int]]:
    label_source = list(train_samples) + list(test_samples)
    feature_source = list(train_samples)
    events = sorted(
        {sample.label for sample in label_source}
        | {token for sample in feature_source for token in sample.prefix}
    )
    clients = sorted({sample.client_id for sample in train_samples})
    return (
        {event: index for index, event in enumerate(events)},
        {client: index for index, client in enumerate(clients)},
    )


def _build_profiles(
    prepared: PreparedData,
    ltl_result: LTLFilterResult,
    *,
    client_profiles: Mapping[str, Mapping[str, float]] | None,
    prefer_client_profiles: bool,
) -> dict[str, dict[str, float]]:
    if prefer_client_profiles and client_profiles:
        profiles = {
            client_id: dict(profile)
            for client_id, profile in client_profiles.items()
            if client_id in ltl_result.matched_clients and profile
        }
        if len(profiles) >= min(2, ltl_result.n_matched):
            return profiles

    filtered_traces = ltl_result.matched_traces_by_client
    train_events_by_client = events_from_traces(filtered_traces)
    train_events_by_client_task = events_by_task_from_traces(
        filtered_traces,
        prepared.task_by_case,
        ltl_result.matched_case_ids_by_client,
    )
    vocabulary = None if prefer_client_profiles else sorted(prepared.event_map)
    return {
        client_id: build_client_profile(
            events,
            vocabulary=vocabulary,
            events_by_task=train_events_by_client_task.get(client_id, {}),
            include_task_breakdown=True,
        )
        for client_id, events in train_events_by_client.items()
        if client_id in ltl_result.matched_clients
    }


def _payload_ltl_filter(result: LTLFilterResult) -> dict[str, Any]:
    return {
        "query": result.query,
        "min_matching_traces": result.min_matching_traces,
        "matched_clients": sorted(result.matched_clients),
        "excluded_clients": sorted(result.excluded_clients),
        "matched_case_ids": sorted(result.matched_case_ids),
        "matched_case_ids_by_client": {
            client: case_ids
            for client, case_ids in sorted(result.matched_case_ids_by_client.items())
        },
    }


def _artifact_files(output_dir: Path) -> list[str]:
    if not output_dir.exists():
        return []
    return sorted(
        path.name
        for path in output_dir.iterdir()
        if path.is_file() and not path.name.startswith(".")
    )


def _run_grouped(
    prepared: PreparedData,
    output_dir: Path,
    *,
    ltl: str,
    n_clusters: int | str,
    protocol: str,
    min_matching_traces: int,
    include_markov_baselines: bool,
    include_per_client_baseline: bool,
    write_workflow_graphs: bool,
    client_profiles: Mapping[str, Mapping[str, float]] | None,
    prefer_client_profiles: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    n_clusters = normalize_n_clusters(n_clusters)

    ltl_result = filter_clients_by_ltl(
        prepared.train_traces_by_client,
        prepared.case_ids_by_trace,
        ltl,
        min_matching_traces=min_matching_traces,
    )
    save_ltl_filter_summary(ltl_result, output_dir)
    if ltl_result.active and ltl_result.n_matched < 2:
        raise ValueError(
            f"LTL filter matched {ltl_result.n_matched} client(s); need at least 2 for grouping. "
            f"Query: {ltl_result.query!r}"
        )

    profiles = _build_profiles(
        prepared,
        ltl_result,
        client_profiles=client_profiles,
        prefer_client_profiles=prefer_client_profiles,
    )
    cluster_result = cluster_clients(profiles, n_clusters=n_clusters)
    save_cluster_outputs(cluster_result, profiles, output_dir)

    x_train_dicts, y_train = vectorize(
        prepared.train_samples,
        prepared.event_map,
        include_client=True,
        client_map=prepared.client_map,
    )
    x_test_dicts, y_test = vectorize(
        prepared.test_samples,
        prepared.event_map,
        include_client=True,
        client_map=prepared.client_map,
    )
    global_vectorizer = DictVectorizer(sparse=False)
    x_train = global_vectorizer.fit_transform(x_train_dicts)
    x_test = global_vectorizer.transform(x_test_dicts)

    global_start = time.perf_counter()
    global_model = train_global_model(x_train, y_train, max_depth=25, min_samples_leaf=5)
    global_train_time = time.perf_counter() - global_start
    y_global = global_model.predict(x_test)

    grouped_train_samples = [
        sample
        for sample in prepared.train_samples
        if sample.client_id in ltl_result.matched_clients
    ]
    group_models, group_vectorizers, group_train_time = _train_group_models(
        grouped_train_samples,
        cluster_result.assignments,
        prepared.event_map,
    )
    y_grouped, _ = _predict_grouped(
        prepared.test_samples,
        cluster_result.assignments,
        group_models,
        group_vectorizers,
        prepared.event_map,
        global_model,
        global_vectorizer,
    )

    inverse_event_map = {index: event for event, index in prepared.event_map.items()}
    test_prefixes = [sample.prefix for sample in prepared.test_samples]

    results: list[ApproachResult] = []
    global_graph = build_workflow_graph_from_predictions(
        test_prefixes,
        y_global,
        inverse_event_map,
    )
    results.append(
        ApproachResult(
            name="Global",
            y_true=y_test,
            y_pred=y_global,
            graph_nodes=global_graph.n_nodes,
            graph_edges=global_graph.n_edges,
            train_seconds=global_train_time,
        )
    )

    prefixes_by_group, predictions_by_group = split_predictions_by_group(
        prepared.test_samples,
        y_grouped,
        cluster_result.assignments,
        matched_clients=ltl_result.matched_clients,
    )
    grouped_graphs: dict[int, WorkflowGraph] = {
        group_id: build_workflow_graph_from_predictions(
            prefixes_by_group[group_id],
            predictions_by_group[group_id],
            inverse_event_map,
        )
        for group_id in sorted(predictions_by_group)
    }

    if write_workflow_graphs:
        save_workflow_graph(global_graph, output_dir, "global", write_png=True)
        for group_id, graph in grouped_graphs.items():
            save_workflow_graph(graph, output_dir, f"group_{group_id}", write_png=True)

    grouped_graph_nodes = (
        int(np.mean([graph.n_nodes for graph in grouped_graphs.values()]))
        if grouped_graphs
        else None
    )
    grouped_graph_edges = (
        int(np.mean([graph.n_edges for graph in grouped_graphs.values()]))
        if grouped_graphs
        else None
    )

    grouped_label = f"Grouped (K={cluster_result.n_clusters})"
    if ltl_result.active:
        grouped_label += f", LTL n={ltl_result.n_matched}"

    matched_indices = [
        index
        for index, sample in enumerate(prepared.test_samples)
        if sample.client_id in ltl_result.matched_clients
    ]
    cluster_metrics: dict[int, float] = {}
    if matched_indices:
        cluster_metrics = per_cluster_accuracy(
            y_test[matched_indices],
            y_grouped[matched_indices],
            [
                cluster_result.assignments[prepared.test_samples[index].client_id]
                for index in matched_indices
            ],
        )

    results.append(
        ApproachResult(
            name=grouped_label,
            y_true=y_test,
            y_pred=y_grouped,
            graph_nodes=grouped_graph_nodes,
            graph_edges=grouped_graph_edges,
            train_seconds=group_train_time,
            cluster_metrics=cluster_metrics,
        )
    )

    if include_per_client_baseline:
        client_models, client_vectorizers = _train_per_client_models(
            prepared.train_samples,
            prepared.event_map,
        )
        y_local = _predict_per_client(
            prepared.test_samples,
            client_models,
            client_vectorizers,
            prepared.event_map,
            global_model,
            global_vectorizer,
        )
        results.append(
            ApproachResult(
                name="Per-client local",
                y_true=y_test,
                y_pred=y_local,
            )
        )
        y_ensemble = _predict_local_tree_ensemble(
            prepared.test_samples,
            client_models,
            client_vectorizers,
            prepared.event_map,
        )
        results.append(
            ApproachResult(
                name="Local tree ensemble",
                y_true=y_test,
                y_pred=y_ensemble,
            )
        )

        y_fedavg, fedavg_seconds = _predict_fedavg_sgd(
            prepared.train_samples,
            prepared.test_samples,
            prepared.event_map,
        )
        results.append(
            ApproachResult(
                name="FedAvg SGD",
                y_true=y_test,
                y_pred=y_fedavg,
                train_seconds=fedavg_seconds,
            )
        )

    if include_markov_baselines:
        markov_global = MarkovPredictor.fit(prepared.global_train_traces, use_trigram=True)
        y_markov_global = predict_samples(
            markov_global,
            test_prefixes,
            label_encoder=prepared.event_map,
        )
        markov_global_graph = build_workflow_graph_from_predictions(
            test_prefixes,
            y_markov_global,
            inverse_event_map,
        )
        results.append(
            ApproachResult(
                name="Markov global",
                y_true=y_test,
                y_pred=y_markov_global,
                graph_nodes=markov_global_graph.n_nodes,
                graph_edges=markov_global_graph.n_edges,
            )
        )

        group_markov = fit_group_markov_models(
            ltl_result.matched_traces_by_client,
            cluster_result.assignments,
        )
        y_markov_grouped = predict_routed_markov(
            test_prefixes,
            [sample.client_id for sample in prepared.test_samples],
            group_markov,
            markov_global,
            cluster_result.assignments,
            label_encoder=prepared.event_map,
        )
        _, markov_predictions_by_group = split_predictions_by_group(
            prepared.test_samples,
            y_markov_grouped,
            cluster_result.assignments,
            matched_clients=ltl_result.matched_clients,
        )
        markov_group_graphs = [
            build_workflow_graph_from_predictions(
                prefixes_by_group[group_id],
                markov_predictions_by_group[group_id],
                inverse_event_map,
            )
            for group_id in sorted(markov_predictions_by_group)
        ]
        markov_grouped_graph_nodes = (
            int(np.mean([graph.n_nodes for graph in markov_group_graphs]))
            if markov_group_graphs
            else None
        )
        markov_grouped_graph_edges = (
            int(np.mean([graph.n_edges for graph in markov_group_graphs]))
            if markov_group_graphs
            else None
        )
        results.append(
            ApproachResult(
                name="Markov grouped",
                y_true=y_test,
                y_pred=y_markov_grouped,
                graph_nodes=markov_grouped_graph_nodes,
                graph_edges=markov_grouped_graph_edges,
            )
        )

    extra_lines = [
        f"LTL query: {ltl_result.query or '(none)'}",
        f"LTL matched clients: {ltl_result.n_matched} / {len(prepared.client_map)}",
        f"Silhouette score: {cluster_result.silhouette:.4f}",
        f"Participants: {len(prepared.client_map)}",
        f"Grouped train samples: {len(grouped_train_samples)}",
        f"Train samples (all): {len(prepared.train_samples)}",
        f"Test samples: {len(prepared.test_samples)}",
    ]
    if ltl_result.excluded_clients:
        extra_lines.append(
            "Excluded by LTL: " + ", ".join(sorted(ltl_result.excluded_clients))
        )
    save_comparison(results, output_dir, extra_lines=extra_lines)

    comparison = comparison_rows(results)
    return {
        "protocol": protocol,
        "clustering": _cluster_payload(cluster_result),
        "comparison": comparison,
        "per_cluster_accuracy": {
            str(cluster_id): round(value, 4)
            for cluster_id, value in sorted(cluster_metrics.items())
        },
        "ltl_filter": _payload_ltl_filter(ltl_result),
        "matched_clients": sorted(ltl_result.matched_clients),
        "excluded_clients": sorted(ltl_result.excluded_clients),
        "matched_count": ltl_result.n_matched,
        "participant_count": len(prepared.client_map),
        "train_samples": len(prepared.train_samples),
        "test_samples": len(prepared.test_samples),
        "grouped_train_samples": len(grouped_train_samples),
        "output_dir": str(output_dir),
        "artifacts": _artifact_files(output_dir),
        "workflow_graphs": workflow_graphs_payload(
            global_graph,
            grouped_graphs,
            cluster_result.assignments,
        ),
    }


def _cluster_payload(result: ClusterResult) -> dict[str, Any]:
    return {
        "n_clusters": result.n_clusters,
        "silhouette": round(result.silhouette, 4),
        "assignments": result.assignments,
    }


def run_grouped_evaluation(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    *,
    ltl: str = "",
    n_clusters: int | str = "auto",
    eval_protocol: str = "casas2",
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    include_errors: bool = True,
    skip_analog: bool = True,
    min_matching_traces: int = 1,
    include_markov_baselines: bool = True,
    include_per_client_baseline: bool = True,
    write_workflow_graphs: bool = True,
    client_profiles: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, Any]:
    """Run grouped model evaluation and persist CASAS2-style artifacts."""
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    protocol = eval_protocol.strip().lower()

    if protocol == "casas2":
        prepared = _prepare_casas2_data(
            data_path,
            train_fraction=train_fraction,
            include_errors=include_errors,
            skip_analog=skip_analog,
        )
        prefer_client_profiles = False
    elif protocol == "federated":
        prepared = _prepare_federated_data(data_path)
        prefer_client_profiles = True
    else:
        raise ValueError("eval_protocol must be 'casas2' or 'federated'")

    return _run_grouped(
        prepared,
        output_path,
        ltl=ltl,
        n_clusters=n_clusters,
        protocol=protocol,
        min_matching_traces=min_matching_traces,
        include_markov_baselines=include_markov_baselines,
        include_per_client_baseline=include_per_client_baseline,
        write_workflow_graphs=write_workflow_graphs,
        client_profiles=client_profiles,
        prefer_client_profiles=prefer_client_profiles,
    )
