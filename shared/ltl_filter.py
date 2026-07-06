"""LTL-based client filtering before behavioral grouping."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from fpm.ltl import PatternQuery


def event_to_ltl_token(event: str) -> str:
    """Map CASAS2 event labels (M07=ON) to LTL atom tokens (M07_ON)."""
    return event.replace("=", "_").replace("-", "_").upper()


def trace_to_ltl(trace: Sequence[str]) -> tuple[str, ...]:
    return tuple(event_to_ltl_token(event) for event in trace)


@dataclass(frozen=True)
class LTLFilterResult:
    query: str
    matched_clients: frozenset[str]
    excluded_clients: frozenset[str]
    matched_traces_by_client: dict[str, list[list[str]]]
    matched_case_ids: frozenset[str]
    min_matching_traces: int

    @property
    def n_matched(self) -> int:
        return len(self.matched_clients)

    @property
    def n_excluded(self) -> int:
        return len(self.excluded_clients)

    @property
    def active(self) -> bool:
        return bool(self.query.strip())


def filter_clients_by_ltl(
    train_traces_by_client: Mapping[str, list[list[str]]],
    case_ids_by_trace: Mapping[str, list[str]],
    query_text: str,
    *,
    min_matching_traces: int = 1,
) -> LTLFilterResult:
    """Keep clients with enough training traces that satisfy the LTL query."""
    stripped = query_text.strip()
    all_clients = frozenset(train_traces_by_client)

    if not stripped:
        matched_traces = {client: list(traces) for client, traces in train_traces_by_client.items()}
        matched_cases = frozenset(
            case_id
            for case_list in case_ids_by_trace.values()
            for case_id in case_list
        )
        return LTLFilterResult(
            query="",
            matched_clients=all_clients,
            excluded_clients=frozenset(),
            matched_traces_by_client=matched_traces,
            matched_case_ids=matched_cases,
            min_matching_traces=min_matching_traces,
        )

    query = PatternQuery.parse(stripped)
    matched_clients: set[str] = set()
    matched_traces: dict[str, list[list[str]]] = {}
    matched_case_ids: set[str] = set()

    for client_id, traces in train_traces_by_client.items():
        case_ids = case_ids_by_trace.get(client_id, [])
        satisfying = sum(
            1 for trace in traces if query.satisfied_by(trace_to_ltl(trace))
        )
        if satisfying >= min_matching_traces:
            matched_clients.add(client_id)
            matched_traces[client_id] = list(traces)
            matched_case_ids.update(case_ids)

    excluded = all_clients - matched_clients
    return LTLFilterResult(
        query=stripped,
        matched_clients=frozenset(matched_clients),
        excluded_clients=frozenset(excluded),
        matched_traces_by_client=matched_traces,
        matched_case_ids=frozenset(matched_case_ids),
        min_matching_traces=min_matching_traces,
    )


def events_from_traces(traces_by_client: Mapping[str, list[list[str]]]) -> dict[str, list[str]]:
    """Flatten matched traces into one event list per client."""
    events_by_client: dict[str, list[str]] = {}
    for client_id, traces in traces_by_client.items():
        events: list[str] = []
        for trace in traces:
            events.extend(trace)
        events_by_client[client_id] = events
    return events_by_client


def events_by_task_from_traces(
    traces_by_client: Mapping[str, list[list[str]]],
    task_by_case: Mapping[str, int],
    case_ids_by_trace: Mapping[str, list[str]],
) -> dict[str, dict[int, list[str]]]:
    """Rebuild per-task event lists from filtered traces."""
    result: dict[str, dict[int, list[str]]] = {}
    for client_id, traces in traces_by_client.items():
        case_ids = case_ids_by_trace.get(client_id, [])
        task_events: dict[int, list[str]] = {}
        for trace, case_id in zip(traces, case_ids):
            task = task_by_case.get(case_id)
            if task is None:
                continue
            task_events.setdefault(task, []).extend(trace)
        result[client_id] = task_events
    return result


def save_ltl_filter_summary(result: LTLFilterResult, output_dir: Path) -> None:
    """Write which clients passed or failed the LTL pre-filter."""
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "query": result.query,
        "min_matching_traces": result.min_matching_traces,
        "matched_clients": sorted(result.matched_clients),
        "excluded_clients": sorted(result.excluded_clients),
        "matched_case_ids": sorted(result.matched_case_ids),
    }
    (output_dir / "ltl_filter.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "LTL pre-filter (before behavioral grouping)",
        f"Query: {result.query or '(none — all clients included)'}",
        f"Min matching traces: {result.min_matching_traces}",
        f"Matched clients: {result.n_matched}",
        f"Excluded clients: {result.n_excluded}",
        "",
    ]
    if result.matched_clients:
        lines.append("Matched: " + ", ".join(sorted(result.matched_clients)))
    if result.excluded_clients:
        lines.append("Excluded: " + ", ".join(sorted(result.excluded_clients)))
    (output_dir / "ltl_filter_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_ltl_query(ltl: str | None, example: str | None, examples: Mapping[str, str]) -> str:
    """Resolve --ltl and --example-query into one query string."""
    if ltl and example:
        raise ValueError("Specify only one of --ltl or --example-query")
    if example:
        if example not in examples:
            known = ", ".join(sorted(examples))
            raise ValueError(f"Unknown example query {example!r}; choose from {known}")
        return examples[example]
    return ltl or ""
