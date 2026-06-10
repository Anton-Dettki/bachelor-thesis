"""Individual process mining component — one Phone per subject."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pm4py
from pm4py.algo.discovery.alpha import algorithm as alpha_algorithm

from fpm.event_log import DEFAULT_EVENT_LOG_DIR, load_event_log, subject_event_log_xes_path
from fpm.loader import ACTIVITY, CASE_ID, SUBJECT_IDS, TIMESTAMP
from fpm.ltl import PatternQuery
from fpm.prefix import EVENT_INDEX


def trace_sequences_from_log(log: pd.DataFrame) -> dict[str, list[str]]:
    """Return ordered activity sequences keyed by trace (case) id."""
    if log.empty:
        return {}

    sort_cols = [CASE_ID, TIMESTAMP]
    if EVENT_INDEX in log.columns:
        sort_cols.append(EVENT_INDEX)
    ordered = log.sort_values(sort_cols, kind="stable")
    sequences: dict[str, list[str]] = {}
    for case_id, group in ordered.groupby(CASE_ID, sort=False):
        sequences[str(case_id)] = group[ACTIVITY].astype(str).tolist()
    return sequences


def select_matching_case_ids(
    log: pd.DataFrame,
    query: str | PatternQuery,
) -> list[str]:
    """Return case ids whose activity sequence satisfies the query."""
    pattern = query if isinstance(query, PatternQuery) else PatternQuery.parse(query)
    return [
        case_id
        for case_id, sequence in trace_sequences_from_log(log).items()
        if pattern.satisfied_by(sequence)
    ]


class Phone:
    """Mock smartphone that holds one subject's event log and discovers a local model."""

    def __init__(
        self,
        subject_id: int,
        *,
        event_log_dir: Path = DEFAULT_EVENT_LOG_DIR,
        event_log_path: Path | None = None,
        log: Any | None = None,
    ) -> None:
        if subject_id not in SUBJECT_IDS:
            raise ValueError(f"subject_id must be one of {SUBJECT_IDS}, got {subject_id!r}")

        self.subject_id = subject_id
        self.event_log_dir = event_log_dir
        self.event_log_path = event_log_path or subject_event_log_xes_path(
            event_log_dir, subject_id
        )
        self.log = log if log is not None else load_event_log(self.event_log_path)
        self._net = None
        self._initial_marking = None
        self._final_marking = None

    @property
    def subject_label(self) -> str:
        return f"subject{self.subject_id}"

    def discover_model(self):
        """Run Alpha+ on this phone's event log and cache the result."""
        self._net, self._initial_marking, self._final_marking = alpha_algorithm.apply(
            self.log,
            variant=alpha_algorithm.Variants.ALPHA_VERSION_PLUS,
        )
        return self._net, self._initial_marking, self._final_marking

    @property
    def model(self):
        if self._net is None:
            self.discover_model()
        return self._net, self._initial_marking, self._final_marking

    def activities_in_log(self) -> set[str]:
        return set(pm4py.get_event_attribute_values(self.log, ACTIVITY))

    def trace_sequences(self) -> dict[str, list[str]]:
        """Return ordered activity sequences keyed by trace (case) id."""
        return trace_sequences_from_log(self.log)

    def select_matching_traces(self, query: str | PatternQuery) -> list[str]:
        """Return case ids whose activity sequence satisfies the query."""
        return select_matching_case_ids(self.log, query)

    def matches_query(self, query: str | PatternQuery, *, min_traces: int = 1) -> bool:
        """True when at least ``min_traces`` day-traces satisfy the query."""
        return len(self.select_matching_traces(query)) >= min_traces

    def filtered_log(self, query: str | PatternQuery):
        """Event log containing only traces that satisfy the query."""
        matching = set(self.select_matching_traces(query))
        if not matching:
            return self.log.iloc[0:0].copy()
        return self.log[self.log[CASE_ID].astype(str).isin(matching)].copy()

    def transitions_in_model(self) -> set[str]:
        net, _, _ = self.model
        return {transition.label for transition in net.transitions if transition.label}

    def model_stats(self) -> dict[str, Any]:
        net, initial_marking, final_marking = self.model
        log_activities = self.activities_in_log()
        model_transitions = self.transitions_in_model()
        trace_count = len(pm4py.get_event_attribute_values(self.log, CASE_ID))

        labeled_transitions = [
            transition.label
            for transition in net.transitions
            if transition.label
        ]

        return {
            "subject_id": self.subject_id,
            "subject_label": self.subject_label,
            "event_log_path": str(self.event_log_path),
            "traces": trace_count,
            "events": sum(pm4py.get_event_attribute_values(self.log, ACTIVITY).values()),
            "activities_in_log": sorted(log_activities),
            "activities_in_model": sorted(model_transitions),
            "places": len(net.places),
            "labeled_transitions": len(labeled_transitions),
            "transitions": len(net.transitions),
            "arcs": len(net.arcs),
            "connected": len(net.arcs) > len(labeled_transitions),
            "initial_marking_places": len(initial_marking),
            "final_marking_places": len(final_marking),
        }
