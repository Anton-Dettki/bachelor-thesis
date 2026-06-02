"""FPM Aggregator — collect filtered traces from phones and integrate one log."""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import pm4py

from fpm.event_log import DEFAULT_EVENT_LOG_DIR
from fpm.loader import ACTIVITY, CASE_ID, SUBJECT_IDS, TIMESTAMP
from fpm.ltl import PatternQuery
from fpm.phone import Phone

RESOURCE = "org:resource"


@dataclass
class PhoneContribution:
    subject_id: int
    subject_label: str
    meets_pattern: bool
    matching_traces: int
    total_traces: int
    matching_case_ids: list[str]
    filtered_log: pd.DataFrame
    size_kb: float
    filtered_events: int = field(init=False)

    def __post_init__(self) -> None:
        self.filtered_events = len(self.filtered_log)


@dataclass
class AggregatorResult:
    query: str
    contributions: list[PhoneContribution]
    integrated_log: pd.DataFrame
    merge_time_s: float
    contributing_subjects: list[str] = field(init=False)

    def __post_init__(self) -> None:
        self.contributing_subjects = [
            c.subject_label for c in self.contributions if c.meets_pattern
        ]


def log_size_kb(log: pd.DataFrame) -> float:
    """Approximate serialized CSV size in kilobytes."""
    if log.empty:
        return 0.0
    buffer = io.StringIO()
    log.to_csv(buffer, index=False)
    return len(buffer.getvalue().encode("utf-8")) / 1024


def namespace_filtered_log(log: pd.DataFrame, subject_label: str) -> pd.DataFrame:
    """Prefix case ids and tag events with their source subject."""
    if log.empty:
        return log.copy()

    namespaced = log.copy()
    namespaced[CASE_ID] = (
        subject_label + ":" + namespaced[CASE_ID].astype(str)
    )
    namespaced[RESOURCE] = subject_label
    return namespaced


class Aggregator:
    """Broadcast LTL queries to phones and merge matching traces."""

    def __init__(self, phones: list[Phone]) -> None:
        if not phones:
            raise ValueError("Aggregator requires at least one Phone.")
        self.phones = phones

    @classmethod
    def from_subject_ids(
        cls,
        subject_ids: list[int] | tuple[int, ...] | None = None,
        *,
        event_log_dir: Path = DEFAULT_EVENT_LOG_DIR,
    ) -> Aggregator:
        ids = list(subject_ids) if subject_ids is not None else list(SUBJECT_IDS)
        phones = [Phone(subject_id, event_log_dir=event_log_dir) for subject_id in ids]
        return cls(phones)

    def collect(
        self,
        query: str | PatternQuery,
        *,
        min_traces: int = 1,
    ) -> list[PhoneContribution]:
        contributions: list[PhoneContribution] = []
        for phone in self.phones:
            matching = phone.select_matching_traces(query)
            meets = len(matching) >= min_traces
            filtered = phone.filtered_log(query) if meets else phone.log.iloc[0:0].copy()
            contributions.append(
                PhoneContribution(
                    subject_id=phone.subject_id,
                    subject_label=phone.subject_label,
                    meets_pattern=meets,
                    matching_traces=len(matching),
                    total_traces=len(phone.trace_sequences()),
                    matching_case_ids=matching,
                    filtered_log=filtered,
                    size_kb=log_size_kb(filtered),
                )
            )
        return contributions

    def integrate(self, contributions: list[PhoneContribution]) -> pd.DataFrame:
        parts: list[pd.DataFrame] = []
        for contribution in contributions:
            if not contribution.meets_pattern or contribution.filtered_log.empty:
                continue
            parts.append(
                namespace_filtered_log(
                    contribution.filtered_log,
                    contribution.subject_label,
                )
            )

        if not parts:
            empty = contributions[0].filtered_log.iloc[0:0].copy()
            return pm4py.format_dataframe(
                empty,
                case_id=CASE_ID,
                activity_key=ACTIVITY,
                timestamp_key=TIMESTAMP,
            )

        merged = pd.concat(parts, ignore_index=True)
        return pm4py.format_dataframe(
            merged,
            case_id=CASE_ID,
            activity_key=ACTIVITY,
            timestamp_key=TIMESTAMP,
        )

    def run(
        self,
        query: str | PatternQuery,
        *,
        min_traces: int = 1,
    ) -> AggregatorResult:
        pattern = query if isinstance(query, PatternQuery) else PatternQuery.parse(query)
        contributions = self.collect(pattern, min_traces=min_traces)

        start = time.perf_counter()
        integrated_log = self.integrate(contributions)
        merge_time_s = time.perf_counter() - start

        return AggregatorResult(
            query=pattern.text,
            contributions=contributions,
            integrated_log=integrated_log,
            merge_time_s=merge_time_s,
        )


def contribution_summary(contribution: PhoneContribution) -> dict[str, Any]:
    return {
        "subject_id": contribution.subject_id,
        "subject_label": contribution.subject_label,
        "meets_pattern": contribution.meets_pattern,
        "matching_traces": contribution.matching_traces,
        "total_traces": contribution.total_traces,
        "matching_case_ids": contribution.matching_case_ids,
        "filtered_events": contribution.filtered_events,
        "size_kb": round(contribution.size_kb, 3),
    }


def aggregator_metrics(result: AggregatorResult) -> dict[str, Any]:
    contributing = [c for c in result.contributions if c.meets_pattern]
    total_individual_kb = sum(c.size_kb for c in contributing)
    integrated_kb = log_size_kb(result.integrated_log)
    trace_count = len(pm4py.get_event_attribute_values(result.integrated_log, CASE_ID))

    return {
        "query": result.query,
        "contributing_subjects": result.contributing_subjects,
        "contributor_count": len(contributing),
        "total_size_individual_logs_kb": round(total_individual_kb, 3),
        "size_integrated_log_kb": round(integrated_kb, 3),
        "total_size_logs_kb": round(total_individual_kb + integrated_kb, 3),
        "integrated_traces": trace_count,
        "integrated_events": len(result.integrated_log),
        "merge_time_s": round(result.merge_time_s, 6),
    }
