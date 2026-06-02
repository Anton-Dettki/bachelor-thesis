"""FPM Aggregator — collect filtered traces from phones and integrate one log."""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pandas as pd
import pm4py

from fpm.event_log import DEFAULT_EVENT_LOG_DIR
from fpm.loader import ACTIVITY, CASE_ID, SUBJECT_IDS, TIMESTAMP
from fpm.ltl import LTLParseError, PatternQuery
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
    error: str | None = None
    bytes_transferred: int | None = None
    request_time_s: float | None = None
    filtered_events: int = field(init=False)

    def __post_init__(self) -> None:
        self.filtered_events = len(self.filtered_log)


def empty_event_log() -> pd.DataFrame:
    """An empty, properly typed event log (used for non-matching / failed phones)."""
    return pd.DataFrame(columns=[CASE_ID, ACTIVITY, TIMESTAMP])


def query_text(query: str | PatternQuery) -> str:
    return query.text if isinstance(query, PatternQuery) else query


@runtime_checkable
class PhoneConnector(Protocol):
    """Anything the Aggregator can broadcast a query to (local or remote)."""

    def resolve(
        self, query: str | PatternQuery, *, min_traces: int = 1
    ) -> PhoneContribution: ...


class LocalPhoneConnector:
    """In-process connector wrapping a :class:`Phone`."""

    def __init__(self, phone: Phone) -> None:
        self.phone = phone

    def resolve(
        self, query: str | PatternQuery, *, min_traces: int = 1
    ) -> PhoneContribution:
        phone = self.phone
        try:
            matching = phone.select_matching_traces(query)
        except LTLParseError as exc:
            return PhoneContribution(
                subject_id=phone.subject_id,
                subject_label=phone.subject_label,
                meets_pattern=False,
                matching_traces=0,
                total_traces=len(phone.trace_sequences()),
                matching_case_ids=[],
                filtered_log=phone.log.iloc[0:0].copy(),
                size_kb=0.0,
                error=str(exc),
            )
        meets = len(matching) >= min_traces
        filtered = phone.filtered_log(query) if meets else phone.log.iloc[0:0].copy()
        return PhoneContribution(
            subject_id=phone.subject_id,
            subject_label=phone.subject_label,
            meets_pattern=meets,
            matching_traces=len(matching),
            total_traces=len(phone.trace_sequences()),
            matching_case_ids=matching,
            filtered_log=filtered,
            size_kb=log_size_kb(filtered),
        )


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
    """Approximate XES serialized size in kilobytes (matches SOWCompact paper metric)."""
    if log.empty:
        return 0.0
    with tempfile.NamedTemporaryFile(suffix=".xes", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        pm4py.write_xes(log, str(tmp_path))
        return tmp_path.stat().st_size / 1024
    finally:
        tmp_path.unlink(missing_ok=True)


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

    def __init__(self, phones: list[Phone | PhoneConnector]) -> None:
        if not phones:
            raise ValueError("Aggregator requires at least one Phone.")
        self.connectors: list[PhoneConnector] = [
            LocalPhoneConnector(p) if isinstance(p, Phone) else p for p in phones
        ]

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

    @classmethod
    def from_endpoints(
        cls,
        urls: list[str],
        *,
        timeout: float = 30.0,
    ) -> Aggregator:
        """Build an aggregator that talks to phone servers over HTTP."""
        from fpm.client import PhoneClient, RemotePhoneConnector

        connectors = [
            RemotePhoneConnector(PhoneClient(url, timeout=timeout)) for url in urls
        ]
        return cls(connectors)

    def collect(
        self,
        query: str | PatternQuery,
        *,
        min_traces: int = 1,
    ) -> list[PhoneContribution]:
        return [
            connector.resolve(query, min_traces=min_traces)
            for connector in self.connectors
        ]

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
            return pm4py.format_dataframe(
                empty_event_log(),
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
        contributions = self.collect(query, min_traces=min_traces)
        qtext = query.text if isinstance(query, PatternQuery) else str(query)

        start = time.perf_counter()
        integrated_log = self.integrate(contributions)
        merge_time_s = time.perf_counter() - start

        return AggregatorResult(
            query=qtext,
            contributions=contributions,
            integrated_log=integrated_log,
            merge_time_s=merge_time_s,
        )


def contribution_summary(contribution: PhoneContribution) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "subject_id": contribution.subject_id,
        "subject_label": contribution.subject_label,
        "meets_pattern": contribution.meets_pattern,
        "matching_traces": contribution.matching_traces,
        "total_traces": contribution.total_traces,
        "matching_case_ids": contribution.matching_case_ids,
        "filtered_events": contribution.filtered_events,
        "size_kb": round(contribution.size_kb, 3),
    }
    if contribution.bytes_transferred is not None:
        summary["bytes_transferred"] = contribution.bytes_transferred
    if contribution.request_time_s is not None:
        summary["request_time_s"] = round(contribution.request_time_s, 6)
    if contribution.error is not None:
        summary["error"] = contribution.error
    return summary


def aggregator_metrics(result: AggregatorResult) -> dict[str, Any]:
    contributing = [c for c in result.contributions if c.meets_pattern]
    total_individual_kb = sum(c.size_kb for c in contributing)
    integrated_kb = log_size_kb(result.integrated_log)
    trace_count = len(pm4py.get_event_attribute_values(result.integrated_log, CASE_ID))

    metrics: dict[str, Any] = {
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

    has_network = any(
        c.bytes_transferred is not None
        or c.request_time_s is not None
        or c.error is not None
        for c in result.contributions
    )
    if has_network:
        metrics["total_bytes_received"] = sum(
            c.bytes_transferred or 0 for c in result.contributions
        )
        metrics["total_request_time_s"] = round(
            sum(c.request_time_s or 0.0 for c in result.contributions), 6
        )
        metrics["phone_errors"] = {
            c.subject_label: c.error
            for c in result.contributions
            if c.error is not None
        }

    return metrics
