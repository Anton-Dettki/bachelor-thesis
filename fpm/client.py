"""HTTP client for remote phone servers."""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import pandas as pd
import pm4py

from fpm.aggregator import PhoneContribution, empty_event_log, log_size_kb, query_text
from fpm.loader import ACTIVITY, CASE_ID, TIMESTAMP
from fpm.ltl import PatternQuery


def subject_label_from_url(url: str) -> str:
    """Infer subject label from default port convention (800N -> subjectN)."""
    parsed = urlparse(url)
    port = parsed.port
    if port is not None and 8001 <= port <= 8099:
        return f"subject{port - 8000}"
    host = parsed.hostname or "unknown"
    return f"phone@{host}:{port or 80}"


def subject_id_from_url(url: str) -> int:
    parsed = urlparse(url)
    port = parsed.port
    if port is not None and 8001 <= port <= 8099:
        return port - 8000
    return -1


def parse_xes_string(xes: str) -> pd.DataFrame:
    if not xes.strip():
        return empty_event_log()

    with tempfile.NamedTemporaryFile(suffix=".xes", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        tmp_path.write_text(xes, encoding="utf-8")
        log = pm4py.read_xes(str(tmp_path))
        if not isinstance(log, pd.DataFrame):
            log = pm4py.convert_to_dataframe(log)
        log[TIMESTAMP] = pd.to_datetime(log[TIMESTAMP], errors="coerce")
        return pm4py.format_dataframe(
            log,
            case_id=CASE_ID,
            activity_key=ACTIVITY,
            timestamp_key=TIMESTAMP,
        )
    finally:
        tmp_path.unlink(missing_ok=True)


@dataclass
class RemoteResolveResult:
    subject_id: int
    subject_label: str
    meets_pattern: bool
    matching_traces: int
    total_traces: int
    matching_case_ids: list[str]
    filtered_log: pd.DataFrame
    bytes_received: int
    request_time_s: float
    error: str | None = None


@dataclass
class RemotePredictParamsResult:
    subject_id: int
    subject_label: str
    model: str
    params: dict
    n_train: int
    bytes_received: int
    request_time_s: float
    error: str | None = None


class PhoneClient:
    """HTTP client for one phone server."""

    def __init__(
        self,
        base_url: str,
        *,
        http_client: httpx.Client | Any | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(timeout=timeout)

    @classmethod
    def from_app(cls, app: Any) -> PhoneClient:
        """Build a client backed by Starlette's in-process TestClient (for tests)."""
        from starlette.testclient import TestClient

        return cls("", http_client=TestClient(app))

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> PhoneClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def info(self) -> dict:
        response = self._client.get(f"{self.base_url}/info")
        response.raise_for_status()
        return response.json()

    def predict_params(self, model: str) -> RemotePredictParamsResult:
        start = time.perf_counter()
        label = subject_label_from_url(self.base_url)
        sid = subject_id_from_url(self.base_url)

        try:
            response = self._client.get(f"{self.base_url}/predict/params/{model}")
            elapsed = time.perf_counter() - start
            body_bytes = len(response.content)

            if response.status_code == 400:
                detail = response.json().get("detail", response.text)
                return RemotePredictParamsResult(
                    subject_id=sid,
                    subject_label=label,
                    model=model,
                    params={},
                    n_train=0,
                    bytes_received=body_bytes,
                    request_time_s=elapsed,
                    error=f"HTTP 400: {detail}",
                )

            if response.status_code == 404:
                detail = response.json().get("detail", response.text)
                return RemotePredictParamsResult(
                    subject_id=sid,
                    subject_label=label,
                    model=model,
                    params={},
                    n_train=0,
                    bytes_received=body_bytes,
                    request_time_s=elapsed,
                    error=f"HTTP 404: {detail}",
                )

            response.raise_for_status()
            data = response.json()
            return RemotePredictParamsResult(
                subject_id=data.get("subject_id", sid),
                subject_label=data.get("subject_label", label),
                model=data.get("model", model),
                params=data.get("params", {}),
                n_train=int(data.get("n_train", 0)),
                bytes_received=body_bytes,
                request_time_s=elapsed,
            )
        except httpx.HTTPError as exc:
            elapsed = time.perf_counter() - start
            return RemotePredictParamsResult(
                subject_id=sid,
                subject_label=label,
                model=model,
                params={},
                n_train=0,
                bytes_received=0,
                request_time_s=elapsed,
                error=str(exc),
            )

    def resolve(
        self,
        query: str | PatternQuery,
        *,
        min_traces: int = 1,
    ) -> RemoteResolveResult:
        q = query_text(query)
        start = time.perf_counter()
        label = subject_label_from_url(self.base_url)
        sid = subject_id_from_url(self.base_url)

        try:
            response = self._client.post(
                f"{self.base_url}/resolve",
                json={"query": q, "min_traces": min_traces},
            )
            elapsed = time.perf_counter() - start
            body_bytes = len(response.content)

            if response.status_code == 400:
                detail = response.json().get("detail", response.text)
                return RemoteResolveResult(
                    subject_id=sid,
                    subject_label=label,
                    meets_pattern=False,
                    matching_traces=0,
                    total_traces=0,
                    matching_case_ids=[],
                    filtered_log=empty_event_log(),
                    bytes_received=body_bytes,
                    request_time_s=elapsed,
                    error=f"HTTP 400: {detail}",
                )

            response.raise_for_status()
            data = response.json()
            filtered = parse_xes_string(data.get("filtered_xes", ""))

            return RemoteResolveResult(
                subject_id=data.get("subject_id", sid),
                subject_label=data.get("subject_label", label),
                meets_pattern=data.get("meets_pattern", False),
                matching_traces=data.get("matching_traces", 0),
                total_traces=data.get("total_traces", 0),
                matching_case_ids=data.get("matching_case_ids", []),
                filtered_log=filtered,
                bytes_received=body_bytes,
                request_time_s=elapsed,
            )
        except httpx.HTTPError as exc:
            elapsed = time.perf_counter() - start
            return RemoteResolveResult(
                subject_id=sid,
                subject_label=label,
                meets_pattern=False,
                matching_traces=0,
                total_traces=0,
                matching_case_ids=[],
                filtered_log=empty_event_log(),
                bytes_received=0,
                request_time_s=elapsed,
                error=str(exc),
            )


class RemotePhoneConnector:
    """Aggregator connector that resolves queries over HTTP."""

    def __init__(self, client: PhoneClient) -> None:
        self.client = client

    def resolve(
        self,
        query: str | PatternQuery,
        *,
        min_traces: int = 1,
    ) -> PhoneContribution:
        result = self.client.resolve(query, min_traces=min_traces)
        return PhoneContribution(
            subject_id=result.subject_id,
            subject_label=result.subject_label,
            meets_pattern=result.meets_pattern and result.error is None,
            matching_traces=result.matching_traces,
            total_traces=result.total_traces,
            matching_case_ids=result.matching_case_ids,
            filtered_log=result.filtered_log,
            size_kb=log_size_kb(result.filtered_log),
            error=result.error,
            bytes_transferred=result.bytes_received,
            request_time_s=result.request_time_s,
        )
