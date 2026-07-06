"""Main coordinator server for the federated sensor workflow."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from fpm.grouped import DEFAULT_OUTPUT_DIR, run_grouped_evaluation
from fpm.queries import EXAMPLE_QUERIES

RESULTS: list[dict[str, Any]] = []


class QueryRequest(BaseModel):
    model: str = Field(default="tree", pattern="^(tree|frequency|markov|logreg)$")
    ltl: str = ""
    min_match_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    group: bool = False
    n_clusters: int | str = "auto"
    include_baselines: bool = True
    eval_protocol: str = Field(default="casas2", pattern="^(casas2|federated)$")


def _client_urls() -> list[str]:
    raw = os.getenv("CLIENTS", "")
    return [url.strip().rstrip("/") for url in raw.split(",") if url.strip()]


def _data_dir() -> Path:
    return Path(os.getenv("DATA_DIR", "data"))


def _output_dir() -> Path:
    return Path(os.getenv("GROUPED_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))


def _train_payload(request: QueryRequest) -> dict[str, Any]:
    return {
        "model": request.model,
        "ltl": request.ltl,
        "min_match_fraction": request.min_match_fraction,
        "eval_protocol": request.eval_protocol,
    }


def _param_summary(params: dict[str, Any] | None) -> str:
    if not params:
        return "no params"
    model_type = params.get("type", "unknown")
    if model_type == "frequency":
        return f"{len(params.get('event_counts', {}))} event counts"
    if model_type == "markov":
        transitions = params.get("transitions", {})
        edge_count = sum(len(targets) for targets in transitions.values())
        return f"{len(transitions)} states, {edge_count} transitions"
    if model_type == "tree" and params.get("fitted"):
        return (
            f"{params.get('n_nodes', 0)} nodes, "
            f"{params.get('n_leaves', 0)} leaves, "
            f"{len(params.get('classes', []))} classes"
        )
    if model_type == "tree":
        return f"tree fallback: {params.get('fallback_reason', 'not fitted')}"
    if model_type == "logreg" and params.get("fitted"):
        return (
            f"{len(params.get('classes', []))} classes, "
            f"{len(params.get('feature_names', []))} features"
        )
    if model_type == "logreg":
        return f"logreg fallback: {params.get('fallback_reason', 'not fitted')}"
    return model_type


async def _query_client(
    client: httpx.AsyncClient,
    url: str,
    request: QueryRequest,
) -> dict[str, Any]:
    try:
        response = await client.post(
            f"{url}/train",
            json=_train_payload(request),
        )
        response.raise_for_status()
        payload = response.json()
        payload["client_url"] = url
        payload["status"] = "ok"
        payload["param_summary"] = _param_summary(payload.get("params"))
        return payload
    except Exception as exc:  # noqa: BLE001 - surfaced in the dashboard.
        return {
            "client_url": url,
            "status": "error",
            "matched": False,
            "error": str(exc),
            "param_summary": "error",
        }


async def _profile_client(
    client: httpx.AsyncClient,
    url: str,
    request: QueryRequest,
) -> dict[str, Any]:
    try:
        response = await client.get(
            f"{url}/profile",
            params={
                "ltl": request.ltl,
                "min_match_fraction": request.min_match_fraction,
            },
        )
        response.raise_for_status()
        payload = response.json()
        payload["client_url"] = url
        payload["status"] = "ok"
        return payload
    except Exception as exc:  # noqa: BLE001 - surfaced in the dashboard.
        return {
            "client_url": url,
            "status": "error",
            "matched": False,
            "error": str(exc),
        }


def _artifact_links(names: list[str]) -> list[dict[str, str]]:
    return [
        {
            "name": name,
            "url": f"/api/artifacts/{name}",
        }
        for name in names
    ]


def _attach_grouping_to_clients(
    client_results: list[dict[str, Any]],
    profile_results: list[dict[str, Any]],
    assignments: dict[str, int],
) -> list[dict[str, Any]]:
    profiles_by_participant = {
        profile["participant"]: profile
        for profile in profile_results
        if profile.get("participant")
    }
    enriched: list[dict[str, Any]] = []
    for result in client_results:
        item = dict(result)
        participant = item.get("participant")
        if participant in assignments:
            item["cluster"] = assignments[participant]
        profile = profiles_by_participant.get(participant)
        if profile is not None:
            item["profile_status"] = profile.get("status", "ok")
            item["profile_matched"] = profile.get("matched")
            item["profile_matched_fraction"] = profile.get("matched_fraction")
            item["profile_traces"] = profile.get("n_matched_traces")
        enriched.append(item)
    return enriched


def create_app() -> FastAPI:
    app = FastAPI(title="Federated Sensor Coordinator")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @app.get("/api/clients")
    def clients() -> dict[str, Any]:
        return {"clients": _client_urls()}

    @app.get("/api/examples")
    def examples() -> dict[str, Any]:
        return {"queries": EXAMPLE_QUERIES}

    @app.get("/api/results")
    def results() -> dict[str, Any]:
        return {"results": RESULTS}

    @app.get("/api/artifacts/{filename:path}")
    def artifact(filename: str) -> FileResponse:
        root = _output_dir().resolve()
        target = (root / filename).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise HTTPException(status_code=404, detail="Artifact not found")
        return FileResponse(target)

    @app.post("/api/query")
    async def query(request: QueryRequest) -> dict[str, Any]:
        started = time.perf_counter()
        urls = _client_urls()
        if not urls:
            run = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "request": request.model_dump(),
                "clients": [],
                "matched_clients": [],
                "error": "CLIENTS environment variable is empty",
                "elapsed_s": 0.0,
            }
            RESULTS.insert(0, run)
            return run

        async with httpx.AsyncClient(timeout=60.0) as client:
            client_results = await asyncio.gather(
                *[_query_client(client, url, request) for url in urls]
            )

        profile_results: list[dict[str, Any]] = []
        grouped_result: dict[str, Any] | None = None
        if request.group:
            async with httpx.AsyncClient(timeout=60.0) as client:
                profile_results = await asyncio.gather(
                    *[_profile_client(client, url, request) for url in urls]
                )
            client_profiles = {
                result["participant"]: result.get("profile", {})
                for result in profile_results
                if result.get("participant") and result.get("matched") is True
            }
            try:
                grouped_result = await asyncio.to_thread(
                    run_grouped_evaluation,
                    _data_dir(),
                    _output_dir(),
                    ltl=request.ltl,
                    n_clusters=request.n_clusters,
                    eval_protocol=request.eval_protocol,
                    include_markov_baselines=request.include_baselines,
                    include_per_client_baseline=request.include_baselines,
                    write_workflow_graphs=True,
                    client_profiles=client_profiles,
                )
            except ValueError as exc:
                grouped_result = {
                    "error": str(exc),
                    "protocol": request.eval_protocol,
                    "output_dir": str(_output_dir()),
                    "artifacts": [],
                }

        matched_clients = [
            result.get("participant", result.get("client_url"))
            for result in client_results
            if result.get("matched") is True
        ]
        assignments = (
            grouped_result.get("clustering", {}).get("assignments", {})
            if grouped_result
            else {}
        )
        if request.group:
            client_results = _attach_grouping_to_clients(
                client_results,
                profile_results,
                assignments,
            )

        run = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "request": request.model_dump(),
            "client_count": len(urls),
            "matched_count": (
                grouped_result.get("matched_count", len(matched_clients))
                if grouped_result and not grouped_result.get("error")
                else len(matched_clients)
            ),
            "matched_clients": matched_clients,
            "clients": sorted(
                client_results,
                key=lambda item: str(item.get("participant", item.get("client_url"))),
            ),
            "eval_protocol": request.eval_protocol,
            "elapsed_s": round(time.perf_counter() - started, 6),
        }
        if request.group and grouped_result is not None:
            artifact_names = grouped_result.get("artifacts", [])
            run.update(
                {
                    "profiles": sorted(
                        profile_results,
                        key=lambda item: str(item.get("participant", item.get("client_url"))),
                    ),
                    "clustering": grouped_result.get("clustering"),
                    "comparison": grouped_result.get("comparison", []),
                    "per_cluster_accuracy": grouped_result.get("per_cluster_accuracy", {}),
                    "ltl_filter": grouped_result.get("ltl_filter"),
                    "artifacts_dir": grouped_result.get("output_dir", str(_output_dir())),
                    "artifacts": _artifact_links(artifact_names),
                    "eval_protocol": grouped_result.get("protocol", request.eval_protocol),
                    "grouped_train_samples": grouped_result.get("grouped_train_samples"),
                    "train_samples": grouped_result.get("train_samples"),
                    "test_samples": grouped_result.get("test_samples"),
                }
            )
            if grouped_result.get("error"):
                run["group_error"] = grouped_result["error"]
        RESULTS.insert(0, run)
        del RESULTS[25:]
        return run

    return app


app = create_app()
