"""FastAPI app representing one federated sensor client."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from fpm.dataset import (
    EVAL_TRIAL,
    event_count,
    filter_traces,
    load_participant,
    split_traces_protocol,
    training_traces,
)
from fpm.casas_client import train_and_evaluate_casas_client_views
from fpm.ltl import LTLParseError
from fpm.models import train_and_evaluate
from shared.grouping import build_client_profile
from shared.ltl_filter import event_to_ltl_token


class TrainRequest(BaseModel):
    model: str = Field(
        default="casas_tree",
        pattern="^(casas_tree|casas_markov|tree|frequency|markov|logreg)$",
    )
    ltl: str = ""
    min_match_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    eval_protocol: str = Field(default="federated", pattern="^(casas2|federated)$")


def _participant() -> str:
    participant = os.getenv("PARTICIPANT", "").strip()
    if not participant:
        raise RuntimeError("PARTICIPANT environment variable is required")
    return participant


def _data_dir() -> Path:
    return Path(os.getenv("DATA_DIR", "data"))


def _client_view_payload(result: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "accuracy": result.get("accuracy"),
        "correct": result.get("correct"),
        "total": result.get("total"),
        "params": result.get("params"),
    }
    if "macro_f1" in result:
        payload["macro_f1"] = result["macro_f1"]
    if "weighted_f1" in result:
        payload["weighted_f1"] = result["weighted_f1"]
    return payload


def _legacy_client_views(
    traces,
    *,
    model_name: str,
    protocol: str,
) -> dict[str, dict[str, Any]]:
    fit_traces = traces if protocol == "casas2" else training_traces(traces)
    views: dict[str, dict[str, Any]] = {}
    for abstraction in ("sensor", "raw"):
        train_traces, eval_traces = split_traces_protocol(
            fit_traces,
            protocol=protocol,
            abstraction=abstraction,
        )
        result = train_and_evaluate(model_name, train_traces, eval_traces)
        views[abstraction] = {
            **_client_view_payload(result),
            "model": result["model"],
            "n_train_traces": len(train_traces),
            "n_eval_traces": len(eval_traces),
            "n_train_events": event_count(train_traces),
            "n_eval_events": event_count(eval_traces),
        }
    return views


def create_app() -> FastAPI:
    app = FastAPI(title="Federated Sensor Client")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "participant": _participant()}

    @app.get("/info")
    def info() -> dict[str, Any]:
        traces = load_participant(_participant(), _data_dir())
        return {
            "participant": _participant(),
            "source": traces[0].source if traces else None,
            "trials": len(traces),
            "events": sum(trace.event_count for trace in traces),
            "tokens": sorted({event for trace in traces for event in trace.events}),
        }

    @app.post("/train")
    def train(request: TrainRequest) -> dict[str, Any]:
        started = time.perf_counter()
        participant = _participant()
        traces = load_participant(participant, _data_dir())
        train_pool = training_traces(traces)

        try:
            matched_traces, matched_fraction = filter_traces(train_pool, request.ltl)
        except LTLParseError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        has_filter = bool(request.ltl.strip())
        matched = (not has_filter) or (
            bool(matched_traces) and matched_fraction >= request.min_match_fraction
        )
        if not matched:
            return {
                "participant": participant,
                "matched": False,
                "matched_fraction": matched_fraction,
                "n_matched_traces": len(matched_traces),
                "message": "client filtered out by LTL query",
                "elapsed_s": round(time.perf_counter() - started, 6),
            }

        if request.model in {"casas_tree", "casas_markov"}:
            raw_views = train_and_evaluate_casas_client_views(
                _data_dir(),
                participant,
                model_name=request.model,
                protocol=request.eval_protocol,
            )
            abstraction_views = {
                level: _client_view_payload(raw_views[level])
                for level in ("sensor", "raw")
            }
            sensor_view = abstraction_views["sensor"]
            return {
                "participant": participant,
                "matched": True,
                "matched_fraction": matched_fraction,
                "n_matched_traces": len(matched_traces),
                "n_total_traces": len(traces),
                "n_train_traces": None,
                "n_eval_traces": None,
                "n_train_events": None,
                "n_eval_events": None,
                "eval_protocol": request.eval_protocol,
                "model": raw_views["sensor"]["model"],
                "abstraction_views": abstraction_views,
                "accuracy": sensor_view["accuracy"],
                "macro_f1": sensor_view.get("macro_f1"),
                "weighted_f1": sensor_view.get("weighted_f1"),
                "correct": sensor_view["correct"],
                "total": sensor_view["total"],
                "params": sensor_view["params"],
                "elapsed_s": round(time.perf_counter() - started, 6),
            }

        legacy_views = _legacy_client_views(
            traces,
            model_name=request.model,
            protocol=request.eval_protocol,
        )
        abstraction_views = {
            level: _client_view_payload(legacy_views[level])
            for level in ("sensor", "raw")
        }
        sensor_view = legacy_views["sensor"]
        return {
            "participant": participant,
            "matched": True,
            "matched_fraction": matched_fraction,
            "n_matched_traces": len(matched_traces),
            "n_total_traces": len(traces),
            "n_train_traces": sensor_view["n_train_traces"],
            "n_eval_traces": sensor_view["n_eval_traces"],
            "n_train_events": sensor_view["n_train_events"],
            "n_eval_events": sensor_view["n_eval_events"],
            "eval_protocol": request.eval_protocol,
            "model": sensor_view["model"],
            "abstraction_views": abstraction_views,
            "accuracy": sensor_view["accuracy"],
            "correct": sensor_view["correct"],
            "total": sensor_view["total"],
            "params": sensor_view["params"],
            "elapsed_s": round(time.perf_counter() - started, 6),
        }

    @app.get("/profile")
    def profile(
        ltl: str = "",
        min_match_fraction: float = Query(default=0.0, ge=0.0, le=1.0),
    ) -> dict[str, Any]:
        participant = _participant()
        traces = load_participant(participant, _data_dir())
        train_traces = training_traces(traces)

        try:
            matched_traces, matched_fraction = filter_traces(train_traces, ltl)
        except LTLParseError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        has_filter = bool(ltl.strip())
        matched = (not has_filter) or (
            bool(matched_traces) and matched_fraction >= min_match_fraction
        )
        selected = matched_traces if matched else []

        events_by_trial = {
            str(trace.trial): trace.event_count
            for trace in selected
        }
        train_event_traces = [
            [event_to_ltl_token(event) for event in trace.events]
            for trace in selected
        ]
        flat_events = [
            event
            for trace_events in train_event_traces
            for event in trace_events
        ]
        events_by_task = {
            trace.trial: [event_to_ltl_token(event) for event in trace.events]
            for trace in selected
        }
        profile_vector = (
            build_client_profile(
                flat_events,
                events_by_task=events_by_task,
                include_task_breakdown=True,
            )
            if matched and flat_events
            else {}
        )

        return {
            "participant": participant,
            "matched": matched,
            "matched_fraction": matched_fraction,
            "n_matched_traces": len(selected),
            "n_total_train_traces": len(train_traces),
            "events_by_trial": events_by_trial,
            "profile": profile_vector,
            "train_traces": train_event_traces,
        }

    return app


app = create_app()
