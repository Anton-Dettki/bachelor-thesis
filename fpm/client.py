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
from fpm.ltl import LTLParseError
from fpm.models import train_and_evaluate
from shared.grouping import build_client_profile
from shared.ltl_filter import event_to_ltl_token


class TrainRequest(BaseModel):
    model: str = Field(default="tree", pattern="^(tree|frequency|markov|logreg)$")
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

        fit_traces = traces if request.eval_protocol == "casas2" else train_pool
        train_traces, eval_traces = split_traces_protocol(
            fit_traces,
            protocol=request.eval_protocol,
        )
        result = train_and_evaluate(request.model, train_traces, eval_traces)

        return {
            "participant": participant,
            "matched": True,
            "matched_fraction": matched_fraction,
            "n_matched_traces": len(matched_traces),
            "n_total_traces": len(traces),
            "n_train_traces": len(train_traces),
            "n_eval_traces": len(eval_traces),
            "n_train_events": event_count(train_traces),
            "n_eval_events": event_count(eval_traces),
            "eval_protocol": request.eval_protocol,
            "model": result["model"],
            "accuracy": result["accuracy"],
            "correct": result["correct"],
            "total": result["total"],
            "params": result["params"],
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
