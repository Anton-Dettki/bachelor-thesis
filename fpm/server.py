"""FastAPI phone server — expose pattern query resolution over HTTP."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pm4py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fpm.ltl import LTLParseError, PatternQuery
from fpm.phone import Phone


class ResolveRequest(BaseModel):
    query: str
    min_traces: int = Field(default=1, ge=1)


def _log_to_xes_string(log) -> str:
    if log.empty:
        return ""
    with tempfile.NamedTemporaryFile(suffix=".xes", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        pm4py.write_xes(log, str(tmp_path))
        return tmp_path.read_text(encoding="utf-8")
    finally:
        tmp_path.unlink(missing_ok=True)


def create_phone_app(phone: Phone) -> FastAPI:
    """Build a FastAPI app serving one phone's LTL resolver."""
    app = FastAPI(title=f"FPM Phone — {phone.subject_label}")

    @app.get("/info")
    def info() -> dict:
        return {
            "subject_id": phone.subject_id,
            "subject_label": phone.subject_label,
            "total_traces": len(phone.trace_sequences()),
            "activities": sorted(phone.activities_in_log()),
        }

    @app.post("/resolve")
    def resolve(body: ResolveRequest) -> dict:
        try:
            pattern = PatternQuery.parse(body.query)
        except LTLParseError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        matching = phone.select_matching_traces(pattern)
        meets = len(matching) >= body.min_traces
        filtered = phone.filtered_log(pattern) if meets else phone.log.iloc[0:0].copy()

        return {
            "subject_id": phone.subject_id,
            "subject_label": phone.subject_label,
            "meets_pattern": meets,
            "matching_traces": len(matching),
            "total_traces": len(phone.trace_sequences()),
            "matching_case_ids": matching,
            "filtered_xes": _log_to_xes_string(filtered),
        }

    return app
