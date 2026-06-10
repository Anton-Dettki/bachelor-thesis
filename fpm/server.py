"""FastAPI phone server — expose pattern query resolution over HTTP."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pm4py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fpm.loader import CASE_ID
from fpm.event_log import load_event_log
from fpm.ltl import LTLParseError, PatternQuery
from fpm.phone import Phone, select_matching_case_ids
from fpm.prefix import DEFAULT_PREFIX_DIR, Vocabulary
from fpm.predict import FEDERATED_MODELS, fit_params
from fpm.split import DEFAULT_SPLIT_DIR, subject_split_dir


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


def create_phone_app(
    phone: Phone,
    *,
    prefix_dir: Path = DEFAULT_PREFIX_DIR,
    split_dir: Path = DEFAULT_SPLIT_DIR,
) -> FastAPI:
    """Build a FastAPI app serving one phone's LTL resolver and predict params."""
    app = FastAPI(title=f"FPM Phone — {phone.subject_label}")

    @app.get("/info")
    def info() -> dict:
        return {
            "subject_id": phone.subject_id,
            "subject_label": phone.subject_label,
            "total_traces": len(phone.trace_sequences()),
            "activities": sorted(phone.activities_in_log()),
        }

    @app.get("/predict/params/{model}")
    def predict_params(model: str, query: str | None = None) -> dict:
        if model not in FEDERATED_MODELS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unknown federated model {model!r}; "
                    f"choose from {sorted(FEDERATED_MODELS)}"
                ),
            )

        scope_dir = prefix_dir / phone.subject_label
        train_path = scope_dir / "train.csv"
        vocab_path = scope_dir / "vocab.json"
        if not train_path.exists() or not vocab_path.exists():
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Prefix dataset not found for {phone.subject_label} under "
                    f"{prefix_dir}. Run build_prefix_datasets.py first."
                ),
            )

        train_df = pd.read_csv(train_path)
        matching_traces = 0
        total_traces = 0
        meets_pattern = True

        if query is not None:
            try:
                PatternQuery.parse(query)
            except LTLParseError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            train_log = load_event_log(
                subject_split_dir(split_dir, phone.subject_id) / "train.xes"
            )
            matching = select_matching_case_ids(train_log, query)
            matching_traces = len(matching)
            total_traces = (
                train_log[CASE_ID].astype(str).nunique() if not train_log.empty else 0
            )
            meets_pattern = matching_traces > 0
            if matching:
                allowed = set(matching)
                train_df = train_df[train_df["case_id"].astype(str).isin(allowed)]
            else:
                train_df = train_df.iloc[0:0]

        vocab = Vocabulary.read_json(vocab_path)
        params = fit_params(model, train_df, vocab)
        payload = {
            "subject_id": phone.subject_id,
            "subject_label": phone.subject_label,
            "model": model,
            "params": params,
            "n_train": len(train_df),
        }
        if query is not None:
            payload.update(
                {
                    "query": query,
                    "matching_traces": matching_traces,
                    "total_traces": total_traces,
                    "meets_pattern": meets_pattern,
                }
            )
        return payload

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
