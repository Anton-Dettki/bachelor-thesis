"""Sensor-level event abstraction for next-activity prediction."""

from __future__ import annotations

from typing import Iterable, Literal, Mapping, Sequence

AbstractionLevel = Literal["raw", "sensor"]

PAD_TOKEN = "<PAD>"
_KNOWN_STATES = frozenset(
    {
        "ON",
        "OFF",
        "PRESENT",
        "ABSENT",
        "OPEN",
        "CLOSE",
        "START",
        "END",
        "STOP_INSTRUCT",
    }
)


def abstract_sensor_event(token: str) -> str:
    """Collapse a raw sensor event to the sensor/activity identifier."""
    if token in {PAD_TOKEN, "<PAD>"}:
        return token

    if "=" in token:
        sensor, message = token.split("=", 1)
        if sensor.lower() == "asterisk":
            return message.upper()
        return sensor

    if "_" in token:
        sensor, state = token.rsplit("_", 1)
        if state.upper() in _KNOWN_STATES:
            if sensor.upper() == "ASTERISK":
                return state.upper()
            return sensor

    return token


def collapse_consecutive_events(events: Sequence[str]) -> list[str]:
    """Drop repeated adjacent activities so toggles become one sensor visit."""
    collapsed: list[str] = []
    for event in events:
        if not collapsed or collapsed[-1] != event:
            collapsed.append(event)
    return collapsed


def abstract_trace(
    events: Sequence[str],
    *,
    collapse_consecutive: bool = True,
) -> list[str]:
    """Map a raw event trace to sensor-level activities."""
    abstracted = [abstract_sensor_event(event) for event in events]
    if collapse_consecutive:
        return collapse_consecutive_events(abstracted)
    return abstracted


def abstract_traces_by_client(
    traces_by_client: Mapping[str, Sequence[Sequence[str]]],
    *,
    collapse_consecutive: bool = True,
) -> dict[str, list[list[str]]]:
    """Abstract every client trace while preserving trace boundaries."""
    return {
        client_id: [
            abstract_trace(trace, collapse_consecutive=collapse_consecutive)
            for trace in traces
        ]
        for client_id, traces in traces_by_client.items()
    }


def flatten_abstracted_events(
    traces_by_client: Mapping[str, Sequence[Sequence[str]]],
    *,
    collapse_consecutive: bool = True,
) -> dict[str, list[str]]:
    """Flatten abstracted traces into one activity list per client."""
    events_by_client: dict[str, list[str]] = {}
    for client_id, traces in traces_by_client.items():
        events: list[str] = []
        for trace in traces:
            events.extend(abstract_trace(trace, collapse_consecutive=collapse_consecutive))
        events_by_client[client_id] = events
    return events_by_client


def normalize_trace(
    events: Sequence[str],
    abstraction: AbstractionLevel = "sensor",
    *,
    collapse_consecutive: bool = True,
) -> list[str]:
    """Return a trace at the requested abstraction level."""
    if abstraction == "raw":
        return list(events)
    return abstract_trace(events, collapse_consecutive=collapse_consecutive)


def workflow_artifact_stem(base: str, abstraction: AbstractionLevel) -> str:
    """Build stable workflow artifact names per abstraction view."""
    if abstraction == "sensor":
        return base
    return f"{base}_{abstraction}"


ABSTRACTION_DESCRIPTIONS: dict[AbstractionLevel, str] = {
    "sensor": (
        "Sensor-level view: ON/OFF and PRESENT/ABSENT are collapsed to sensor IDs "
        "(e.g. M07=ON and M07=OFF both become M07), and rapid repeats are removed."
    ),
    "raw": (
        "Raw event view: predicts the next low-level sensor state token "
        "(e.g. M07=ON, M13=OFF, I08=PRESENT) without collapsing toggles."
    ),
}
