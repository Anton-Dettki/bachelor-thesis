"""Shared sensor filtering for CASAS/Chinook event loading."""

from __future__ import annotations

from typing import Iterable

# Rare sensors with negligible coverage (<0.5% of sensor-level labels).
EXCLUDED_SENSORS = frozenset({"M06", "M10", "M21", "M22", "I09", "E01"})


def normalize_sensor_id(sensor: object) -> str:
    return str(sensor).strip().upper()


def should_skip_sensor(
    sensor: object,
    *,
    skip_analog: bool = True,
    excluded_sensors: Iterable[str] | None = EXCLUDED_SENSORS,
) -> bool:
    """Return True when a raw sensor row should be dropped during loading."""
    sensor_id = normalize_sensor_id(sensor)
    if skip_analog and sensor_id.startswith("AD1"):
        return True
    if excluded_sensors is not None and sensor_id in {item.upper() for item in excluded_sensors}:
        return True
    return False
