"""Load Chinook ADL sensor CSVs from zip archives into a pm4py-compatible DataFrame."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

CASE_ID = "case:concept:name"
ACTIVITY = "concept:name"
TIMESTAMP = "time:timestamp"

DATASET_VARIANTS = ("adl_noerror", "adl_error")


def _parse_case_id(csv_path: str) -> str:
    return Path(csv_path).stem


def load_event_log_dataframe(
    zip_path: Path,
    *,
    task_filter: int | None = None,
) -> pd.DataFrame:
    """Read all trace CSVs from a Chinook zip and return a flat event-log DataFrame.

    Each CSV file is one trace (case). Activities are encoded as ``sensor_message``
    (e.g. ``M07_ON``). Rows are sorted by timestamp within each case.
    """
    if not zip_path.exists():
        raise FileNotFoundError(f"Dataset archive not found: {zip_path}")

    frames: list[pd.DataFrame] = []
    with zipfile.ZipFile(zip_path) as archive:
        csv_names = sorted(
            name for name in archive.namelist() if name.endswith(".csv")
        )
        for csv_name in csv_names:
            case_id = _parse_case_id(csv_name)
            if task_filter is not None and not case_id.endswith(f".t{task_filter}"):
                continue

            with archive.open(csv_name) as handle:
                trace = pd.read_csv(handle)

            trace[CASE_ID] = case_id
            trace[ACTIVITY] = (
                trace["sensor"].astype(str) + "_" + trace["message"].astype(str)
            )
            trace[TIMESTAMP] = pd.to_datetime(
                trace["date"].astype(str) + " " + trace["time"].astype(str),
                errors="coerce",
            )
            frames.append(trace)

    if not frames:
        raise ValueError(f"No CSV traces found in {zip_path}")

    event_log = pd.concat(frames, ignore_index=True)
    event_log = event_log.dropna(subset=[TIMESTAMP])
    event_log = event_log.sort_values([CASE_ID, TIMESTAMP]).reset_index(drop=True)
    return event_log


def resolve_dataset_zip(dataset_dir: Path, variant: str) -> Path:
    if variant not in DATASET_VARIANTS:
        raise ValueError(
            f"Unknown variant {variant!r}. Expected one of {DATASET_VARIANTS}."
        )
    return dataset_dir / f"{variant}.zip"
