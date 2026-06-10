"""Load per-subject ADL event logs from the dailylog2016 dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pm4py

CASE_ID = "case:concept:name"
ACTIVITY = "concept:name"
TIMESTAMP = "time:timestamp"

DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[1] / "dailylog2016_dataset"
SUBJECT_IDS = tuple(range(1, 8))
EXCLUDED_ACTIVITIES = ("TakeMedication", "Functionalmobility")

#: Canonical activity alphabet for the dailylog2016 ADL domain (after name
#: normalization and ``EXCLUDED_ACTIVITIES``). This is a *declared taxonomy*:
#: it is fixed independent of any train/validation split, so encoding a split
#: never leaks information about which activities appear only in validation.
#: It is identical across timestamp sources (xes/csv) and is the union of all
#: activities observed across every subject, giving a shared integer id space
#: so per-subject Markov counts remain summable for federated aggregation.
ACTIVITY_TAXONOMY = (
    "DeskWork",
    "EatingDrinking",
    "Housework",
    "Mealpreparation",
    "Movement",
    "PersonalGrooming",
    "Relaxing",
    "Shopping",
    "Sleeping",
    "Socializing",
    "Sport",
    "Transportation",
)


def subject_xes_path(dataset_root: Path, subject_id: int) -> Path:
    """Return the activity.xes path for a given subject (1-7)."""
    if subject_id not in SUBJECT_IDS:
        raise ValueError(f"subject_id must be one of {SUBJECT_IDS}, got {subject_id!r}")

    path = dataset_root / f"subject{subject_id}" / "data" / "activity.xes"
    if not path.exists():
        raise FileNotFoundError(f"Event log not found for subject {subject_id}: {path}")
    return path


def collapse_consecutive_activities(event_log: pd.DataFrame) -> pd.DataFrame:
    """Remove back-to-back repeats of the same activity within a trace.

    ADL logs often contain consecutive identical activities (e.g. three
    Housework events in a row). Alpha/Alpha+ infers causality from direct
    succession between *different* activities, so these repeats prevent
    causal relations from being discovered and yield disconnected models.
    """
    mask = event_log[ACTIVITY] != event_log.groupby(CASE_ID)[ACTIVITY].shift(1)
    collapsed = event_log.loc[mask.fillna(True)].copy()
    return collapsed.reset_index(drop=True)


def _normalize_activity_names(event_log: pd.DataFrame) -> pd.DataFrame:
    normalized = event_log.copy()
    normalized[ACTIVITY] = (
        normalized[ACTIVITY].astype(str).str.replace("/", "", regex=False)
    )
    return normalized


def _add_order_timestamps(event_log: pd.DataFrame) -> pd.DataFrame:
    """Assign synthetic timestamps when the XES log has none.

    Alpha+ requires a datetime column. Event order within each trace is
    preserved using one-second increments from a fixed epoch.
    """
    if TIMESTAMP in event_log.columns:
        event_log = event_log.copy()
        event_log[TIMESTAMP] = pd.to_datetime(event_log[TIMESTAMP], errors="coerce")
        if event_log[TIMESTAMP].notna().all():
            return event_log

    ordered = event_log.copy()
    ordered["_event_order"] = ordered.groupby(CASE_ID).cumcount()
    base = pd.Timestamp("2015-01-01")
    ordered[TIMESTAMP] = base + pd.to_timedelta(ordered["_event_order"], unit="s")
    return ordered.drop(columns=["_event_order"])


def subject_csv_path(dataset_root: Path, subject_id: int) -> Path:
    if subject_id not in SUBJECT_IDS:
        raise ValueError(f"subject_id must be one of {SUBJECT_IDS}, got {subject_id!r}")

    path = dataset_root / f"subject{subject_id}" / "data" / "activity.csv"
    if not path.exists():
        raise FileNotFoundError(f"Activity CSV not found for subject {subject_id}: {path}")
    return path


def load_subject_csv(csv_path: Path) -> pd.DataFrame:
    """Read activity.csv: one trace per day, timestamped at activity end time."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Activity CSV not found: {csv_path}")

    raw = pd.read_csv(csv_path)
    raw["_event_order"] = range(len(raw))
    raw[CASE_ID] = "day" + raw["dayID"].astype(str)
    raw[ACTIVITY] = raw["label_activity"].astype(str)
    raw[TIMESTAMP] = pd.to_datetime(
        raw["attr_endtime"], format="%d.%m.%y %H:%M:%S", errors="coerce"
    )
    raw = raw.dropna(subset=[TIMESTAMP])
    raw = raw.sort_values(
        [CASE_ID, TIMESTAMP, "_event_order"],
        kind="stable",
    ).reset_index(drop=True)
    raw = _normalize_activity_names(raw)
    raw = raw[~raw[ACTIVITY].isin(EXCLUDED_ACTIVITIES)].reset_index(drop=True)
    return raw[[CASE_ID, ACTIVITY, TIMESTAMP]]


def load_subject_log(xes_path: Path):
    """Read a subject's XES file into a pm4py-compatible event log.

    XES logs carry no usable timestamps, so synthetic per-trace order
    timestamps are assigned and case ids stay as ``caseN``.
    """
    if not xes_path.exists():
        raise FileNotFoundError(f"Event log not found: {xes_path}")

    raw = pm4py.read_xes(str(xes_path))
    if not isinstance(raw, pd.DataFrame):
        raw = pm4py.convert_to_dataframe(raw)

    raw = _normalize_activity_names(raw)
    with_timestamps = _add_order_timestamps(raw)
    return pm4py.format_dataframe(
        with_timestamps,
        case_id=CASE_ID,
        activity_key=ACTIVITY,
        timestamp_key=TIMESTAMP,
    )


def load_subject_csv_log(csv_path: Path):
    """Read a subject's activity.csv into a pm4py-compatible event log.

    Unlike :func:`load_subject_log`, this carries real ``attr_endtime``
    timestamps and uses ``dayN`` case ids. Activity normalization and the
    ``EXCLUDED_ACTIVITIES`` filter are already applied by
    :func:`load_subject_csv`.
    """
    raw = load_subject_csv(csv_path)
    return pm4py.format_dataframe(
        raw,
        case_id=CASE_ID,
        activity_key=ACTIVITY,
        timestamp_key=TIMESTAMP,
    )
