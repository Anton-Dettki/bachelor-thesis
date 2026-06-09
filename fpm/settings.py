"""Pipeline configuration knobs shared across the FPM/predictive steps."""

from __future__ import annotations

#: Where event timestamps come from when building per-subject event logs.
#:
#: - ``"xes"`` (default): synthetic per-trace order timestamps derived from
#:   ``activity.xes``. Every trace starts at 2015-01-01 with one-second
#:   increments, and case ids are ``caseN``. This is what the SOWCompact
#:   Section 7 comparison reproduces, so it must stay the default.
#: - ``"csv"``: real wall-clock timestamps from ``activity.csv``
#:   (``attr_endtime``). Case ids are ``dayN``. Use this for predictive /
#:   temporal work where event and trace ordering must reflect actual time.
TIMESTAMP_SOURCE = "xes"

VALID_TIMESTAMP_SOURCES = ("xes", "csv")
