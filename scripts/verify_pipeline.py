#!/usr/bin/env python3
"""Verify the FPM pipeline (steps 3–4): invariants, edge cases, and scenario sweep."""

from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pm4py

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fpm.aggregator import (  # noqa: E402
    RESOURCE,
    Aggregator,
    namespace_filtered_log,
)
from fpm.event_log import DEFAULT_EVENT_LOG_DIR, subject_event_log_xes_path  # noqa: E402
from fpm.loader import ACTIVITY, CASE_ID, SUBJECT_IDS, TIMESTAMP  # noqa: E402
from fpm.ltl import PatternQuery  # noqa: E402
from fpm.phone import Phone  # noqa: E402
from fpm.queries import SCENARIO_QUERIES  # noqa: E402
from fpm.social import SocialProcessMiner  # noqa: E402


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class Report:
    results: list[CheckResult] = field(default_factory=list)

    def ok(self, name: str, detail: str = "") -> None:
        self.results.append(CheckResult(name, True, detail))

    def fail(self, name: str, detail: str = "") -> None:
        self.results.append(CheckResult(name, False, detail))

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    def print_summary(self) -> None:
        width = max(len(r.name) for r in self.results) if self.results else 20
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            line = f"  [{status}] {r.name:<{width}}"
            if r.detail:
                line += f"  —  {r.detail}"
            print(line)
        print()
        total = len(self.results)
        ok = sum(1 for r in self.results if r.passed)
        print(f"Result: {ok}/{total} checks passed")
        if not self.passed:
            print("FAILED — see [FAIL] lines above.")


def event_logs_available() -> bool:
    return all(
        subject_event_log_xes_path(DEFAULT_EVENT_LOG_DIR, sid).exists()
        for sid in SUBJECT_IDS
    )


def make_synthetic_log(
    subject_label: str,
    traces: dict[str, list[tuple[str, str]]],
) -> pd.DataFrame:
    """Build a minimal pm4py log. traces: case_id -> [(activity, timestamp), ...]."""
    rows: list[dict] = []
    for case_id, events in traces.items():
        for activity, ts in events:
            rows.append(
                {
                    CASE_ID: case_id,
                    ACTIVITY: activity,
                    TIMESTAMP: pd.Timestamp(ts, tz="UTC"),
                }
            )
    log = pd.DataFrame(rows)
    return pm4py.format_dataframe(
        log,
        case_id=CASE_ID,
        activity_key=ACTIVITY,
        timestamp_key=TIMESTAMP,
    )


def assert_integrated_invariants(
    report: Report,
    *,
    prefix: str,
    result,
    query: str | PatternQuery,
) -> None:
    pattern = query if isinstance(query, PatternQuery) else PatternQuery.parse(query)
    integrated = result.integrated_log
    contributors = [c for c in result.contributions if c.meets_pattern]

    expected_traces = sum(c.matching_traces for c in contributors)
    expected_events = sum(c.filtered_events for c in contributors)
    actual_traces = len(pm4py.get_event_attribute_values(integrated, CASE_ID))

    if expected_traces == actual_traces:
        report.ok(f"{prefix}: trace count", f"{actual_traces} traces")
    else:
        report.fail(
            f"{prefix}: trace count",
            f"expected {expected_traces}, got {actual_traces}",
        )

    if expected_events == len(integrated):
        report.ok(f"{prefix}: event count", f"{len(integrated)} events")
    else:
        report.fail(
            f"{prefix}: event count",
            f"expected {expected_events}, got {len(integrated)}",
        )

    if integrated.empty:
        return

    unique_cases = integrated[CASE_ID].astype(str).unique()
    if len(unique_cases) == actual_traces:
        report.ok(f"{prefix}: unique case ids")
    else:
        report.fail(
            f"{prefix}: unique case ids",
            f"{len(unique_cases)} unique vs {actual_traces} traces",
        )

    bare = [c for c in unique_cases if c.startswith("day") or ":" not in c]
    if not bare:
        report.ok(f"{prefix}: namespaced case ids")
    else:
        report.fail(f"{prefix}: namespaced case ids", f"bare ids: {bare[:3]}")

    if RESOURCE in integrated.columns:
        mismatches = []
        for case_id, group in integrated.groupby(CASE_ID, sort=False):
            expected = str(case_id).split(":")[0]
            if not (group[RESOURCE].astype(str) == expected).all():
                mismatches.append(str(case_id))
        if not mismatches:
            report.ok(f"{prefix}: org:resource provenance")
        else:
            report.fail(f"{prefix}: org:resource provenance", str(mismatches[:3]))
    else:
        report.fail(f"{prefix}: org:resource provenance", "column missing")

    for c in result.contributions:
        phone = Phone(c.subject_id)
        direct = len(phone.select_matching_traces(pattern))
        if direct == c.matching_traces:
            continue
        report.fail(
            f"{prefix}: step3 agrees ({c.subject_label})",
            f"phone={direct}, aggregator={c.matching_traces}",
        )
        return
    report.ok(f"{prefix}: step3 agrees with aggregator")


def check_synthetic_edge_cases(report: Report) -> None:
    # Case ID collision across two synthetic phones
    log_a = make_synthetic_log(
        "subject1",
        {"day1": [("A", "2020-01-01 08:00:00"), ("B", "2020-01-01 09:00:00")]},
    )
    log_b = make_synthetic_log(
        "subject2",
        {"day1": [("C", "2020-01-02 08:00:00"), ("D", "2020-01-02 09:00:00")]},
    )
    phones = [
        Phone(1, log=log_a),
        Phone(2, log=log_b),
    ]
    agg = Aggregator(phones)
    result = agg.run("true")
    cases = set(result.integrated_log[CASE_ID].astype(str))
    if cases == {"subject1:day1", "subject2:day1"}:
        report.ok("synthetic: case id collision avoided")
    else:
        report.fail("synthetic: case id collision avoided", str(sorted(cases)))

    # min_traces gate
    log_partial = make_synthetic_log(
        "subject1",
        {
            "day1": [("A", "2020-01-01 08:00:00")],
            "day2": [("B", "2020-01-02 08:00:00")],
        },
    )
    agg2 = Aggregator([Phone(1, log=log_partial)])
    gated = agg2.run("true", min_traces=3)
    if not gated.contributing_subjects and gated.integrated_log.empty:
        report.ok("synthetic: min_traces excludes contributor")
    else:
        report.fail("synthetic: min_traces excludes contributor")

    # No matches
    nomatch = agg2.run("F(NonExistentActivityXYZ)")
    if nomatch.integrated_log.empty and not nomatch.contributing_subjects:
        report.ok("synthetic: no-match query yields empty log")
    else:
        report.fail("synthetic: no-match query yields empty log")

    # Empty aggregator rejected
    try:
        Aggregator([])
        report.fail("synthetic: empty phone list raises")
    except ValueError:
        report.ok("synthetic: empty phone list raises")

    # Social miner rejects empty log
    try:
        SocialProcessMiner().discover_with_stats(nomatch.integrated_log)
        report.fail("synthetic: discovery on empty log raises")
    except ValueError:
        report.ok("synthetic: discovery on empty log raises")

    # Namespace helper
    empty = namespace_filtered_log(log_a.iloc[0:0], "subject1")
    if empty.empty:
        report.ok("synthetic: namespace empty log")
    else:
        report.fail("synthetic: namespace empty log")

    # XES round-trip on synthetic integrated log
    with tempfile.TemporaryDirectory() as tmp:
        xes_path = Path(tmp) / "test.xes"
        pm4py.write_xes(result.integrated_log, str(xes_path))
        reloaded = pm4py.read_xes(str(xes_path))
        if not isinstance(reloaded, pd.DataFrame):
            reloaded = pm4py.convert_to_dataframe(reloaded)
        if len(reloaded) == len(result.integrated_log):
            report.ok("synthetic: XES round-trip preserves events")
        else:
            report.fail(
                "synthetic: XES round-trip preserves events",
                f"{len(reloaded)} vs {len(result.integrated_log)}",
            )


def check_real_scenarios(report: Report) -> None:
    for scenario, query_text in SCENARIO_QUERIES.items():
        pattern = PatternQuery.parse(query_text)
        agg = Aggregator.from_subject_ids(SUBJECT_IDS)
        result = agg.run(pattern)
        assert_integrated_invariants(
            report,
            prefix=scenario,
            result=result,
            query=pattern,
        )

        if result.integrated_log.empty:
            continue

        try:
            discovery = SocialProcessMiner().discover_with_stats(result.integrated_log)
            if discovery.stats["sum_arc_weights"] > 0:
                report.ok(f"{scenario}: heuristic discovery")
            else:
                report.fail(f"{scenario}: heuristic discovery", "zero arc weights")
        except Exception as exc:
            report.fail(f"{scenario}: heuristic discovery", str(exc))


def check_g_sport_hand_sample(report: Report) -> None:
    """Spot-check G(!Sport): matching days must not contain Sport."""
    query = PatternQuery.parse("G(!Sport)")
    phone = Phone(1)
    matching = set(phone.select_matching_traces(query))
    sequences = phone.trace_sequences()

    bad = []
    for case_id in matching:
        if "Sport" in sequences[case_id]:
            bad.append(case_id)

    if not bad:
        report.ok(
            "G(!Sport) hand-check subject1",
            f"{len(matching)}/{len(sequences)} days without Sport",
        )
    else:
        report.fail("G(!Sport) hand-check subject1", f"Sport found in {bad}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FPM pipeline verification checks (synthetic + real data)."
    )
    parser.add_argument(
        "--skip-real-data",
        action="store_true",
        help="Only run synthetic edge-case checks (no event logs required)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = Report()

    print("Running synthetic edge-case checks ...")
    check_synthetic_edge_cases(report)

    if args.skip_real_data:
        print("(Skipping real-data scenarios — use without --skip-real-data for full run)\n")
    elif not event_logs_available():
        report.fail(
            "real data available",
            "Run: python scripts/build_event_logs.py",
        )
        print()
    else:
        print("Running 5 SOWCompact scenario checks on real event logs ...")
        check_real_scenarios(report)
        check_g_sport_hand_sample(report)

    print()
    report.print_summary()
    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
