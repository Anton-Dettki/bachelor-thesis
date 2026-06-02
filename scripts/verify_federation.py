#!/usr/bin/env python3
"""Verify HTTP federation parity with in-process aggregation (ASGI transport)."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pm4py

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fpm.aggregator import Aggregator  # noqa: E402
from fpm.client import PhoneClient, RemotePhoneConnector  # noqa: E402
from fpm.event_log import DEFAULT_EVENT_LOG_DIR, subject_event_log_xes_path  # noqa: E402
from fpm.loader import CASE_ID, SUBJECT_IDS  # noqa: E402
from fpm.ltl import PatternQuery  # noqa: E402
from fpm.phone import Phone  # noqa: E402
from fpm.queries import SCENARIO_QUERIES  # noqa: E402
from fpm.server import create_phone_app  # noqa: E402
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
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            line = f"  [{status}] {r.name}"
            if r.detail:
                line += f"  —  {r.detail}"
            print(line)
        print()
        ok = sum(1 for r in self.results if r.passed)
        print(f"Result: {ok}/{len(self.results)} checks passed")
        if not self.passed:
            print("FAILED")


def event_logs_available() -> bool:
    return all(
        subject_event_log_xes_path(DEFAULT_EVENT_LOG_DIR, sid).exists()
        for sid in SUBJECT_IDS
    )


def build_asgi_connectors() -> list[RemotePhoneConnector]:
    connectors: list[RemotePhoneConnector] = []
    for sid in SUBJECT_IDS:
        phone = Phone(sid)
        app = create_phone_app(phone)
        connectors.append(
            RemotePhoneConnector(PhoneClient.from_app(app))
        )
    return connectors


def compare_results(
    report: Report,
    prefix: str,
    local: Aggregator,
    remote: Aggregator,
    pattern: PatternQuery,
) -> None:
    local_result = local.run(pattern)
    remote_result = remote.run(pattern)

    if local_result.contributing_subjects == remote_result.contributing_subjects:
        report.ok(f"{prefix}: contributing subjects", str(local_result.contributing_subjects))
    else:
        report.fail(
            f"{prefix}: contributing subjects",
            f"local={local_result.contributing_subjects} "
            f"remote={remote_result.contributing_subjects}",
        )

    local_traces = len(pm4py.get_event_attribute_values(local_result.integrated_log, CASE_ID))
    remote_traces = len(
        pm4py.get_event_attribute_values(remote_result.integrated_log, CASE_ID)
    )
    if local_traces == remote_traces:
        report.ok(f"{prefix}: integrated traces", str(local_traces))
    else:
        report.fail(f"{prefix}: integrated traces", f"{local_traces} vs {remote_traces}")

    if len(local_result.integrated_log) == len(remote_result.integrated_log):
        report.ok(f"{prefix}: integrated events", str(len(local_result.integrated_log)))
    else:
        report.fail(
            f"{prefix}: integrated events",
            f"{len(local_result.integrated_log)} vs {len(remote_result.integrated_log)}",
        )

    if local_result.integrated_log.empty:
        return

    local_miner = SocialProcessMiner()
    remote_miner = SocialProcessMiner()
    local_stats = local_miner.discover_with_stats(local_result.integrated_log).stats
    remote_stats = remote_miner.discover_with_stats(remote_result.integrated_log).stats

    if local_stats["sum_arc_weights"] == remote_stats["sum_arc_weights"]:
        report.ok(f"{prefix}: sum_arc_weights", str(local_stats["sum_arc_weights"]))
    else:
        report.fail(
            f"{prefix}: sum_arc_weights",
            f"{local_stats['sum_arc_weights']} vs {remote_stats['sum_arc_weights']}",
        )

    total_bytes = sum(c.bytes_transferred or 0 for c in remote_result.contributions)
    if total_bytes > 0:
        report.ok(f"{prefix}: network bytes received", str(total_bytes))
    else:
        report.fail(f"{prefix}: network bytes received", "zero bytes")


def check_malformed_query(report: Report, remote: Aggregator) -> None:
    result = remote.run("F(A &")
    errors = [c for c in result.contributions if c.error]
    if len(errors) == len(SUBJECT_IDS):
        report.ok("malformed query: all phones report error")
    else:
        report.fail("malformed query: all phones report error", f"{len(errors)} errors")


def check_down_phone(report: Report, pattern: PatternQuery) -> None:
    """One unreachable phone is skipped; others still contribute."""
    good = build_asgi_connectors()[:2]
    bad = RemotePhoneConnector(PhoneClient("http://127.0.0.1:59999"))
    agg = Aggregator([*good, bad])
    result = agg.run(pattern)

    bad_contrib = result.contributions[-1]
    if bad_contrib.error:
        report.ok("down phone: unreachable phone recorded error")
    else:
        report.fail("down phone: unreachable phone recorded error", "no error set")

    if any(c.error for c in result.contributions[:-1]):
        report.fail("down phone: good phones unaffected")
    else:
        report.ok("down phone: good phones unaffected")

    local = Aggregator.from_subject_ids([1, 2])
    local_result = local.run(pattern)
    remote_traces = len(pm4py.get_event_attribute_values(result.integrated_log, CASE_ID))
    local_traces = len(pm4py.get_event_attribute_values(local_result.integrated_log, CASE_ID))
    if remote_traces == local_traces:
        report.ok("down phone: integration unchanged", str(local_traces))
    else:
        report.fail("down phone: integration unchanged", f"{local_traces} vs {remote_traces}")


def main() -> None:
    report = Report()

    if not event_logs_available():
        report.fail("event logs", "Run: python scripts/build_event_logs.py")
        report.print_summary()
        sys.exit(1)

    local = Aggregator.from_subject_ids(SUBJECT_IDS)
    remote = Aggregator(build_asgi_connectors())

    print("Checking scenario parity (local vs HTTP/ASGI) ...")
    for scenario, query_text in SCENARIO_QUERIES.items():
        compare_results(report, scenario, local, remote, PatternQuery.parse(query_text))

    print("Checking edge cases ...")
    check_malformed_query(report, remote)
    check_down_phone(
        report,
        PatternQuery.parse(SCENARIO_QUERIES["scenario1_shopping_mealprep"]),
    )

    print()
    report.print_summary()
    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
