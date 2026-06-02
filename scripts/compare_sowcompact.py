#!/usr/bin/env python3
"""Compare FPM results against a full-log baseline and SOWCompact paper (Section 7)."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pm4py

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fpm.aggregator import Aggregator, aggregator_metrics, contribution_summary  # noqa: E402
from fpm.event_log import DEFAULT_EVENT_LOG_DIR  # noqa: E402
from fpm.ltl import PatternQuery  # noqa: E402
from fpm.queries import SCENARIO_QUERIES, query_slug  # noqa: E402
from fpm.social import SocialProcessMiner  # noqa: E402

# SOWCompact paper (Section 7) reference values for the ADL case study.
PAPER_BASELINE = {
    "integrated_log_kb": 201.0,
    "activities": 12,
    "arcs": 116,
    "sum_arc_weights": 1309,
}

PAPER_FPM = {
    "scenario1_shopping_mealprep": {
        "integrated_log_kb": 24.0,
        "activities": 11,
        "arcs": 59,
        "sum_arc_weights": 150,
    },
    "scenario2_no_sport": {
        "integrated_log_kb": 47.0,
        "activities": 11,
        "arcs": 78,
        "sum_arc_weights": 298,
    },
    "scenario3_movement_transportation": {
        "integrated_log_kb": 105.0,
        "activities": 12,
        "arcs": 96,
        "sum_arc_weights": 642,
    },
    "scenario4_social_eat_transport": {
        "integrated_log_kb": 51.0,
        "activities": 12,
        "arcs": 81,
        "sum_arc_weights": 313,
    },
    "scenario5_no_eat_no_social": {
        "integrated_log_kb": 4.0,
        "activities": 6,
        "arcs": 12,
        "sum_arc_weights": 22,
    },
}

BASELINE_QUERY = "true"


@dataclass
class RunMetrics:
    label: str
    query: str
    metrics_path: Path
    data: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare SOWCompact FPM scenario results to a full-log baseline "
            "and published paper reference values."
        )
    )
    parser.add_argument(
        "--sow-dir",
        type=Path,
        default=ROOT / "output" / "sow",
        help="Directory containing per-query SOW outputs (metrics.json)",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=ROOT / "output" / "baseline_full",
        help="Directory for the non-federated full-log baseline",
    )
    parser.add_argument(
        "--event-log-dir",
        type=Path,
        default=DEFAULT_EVENT_LOG_DIR,
        help="Event log directory (used with --run)",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run baseline + all 5 scenarios before comparing (may take a minute)",
    )
    parser.add_argument(
        "--with-quality",
        action="store_true",
        help="Compute fitness/precision via alignments (slow)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=ROOT / "output" / "comparison" / "sowcompact_comparison.json",
        help="Write structured comparison results to this path",
    )
    return parser.parse_args()


def metrics_path(output_dir: Path, query_text: str) -> Path:
    return output_dir / query_slug(query_text) / "metrics.json"


def run_one_query(
    query_text: str,
    output_dir: Path,
    *,
    event_log_dir: Path,
) -> Path:
    """Aggregate, discover, and write metrics for one query. Returns metrics path."""
    pattern = PatternQuery.parse(query_text)
    output_dir.mkdir(parents=True, exist_ok=True)
    query_dir = output_dir / query_slug(pattern.text)
    query_dir.mkdir(parents=True, exist_ok=True)

    aggregator = Aggregator.from_subject_ids(event_log_dir=event_log_dir)
    result = aggregator.run(pattern)

    metrics = aggregator_metrics(result)
    metrics["contributions"] = [
        contribution_summary(c) for c in result.contributions
    ]

    query_payload = {
        "query": result.query,
        "contributing_subjects": result.contributing_subjects,
        "contributions": metrics["contributions"],
    }

    if not result.integrated_log.empty:
        pm4py.write_xes(result.integrated_log, str(query_dir / "integrated_log.xes"))
        result.integrated_log.to_csv(query_dir / "integrated_log.csv", index=False)

        miner = SocialProcessMiner()
        discovery = miner.discover_with_stats(result.integrated_log)
        miner.write_artifacts(
            discovery.net,
            discovery.initial_marking,
            discovery.final_marking,
            query_dir,
        )
        metrics["model"] = discovery.stats
        metrics["discovery_time_s"] = discovery.stats["discovery_time_s"]

    out = metrics_path(output_dir, pattern.text)
    (query_dir / "query.json").write_text(
        json.dumps(query_payload, indent=2),
        encoding="utf-8",
    )
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return out


def ensure_runs(args: argparse.Namespace) -> None:
    if not args.event_log_dir.exists():
        raise SystemExit(
            f"Event logs not found under {args.event_log_dir}. "
            "Run: python scripts/build_event_logs.py"
        )

    print("Running full-log baseline (F(true)) ...")
    run_one_query(BASELINE_QUERY, args.baseline_dir, event_log_dir=args.event_log_dir)

    for name, query in SCENARIO_QUERIES.items():
        print(f"Running {name} ...")
        run_one_query(query, args.sow_dir, event_log_dir=args.event_log_dir)


def load_metrics(path: Path, label: str) -> RunMetrics:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run with --run or execute scripts/run_social_mining.py first."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    return RunMetrics(label=label, query=data.get("query", ""), metrics_path=path, data=data)


def model_fields(data: dict[str, Any]) -> dict[str, Any]:
    model = data.get("model") or {}
    return {
        "integrated_log_kb": data.get("size_integrated_log_kb"),
        "integrated_traces": data.get("integrated_traces"),
        "integrated_events": data.get("integrated_events"),
        "contributor_count": data.get("contributor_count"),
        "activities": model.get("activities"),
        "arcs": model.get("arcs"),
        "sum_arc_weights": model.get("sum_arc_weights"),
        "discovery_time_s": data.get("discovery_time_s"),
    }


def pct(part: float | None, whole: float | None) -> float | None:
    if part is None or whole is None or whole == 0:
        return None
    return part / whole


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def fmt_num(value: Any, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def quality_metrics(query_dir: Path) -> dict[str, Any] | None:
    xes = query_dir / "integrated_log.xes"
    pnml = query_dir / "model.pnml"
    if not xes.exists() or not pnml.exists():
        return None

    log = pm4py.read_xes(str(xes))
    if not isinstance(log, pd.DataFrame):
        log = pm4py.convert_to_dataframe(log)

    net, im, fm = pm4py.read_pnml(str(pnml))
    try:
        fitness = pm4py.fitness_alignments(log, net, im, fm)
        precision = pm4py.precision_alignments(log, net, im, fm)
    except Exception as exc:  # alignment can fail on dense models
        return {"error": str(exc)}

    def _scalar(result: Any) -> float | None:
        if isinstance(result, dict):
            for key in ("log_fitness", "average_trace_fitness", "precision"):
                if key in result:
                    return float(result[key])
            if len(result) == 1:
                return float(next(iter(result.values())))
        if isinstance(result, (int, float)):
            return float(result)
        return None

    return {
        "fitness": _scalar(fitness),
        "precision": _scalar(precision),
        "raw_fitness": fitness if isinstance(fitness, dict) else str(fitness),
        "raw_precision": precision if isinstance(precision, dict) else str(precision),
    }


def print_table(rows: list[dict[str, Any]], baseline_kb: float) -> None:
    paper_base = PAPER_BASELINE["integrated_log_kb"]
    header = (
        f"{'Scenario':<28} {'IntKB':>7} {'%Yours':>7} {'Paper%':>7} "
        f"{'PaperKB':>7} {'Trc':>4}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        yours_pct = row.get("pct_your_baseline")
        paper_pct = row.get("paper_pct_baseline")
        print(
            f"{row['label']:<28} "
            f"{fmt_num(row['integrated_log_kb']):>7} "
            f"{fmt_pct(yours_pct):>7} "
            f"{fmt_pct(paper_pct):>7} "
            f"{fmt_num(row['paper_integrated_log_kb'], 0):>7} "
            f"{fmt_num(row['integrated_traces'], 0):>4}"
        )
    print()
    print(f"Your baseline integrated log: {baseline_kb:.1f} KB")
    print(f"Paper baseline integrated log:  {paper_base:.1f} KB")
    print(
        "Compare %Yours vs Paper% (same unit). Absolute IntKB differs because "
        "your baseline XES is ~1.7× larger (namespaced case ids, pm4py encoding)."
    )


def main() -> None:
    args = parse_args()

    if args.run:
        ensure_runs(args)

    baseline = load_metrics(
        metrics_path(args.baseline_dir, BASELINE_QUERY),
        "baseline_full (F(true))",
    )
    baseline_fields = model_fields(baseline.data)
    baseline_kb = float(baseline_fields["integrated_log_kb"] or 0)

    rows: list[dict[str, Any]] = []
    comparison: dict[str, Any] = {
        "your_baseline": {
            **baseline_fields,
            "metrics_path": str(baseline.metrics_path),
            "query": baseline.query,
        },
        "paper_baseline": PAPER_BASELINE,
        "scenarios": [],
    }

    if args.with_quality:
        baseline_dir = args.baseline_dir / query_slug(BASELINE_QUERY)
        comparison["your_baseline"]["quality"] = quality_metrics(baseline_dir)

    for scenario_name, query_text in SCENARIO_QUERIES.items():
        path = metrics_path(args.sow_dir, query_text)
        run = load_metrics(path, scenario_name)
        fields = model_fields(run.data)
        paper = PAPER_FPM[scenario_name]

        row = {
            "label": scenario_name,
            "query": run.query,
            **fields,
            "pct_your_baseline": pct(fields["integrated_log_kb"], baseline_kb),
            "pct_paper_baseline": pct(
                fields["integrated_log_kb"], PAPER_BASELINE["integrated_log_kb"]
            ),
            "paper_integrated_log_kb": paper["integrated_log_kb"],
            "paper_pct_baseline": pct(
                paper["integrated_log_kb"], PAPER_BASELINE["integrated_log_kb"]
            ),
            "paper_activities": paper["activities"],
            "paper_arcs": paper["arcs"],
            "paper_sum_arc_weights": paper["sum_arc_weights"],
            "metrics_path": str(run.metrics_path),
        }

        if args.with_quality:
            row["quality"] = quality_metrics(args.sow_dir / query_slug(query_text))

        rows.append(row)
        comparison["scenarios"].append(row)

    print("SOWCompact comparison (your pipeline vs baseline vs paper Section 7)\n")
    print_table(rows, baseline_kb)

    print("Paper FPM integrated-log sizes (KB): "
          + ", ".join(str(PAPER_FPM[s]["integrated_log_kb"]) for s in SCENARIO_QUERIES))
    print("Paper claim: federated integrated logs ~23% of full-log size on average.\n")

    if args.with_quality:
        print("Quality (alignments):")
        for row in rows:
            q = row.get("quality") or {}
            if "error" in q:
                print(f"  {row['label']}: error — {q['error'][:60]}")
            else:
                print(
                    f"  {row['label']}: fitness={fmt_num(q.get('fitness'), 3)} "
                    f"precision={fmt_num(q.get('precision'), 3)}"
                )
        bq = comparison["your_baseline"].get("quality") or {}
        if bq:
            print(
                f"  baseline: fitness={fmt_num(bq.get('fitness'), 3)} "
                f"precision={fmt_num(bq.get('precision'), 3)}"
            )
        print()

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
