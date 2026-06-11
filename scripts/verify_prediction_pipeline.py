#!/usr/bin/env python3
"""Verify Phase 3 prediction artifacts, parity, and comparison completeness."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fpm.loader import SUBJECT_IDS  # noqa: E402
from fpm.predict import DEFAULT_FEDERATED_MODEL_DIR, DEFAULT_MODEL_DIR, FEDERATED_MODELS  # noqa: E402
from fpm.prefix import DEFAULT_PREFIX_DIR  # noqa: E402

DEFAULT_GROUP_PREFIX_DIR = ROOT / "output" / "prefix" / "group"
DEFAULT_GROUP_MODEL_DIR = ROOT / "output" / "models" / "group"
DEFAULT_LOCAL_BASELINES = ("frequency", "markov", "markov_order3", "tree")
DEFAULT_FEDERATED_MODELS = tuple(FEDERATED_MODELS)
DEBUG_LOG_PATH = ROOT / ".cursor" / "debug-1f86c4.log"
DEBUG_SESSION_ID = "1f86c4"
GROUP_SCENARIO_VARIANTS_BASE = (
    "group_centralized",
    "group_federated",
    "global_centralized",
    "global_federated",
    "local_pooled",
)
GROUP_SCENARIO_VARIANTS_WITH_LOCAL_GROUP = (*GROUP_SCENARIO_VARIANTS_BASE, "local_group_pooled")
FEDERATED_GLOBAL_VARIANTS = ("centralized", "federated")
FEDERATED_SUBJECT_VARIANTS = ("local", "federated")
METRICS_SKIP_STEMS = frozenset({"metrics", "predictions"})


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
        for result in self.results:
            status = "PASS" if result.passed else "FAIL"
            line = f"  [{status}] {result.name:<{width}}"
            if result.detail:
                line += f"  —  {result.detail}"
            print(line)
        print()
        ok = sum(1 for r in self.results if r.passed)
        print(f"Result: {ok}/{len(self.results)} checks passed")
        if not self.passed:
            print("FAILED — see [FAIL] lines above.")


def _debug_log(*, hypothesis_id: str, location: str, message: str, data: dict) -> None:
    # region agent log
    payload = {
        "sessionId": DEBUG_SESSION_ID,
        "runId": "verify",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with DEBUG_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
    except OSError:
        pass
    # endregion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test Phase 3 prediction outputs: prefix datasets, metrics, "
            "federated parity, group membership, and comparison completeness."
        )
    )
    parser.add_argument(
        "--prefix-dir",
        type=Path,
        default=DEFAULT_PREFIX_DIR,
        help="Directory containing per-subject/global prefix datasets",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory containing local model artifacts",
    )
    parser.add_argument(
        "--federated-dir",
        type=Path,
        default=DEFAULT_FEDERATED_MODEL_DIR,
        help="Directory containing global federated model artifacts",
    )
    parser.add_argument(
        "--group-prefix-dir",
        type=Path,
        default=DEFAULT_GROUP_PREFIX_DIR,
        help="Directory containing group prefix datasets",
    )
    parser.add_argument(
        "--group-models-dir",
        type=Path,
        default=DEFAULT_GROUP_MODEL_DIR,
        help="Directory containing group model artifacts",
    )
    parser.add_argument(
        "--skip-group",
        action="store_true",
        help="Skip group membership and group comparison checks",
    )
    return parser.parse_args()


def expected_scopes() -> list[str]:
    return [f"subject{sid}" for sid in SUBJECT_IDS] + ["global"]


def discover_scope_baselines(scope_dir: Path) -> list[str]:
    """Baseline names from persisted model JSON files in a scope directory."""
    if not scope_dir.exists():
        return []
    return sorted(
        path.stem
        for path in scope_dir.glob("*.json")
        if path.stem not in METRICS_SKIP_STEMS
    )


def load_local_metric_baselines(models_dir: Path, scope: str) -> set[str]:
    metrics_path = models_dir / scope / "metrics.json"
    if not metrics_path.exists():
        return set()
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    return set(payload.get("baselines", {}))


def group_scenario_variants(frame: pd.DataFrame) -> tuple[str, ...]:
    variants = set(frame["variant"].astype(str))
    if "local_group" in variants or "local_group_pooled" in variants:
        return GROUP_SCENARIO_VARIANTS_WITH_LOCAL_GROUP
    return GROUP_SCENARIO_VARIANTS_BASE


def check_prefix_artifacts(report: Report, prefix_dir: Path) -> None:
    required = ("train.csv", "val.csv", "vocab.json", "prefix_manifest.json")
    for scope in expected_scopes():
        scope_dir = prefix_dir / scope
        missing = [name for name in required if not (scope_dir / name).exists()]
        if missing:
            report.fail(f"prefix artifacts ({scope})", f"missing {', '.join(missing)}")
        else:
            report.ok(f"prefix artifacts ({scope})")


def check_local_metrics(report: Report, models_dir: Path) -> None:
    for scope in expected_scopes():
        scope_dir = models_dir / scope
        metrics_path = scope_dir / "metrics.json"
        baselines_on_disk = discover_scope_baselines(scope_dir)

        if not metrics_path.exists():
            report.fail(f"local metrics ({scope})", "metrics.json missing")
            continue
        if not baselines_on_disk:
            report.fail(f"local model artifacts ({scope})", "no baseline JSON files")
            continue

        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        baselines_in_metrics = set(payload.get("baselines", {}))
        missing_artifacts = sorted(baselines_in_metrics - set(baselines_on_disk))
        stale_artifacts = sorted(set(baselines_on_disk) - baselines_in_metrics)
        _debug_log(
            hypothesis_id="H1",
            location="verify_prediction_pipeline.py:check_local_metrics",
            message="local baseline coverage",
            data={
                "scope": scope,
                "on_disk": baselines_on_disk,
                "in_metrics": sorted(baselines_in_metrics),
                "missing_artifacts": missing_artifacts,
                "stale_artifacts": stale_artifacts,
            },
        )
        if not baselines_in_metrics:
            report.fail(f"local metrics ({scope})", "metrics.json has no baselines")
        elif missing_artifacts:
            report.fail(
                f"local metrics ({scope})",
                f"missing artifacts for: {', '.join(missing_artifacts)}",
            )
        elif stale_artifacts:
            report.ok(
                f"local metrics ({scope})",
                f"stale artifacts without metrics: {', '.join(stale_artifacts)}",
            )
        else:
            report.ok(f"local metrics ({scope})")

        for baseline in baselines_on_disk:
            report.ok(f"local model artifact ({scope}/{baseline})")

    comparison_path = models_dir / "comparison.csv"
    if comparison_path.exists():
        report.ok("local comparison.csv")
    else:
        report.fail("local comparison.csv", str(comparison_path))


def check_federated_parity(report: Report, federated_dir: Path) -> None:
    parity_path = federated_dir / "parity.json"
    if not parity_path.exists():
        report.fail("federated parity.json", "file missing")
        return

    parity = json.loads(parity_path.read_text(encoding="utf-8"))
    for model_name in sorted(parity):
        checks = parity[model_name]
        params_ok = bool(checks.get("params_equal"))
        metrics_ok = bool(checks.get("metrics_equal"))
        if params_ok and metrics_ok:
            report.ok(f"federated parity ({model_name})")
        else:
            report.fail(
                f"federated parity ({model_name})",
                f"params_equal={params_ok}, metrics_equal={metrics_ok}",
            )

        model_path = federated_dir / f"{model_name}.json"
        if model_path.exists():
            report.ok(f"federated model artifact ({model_name})")
        else:
            report.fail(f"federated model artifact ({model_name})", str(model_path))


def expected_group_rows(
    membership: dict,
    *,
    prefix_dir: Path,
    split_key: str,
) -> int:
    total = 0
    for subject_label, payload in membership.get("subjects", {}).items():
        case_ids = payload.get(f"{split_key}_case_ids", [])
        if not case_ids:
            continue
        subject_dir = prefix_dir / subject_label
        split_csv = subject_dir / f"{split_key}.csv"
        if not split_csv.exists():
            raise FileNotFoundError(f"Missing prefix split file: {split_csv}")
        subject_frame = pd.read_csv(split_csv)
        allowed = set(case_ids)
        total += int(subject_frame["case_id"].astype(str).isin(allowed).sum())
    return total


def check_group_membership(
    report: Report,
    *,
    group_prefix_dir: Path,
    prefix_dir: Path,
) -> None:
    if not group_prefix_dir.exists():
        report.fail("group prefix root", f"missing {group_prefix_dir}")
        return

    scenarios = sorted(path.name for path in group_prefix_dir.iterdir() if path.is_dir())
    if not scenarios:
        report.fail("group scenarios", "no scenario directories found")
        return

    for scenario in scenarios:
        scenario_dir = group_prefix_dir / scenario
        membership_path = scenario_dir / "membership.json"
        train_path = scenario_dir / "train.csv"
        val_path = scenario_dir / "val.csv"

        if not membership_path.exists():
            report.fail(f"group membership ({scenario})", "membership.json missing")
            continue

        membership = json.loads(membership_path.read_text(encoding="utf-8"))
        for split_key, csv_path in (("train", train_path), ("val", val_path)):
            if not csv_path.exists():
                report.fail(f"group {split_key}.csv ({scenario})", "file missing")
                continue

            expected_rows = expected_group_rows(
                membership,
                prefix_dir=prefix_dir,
                split_key=split_key,
            )
            actual_rows = len(pd.read_csv(csv_path))
            if expected_rows == actual_rows:
                report.ok(
                    f"group membership rows ({scenario}/{split_key})",
                    str(actual_rows),
                )
            else:
                report.fail(
                    f"group membership rows ({scenario}/{split_key})",
                    f"expected {expected_rows}, got {actual_rows}",
                )


def _comparison_has_rows(
    frame: pd.DataFrame,
    *,
    scope: str,
    model: str,
    variants: tuple[str, ...],
) -> bool:
    subset = frame[
        (frame["scope"] == scope)
        & (frame["model"] == model)
        & (frame["variant"].isin(variants))
    ]
    return len(subset) == len(variants)


def check_federated_comparison(report: Report, federated_dir: Path, models_dir: Path) -> None:
    comparison_path = federated_dir / "comparison.csv"
    if not comparison_path.exists():
        report.fail("federated comparison.csv", "file missing")
        return

    frame = pd.read_csv(comparison_path)
    required_cols = {"scope", "model", "variant", "accuracy", "macro_f1"}
    missing_cols = required_cols - set(frame.columns)
    if missing_cols:
        report.fail("federated comparison columns", f"missing {sorted(missing_cols)}")
        return
    report.ok("federated comparison columns")

    models = sorted(frame["model"].astype(str).unique())
    for model_name in models:
        if _comparison_has_rows(
            frame,
            scope="global",
            model=model_name,
            variants=FEDERATED_GLOBAL_VARIANTS,
        ):
            report.ok(f"federated comparison global ({model_name})")
        else:
            report.fail(
                f"federated comparison global ({model_name})",
                f"expected variants {FEDERATED_GLOBAL_VARIANTS}",
            )

        for subject_label in [f"subject{sid}" for sid in SUBJECT_IDS]:
            local_baselines = load_local_metric_baselines(models_dir, subject_label)
            require_local = model_name in local_baselines
            required_variants = ("federated",) if not require_local else FEDERATED_SUBJECT_VARIANTS
            _debug_log(
                hypothesis_id="H2",
                location="verify_prediction_pipeline.py:check_federated_comparison",
                message="subject federated comparison expectations",
                data={
                    "subject": subject_label,
                    "model": model_name,
                    "require_local": require_local,
                    "required_variants": required_variants,
                },
            )
            if _comparison_has_rows(
                frame,
                scope=subject_label,
                model=model_name,
                variants=required_variants,
            ):
                detail = ""
                if not require_local:
                    detail = "local skipped (not in metrics.json)"
                report.ok(
                    f"federated comparison {subject_label} ({model_name})",
                    detail,
                )
            else:
                report.fail(
                    f"federated comparison {subject_label} ({model_name})",
                    f"expected variants {required_variants}",
                )


def check_group_comparison(report: Report, group_models_dir: Path) -> None:
    if not group_models_dir.exists():
        report.fail("group models root", f"missing {group_models_dir}")
        return

    scenarios = sorted(path.name for path in group_models_dir.iterdir() if path.is_dir())
    if not scenarios:
        report.fail("group model scenarios", "no scenario directories found")
        return

    for scenario in scenarios:
        scenario_dir = group_models_dir / scenario
        comparison_path = scenario_dir / "comparison.csv"
        parity_path = scenario_dir / "parity.json"

        if not comparison_path.exists():
            report.fail(f"group comparison.csv ({scenario})", "file missing")
            continue

        frame = pd.read_csv(comparison_path)
        scenario_variants = group_scenario_variants(frame)
        models_in_comparison = sorted(frame["model"].astype(str).unique())
        federated_models = [name for name in models_in_comparison if name in FEDERATED_MODELS]
        has_local_group = "local_group" in set(frame["variant"].astype(str))

        _debug_log(
            hypothesis_id="H3",
            location="verify_prediction_pipeline.py:check_group_comparison",
            message="group comparison expectations",
            data={
                "scenario": scenario,
                "models": federated_models,
                "scenario_variants": scenario_variants,
                "has_local_group": has_local_group,
            },
        )

        for model_name in federated_models:
            if _comparison_has_rows(
                frame,
                scope=scenario,
                model=model_name,
                variants=scenario_variants,
            ):
                report.ok(f"group comparison scenario ({scenario}/{model_name})")
            else:
                report.fail(
                    f"group comparison scenario ({scenario}/{model_name})",
                    f"expected variants {scenario_variants}",
                )

            if has_local_group:
                if ((frame["model"] == model_name) & (frame["variant"] == "local_group")).any():
                    report.ok(f"group comparison local_group rows ({scenario}/{model_name})")
                else:
                    report.fail(
                        f"group comparison local_group rows ({scenario}/{model_name})",
                        "no local_group rows",
                    )

        if parity_path.exists():
            parity = json.loads(parity_path.read_text(encoding="utf-8"))
            _debug_log(
                hypothesis_id="H4",
                location="verify_prediction_pipeline.py:check_group_comparison",
                message="group parity models",
                data={"scenario": scenario, "parity_models": sorted(parity)},
            )
            bad = [
                model_name
                for model_name, checks in parity.items()
                if not checks.get("params_equal")
                or not checks.get("metrics_equal")
            ]
            if bad:
                report.fail(
                    f"group parity ({scenario})",
                    f"failed for {', '.join(bad)}",
                )
            else:
                report.ok(f"group parity ({scenario})")
        else:
            report.fail(f"group parity ({scenario})", "parity.json missing")


def main() -> None:
    args = parse_args()
    report = Report()

    print("Checking prefix datasets ...")
    check_prefix_artifacts(report, args.prefix_dir)

    print("Checking local model artifacts ...")
    check_local_metrics(report, args.models_dir)

    print("Checking global federated parity ...")
    check_federated_parity(report, args.federated_dir)
    check_federated_comparison(report, args.federated_dir, args.models_dir)

    if args.skip_group:
        print("(Skipping group checks — use without --skip-group for full run)\n")
    else:
        print("Checking group membership consistency ...")
        check_group_membership(
            report,
            group_prefix_dir=args.group_prefix_dir,
            prefix_dir=args.prefix_dir,
        )
        print("Checking group comparison completeness ...")
        check_group_comparison(report, args.group_models_dir)

    print()
    report.print_summary()
    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
