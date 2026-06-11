#!/usr/bin/env python3
"""LTL-group next-activity prediction: centralized + federated evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fpm.client import PhoneClient, RemotePredictParamsResult  # noqa: E402
from fpm.event_log import load_event_log  # noqa: E402
from fpm.loader import SUBJECT_IDS  # noqa: E402
from fpm.phone import Phone, select_matching_case_ids  # noqa: E402
from fpm.prefix import DEFAULT_PREFIX_DIR  # noqa: E402
from fpm.predict import (  # noqa: E402
    DEFAULT_FEDERATED_MODEL_DIR,
    DecisionTreeModel,
    FEDERATED_MODELS,
    evaluate_predictor,
    fit_model,
    load_scope,
    merge_params,
    params_equal,
    write_json,
)
from fpm.queries import SCENARIO_QUERIES, query_slug  # noqa: E402
from fpm.server import create_phone_app  # noqa: E402
from fpm.split import DEFAULT_SPLIT_DIR, subject_split_dir  # noqa: E402

DEFAULT_GROUP_PREFIX_DIR = ROOT / "output" / "prefix" / "group"
DEFAULT_GROUP_MODEL_DIR = ROOT / "output" / "models" / "group"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run LTL-group next-activity prediction: train/evaluate group models "
            "(centralized + federated) and compare against local and global variants."
        )
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Scenario key from SCENARIO_QUERIES (e.g. scenario2_no_sport).",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Raw LTL query text (alternative to --scenario).",
    )
    parser.add_argument(
        "--prefix-dir",
        type=Path,
        default=DEFAULT_PREFIX_DIR,
        help="Directory containing per-subject/global prefix datasets",
    )
    parser.add_argument(
        "--group-prefix-dir",
        type=Path,
        default=DEFAULT_GROUP_PREFIX_DIR,
        help="Directory containing group prefix datasets",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for group model artifacts (default: output/models/group/<scenario>/)",
    )
    parser.add_argument(
        "--federated-models-dir",
        type=Path,
        default=DEFAULT_FEDERATED_MODEL_DIR,
        help="Directory with global federated models from run_federated_prediction.py",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=DEFAULT_SPLIT_DIR,
        help="Directory containing train/val splits (for on-device LTL filtering)",
    )
    parser.add_argument(
        "--phones",
        nargs="+",
        default=None,
        help="Phone base URLs (default: in-process ASGI clients for all subjects)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="markov,markov_order3,frequency",
        help="Comma-separated additive models (default: markov,markov_order3,frequency)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout per phone (seconds)",
    )
    return parser.parse_args()


def resolve_scenario(args: argparse.Namespace) -> tuple[str, str]:
    if args.scenario and args.query:
        raise ValueError("Specify only one of --scenario or --query")
    if args.query:
        return query_slug(args.query), args.query
    if args.scenario:
        if args.scenario not in SCENARIO_QUERIES:
            known = ", ".join(sorted(SCENARIO_QUERIES))
            raise ValueError(f"Unknown scenario {args.scenario!r}; choose from {known}")
        return args.scenario, SCENARIO_QUERIES[args.scenario]
    raise ValueError("One of --scenario or --query is required")


def parse_models(raw: str) -> list[str]:
    names = [name.strip() for name in raw.split(",") if name.strip()]
    if not names:
        raise ValueError("At least one model must be specified")
    unknown = [name for name in names if name not in FEDERATED_MODELS]
    if unknown:
        raise ValueError(f"Unknown federated models: {', '.join(unknown)}")
    return names


def build_clients(
    args: argparse.Namespace,
) -> list[tuple[str, PhoneClient]]:
    if args.phones:
        return [(url, PhoneClient(url, timeout=args.timeout)) for url in args.phones]

    clients: list[tuple[str, PhoneClient]] = []
    for subject_id in SUBJECT_IDS:
        phone = Phone(subject_id)
        app = create_phone_app(
            phone,
            prefix_dir=args.prefix_dir,
            split_dir=args.split_dir,
        )
        clients.append((phone.subject_label, PhoneClient.from_app(app)))
    return clients


def collect_params(
    clients: list[tuple[str, PhoneClient]],
    model_name: str,
    query: str,
) -> tuple[list[dict], list[RemotePredictParamsResult]]:
    parts: list[dict] = []
    results: list[RemotePredictParamsResult] = []
    for _label, client in clients:
        result = client.predict_params(model_name, query=query)
        results.append(result)
        if result.error:
            raise RuntimeError(
                f"{result.subject_label}: failed to fetch {model_name} params: "
                f"{result.error}"
            )
        parts.append(result.params)
    return parts, results


def metrics_equal(left: dict[str, float], right: dict[str, float]) -> bool:
    keys = set(left) | set(right)
    return all(abs(left.get(k, 0.0) - right.get(k, 0.0)) < 1e-12 for k in keys)


def comparison_row(
    *,
    scope: str,
    model: str,
    variant: str,
    metrics: dict[str, float],
) -> dict:
    row = {
        "scope": scope,
        "model": model,
        "variant": variant,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
    }
    if "top3_accuracy" in metrics:
        row["top3_accuracy"] = metrics["top3_accuracy"]
    return row


def subject_group_val(val_df: pd.DataFrame, subject_label: str) -> pd.DataFrame:
    prefix = f"{subject_label}:"
    mask = val_df["case_id"].astype(str).str.startswith(prefix)
    return val_df[mask].copy()


def filter_train_by_query(
    train_df: pd.DataFrame,
    *,
    subject_id: int,
    split_dir: Path,
    query: str,
) -> pd.DataFrame:
    """Keep only train prefix rows whose case id matches the LTL query on train.xes."""
    train_log = load_event_log(subject_split_dir(split_dir, subject_id) / "train.xes")
    matching = select_matching_case_ids(train_log, query)
    if not matching:
        return train_df.iloc[0:0].copy()
    allowed = set(matching)
    return train_df[train_df["case_id"].astype(str).isin(allowed)].copy()


def weighted_average_metrics(
    weighted: list[tuple[int, dict[str, float]]],
) -> dict[str, float]:
    total = sum(n for n, _ in weighted if n > 0)
    if total == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "top3_accuracy": 0.0}

    keys = set().union(*(m.keys() for _, m in weighted if _))
    result: dict[str, float] = {}
    for key in keys:
        result[key] = sum(n * m.get(key, 0.0) for n, m in weighted if n > 0) / total
    return result


def load_global_federated(model_name: str, federated_dir: Path):
    path = federated_dir / f"{model_name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Global federated model not found at {path}. "
            "Run scripts/run_federated_prediction.py first."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    cls, _ = FEDERATED_MODELS[model_name]
    return cls.from_dict(data)


def print_summary(rows: list[dict]) -> None:
    header = (
        f"{'Scope':<34} {'Model':<10} {'Variant':<20} "
        f"{'Accuracy':>10} {'Macro-F1':>10} {'Top-3':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        top3 = row.get("top3_accuracy")
        top3_str = f"{top3:>10.4f}" if top3 is not None else f"{'—':>10}"
        print(
            f"{row['scope']:<34} "
            f"{row['model']:<10} "
            f"{row['variant']:<20} "
            f"{row['accuracy']:>10.4f} "
            f"{row['macro_f1']:>10.4f} "
            f"{top3_str}"
        )


def evaluate_local_variants(
    *,
    model_name: str,
    val_df: pd.DataFrame,
    vocab,
    prefix_dir: Path,
    scenario: str,
    fit_fn,
    variant: str = "local",
    pooled_variant: str = "local_pooled",
    train_filter=None,
) -> list[dict]:
    rows: list[dict] = []
    weighted: list[tuple[int, dict[str, float]]] = []

    for subject_id in SUBJECT_IDS:
        subject_label = f"subject{subject_id}"
        subject_val = subject_group_val(val_df, subject_label)
        if subject_val.empty:
            continue

        train_df, _, subject_vocab = load_scope(prefix_dir, subject_label)
        if train_filter is not None:
            train_df = train_filter(train_df, subject_id=subject_id)
        model = fit_fn(model_name, train_df, subject_vocab)
        metrics = evaluate_predictor(model, subject_val, subject_vocab)
        rows.append(
            comparison_row(
                scope=subject_label,
                model=model_name,
                variant=variant,
                metrics=metrics,
            )
        )
        weighted.append((len(subject_val), metrics))

    if weighted:
        pooled = weighted_average_metrics(weighted)
        rows.append(
            comparison_row(
                scope=scenario,
                model=model_name,
                variant=pooled_variant,
                metrics=pooled,
            )
        )
    return rows


def evaluate_local_group_variants(
    *,
    model_name: str,
    val_df: pd.DataFrame,
    vocab,
    prefix_dir: Path,
    split_dir: Path,
    scenario: str,
    query: str,
    fit_fn,
) -> list[dict]:
    def train_filter(train_df: pd.DataFrame, *, subject_id: int) -> pd.DataFrame:
        return filter_train_by_query(
            train_df,
            subject_id=subject_id,
            split_dir=split_dir,
            query=query,
        )

    return evaluate_local_variants(
        model_name=model_name,
        val_df=val_df,
        vocab=vocab,
        prefix_dir=prefix_dir,
        scenario=scenario,
        fit_fn=fit_fn,
        variant="local_group",
        pooled_variant="local_group_pooled",
        train_filter=train_filter,
    )


def main() -> None:
    args = parse_args()
    scenario, query_text = resolve_scenario(args)
    model_names = parse_models(args.models)
    output_dir = args.output_dir or (DEFAULT_GROUP_MODEL_DIR / scenario)
    output_dir.mkdir(parents=True, exist_ok=True)

    group_scope_dir = args.group_prefix_dir / scenario
    if not (group_scope_dir / "train.csv").exists():
        raise FileNotFoundError(
            f"Group prefix dataset not found at {group_scope_dir}. "
            "Run scripts/build_group_prefix_datasets.py first."
        )

    group_train, group_val, group_vocab = load_scope(args.group_prefix_dir, scenario)
    global_train, _global_val, global_vocab = load_scope(args.prefix_dir, "global")

    clients = build_clients(args)
    mode = "live HTTP" if args.phones else "in-process ASGI"
    print(f"Group prediction ({mode}) for {scenario}")
    print(f"Query: {query_text}")
    print(f"Models: {', '.join(model_names)} + tree")
    print(f"Phones: {', '.join(label for label, _ in clients)}")

    comparison_rows: list[dict] = []
    parity_payload: dict[str, dict[str, bool | str]] = {}
    contributions: list[dict] = []
    federated_metrics: dict[str, dict[str, float]] = {}

    for model_name in model_names:
        print(f"\nCollecting {model_name} params for group query ...")
        parts, fetch_results = collect_params(clients, model_name, query_text)
        for result in fetch_results:
            contributions.append(
                {
                    "subject_label": result.subject_label,
                    "model": model_name,
                    "matching_traces": result.matching_traces,
                    "meets_pattern": result.meets_pattern,
                    "n_train": result.n_train,
                    "bytes_received": result.bytes_received,
                    "request_time_s": result.request_time_s,
                }
            )

        federated_model = merge_params(model_name, parts)
        federated_dict = federated_model.to_dict()
        write_json(output_dir / f"{model_name}.json", federated_dict)

        centralized_model = fit_model(model_name, group_train, group_vocab)
        centralized_dict = centralized_model.to_dict()

        group_centralized_metrics = evaluate_predictor(
            centralized_model, group_val, group_vocab
        )
        group_federated_metrics = evaluate_predictor(
            federated_model, group_val, group_vocab
        )
        parity_payload[model_name] = {
            "params_equal": params_equal(federated_dict, centralized_dict),
            "metrics_equal": metrics_equal(
                group_federated_metrics,
                group_centralized_metrics,
            ),
        }
        federated_metrics[model_name] = group_federated_metrics

        global_centralized_model = fit_model(model_name, global_train, global_vocab)
        global_federated_model = load_global_federated(
            model_name, args.federated_models_dir
        )
        global_centralized_metrics = evaluate_predictor(
            global_centralized_model, group_val, group_vocab
        )
        global_federated_metrics = evaluate_predictor(
            global_federated_model, group_val, group_vocab
        )

        for variant, metrics in (
            ("group_centralized", group_centralized_metrics),
            ("group_federated", group_federated_metrics),
            ("global_centralized", global_centralized_metrics),
            ("global_federated", global_federated_metrics),
        ):
            comparison_rows.append(
                comparison_row(
                    scope=scenario,
                    model=model_name,
                    variant=variant,
                    metrics=metrics,
                )
            )

        comparison_rows.extend(
            evaluate_local_variants(
                model_name=model_name,
                val_df=group_val,
                vocab=group_vocab,
                prefix_dir=args.prefix_dir,
                scenario=scenario,
                fit_fn=fit_model,
            )
        )
        comparison_rows.extend(
            evaluate_local_group_variants(
                model_name=model_name,
                val_df=group_val,
                vocab=group_vocab,
                prefix_dir=args.prefix_dir,
                split_dir=args.split_dir,
                scenario=scenario,
                query=query_text,
                fit_fn=fit_model,
            )
        )

    print("\nTraining decision tree (group-centralized only; not federated) ...")
    tree_model = DecisionTreeModel()
    tree_model.fit(group_train, group_vocab)
    write_json(output_dir / "tree.json", tree_model.to_dict())
    tree_group_metrics = evaluate_predictor(tree_model, group_val, group_vocab)
    comparison_rows.append(
        comparison_row(
            scope=scenario,
            model="tree",
            variant="group_centralized",
            metrics=tree_group_metrics,
        )
    )

    global_tree = DecisionTreeModel()
    global_tree.fit(global_train, global_vocab)
    comparison_rows.append(
        comparison_row(
            scope=scenario,
            model="tree",
            variant="global_centralized",
            metrics=evaluate_predictor(global_tree, group_val, group_vocab),
        )
    )

    def fit_tree(_name: str, train_df, vocab):
        model = DecisionTreeModel()
        model.fit(train_df, vocab)
        return model

    comparison_rows.extend(
        evaluate_local_variants(
            model_name="tree",
            val_df=group_val,
            vocab=group_vocab,
            prefix_dir=args.prefix_dir,
            scenario=scenario,
            fit_fn=fit_tree,
        )
    )
    comparison_rows.extend(
        evaluate_local_group_variants(
            model_name="tree",
            val_df=group_val,
            vocab=group_vocab,
            prefix_dir=args.prefix_dir,
            split_dir=args.split_dir,
            scenario=scenario,
            query=query_text,
            fit_fn=fit_tree,
        )
    )

    write_json(
        output_dir / "metrics.json",
        {
            "scenario": scenario,
            "query": query_text,
            "models": model_names,
            "mode": mode,
            "n_group_train": len(group_train),
            "n_group_val": len(group_val),
            "federated": federated_metrics,
            "contributions": contributions,
            "tree_not_federated": (
                "Decision trees are not additive; federated sum-merge does not apply."
            ),
        },
    )
    write_json(output_dir / "parity.json", parity_payload)
    write_json(output_dir / "comparison.json", {"results": comparison_rows})
    pd.DataFrame(comparison_rows).to_csv(output_dir / "comparison.csv", index=False)

    print()
    print_summary(comparison_rows)
    print()
    print("Parity (group federated vs group centralized train):")
    for model_name, checks in parity_payload.items():
        print(
            f"  {model_name}: params_equal={checks['params_equal']} "
            f"metrics_equal={checks['metrics_equal']}"
        )
    if not all(
        checks["params_equal"] and checks["metrics_equal"]
        for checks in parity_payload.values()
    ):
        raise SystemExit("Parity check failed: group federated != group centralized")
    print()
    print(f"Wrote {output_dir / 'comparison.csv'}")
    print(f"Wrote {output_dir / 'parity.json'}")


if __name__ == "__main__":
    main()
