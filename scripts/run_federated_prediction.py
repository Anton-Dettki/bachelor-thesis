#!/usr/bin/env python3
"""Global federated next-activity prediction over HTTP (additive Markov/Frequency)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fpm.client import (  # noqa: E402
    PhoneClient,
    RemoteFedAvgUpdateResult,
    RemotePredictParamsResult,
)
from fpm.loader import SUBJECT_IDS  # noqa: E402
from fpm.phone import Phone  # noqa: E402
from fpm.prefix import DEFAULT_PREFIX_DIR  # noqa: E402
from fpm.predict import (  # noqa: E402
    DEFAULT_FEDERATED_MODEL_DIR,
    DEFAULT_MODEL_DIR,
    DEFAULT_LOGREG_BATCH_SIZE,
    DEFAULT_LOGREG_L2,
    DEFAULT_LOGREG_LEARNING_RATE,
    DEFAULT_LOGREG_SEED,
    average_fedavg_models,
    evaluate_predictor,
    fit_model,
    federated_model_names,
    initialize_fedavg_model,
    is_additive_model,
    is_fedavg_model,
    load_scope,
    merge_params,
    params_equal,
    write_json,
)
from fpm.server import create_phone_app  # noqa: E402


def default_phone_urls(subject_ids: list[int] | None = None) -> list[str]:
    ids = subject_ids if subject_ids is not None else list(SUBJECT_IDS)
    return [f"http://127.0.0.1:{8000 + sid}" for sid in ids]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run global federated next-activity prediction: collect additive "
            "Markov/Frequency params or iterative FedAvg updates from phones, "
            "evaluate, and compare local vs centralized vs federated."
        )
    )
    parser.add_argument(
        "--prefix-dir",
        type=Path,
        default=DEFAULT_PREFIX_DIR,
        help="Directory containing prefix datasets from build_prefix_datasets.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_FEDERATED_MODEL_DIR,
        help="Directory for federated model artifacts",
    )
    parser.add_argument(
        "--local-models-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory with per-subject local metrics from train_local_models.py",
    )
    parser.add_argument(
        "--phones",
        nargs="+",
        default=None,
        help="Phone base URLs (default: in-process ASGI clients for all subjects)",
    )
    parser.add_argument(
        "--subject",
        type=int,
        choices=SUBJECT_IDS,
        default=None,
        help="Use only this subject's phone (must be running when using --phones)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="markov,markov_order3,frequency,logreg",
        help=(
            "Comma-separated federated models "
            "(default: markov,markov_order3,frequency,logreg)"
        ),
    )
    parser.add_argument("--rounds", type=int, default=50, help="FedAvg rounds")
    parser.add_argument(
        "--local-epochs",
        type=int,
        default=1,
        help="Local epochs per FedAvg round",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LOGREG_LEARNING_RATE,
        help="FedAvg logistic regression learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_LOGREG_BATCH_SIZE,
        help="FedAvg logistic regression local batch size",
    )
    parser.add_argument(
        "--l2",
        type=float,
        default=DEFAULT_LOGREG_L2,
        help="FedAvg logistic regression L2 penalty",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_LOGREG_SEED,
        help="FedAvg logistic regression random seed",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout per phone (seconds)",
    )
    return parser.parse_args()


def parse_models(raw: str) -> list[str]:
    names = [name.strip() for name in raw.split(",") if name.strip()]
    if not names:
        raise ValueError("At least one model must be specified")
    unknown = [name for name in names if name not in federated_model_names()]
    if unknown:
        raise ValueError(f"Unknown federated models: {', '.join(unknown)}")
    return names


def build_clients(
    args: argparse.Namespace,
) -> list[tuple[str, PhoneClient]]:
    subject_ids = [args.subject] if args.subject is not None else list(SUBJECT_IDS)

    if args.phones:
        urls = args.phones
        return [(url, PhoneClient(url, timeout=args.timeout)) for url in urls]

    clients: list[tuple[str, PhoneClient]] = []
    for subject_id in subject_ids:
        phone = Phone(subject_id)
        app = create_phone_app(phone, prefix_dir=args.prefix_dir)
        label = phone.subject_label
        clients.append((label, PhoneClient.from_app(app)))
    return clients


def collect_params(
    clients: list[tuple[str, PhoneClient]],
    model_name: str,
) -> tuple[list[dict], list[RemotePredictParamsResult]]:
    parts: list[dict] = []
    results: list[RemotePredictParamsResult] = []
    for _label, client in clients:
        result = client.predict_params(model_name)
        results.append(result)
        if result.error:
            raise RuntimeError(
                f"{result.subject_label}: failed to fetch {model_name} params: "
                f"{result.error}"
            )
        parts.append(result.params)
    return parts, results


def collect_fedavg_updates(
    clients: list[tuple[str, PhoneClient]],
    model_name: str,
    state: dict,
    args: argparse.Namespace,
    *,
    round_index: int,
    query: str | None = None,
) -> tuple[list[tuple[int, dict]], list[RemoteFedAvgUpdateResult]]:
    updates: list[tuple[int, dict]] = []
    results: list[RemoteFedAvgUpdateResult] = []
    for _label, client in clients:
        result = client.fedavg_update(
            model_name,
            state=state,
            round_index=round_index,
            query=query,
            local_epochs=args.local_epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            l2=args.l2,
            seed=args.seed,
        )
        results.append(result)
        if result.error:
            raise RuntimeError(
                f"{result.subject_label}: failed to update {model_name}: "
                f"{result.error}"
            )
        if result.n_train > 0:
            updates.append((result.n_train, result.params))
    return updates, results


def metrics_equal(left: dict[str, float], right: dict[str, float]) -> bool:
    keys = set(left) | set(right)
    return all(abs(left.get(k, 0.0) - right.get(k, 0.0)) < 1e-12 for k in keys)


def load_local_metrics(local_models_dir: Path, scope: str, model_name: str) -> dict | None:
    metrics_path = local_models_dir / scope / "metrics.json"
    if not metrics_path.exists():
        return None
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    return payload.get("baselines", {}).get(model_name)


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


def logreg_fit_kwargs(args: argparse.Namespace, *, epochs: int | None = None) -> dict:
    return {
        "epochs": epochs if epochs is not None else args.rounds * args.local_epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "l2": args.l2,
        "seed": args.seed,
    }


def print_summary(rows: list[dict]) -> None:
    header = (
        f"{'Scope':<12} {'Model':<10} {'Variant':<13} "
        f"{'Accuracy':>10} {'Macro-F1':>10} {'Top-3':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        top3 = row.get("top3_accuracy")
        top3_str = f"{top3:>10.4f}" if top3 is not None else f"{'—':>10}"
        print(
            f"{row['scope']:<12} "
            f"{row['model']:<10} "
            f"{row['variant']:<13} "
            f"{row['accuracy']:>10.4f} "
            f"{row['macro_f1']:>10.4f} "
            f"{top3_str}"
        )


def main() -> None:
    args = parse_args()
    if args.rounds < 1:
        raise ValueError("--rounds must be >= 1")
    if args.local_epochs < 1:
        raise ValueError("--local-epochs must be >= 1")
    model_names = parse_models(args.models)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    subject_ids = [args.subject] if args.subject is not None else list(SUBJECT_IDS)
    eval_scopes = [f"subject{sid}" for sid in subject_ids] + (
        [] if args.subject is not None else ["global"]
    )

    clients = build_clients(args)
    mode = "live HTTP" if args.phones else "in-process ASGI"
    print(f"Federated prediction ({mode}) for models: {', '.join(model_names)}")
    print(f"Phones: {', '.join(label for label, _ in clients)}")

    comparison_rows: list[dict] = []
    parity_payload: dict[str, dict[str, Any]] = {}
    federated_metrics: dict[str, dict[str, dict[str, float]]] = {}
    contributions: list[dict] = []
    run_parity = args.subject is None and len(clients) == len(SUBJECT_IDS)

    global_train, global_val, global_vocab = load_scope(args.prefix_dir, "global")

    for model_name in model_names:
        if is_additive_model(model_name):
            print(f"\nCollecting {model_name} params ...")
            parts, fetch_results = collect_params(clients, model_name)
            for result in fetch_results:
                contributions.append(
                    {
                        "subject_label": result.subject_label,
                        "model": model_name,
                        "n_train": result.n_train,
                        "bytes_received": result.bytes_received,
                        "request_time_s": result.request_time_s,
                    }
                )
            federated_model = merge_params(model_name, parts)
            centralized_model = fit_model(model_name, global_train, global_vocab)
            exact_parity_applicable = run_parity
        elif is_fedavg_model(model_name):
            print(f"\nRunning {model_name} FedAvg ({args.rounds} rounds) ...")
            federated_model = initialize_fedavg_model(
                model_name,
                global_train,
                global_vocab,
            )
            for round_index in range(args.rounds):
                updates, update_results = collect_fedavg_updates(
                    clients,
                    model_name,
                    federated_model.to_dict(),
                    args,
                    round_index=round_index,
                )
                for result in update_results:
                    contributions.append(
                        {
                            "subject_label": result.subject_label,
                            "model": model_name,
                            "round_index": result.round_index,
                            "n_train": result.n_train,
                            "bytes_received": result.bytes_received,
                            "request_time_s": result.request_time_s,
                        }
                    )
                federated_model = average_fedavg_models(
                    model_name,
                    updates,
                    fallback_state=federated_model.to_dict(),
                )
            centralized_model = fit_model(
                model_name,
                global_train,
                global_vocab,
                **logreg_fit_kwargs(args),
            )
            exact_parity_applicable = False
        else:
            raise ValueError(f"Unsupported federated model: {model_name}")

        federated_dict = federated_model.to_dict()
        write_json(args.output_dir / f"{model_name}.json", federated_dict)

        centralized_dict = centralized_model.to_dict()

        if exact_parity_applicable:
            parity_payload[model_name] = {
                "exact_parity_applicable": True,
                "params_equal": params_equal(federated_dict, centralized_dict),
            }
        elif is_fedavg_model(model_name):
            parity_payload[model_name] = {
                "exact_parity_applicable": False,
                "params_equal": False,
                "metrics_equal": False,
                "reason": "FedAvg logistic regression is iterative optimization, not an exact additive sufficient-statistic merge.",
            }
        else:
            parity_payload[model_name] = {
                "exact_parity_applicable": False,
                "params_equal": False,
                "skipped": True,
            }

        federated_metrics[model_name] = {}
        centralized_global_metrics = evaluate_predictor(
            centralized_model, global_val, global_vocab
        )
        federated_global_metrics = evaluate_predictor(
            federated_model, global_val, global_vocab
        )
        if exact_parity_applicable:
            parity_payload[model_name]["metrics_equal"] = metrics_equal(
                federated_global_metrics,
                centralized_global_metrics,
            )
        elif not is_fedavg_model(model_name):
            parity_payload[model_name]["metrics_equal"] = False

        if "global" in eval_scopes:
            for variant, metrics in (
                ("centralized", centralized_global_metrics),
                ("federated", federated_global_metrics),
            ):
                comparison_rows.append(
                    comparison_row(
                        scope="global",
                        model=model_name,
                        variant=variant,
                        metrics=metrics,
                    )
                )

        for scope in eval_scopes:
            if scope == "global":
                federated_metrics[model_name][scope] = federated_global_metrics
                continue

            _train_df, val_df, vocab = load_scope(args.prefix_dir, scope)
            scope_metrics = evaluate_predictor(federated_model, val_df, vocab)
            federated_metrics[model_name][scope] = scope_metrics

            local_metrics = load_local_metrics(args.local_models_dir, scope, model_name)
            if local_metrics is not None:
                comparison_rows.append(
                    comparison_row(
                        scope=scope,
                        model=model_name,
                        variant="local",
                        metrics=local_metrics,
                    )
                )

            comparison_rows.append(
                comparison_row(
                    scope=scope,
                    model=model_name,
                    variant="federated",
                    metrics=scope_metrics,
                )
            )

    write_json(
        args.output_dir / "metrics.json",
        {
            "models": model_names,
            "mode": mode,
            "federated": federated_metrics,
            "contributions": contributions,
        },
    )
    write_json(args.output_dir / "parity.json", parity_payload)
    write_json(args.output_dir / "comparison.json", {"results": comparison_rows})
    pd.DataFrame(comparison_rows).to_csv(args.output_dir / "comparison.csv", index=False)

    print()
    print_summary(comparison_rows)
    print()
    if run_parity:
        print("Parity (federated vs centralized global train):")
        for model_name, checks in parity_payload.items():
            if checks.get("exact_parity_applicable", True):
                print(
                    f"  {model_name}: params_equal={checks['params_equal']} "
                    f"metrics_equal={checks['metrics_equal']}"
                )
            else:
                print(f"  {model_name}: skipped ({checks.get('reason', 'not applicable')})")
        if not all(
            (not checks.get("exact_parity_applicable", True))
            or (checks["params_equal"] and checks["metrics_equal"])
            for checks in parity_payload.values()
        ):
            raise SystemExit("Parity check failed: federated != centralized")
    else:
        print("Parity check skipped (not all subjects contributed).")
    print()
    print(f"Wrote {args.output_dir / 'comparison.csv'}")
    print(f"Wrote {args.output_dir / 'parity.json'}")


if __name__ == "__main__":
    main()
