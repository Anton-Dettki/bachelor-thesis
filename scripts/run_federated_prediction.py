#!/usr/bin/env python3
"""Global federated next-activity prediction over HTTP (additive Markov/Frequency)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fpm.client import PhoneClient, RemotePredictParamsResult  # noqa: E402
from fpm.loader import SUBJECT_IDS  # noqa: E402
from fpm.phone import Phone  # noqa: E402
from fpm.prefix import DEFAULT_PREFIX_DIR  # noqa: E402
from fpm.predict import (  # noqa: E402
    DEFAULT_FEDERATED_MODEL_DIR,
    DEFAULT_MODEL_DIR,
    FEDERATED_MODELS,
    evaluate_predictor,
    fit_model,
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
            "Markov/Frequency params from phones, merge, evaluate, and compare "
            "local vs centralized vs federated."
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


def metrics_equal(left: dict[str, float], right: dict[str, float]) -> bool:
    keys = set(left) | set(right)
    return all(abs(left.get(k, 0.0) - right.get(k, 0.0)) < 1e-12 for k in keys)


def load_local_metrics(local_models_dir: Path, scope: str, model_name: str) -> dict | None:
    metrics_path = local_models_dir / scope / "metrics.json"
    if not metrics_path.exists():
        return None
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    return payload.get("baselines", {}).get(model_name)


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
    parity_payload: dict[str, dict[str, bool]] = {}
    federated_metrics: dict[str, dict[str, dict[str, float]]] = {}
    contributions: list[dict] = []
    run_parity = args.subject is None and len(clients) == len(SUBJECT_IDS)

    global_train, global_val, global_vocab = load_scope(args.prefix_dir, "global")

    for model_name in model_names:
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
        federated_dict = federated_model.to_dict()
        write_json(args.output_dir / f"{model_name}.json", federated_dict)

        centralized_model = fit_model(model_name, global_train, global_vocab)
        centralized_dict = centralized_model.to_dict()

        if run_parity:
            parity_payload[model_name] = {
                "params_equal": params_equal(federated_dict, centralized_dict),
            }
        else:
            parity_payload[model_name] = {
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
        if run_parity:
            parity_payload[model_name]["metrics_equal"] = metrics_equal(
                federated_global_metrics,
                centralized_global_metrics,
            )
        else:
            parity_payload[model_name]["metrics_equal"] = False

        if "global" in eval_scopes:
            for variant, metrics in (
                ("centralized", centralized_global_metrics),
                ("federated", federated_global_metrics),
            ):
                row = {
                    "scope": "global",
                    "model": model_name,
                    "variant": variant,
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                }
                if "top3_accuracy" in metrics:
                    row["top3_accuracy"] = metrics["top3_accuracy"]
                comparison_rows.append(row)

        for scope in eval_scopes:
            if scope == "global":
                federated_metrics[model_name][scope] = federated_global_metrics
                continue

            _train_df, val_df, vocab = load_scope(args.prefix_dir, scope)
            scope_metrics = evaluate_predictor(federated_model, val_df, vocab)
            federated_metrics[model_name][scope] = scope_metrics

            local_metrics = load_local_metrics(args.local_models_dir, scope, model_name)
            if local_metrics is not None:
                row = {
                    "scope": scope,
                    "model": model_name,
                    "variant": "local",
                    "accuracy": local_metrics["accuracy"],
                    "macro_f1": local_metrics["macro_f1"],
                }
                if "top3_accuracy" in local_metrics:
                    row["top3_accuracy"] = local_metrics["top3_accuracy"]
                comparison_rows.append(row)

            row = {
                "scope": scope,
                "model": model_name,
                "variant": "federated",
                "accuracy": scope_metrics["accuracy"],
                "macro_f1": scope_metrics["macro_f1"],
            }
            if "top3_accuracy" in scope_metrics:
                row["top3_accuracy"] = scope_metrics["top3_accuracy"]
            comparison_rows.append(row)

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
            print(
                f"  {model_name}: params_equal={checks['params_equal']} "
                f"metrics_equal={checks['metrics_equal']}"
            )
        if not all(
            checks["params_equal"] and checks["metrics_equal"]
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
