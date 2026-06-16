#!/usr/bin/env python3
"""Train and evaluate local next-activity baseline predictors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fpm.loader import SUBJECT_IDS  # noqa: E402
from fpm.prefix import DEFAULT_PREFIX_DIR  # noqa: E402
from fpm.predict import (  # noqa: E402
    DEFAULT_MODEL_DIR,
    DecisionTreeModel,
    FrequencyBaseline,
    MarkovOrder1Baseline,
    MarkovOrder3Baseline,
    aux_feature_columns,
    evaluate,
    load_scope,
    prefix_columns,
    split_xy,
    TARGET,
    write_json,
)

BASELINE_REGISTRY = {
    "frequency": FrequencyBaseline,
    "markov": MarkovOrder1Baseline,
    "markov_order3": MarkovOrder3Baseline,
    "tree": DecisionTreeModel,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train local next-activity predictors (frequency, Markov order-1/order-3, "
            "decision tree) on prefix datasets and evaluate on held-out validation data."
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
        default=DEFAULT_MODEL_DIR,
        help="Directory for model artifacts and metrics",
    )
    parser.add_argument(
        "--subject",
        type=int,
        choices=SUBJECT_IDS,
        default=None,
        help="Process only this subject (1-7). Default: all subjects + global.",
    )
    parser.add_argument(
        "--baselines",
        type=str,
        default="frequency,markov,markov_order3,tree",
        help="Comma-separated model names (default: frequency,markov,markov_order3,tree)",
    )
    parser.add_argument(
        "--no-predictions",
        action="store_true",
        help="Skip writing predictions.csv",
    )
    return parser.parse_args()


def parse_baselines(raw: str) -> list[str]:
    names = [name.strip() for name in raw.split(",") if name.strip()]
    if not names:
        raise ValueError("At least one baseline must be specified")
    unknown = [name for name in names if name not in BASELINE_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown baselines: {', '.join(unknown)}")
    return names


def print_summary(rows: list[dict]) -> None:
    header = f"{'Scope':<12} {'Baseline':<12} {'Accuracy':>10} {'Macro-F1':>10} {'Top-3':>10}"
    print(header)
    print("-" * len(header))
    for row in rows:
        top3 = row.get("top3_accuracy")
        top3_str = f"{top3:>10.4f}" if top3 is not None else f"{'—':>10}"
        print(
            f"{row['scope']:<12} "
            f"{row['baseline']:<12} "
            f"{row['accuracy']:>10.4f} "
            f"{row['macro_f1']:>10.4f} "
            f"{top3_str}"
        )


def train_scope(
    scope: str,
    prefix_dir: Path,
    output_dir: Path,
    *,
    baseline_names: list[str],
    write_predictions: bool,
) -> list[dict]:
    train_df, val_df, vocab = load_scope(prefix_dir, scope)
    X_val, y_val = split_xy(val_df)

    scope_out = output_dir / scope
    scope_out.mkdir(parents=True, exist_ok=True)

    metrics_payload: dict = {
        "scope": scope,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "baselines": {},
    }
    prediction_rows: list[dict] = []
    summary_rows: list[dict] = []

    for name in baseline_names:
        model = BASELINE_REGISTRY[name]()
        model.fit(train_df, vocab)

        if name == "tree":
            aux_cols = aux_feature_columns(val_df)
            prefix_cols = prefix_columns(val_df)
            X_val = val_df[[*prefix_cols, *aux_cols]]
            y_val = val_df[TARGET].to_numpy(dtype=int)
        else:
            X_val, y_val = split_xy(val_df)
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)
        baseline_metrics = evaluate(y_val, y_pred, vocab=vocab, y_proba=y_proba)
        metrics_payload["baselines"][name] = baseline_metrics

        model_path = scope_out / f"{name}.json"
        write_json(model_path, model.to_dict())

        summary_row = {
            "scope": scope,
            "baseline": name,
            "accuracy": baseline_metrics["accuracy"],
            "macro_f1": baseline_metrics["macro_f1"],
        }
        if "top3_accuracy" in baseline_metrics:
            summary_row["top3_accuracy"] = baseline_metrics["top3_accuracy"]
        summary_rows.append(summary_row)

        if write_predictions and len(val_df) > 0:
            for idx, (_, row) in enumerate(val_df.iterrows()):
                prediction_rows.append(
                    {
                        "case_id": row["case_id"],
                        "position": int(row["position"]),
                        "baseline": name,
                        "y_true": int(y_val[idx]),
                        "y_pred": int(y_pred[idx]),
                    }
                )

    write_json(scope_out / "metrics.json", metrics_payload)

    if write_predictions and prediction_rows:
        pred_df = pd.DataFrame(prediction_rows)
        pred_df.to_csv(scope_out / "predictions.csv", index=False)

    return summary_rows


def write_comparison(output_dir: Path, rows: list[dict]) -> None:
    """Persist a cross-scope/model metrics table for thesis comparison."""
    comparison: list[dict] = []
    for row in rows:
        entry = {
            "scope": row["scope"],
            "model": row["baseline"],
            "accuracy": row["accuracy"],
            "macro_f1": row["macro_f1"],
        }
        if "top3_accuracy" in row:
            entry["top3_accuracy"] = row["top3_accuracy"]
        comparison.append(entry)

    write_json(output_dir / "comparison.json", {"results": comparison})
    pd.DataFrame(comparison).to_csv(output_dir / "comparison.csv", index=False)


def main() -> None:
    args = parse_args()
    baseline_names = parse_baselines(args.baselines)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scopes = [f"subject{sid}" for sid in ([args.subject] if args.subject else SUBJECT_IDS)]
    if args.subject is None:
        scopes.append("global")

    all_summary: list[dict] = []
    for scope in scopes:
        print(f"Training baselines for {scope} ...")
        rows = train_scope(
            scope,
            args.prefix_dir,
            args.output_dir,
            baseline_names=baseline_names,
            write_predictions=not args.no_predictions,
        )
        all_summary.extend(rows)
        print(f"  Wrote {args.output_dir / scope / 'metrics.json'}")

    write_comparison(args.output_dir, all_summary)
    print(f"Wrote {args.output_dir / 'comparison.csv'}")
    print(f"Wrote {args.output_dir / 'comparison.json'}")

    print()
    print_summary(all_summary)


if __name__ == "__main__":
    main()
