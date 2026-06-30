"""Tests for prediction-level ensemble comparison in federated prediction."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fpm.predict import (  # noqa: E402
    FrequencyBaseline,
    RandomForestModel,
    build_subject_ensemble_models,
    confusion_matrix_counts,
    ensemble_soft_vote,
    evaluate,
    evaluate_ensemble_soft_vote,
    fit_subject_model,
    learning_curve_rows,
    prediction_diagnostics,
    predictor_feature_frame,
    write_confusion_matrix_artifacts,
    write_learning_curve_artifacts,
    write_prediction_diagnostics_artifacts,
)
from fpm.prefix import Vocabulary  # noqa: E402


def _vocab() -> Vocabulary:
    return Vocabulary(activities=["<PAD>", "A", "B", "C"])


def _prefix_rows(
    *,
    case_prefix: str,
    e2_values: list[int],
    targets: list[int],
) -> list[dict]:
    rows: list[dict] = []
    for position, (e2, target) in enumerate(zip(e2_values, targets, strict=True)):
        rows.append(
            {
                "case_id": f"{case_prefix}:day1",
                "position": position,
                "e0": 0,
                "e1": 0,
                "e2": e2,
                "next_activity": target,
            }
        )
    return rows


def _empty_val_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["case_id", "position", "e0", "e1", "e2", "next_activity"]
    )


def _write_scope(
    prefix_dir: Path,
    scope: str,
    train_rows: list[dict],
    val_rows: list[dict],
    vocab: Vocabulary,
) -> None:
    scope_dir = prefix_dir / scope
    scope_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train_rows).to_csv(scope_dir / "train.csv", index=False)
    val_frame = pd.DataFrame(val_rows) if val_rows else _empty_val_frame()
    val_frame.to_csv(scope_dir / "val.csv", index=False)
    vocab.write_json(scope_dir / "vocab.json")


class EnsembleSoftVoteTests(unittest.TestCase):
    def test_averages_aligned_probability_matrices(self) -> None:
        left = np.array([[0.8, 0.2], [0.1, 0.9]], dtype=float)
        right = np.array([[0.2, 0.8], [0.9, 0.1]], dtype=float)

        averaged = ensemble_soft_vote([left, right])

        np.testing.assert_allclose(averaged, np.array([[0.5, 0.5], [0.5, 0.5]]))

    def test_rejects_empty_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least one probability matrix"):
            ensemble_soft_vote([])

    def test_rejects_mismatched_shapes(self) -> None:
        left = np.array([[0.8, 0.2]], dtype=float)
        right = np.array([[0.2, 0.8, 0.0]], dtype=float)
        with self.assertRaisesRegex(ValueError, "expected"):
            ensemble_soft_vote([left, right])


class EnsemblePredictionTests(unittest.TestCase):
    def test_ensemble_matches_manual_probability_mean(self) -> None:
        vocab = _vocab()
        val_rows = _prefix_rows(case_prefix="global", e2_values=[1, 2], targets=[2, 3])
        subject1_train = _prefix_rows(
            case_prefix="subject1",
            e2_values=[1, 1, 1],
            targets=[2, 2, 2],
        )
        subject2_train = _prefix_rows(
            case_prefix="subject2",
            e2_values=[2, 2, 2],
            targets=[3, 3, 3],
        )

        with tempfile.TemporaryDirectory() as tmp:
            prefix_dir = Path(tmp) / "prefix"
            _write_scope(prefix_dir, "subject1", subject1_train, [], vocab)
            _write_scope(prefix_dir, "subject2", subject2_train, [], vocab)
            _write_scope(prefix_dir, "global", subject1_train + subject2_train, val_rows, vocab)

            models = build_subject_ensemble_models(
                "frequency",
                [1, 2],
                prefix_dir,
                vocab,
            )
            val_df = pd.read_csv(prefix_dir / "global" / "val.csv")
            metrics = evaluate_ensemble_soft_vote(models, val_df, vocab)

            manual_parts = []
            for model in models:
                X_val = predictor_feature_frame(model, val_df)
                manual_parts.append(model.predict_proba(X_val))
            manual_proba = ensemble_soft_vote(manual_parts)
            y_true = val_df["next_activity"].to_numpy(dtype=int)
            expected = evaluate(
                y_true,
                manual_proba.argmax(axis=1),
                vocab=vocab,
                y_proba=manual_proba,
            )

            self.assertEqual(metrics["accuracy"], expected["accuracy"])
            self.assertEqual(metrics["macro_f1"], expected["macro_f1"])
            self.assertEqual(metrics["top3_accuracy"], expected["top3_accuracy"])

    def test_fit_subject_model_supports_tree_with_shared_vocab(self) -> None:
        vocab = _vocab()
        train_rows = _prefix_rows(case_prefix="subject1", e2_values=[1, 2], targets=[2, 3])
        train_df = pd.DataFrame(train_rows)

        model = fit_subject_model("tree", train_df, vocab)
        proba = model.predict_proba(predictor_feature_frame(model, train_df))

        self.assertEqual(proba.shape, (len(train_df), vocab.size))
        self.assertTrue(np.all(proba[:, 0] == 0.0))


class EvaluationArtifactTests(unittest.TestCase):
    def test_confusion_matrix_counts_uses_activity_labels_without_pad(self) -> None:
        vocab = _vocab()
        y_true = np.array([1, 1, 2, 3])
        y_pred = np.array([1, 2, 2, 1])

        matrix, class_ids, labels = confusion_matrix_counts(y_true, y_pred, vocab)

        self.assertEqual(class_ids, [1, 2, 3])
        self.assertEqual(labels, ["A", "B", "C"])
        np.testing.assert_array_equal(
            matrix,
            np.array(
                [
                    [1, 1, 0],
                    [0, 1, 0],
                    [1, 0, 0],
                ]
            ),
        )

    def test_confusion_and_learning_curve_artifacts_are_written(self) -> None:
        vocab = _vocab()
        train_df = pd.DataFrame(
            _prefix_rows(case_prefix="subject1", e2_values=[1, 1, 2, 2], targets=[2, 2, 3, 3])
        )
        val_df = pd.DataFrame(
            _prefix_rows(case_prefix="global", e2_values=[1, 2], targets=[2, 3])
        )
        model = FrequencyBaseline()
        model.fit(train_df, vocab)
        y_true = val_df["next_activity"].to_numpy(dtype=int)
        y_pred = model.predict(predictor_feature_frame(model, val_df))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            confusion = write_confusion_matrix_artifacts(
                root / "confusion_matrices",
                "frequency",
                y_true,
                y_pred,
                vocab,
            )
            curve_rows = learning_curve_rows(FrequencyBaseline, train_df, val_df, vocab)
            curve = write_learning_curve_artifacts(
                root / "learning_curves",
                "frequency",
                curve_rows,
            )

            for artifact in [confusion, curve]:
                self.assertTrue(Path(artifact["csv"]).exists())
                self.assertTrue(Path(artifact["png"]).exists())
            self.assertTrue(Path(confusion["normalized_csv"]).exists())
            self.assertTrue(Path(confusion["normalized_png"]).exists())
            self.assertGreater(len(curve_rows), 0)

    def test_learning_curve_uses_whole_cases(self) -> None:
        vocab = _vocab()
        rows = []
        rows.extend(_prefix_rows(case_prefix="case1", e2_values=[1, 1], targets=[2, 2]))
        rows.extend(_prefix_rows(case_prefix="case2", e2_values=[2, 2], targets=[3, 3]))
        train_df = pd.DataFrame(rows)
        val_df = pd.DataFrame(
            _prefix_rows(case_prefix="val", e2_values=[1, 2], targets=[2, 3])
        )

        curve_rows = learning_curve_rows(
            FrequencyBaseline,
            train_df,
            val_df,
            vocab,
            fractions=(0.5, 1.0),
        )

        self.assertEqual(curve_rows[0]["n_train_cases"], 1)
        self.assertEqual(curve_rows[0]["n_train"], 2)
        self.assertEqual(curve_rows[1]["n_train_cases"], 2)
        self.assertEqual(curve_rows[1]["n_train"], 4)

    def test_prediction_diagnostics_are_written(self) -> None:
        vocab = _vocab()
        val_df = pd.DataFrame(
            _prefix_rows(case_prefix="val", e2_values=[1, 2, 2], targets=[2, 3, 3])
        )
        y_true = val_df["next_activity"].to_numpy(dtype=int)
        y_pred = np.array([2, 2, 2])
        diagnostics = prediction_diagnostics(
            object(),
            val_df,
            y_true,
            y_pred,
            vocab,
        )

        self.assertEqual(diagnostics["majority_predicted_label"], "B")
        self.assertEqual(diagnostics["nonzero_predicted_classes"], 1)

        with tempfile.TemporaryDirectory() as tmp:
            artifact = write_prediction_diagnostics_artifacts(
                Path(tmp),
                "dummy",
                diagnostics,
            )
            self.assertTrue(Path(artifact["json"]).exists())
            self.assertTrue(Path(artifact["csv"]).exists())

    def test_random_forest_probabilities_use_full_vocab_without_pad(self) -> None:
        vocab = _vocab()
        train_df = pd.DataFrame(
            _prefix_rows(case_prefix="subject1", e2_values=[1, 1, 2, 2], targets=[2, 2, 3, 3])
        )
        model = RandomForestModel(n_estimators=10, random_state=0, n_jobs=1)
        model.fit(train_df, vocab)

        proba = model.predict_proba(predictor_feature_frame(model, train_df))

        self.assertEqual(proba.shape, (len(train_df), vocab.size))
        self.assertTrue(np.all(proba[:, 0] == 0.0))


if __name__ == "__main__":
    unittest.main()
