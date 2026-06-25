"""Tests for FedAvg logistic regression next-activity prediction."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fpm.client import PhoneClient  # noqa: E402
from fpm.predict import (  # noqa: E402
    LOGREG_MODEL,
    LogisticRegressionModel,
    accuracy,
    average_fedavg_models,
    initialize_fedavg_model,
)
from fpm.prefix import Vocabulary  # noqa: E402
from fpm.server import create_phone_app  # noqa: E402


def _vocab() -> Vocabulary:
    return Vocabulary(activities=["<PAD>", "A", "B", "C"])


def _sample_frame() -> pd.DataFrame:
    rows = []
    for index in range(12):
        if index % 2 == 0:
            e2 = 1
            target = 2
        else:
            e2 = 2
            target = 3
        rows.append(
            {
                "case_id": f"day{index}",
                "position": index % 3,
                "e0": 0,
                "e1": 0,
                "e2": e2,
                "next_activity": target,
            }
        )
    return pd.DataFrame(rows)


class FakePhone:
    def __init__(self, subject_id: int) -> None:
        self.subject_id = subject_id
        self.subject_label = f"subject{subject_id}"

    def trace_sequences(self) -> list:
        return []

    def activities_in_log(self) -> set:
        return set()


class LogisticRegressionTests(unittest.TestCase):
    def test_probability_rows_sum_and_pad_is_zero(self) -> None:
        frame = _sample_frame()
        model = LogisticRegressionModel(epochs=5, learning_rate=0.2, l2=0.0, seed=1)
        model.fit(frame, _vocab())

        proba = model.predict_proba(frame[["e0", "e1", "e2", "position"]])
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(frame)))
        self.assertTrue(np.all(proba[:, 0] == 0.0))

    def test_json_roundtrip_preserves_predictions(self) -> None:
        frame = _sample_frame()
        model = LogisticRegressionModel(epochs=10, learning_rate=0.3, l2=0.0, seed=2)
        model.fit(frame, _vocab())

        payload = json.loads(json.dumps(model.to_dict()))
        restored = LogisticRegressionModel.from_dict(payload)

        X = frame[["e0", "e1", "e2", "position"]]
        np.testing.assert_allclose(restored.predict_proba(X), model.predict_proba(X))
        np.testing.assert_array_equal(restored.predict(X), model.predict(X))

    def test_empty_training_data_initializes_valid_model(self) -> None:
        frame = _sample_frame().iloc[0:0]
        model = LogisticRegressionModel()
        model.fit(frame, _vocab())

        X = pd.DataFrame([{"e0": 0, "e1": 0, "e2": 1, "position": 0}])
        proba = model.predict_proba(X)
        self.assertEqual(proba.shape, (1, _vocab().size))
        self.assertEqual(float(proba[0, 0]), 0.0)
        self.assertAlmostEqual(float(proba.sum()), 1.0)

    def test_training_improves_over_zero_weight_initial_state(self) -> None:
        frame = _sample_frame()
        vocab = _vocab()
        X = frame[["e0", "e1", "e2", "position"]]
        y = frame["next_activity"].to_numpy(dtype=int)

        baseline = LogisticRegressionModel(epochs=0)
        baseline.fit(frame, vocab)
        baseline_acc = accuracy(y, baseline.predict(X))

        trained = LogisticRegressionModel(
            epochs=80,
            learning_rate=0.5,
            batch_size=4,
            l2=0.0,
            seed=3,
        )
        trained.fit(frame, vocab)
        trained_acc = accuracy(y, trained.predict(X))

        self.assertGreater(trained_acc, baseline_acc)

    def test_weighted_fedavg_uses_train_counts(self) -> None:
        frame = _sample_frame()
        vocab = _vocab()
        base = initialize_fedavg_model(LOGREG_MODEL, frame, vocab)

        left = LogisticRegressionModel.from_dict(base.to_dict())
        right = LogisticRegressionModel.from_dict(base.to_dict())
        left.weights.fill(1.0)
        left.intercept.fill(1.0)
        right.weights.fill(3.0)
        right.intercept.fill(3.0)

        averaged = average_fedavg_models(
            LOGREG_MODEL,
            [(1, left.to_dict()), (3, right.to_dict())],
            fallback_state=base.to_dict(),
        )

        np.testing.assert_allclose(averaged.weights, np.full_like(averaged.weights, 2.5))
        np.testing.assert_allclose(
            averaged.intercept,
            np.full_like(averaged.intercept, 2.5),
        )

    def test_in_process_phone_fedavg_update(self) -> None:
        frame = _sample_frame()
        vocab = _vocab()
        with tempfile.TemporaryDirectory() as tmp:
            prefix_dir = Path(tmp) / "prefix"
            clients: list[PhoneClient] = []
            for subject_id in (1, 2):
                subject_dir = prefix_dir / f"subject{subject_id}"
                subject_dir.mkdir(parents=True)
                frame.iloc[subject_id - 1 :: 2].to_csv(
                    subject_dir / "train.csv",
                    index=False,
                )
                frame.iloc[0:0].to_csv(subject_dir / "val.csv", index=False)
                vocab.write_json(subject_dir / "vocab.json")

                app = create_phone_app(FakePhone(subject_id), prefix_dir=prefix_dir)
                clients.append(PhoneClient.from_app(app))

            global_model = initialize_fedavg_model(LOGREG_MODEL, frame, vocab)
            updates = []
            for index, client in enumerate(clients):
                result = client.fedavg_update(
                    LOGREG_MODEL,
                    state=global_model.to_dict(),
                    round_index=0,
                    local_epochs=1,
                    learning_rate=0.2,
                    batch_size=4,
                    l2=0.0,
                    seed=index,
                )
                self.assertIsNone(result.error)
                self.assertGreater(result.n_train, 0)
                updates.append((result.n_train, result.params))

            averaged = average_fedavg_models(
                LOGREG_MODEL,
                updates,
                fallback_state=global_model.to_dict(),
            )
            proba = averaged.predict_proba(frame[["e0", "e1", "e2", "position"]])
            self.assertEqual(proba.shape, (len(frame), vocab.size))


if __name__ == "__main__":
    unittest.main()
