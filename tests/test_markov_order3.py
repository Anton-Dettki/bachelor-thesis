"""Tests for order-1 and order-3 Markov baselines and federated merge parity."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fpm.predict import (  # noqa: E402
    MarkovBaseline,
    MarkovOrder1Baseline,
    MarkovOrder3Baseline,
    fit_model,
    merge_params,
    params_equal,
)
from fpm.prefix import Vocabulary  # noqa: E402


def _vocab(size: int = 5) -> Vocabulary:
    names = ["<PAD>"] + [f"A{i}" for i in range(1, size)]
    return Vocabulary(activities=names)


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": ["day1", "day1", "day2", "day2"],
            "position": [1, 2, 1, 2],
            "e0": [0, 1, 0, 2],
            "e1": [0, 0, 0, 1],
            "e2": [1, 2, 2, 3],
            "next_activity": [2, 3, 3, 4],
        }
    )


class MarkovOrder3Tests(unittest.TestCase):
    def test_order1_uses_last_prefix_column_only(self) -> None:
        frame = _sample_frame()
        vocab = _vocab()
        model = MarkovOrder1Baseline()
        model.fit(frame, vocab)

        self.assertEqual(model.context_cols, ["e2"])
        self.assertEqual(set(model.transitions.keys()), {"1", "2", "3"})

    def test_order3_uses_full_prefix_window(self) -> None:
        frame = _sample_frame()
        vocab = _vocab()
        model = MarkovOrder3Baseline()
        model.fit(frame, vocab)

        self.assertEqual(model.context_cols, ["e0", "e1", "e2"])
        self.assertIn("0,0,1", model.transitions)
        self.assertIn("1,0,2", model.transitions)
        self.assertNotIn("1", model.transitions)

    def test_order3_pad_context_falls_back_to_marginal(self) -> None:
        frame = _sample_frame()
        vocab = _vocab()
        model = MarkovOrder3Baseline()
        model.fit(frame, vocab)

        row = pd.DataFrame([{"e0": 0, "e1": 0, "e2": 0}])
        proba = model.predict_proba(row)[0]
        marginal = model._smoothed_marginal()
        self.assertTrue((proba == marginal).all())

    def test_json_roundtrip_order3(self) -> None:
        frame = _sample_frame()
        vocab = _vocab()
        model = MarkovOrder3Baseline()
        model.fit(frame, vocab)

        payload = model.to_dict()
        self.assertEqual(payload["type"], "markov_order3")
        self.assertEqual(payload["order"], 3)

        restored = MarkovBaseline.from_dict(payload)
        self.assertEqual(restored.order, 3)
        self.assertEqual(restored.transitions, model.transitions)
        self.assertEqual(restored.context_cols, model.context_cols)

        serialized = json.dumps(payload)
        roundtrip = MarkovBaseline.from_dict(json.loads(serialized))
        self.assertEqual(roundtrip.transitions, model.transitions)

    def test_legacy_order1_json_loads(self) -> None:
        legacy = {
            "type": "markov",
            "order": 1,
            "alpha": 1.0,
            "transitions": {"2": {"3": 4}},
            "marginal": {"3": 4},
            "context_col": "e2",
            "vocab_size": 5,
        }
        model = MarkovBaseline.from_dict(legacy)
        self.assertEqual(model.order, 1)
        self.assertEqual(model.context_cols, ["e2"])
        self.assertEqual(model.transitions, {"2": {3: 4}})

    def test_federated_merge_parity_order3(self) -> None:
        frame = _sample_frame()
        vocab = _vocab()

        part_a = fit_model("markov_order3", frame.iloc[:2], vocab).to_dict()
        part_b = fit_model("markov_order3", frame.iloc[2:], vocab).to_dict()

        federated = merge_params("markov_order3", [part_a, part_b])
        centralized = fit_model("markov_order3", frame, vocab)

        self.assertTrue(
            params_equal(federated.to_dict(), centralized.to_dict()),
            msg="Federated order-3 counts should equal centralized counts",
        )


if __name__ == "__main__":
    unittest.main()
