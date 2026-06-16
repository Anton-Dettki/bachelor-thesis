"""Tests for timestamp-derived prefix features and tree consumption."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fpm.loader import ACTIVITY, CASE_ID, TIMESTAMP  # noqa: E402
from fpm.predict import DecisionTreeModel, aux_feature_columns  # noqa: E402
from fpm.prefix import (  # noqa: E402
    TIME_FEATURE_COLUMNS,
    Vocabulary,
    build_prefix_frame,
    encode_frame,
    prefix_manifest,
)


def _tiny_log() -> pd.DataFrame:
    return pd.DataFrame(
        {
            CASE_ID: ["day1", "day1", "day1", "day2", "day2"],
            ACTIVITY: ["A", "B", "C", "D", "E"],
            TIMESTAMP: pd.to_datetime(
                [
                    "2020-01-01 08:00:00",
                    "2020-01-01 08:30:00",
                    "2020-01-01 09:15:00",
                    "2020-01-02 18:00:00",
                    "2020-01-02 19:00:00",
                ],
                utc=True,
            ),
        }
    )


class PrefixTimeFeatureTests(unittest.TestCase):
    def test_build_prefix_frame_emits_timestamp_columns(self) -> None:
        frame = build_prefix_frame(_tiny_log(), window=3, include_time_features=True)

        self.assertEqual(len(frame), 3)
        for col in TIME_FEATURE_COLUMNS:
            self.assertIn(col, frame.columns)

        first = frame.iloc[0]
        self.assertEqual(first["case_id"], "day1")
        self.assertEqual(first["position"], 0)
        self.assertEqual(first["hour"], 8)
        self.assertEqual(first["hour_bin"], 1)
        self.assertEqual(first["day_of_week"], 2)
        self.assertEqual(first["minutes_since_day_start"], 0.0)
        self.assertEqual(first["minutes_since_prev_event"], 0.0)

        second = frame.iloc[1]
        self.assertEqual(second["minutes_since_day_start"], 30.0)
        self.assertEqual(second["minutes_since_prev_event"], 30.0)

    def test_features_use_current_event_timestamp_not_next(self) -> None:
        frame = build_prefix_frame(_tiny_log(), window=3, include_time_features=True)
        row = frame.iloc[0]

        self.assertEqual(row["next_activity"], "B")
        self.assertEqual(row["hour"], 8)
        self.assertEqual(row["minutes_since_day_start"], 0.0)

    def test_encode_frame_preserves_numeric_time_columns(self) -> None:
        vocab = Vocabulary(["<PAD>", "A", "B", "C", "D", "E"])
        raw = build_prefix_frame(_tiny_log(), window=3, include_time_features=True)
        encoded = encode_frame(raw, vocab, window=3, include_time_features=True)

        self.assertEqual(encoded.loc[0, "hour"], 8)
        self.assertEqual(encoded.loc[0, "hour_bin"], 1)
        self.assertEqual(encoded.loc[0, "e2"], vocab.encode("A"))
        self.assertEqual(encoded.loc[0, "next_activity"], vocab.encode("B"))

    def test_prefix_manifest_records_time_features(self) -> None:
        manifest = prefix_manifest(
            scope="subject1",
            window=3,
            train_samples=10,
            val_samples=4,
            n_activities=12,
            time_features=True,
        )

        self.assertTrue(manifest["time_features"])
        self.assertEqual(manifest["time_feature_columns"], TIME_FEATURE_COLUMNS)


class TreeTimeFeatureTests(unittest.TestCase):
    def test_tree_uses_auxiliary_timestamp_columns(self) -> None:
        vocab = Vocabulary(["<PAD>", "A", "B", "C", "D", "E"])
        train_df = pd.DataFrame(
            {
                "case_id": ["day1", "day1", "day2", "day2"],
                "position": [0, 1, 0, 1],
                "e0": [0, 1, 0, 4],
                "e1": [0, 0, 0, 1],
                "e2": [1, 2, 4, 2],
                "next_activity": [2, 3, 2, 3],
                "hour": [8, 9, 18, 19],
                "hour_bin": [1, 1, 3, 3],
                "day_of_week": [2, 2, 3, 3],
                "minutes_since_day_start": [0.0, 30.0, 0.0, 60.0],
                "minutes_since_prev_event": [0.0, 30.0, 0.0, 60.0],
            }
        )
        val_df = train_df.iloc[:1].copy()

        model = DecisionTreeModel()
        model.fit(train_df, vocab)
        payload = model.to_dict()

        self.assertEqual(payload["aux_feature_cols"], aux_feature_columns(train_df))
        self.assertGreater(payload["n_features"], vocab.size * 3)
        self.assertEqual(len(model.predict(val_df)), 1)


if __name__ == "__main__":
    unittest.main()
