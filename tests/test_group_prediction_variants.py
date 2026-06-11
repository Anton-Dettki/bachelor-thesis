"""Tests for local_group variants and Phase 3 prediction verification helpers."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_group_prediction import filter_train_by_query  # noqa: E402
from scripts.verify_prediction_pipeline import (  # noqa: E402
    expected_group_rows,
    _comparison_has_rows,
)


class LocalGroupVariantTests(unittest.TestCase):
    def test_filter_train_by_query_keeps_matching_case_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split_dir = root / "splits" / "subject1"
            split_dir.mkdir(parents=True)

            train_log = pd.DataFrame(
                {
                    "case:concept:name": ["day1", "day1", "day2", "day2"],
                    "concept:name": ["A", "B", "C", "D"],
                    "time:timestamp": pd.to_datetime(
                        [
                            "2020-01-01 08:00:00",
                            "2020-01-01 09:00:00",
                            "2020-01-02 08:00:00",
                            "2020-01-02 09:00:00",
                        ],
                        utc=True,
                    ),
                }
            )
            import pm4py

            pm4py.write_xes(train_log, str(split_dir / "train.xes"))

            train_df = pd.DataFrame(
                {
                    "case_id": ["day1", "day1", "day2"],
                    "position": [0, 1, 0],
                    "e0": [0, 1, 0],
                    "e1": [0, 0, 0],
                    "e2": [1, 2, 3],
                    "next_activity": [2, 3, 4],
                }
            )

            filtered = filter_train_by_query(
                train_df,
                subject_id=1,
                split_dir=root / "splits",
                query="F(C)",
            )
            self.assertEqual(filtered["case_id"].tolist(), ["day2"])

    def test_expected_group_rows_matches_membership(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prefix_dir = root / "prefix"
            subject_dir = prefix_dir / "subject1"
            subject_dir.mkdir(parents=True)

            pd.DataFrame(
                {
                    "case_id": ["day1", "day1", "day2", "day3"],
                    "position": [0, 1, 0, 0],
                    "next_activity": [1, 2, 3, 4],
                }
            ).to_csv(subject_dir / "train.csv", index=False)
            pd.DataFrame(
                {
                    "case_id": ["day4"],
                    "position": [0],
                    "next_activity": [1],
                }
            ).to_csv(subject_dir / "val.csv", index=False)

            membership = {
                "subjects": {
                    "subject1": {
                        "train_case_ids": ["day1", "day2"],
                        "val_case_ids": ["day4"],
                    }
                }
            }

            train_rows = expected_group_rows(
                membership,
                prefix_dir=prefix_dir,
                split_key="train",
            )
            val_rows = expected_group_rows(
                membership,
                prefix_dir=prefix_dir,
                split_key="val",
            )
            self.assertEqual(train_rows, 3)
            self.assertEqual(val_rows, 1)

    def test_comparison_has_rows(self) -> None:
        frame = pd.DataFrame(
            {
                "scope": ["scenario2", "scenario2", "subject1"],
                "model": ["markov", "markov", "markov"],
                "variant": ["local_group", "local_group_pooled", "local_group"],
            }
        )
        self.assertTrue(
            _comparison_has_rows(
                frame,
                scope="scenario2",
                model="markov",
                variants=("local_group", "local_group_pooled"),
            )
        )
        self.assertFalse(
            _comparison_has_rows(
                frame,
                scope="subject1",
                model="markov",
                variants=("local_group", "local_group_pooled"),
            )
        )


if __name__ == "__main__":
    unittest.main()
