"""Focused tests for the compact rewrite prediction pipeline."""

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

from rewrite.compare_prediction import (  # noqa: E402
    ACTIVITY_TO_ID,
    SoftmaxModel,
    Trace,
    aggregate_models,
    build_prefix_frame,
    run_comparison,
    temporal_split,
)


def _trace(day: int, activities: tuple[str, ...]) -> Trace:
    return Trace(
        subject_id=1,
        case_id=f"subject1:day{day}",
        start_time=pd.Timestamp(f"2020-01-{day:02d} 08:00:00"),
        activities=activities,
    )


def _write_activity_csv(root: Path, subject_id: int, days: list[list[str]]) -> None:
    rows = []
    row_id = 1
    for day_index, activities in enumerate(days, start=1):
        for position, activity in enumerate(activities):
            rows.append(
                {
                    "id": row_id,
                    "dayID": day_index,
                    "subjectID": subject_id,
                    "attr_starttime": f"01.01.20 08:{position:02d}:00",
                    "attr_endtime": f"{day_index:02d}.01.20 08:{position:02d}:30",
                    "label_activity": activity,
                    "label_subactivity": "",
                }
            )
            row_id += 1

    out_dir = root / f"subject{subject_id}" / "data"
    out_dir.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(out_dir / "activity.csv", index=False)


class CompactRewriteTests(unittest.TestCase):
    def test_temporal_split_holds_out_newest_traces_without_overlap(self) -> None:
        traces = [_trace(day, ("DeskWork", "Relaxing")) for day in range(1, 6)]

        train, val = temporal_split(traces, val_fraction=0.4)

        self.assertEqual(
            [trace.case_id for trace in train],
            [f"subject1:day{day}" for day in (1, 2, 3)],
        )
        self.assertEqual(
            [trace.case_id for trace in val],
            [f"subject1:day{day}" for day in (4, 5)],
        )
        self.assertFalse(
            {trace.case_id for trace in train} & {trace.case_id for trace in val}
        )

    def test_prefix_generation_uses_shared_vocabulary_ids(self) -> None:
        frame = build_prefix_frame(
            [_trace(1, ("DeskWork", "Relaxing", "Sleeping"))],
            window=3,
        )

        self.assertEqual(len(frame), 2)
        self.assertEqual(frame.loc[0, "e0"], ACTIVITY_TO_ID["<PAD>"])
        self.assertEqual(frame.loc[0, "e1"], ACTIVITY_TO_ID["<PAD>"])
        self.assertEqual(frame.loc[0, "e2"], ACTIVITY_TO_ID["DeskWork"])
        self.assertEqual(frame.loc[0, "next_activity"], ACTIVITY_TO_ID["Relaxing"])
        self.assertEqual(frame.loc[1, "e1"], ACTIVITY_TO_ID["DeskWork"])
        self.assertEqual(frame.loc[1, "e2"], ACTIVITY_TO_ID["Relaxing"])
        self.assertEqual(frame.loc[1, "next_activity"], ACTIVITY_TO_ID["Sleeping"])

    def test_fedavg_aggregation_weights_by_local_sample_count(self) -> None:
        left = SoftmaxModel.initialize(vocab_size=4, window=3)
        right = SoftmaxModel.initialize(vocab_size=4, window=3)
        left.weights.fill(1.0)
        left.intercept.fill(1.0)
        right.weights.fill(3.0)
        right.intercept.fill(3.0)

        averaged = aggregate_models([(1, left), (3, right)])

        np.testing.assert_allclose(averaged.weights, np.full_like(averaged.weights, 2.5))
        np.testing.assert_allclose(
            averaged.intercept,
            np.full_like(averaged.intercept, 2.5),
        )

    def test_run_comparison_writes_expected_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "dataset"
            output_dir = root / "out"
            _write_activity_csv(
                dataset_root,
                1,
                [
                    ["DeskWork", "Relaxing", "Sleeping"],
                    ["DeskWork", "Relaxing", "Sleeping"],
                    ["Shopping", "Mealpreparation", "Eating/Drinking"],
                ],
            )
            _write_activity_csv(
                dataset_root,
                2,
                [
                    ["Movement", "Transportation", "DeskWork"],
                    ["Movement", "Transportation", "DeskWork"],
                    ["Socializing", "Eating/Drinking", "Relaxing"],
                ],
            )

            payload = run_comparison(
                dataset_root=dataset_root,
                output_dir=output_dir,
                subject_ids=(1, 2),
                rounds=2,
                local_epochs=1,
                seed=7,
            )

            variants = {row["variant"] for row in payload["results"]}
            self.assertEqual(variants, {"centralized", "federated_fedavg"})
            for row in payload["results"]:
                self.assertEqual(
                    set(row),
                    {
                        "variant",
                        "accuracy",
                        "macro_f1",
                        "top3_accuracy",
                        "n_train",
                        "n_val",
                    },
                )
                self.assertGreater(row["n_train"], 0)
                self.assertGreater(row["n_val"], 0)

            comparison_csv = output_dir / "comparison.csv"
            comparison_json = output_dir / "comparison.json"
            self.assertTrue(comparison_csv.exists())
            self.assertTrue(comparison_json.exists())
            saved = json.loads(comparison_json.read_text(encoding="utf-8"))
            self.assertEqual(saved["results"], payload["results"])
            self.assertEqual(saved["fedavg_contributions"], {"subject1": 4, "subject2": 4})


if __name__ == "__main__":
    unittest.main()
