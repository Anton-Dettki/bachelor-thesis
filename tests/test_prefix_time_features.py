"""Tests for timestamp-derived prefix features and tree consumption."""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import pm4py

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fpm.event_log import load_event_log  # noqa: E402
from fpm.loader import (  # noqa: E402
    ACTIVITY,
    CASE_ID,
    START_TIMESTAMP,
    TIMESTAMP,
    load_subject_csv,
)
from fpm.predict import (  # noqa: E402
    DecisionTreeModel,
    LogisticRegressionModel,
    RandomForestModel,
    aux_feature_columns,
)
from fpm.prefix import (  # noqa: E402
    CONTEXT_FEATURE_COLUMNS,
    DURATION_FEATURE_COLUMNS,
    FEATURE_SET_ENHANCED,
    HISTORY_FEATURE_COLUMNS,
    RECENCY_FEATURE_COLUMNS,
    TIME_FEATURE_COLUMNS,
    TRANSITION_FEATURE_COLUMNS,
    Vocabulary,
    build_prefix_frame,
    encode_frame,
    feature_columns_for_set,
    filter_trainable_target_classes,
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


def _tiny_log_with_start() -> pd.DataFrame:
    return pd.DataFrame(
        {
            CASE_ID: ["day1", "day1", "day1"],
            ACTIVITY: ["A", "B", "C"],
            START_TIMESTAMP: pd.to_datetime(
                [
                    "2020-01-01 07:50:00",
                    "2020-01-01 08:25:00",
                    "2020-01-01 09:00:00",
                ],
                utc=True,
            ),
            TIMESTAMP: pd.to_datetime(
                [
                    "2020-01-01 08:00:00",
                    "2020-01-01 08:30:00",
                    "2020-01-01 09:15:00",
                ],
                utc=True,
            ),
        }
    )


def _history_log() -> pd.DataFrame:
    return pd.DataFrame(
        {
            CASE_ID: ["day1"] * 5,
            ACTIVITY: ["A", "B", "A", "A", "C"],
            TIMESTAMP: pd.to_datetime(
                [
                    "2020-01-01 08:00:00",
                    "2020-01-01 08:10:00",
                    "2020-01-01 08:20:00",
                    "2020-01-01 08:30:00",
                    "2020-01-01 08:40:00",
                ],
                utc=True,
            ),
        }
    )


def _namespaced_log() -> pd.DataFrame:
    return pd.DataFrame(
        {
            CASE_ID: ["subject3:day1", "subject3:day1"],
            ACTIVITY: ["A", "B"],
            TIMESTAMP: pd.to_datetime(
                ["2020-01-01 08:00:00", "2020-01-01 08:10:00"],
                utc=True,
            ),
        }
    )


class PrefixTimeFeatureTests(unittest.TestCase):
    def test_build_prefix_frame_emits_timestamp_columns(self) -> None:
        frame = build_prefix_frame(_tiny_log(), window=3)

        self.assertEqual(len(frame), 3)
        for col in TIME_FEATURE_COLUMNS:
            self.assertIn(col, frame.columns)
        for col in feature_columns_for_set(FEATURE_SET_ENHANCED):
            self.assertIn(col, frame.columns)

        first = frame.iloc[0]
        self.assertEqual(first["case_id"], "day1")
        self.assertEqual(first["position"], 0)
        self.assertEqual(first["hour"], 8)
        self.assertEqual(first["hour_bin"], 1)
        self.assertEqual(first["day_of_week"], 2)
        self.assertEqual(first["minutes_since_day_start"], 0.0)
        self.assertEqual(first["minutes_since_prev_event"], 0.0)
        self.assertEqual(first["minutes_since_midnight"], 480.0)
        self.assertEqual(first["log_minutes_since_day_start"], 0.0)
        self.assertAlmostEqual(first["hour_sin"], math.sin(2.0 * math.pi / 3.0))
        self.assertAlmostEqual(first["hour_cos"], math.cos(2.0 * math.pi / 3.0))
        self.assertEqual(first["is_weekend"], 0)
        self.assertEqual(first["month"], 1)
        self.assertEqual(first["day_of_month"], 1)
        self.assertEqual(first["week_of_year"], 1)
        self.assertEqual(first["trace_start_hour"], 8)
        self.assertEqual(first["trace_start_minutes_since_midnight"], 480.0)

        second = frame.iloc[1]
        self.assertEqual(second["minutes_since_day_start"], 30.0)
        self.assertEqual(second["minutes_since_prev_event"], 30.0)
        self.assertAlmostEqual(second["log_minutes_since_day_start"], math.log1p(30.0))
        self.assertAlmostEqual(second["log_minutes_since_prev_event"], math.log1p(30.0))

    def test_features_use_current_event_timestamp_not_next(self) -> None:
        frame = build_prefix_frame(_tiny_log(), window=3)
        row = frame.iloc[0]

        self.assertEqual(row["next_activity"], "B")
        self.assertEqual(row["hour"], 8)
        self.assertEqual(row["minutes_since_day_start"], 0.0)
        self.assertEqual(row["minutes_since_midnight"], 480.0)

    def test_encode_frame_preserves_numeric_time_columns(self) -> None:
        vocab = Vocabulary(["<PAD>", "A", "B", "C", "D", "E"])
        raw = build_prefix_frame(_tiny_log(), window=3)
        encoded = encode_frame(raw, vocab, window=3)

        self.assertEqual(encoded.loc[0, "hour"], 8)
        self.assertEqual(encoded.loc[0, "hour_bin"], 1)
        self.assertEqual(encoded.loc[0, "e2"], vocab.encode("A"))
        self.assertEqual(encoded.loc[0, "next_activity"], vocab.encode("B"))

    def test_filter_trainable_target_classes_drops_sparse_and_unseen_targets(self) -> None:
        train = pd.DataFrame(
            {
                "case_id": ["day1", "day1", "day2"],
                "position": [0, 1, 0],
                "e0": ["<PAD>", "<PAD>", "<PAD>"],
                "e1": ["<PAD>", "A", "<PAD>"],
                "e2": ["A", "Shopping", "B"],
                "next_activity": ["Shopping", "B", "C"],
            }
        )
        val = pd.DataFrame(
            {
                "case_id": ["day3", "day3", "day4", "day4"],
                "position": [0, 1, 0, 1],
                "e0": ["<PAD>", "<PAD>", "<PAD>", "<PAD>"],
                "e1": ["<PAD>", "A", "<PAD>", "D"],
                "e2": ["A", "B", "D", "Sport"],
                "next_activity": ["B", "DeskWork", "Sport", "C"],
            }
        )

        filtered_train, filtered_val, summary = filter_trainable_target_classes(train, val)

        self.assertEqual(filtered_train["next_activity"].tolist(), ["B", "C"])
        self.assertEqual(filtered_val["next_activity"].tolist(), ["B", "C"])
        self.assertEqual(summary["removed_train_excluded"], 1)
        self.assertEqual(summary["removed_val_excluded"], 1)
        self.assertEqual(summary["removed_val_unseen_classes"], ["DeskWork"])

    def test_prefix_manifest_records_automatic_enhanced_features(self) -> None:
        manifest = prefix_manifest(
            scope="subject1",
            window=3,
            train_samples=10,
            val_samples=4,
            n_activities=12,
        )

        self.assertTrue(manifest["time_features"])
        self.assertEqual(manifest["time_feature_columns"], TIME_FEATURE_COLUMNS)
        self.assertEqual(manifest["feature_set"], FEATURE_SET_ENHANCED)
        self.assertEqual(
            manifest["feature_columns"],
            feature_columns_for_set(FEATURE_SET_ENHANCED),
        )

    def test_enhanced_features_include_duration_from_start_timestamp(self) -> None:
        frame = build_prefix_frame(
            _tiny_log_with_start(),
            window=3,
        )

        first = frame.iloc[0]
        self.assertEqual(first["activity_duration_minutes"], 10.0)
        self.assertEqual(first["previous_activity_duration_minutes"], 0.0)
        self.assertEqual(first["log_previous_activity_duration_minutes"], 0.0)
        self.assertEqual(first["gap_since_prev_event_minutes"], 0.0)
        self.assertEqual(first["cumulative_activity_duration_minutes"], 10.0)
        self.assertEqual(first["mean_activity_duration_minutes_so_far"], 10.0)

        second = frame.iloc[1]
        self.assertEqual(second["next_activity"], "C")
        self.assertEqual(second["activity_duration_minutes"], 5.0)
        self.assertAlmostEqual(second["log_activity_duration_minutes"], math.log1p(5.0))
        self.assertEqual(second["previous_activity_duration_minutes"], 10.0)
        self.assertAlmostEqual(
            second["log_previous_activity_duration_minutes"],
            math.log1p(10.0),
        )
        self.assertEqual(second["gap_since_prev_event_minutes"], 25.0)
        self.assertEqual(second["cumulative_activity_duration_minutes"], 15.0)
        self.assertEqual(second["mean_activity_duration_minutes_so_far"], 7.5)

    def test_enhanced_history_counts_use_prefix_only(self) -> None:
        frame = build_prefix_frame(
            _history_log(),
            window=3,
        )

        third = frame.iloc[2]
        self.assertEqual(third["next_activity"], "A")
        self.assertEqual(third["current_activity_count_so_far"], 2)
        self.assertEqual(third["current_activity_seen_before"], 1)
        self.assertAlmostEqual(third["current_activity_frequency_so_far"], 2 / 3)
        self.assertEqual(third["unique_activities_so_far"], 2)
        self.assertAlmostEqual(third["unique_activity_ratio_so_far"], 2 / 3)
        self.assertEqual(third["dominant_activity_count_so_far"], 2)
        self.assertAlmostEqual(third["dominant_activity_ratio_so_far"], 2 / 3)
        self.assertEqual(third["activity_repetition_count_so_far"], 1)
        self.assertEqual(third["events_seen_so_far"], 3)
        self.assertAlmostEqual(third["log_events_seen_so_far"], math.log1p(3))
        self.assertEqual(third["current_activity_run_length"], 1)
        self.assertEqual(third["prefix_length_ratio"], 1.0)
        self.assertEqual(third["events_since_last_same_activity"], 2)
        self.assertEqual(third["minutes_since_last_same_activity"], 20.0)
        self.assertAlmostEqual(
            third["log_minutes_since_last_same_activity"],
            math.log1p(20.0),
        )
        self.assertEqual(third["same_as_previous_activity"], 0)
        self.assertEqual(third["activity_switch_count_so_far"], 2)
        self.assertEqual(third["activity_switch_ratio_so_far"], 1.0)
        self.assertEqual(third["window_unique_activities"], 2)
        self.assertEqual(third["window_switch_count"], 2)
        self.assertEqual(third["window_switch_ratio"], 1.0)
        self.assertEqual(third["window_repetition_count"], 1)
        self.assertAlmostEqual(third["window_repetition_ratio"], 1 / 3)

        fourth = frame.iloc[3]
        self.assertEqual(fourth["next_activity"], "C")
        self.assertEqual(fourth["current_activity_count_so_far"], 3)
        self.assertEqual(fourth["current_activity_run_length"], 2)
        self.assertEqual(fourth["events_seen_so_far"], 4)
        self.assertEqual(fourth["activity_repetition_count_so_far"], 2)
        self.assertEqual(fourth["events_since_last_same_activity"], 1)
        self.assertEqual(fourth["minutes_since_last_same_activity"], 10.0)
        self.assertEqual(fourth["same_as_previous_activity"], 1)
        self.assertEqual(fourth["activity_switch_count_so_far"], 2)
        self.assertAlmostEqual(fourth["activity_switch_ratio_so_far"], 2 / 3)
        self.assertEqual(fourth["window_unique_activities"], 2)
        self.assertEqual(fourth["window_switch_count"], 1)
        self.assertAlmostEqual(fourth["window_switch_ratio"], 1 / 2)
        self.assertEqual(fourth["window_repetition_count"], 1)
        self.assertAlmostEqual(fourth["window_repetition_ratio"], 1 / 3)

    def test_enhanced_context_extracts_subject_id_from_namespaced_case(self) -> None:
        frame = build_prefix_frame(
            _namespaced_log(),
            window=3,
        )

        self.assertEqual(frame.loc[0, "subject_id"], 3)
        self.assertEqual(frame.loc[0, "case_id_number"], 1)
        self.assertAlmostEqual(frame.loc[0, "log_case_id_number"], math.log1p(1))

    def test_enhanced_context_uses_subject_override_for_local_scope(self) -> None:
        frame = build_prefix_frame(
            _tiny_log(),
            window=3,
            subject_id=5,
        )

        self.assertTrue((frame["subject_id"] == 5).all())

    def test_enhanced_empty_frame_includes_all_feature_columns(self) -> None:
        frame = build_prefix_frame(
            pd.DataFrame(columns=[CASE_ID, ACTIVITY, TIMESTAMP]),
            window=3,
        )

        for col in feature_columns_for_set(FEATURE_SET_ENHANCED):
            self.assertIn(col, frame.columns)

    def test_enhanced_xes_without_start_timestamps_uses_zero_durations(self) -> None:
        frame = build_prefix_frame(
            _tiny_log(),
            window=3,
        )

        for col in DURATION_FEATURE_COLUMNS:
            self.assertTrue((frame[col] == 0.0).all())

    def test_prefix_manifest_records_enhanced_feature_groups(self) -> None:
        manifest = prefix_manifest(
            scope="subject1",
            window=3,
            train_samples=10,
            val_samples=4,
            n_activities=12,
            feature_set=FEATURE_SET_ENHANCED,
        )

        self.assertEqual(manifest["feature_set"], FEATURE_SET_ENHANCED)
        self.assertEqual(
            manifest["feature_columns"],
            feature_columns_for_set(FEATURE_SET_ENHANCED),
        )
        self.assertEqual(manifest["temporal_feature_columns"], TIME_FEATURE_COLUMNS)
        self.assertEqual(manifest["duration_feature_columns"], DURATION_FEATURE_COLUMNS)
        self.assertEqual(manifest["history_feature_columns"], HISTORY_FEATURE_COLUMNS)
        self.assertEqual(manifest["recency_feature_columns"], RECENCY_FEATURE_COLUMNS)
        self.assertEqual(
            manifest["transition_feature_columns"],
            TRANSITION_FEATURE_COLUMNS,
        )
        self.assertEqual(manifest["context_feature_columns"], CONTEXT_FEATURE_COLUMNS)
        self.assertTrue(manifest["time_features"])

    def test_csv_loader_preserves_start_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "activity.csv"
            pd.DataFrame(
                [
                    {
                        "id": 1,
                        "dayID": 1,
                        "subjectID": 1,
                        "attr_starttime": "01.01.20 08:00:00",
                        "attr_endtime": "01.01.20 08:10:00",
                        "label_activity": "DeskWork",
                        "label_subactivity": "",
                    }
                ]
            ).to_csv(path, index=False)

            log = load_subject_csv(path)

        self.assertIn(START_TIMESTAMP, log.columns)
        self.assertEqual(log.loc[0, START_TIMESTAMP], pd.Timestamp("2020-01-01 08:00:00"))

    def test_xes_roundtrip_preserves_start_timestamp_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.xes"
            log = _tiny_log_with_start()
            pm4py.write_xes(log, str(path))

            loaded = load_event_log(path)

        self.assertIn(START_TIMESTAMP, loaded.columns)
        self.assertFalse(pd.isna(loaded.loc[0, START_TIMESTAMP]))


class TreeTimeFeatureTests(unittest.TestCase):
    def test_tree_depth_defaults_are_conservative(self) -> None:
        self.assertEqual(DecisionTreeModel().max_depth, 4)
        self.assertEqual(RandomForestModel().max_depth, 5)

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
                "hour_sin": [0.8, 0.7, -1.0, -0.9],
                "hour_cos": [-0.5, -0.7, 0.0, 0.2],
                "day_of_week_sin": [0.9, 0.9, 0.4, 0.4],
                "day_of_week_cos": [-0.2, -0.2, -0.9, -0.9],
                "is_weekend": [0, 0, 0, 0],
                "minutes_since_midnight": [480.0, 540.0, 1080.0, 1140.0],
                "log_minutes_since_prev_event": [0.0, 3.43, 0.0, 4.11],
                "activity_duration_minutes": [10.0, 15.0, 5.0, 10.0],
                "log_activity_duration_minutes": [2.40, 2.77, 1.79, 2.40],
                "previous_activity_duration_minutes": [0.0, 10.0, 0.0, 5.0],
                "log_previous_activity_duration_minutes": [0.0, 2.40, 0.0, 1.79],
                "gap_since_prev_event_minutes": [0.0, 5.0, 0.0, 10.0],
                "cumulative_activity_duration_minutes": [10.0, 25.0, 5.0, 15.0],
                "mean_activity_duration_minutes_so_far": [10.0, 12.5, 5.0, 7.5],
                "current_activity_count_so_far": [1, 1, 1, 1],
                "current_activity_seen_before": [0, 0, 0, 0],
                "unique_activities_so_far": [1, 2, 1, 2],
                "current_activity_run_length": [1, 1, 1, 1],
                "prefix_length_ratio": [1 / 3, 2 / 3, 1 / 3, 2 / 3],
                "events_since_last_same_activity": [0, 0, 0, 0],
                "minutes_since_last_same_activity": [0.0, 0.0, 0.0, 0.0],
                "log_minutes_since_last_same_activity": [0.0, 0.0, 0.0, 0.0],
                "same_as_previous_activity": [0, 0, 0, 0],
                "activity_switch_count_so_far": [0, 1, 0, 1],
                "activity_switch_ratio_so_far": [0.0, 1.0, 0.0, 1.0],
                "window_unique_activities": [1, 2, 1, 2],
                "window_switch_count": [0, 1, 0, 1],
                "subject_id": [1, 1, 2, 2],
            }
        )
        val_df = train_df.iloc[:1].copy()

        model = DecisionTreeModel()
        model.fit(train_df, vocab)
        payload = model.to_dict()

        self.assertEqual(payload["aux_feature_cols"], aux_feature_columns(train_df))
        self.assertEqual(
            payload["n_features"],
            vocab.size * 3 + len(aux_feature_columns(train_df)),
        )
        self.assertEqual(len(model.predict(val_df)), 1)

    def test_logreg_uses_enhanced_auxiliary_columns(self) -> None:
        vocab = Vocabulary(["<PAD>", "A", "B", "C"])
        raw = build_prefix_frame(
            _tiny_log_with_start(),
            window=3,
        )
        encoded = encode_frame(
            raw,
            vocab,
            window=3,
        )

        model = LogisticRegressionModel(epochs=0)
        model.fit(encoded, vocab)

        self.assertEqual(model._aux_feature_cols, aux_feature_columns(encoded))
        self.assertEqual(
            model.weights.shape[0],
            vocab.size * 3 + len(aux_feature_columns(encoded)),
        )


if __name__ == "__main__":
    unittest.main()
