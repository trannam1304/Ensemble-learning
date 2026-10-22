import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import pandas as pd

from src.model import majority_vote
from src.preprocess import prepare_dataset, save_processed_splits


class MajorityVoteTests(unittest.TestCase):
    def test_votes_by_majority(self):
        predictions = [[0, 1, 1], [1, 0, 1], [0, 0, 1]]
        self.assertEqual(majority_vote(predictions).tolist(), [1, 1, 0])

    def test_tie_prefers_lower_label(self):
        self.assertEqual(majority_vote([[0, 1]]).tolist(), [0])

    def test_rejects_invalid_prediction_shape(self):
        with self.assertRaises(ValueError):
            majority_vote([0, 1])


class PreprocessingTests(unittest.TestCase):
    def test_drops_id_and_maps_diagnosis(self):
        frame = pd.DataFrame(
            {
                "ID": [101, 102],
                "radius1": [1.2, 3.4],
                "Diagnosis": ["B", "M"],
            }
        )

        features, labels, sample_ids = prepare_dataset(frame)

        self.assertEqual(list(features.columns), ["radius1"])
        self.assertEqual(labels.tolist(), [0, 1])
        self.assertEqual(sample_ids.tolist(), [101, 102])

    def test_drops_feature_with_no_numeric_values(self):
        frame = pd.DataFrame(
            {
                "ID": [101, 102],
                "radius1": [1.2, 3.4],
                "invalid_feature": ["x", "y"],
                "Diagnosis": ["B", "M"],
            }
        )

        features, _, _ = prepare_dataset(frame)

        self.assertEqual(list(features.columns), ["radius1"])

    def test_saves_encoded_train_and_test_splits(self):
        features = pd.DataFrame({"radius1": [1.2, 3.4, 5.6]})
        labels = pd.Series([0, 1, 0])
        ids = pd.Series([101, 102, 103])

        with TemporaryDirectory() as temp_dir:
            train_path, test_path = save_processed_splits(
                features.iloc[:2],
                features.iloc[2:],
                labels.iloc[:2],
                labels.iloc[2:],
                ids.iloc[:2],
                ids.iloc[2:],
                Path(temp_dir),
            )
            train = pd.read_csv(train_path)
            test = pd.read_csv(test_path)

        self.assertEqual(train.columns.tolist(), ["ID", "radius1", "Diagnosis"])
        self.assertEqual(train["Diagnosis"].tolist(), [0, 1])
        self.assertEqual(test["ID"].tolist(), [103])


if __name__ == "__main__":
    unittest.main()
