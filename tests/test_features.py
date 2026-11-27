import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.sparse import isspmatrix_csr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from features import SequenceEncoder, check_disjoint
from fitness_model import FitnessPredictor


class FeatureTests(unittest.TestCase):
    def test_fixed_feature_shape_and_dtype(self):
        encoder = SequenceEncoder("A" * 238)
        plain = encoder.transform("C" + "A" * 237)
        extended = encoder.transform("C" + "A" * 237, include_count=True)
        self.assertTrue(isspmatrix_csr(plain))
        self.assertEqual(plain.shape, (1, 4760))
        self.assertEqual(extended.shape, (1, 4761))
        self.assertEqual(plain.dtype, np.float32)
        self.assertEqual(plain.nnz, 238)
        self.assertEqual(extended[0, -1], 1)

    def test_alphabet_and_position_order(self):
        order = "ACDEFGHIKLMNPQRSTVWY"
        encoder = SequenceEncoder("AA")
        one_hot = encoder.transform([f"A{aa}" for aa in order])
        for i in range(20):
            self.assertEqual(list(one_hot[i].indices), [0, 20+i])
            self.assertEqual(list(one_hot[i].data), [1, 1])

    def test_bad_sequence(self):
        encoder = SequenceEncoder("AAA")
        for invalid in ("AA", "AAAA"):
            with self.assertRaisesRegex(ValueError, "length"):
                encoder.transform(invalid)
        for invalid in ("AXA", "A-A", "ACa", "AéA"):
            with self.assertRaisesRegex(ValueError, "amino acid"):
                encoder.transform(invalid)

    def test_counts_and_single_batch_consistency(self):
        encoder = SequenceEncoder("AAAA")
        sequences = ["AAAA", "ACAA", "ACDG"]
        np.testing.assert_array_equal(encoder.mutation_counts(sequences), [0, 1, 3])
        for include_count in (False, True):
            batch = encoder.transform(iter(sequences), include_count)
            for index, sequence in enumerate(sequences):
                self.assertEqual((encoder.transform(sequence, include_count) != batch[index]).nnz, 0)

    def test_labels_and_identifiers_do_not_change_feature_encoding(self):
        a = [{"sequence": "AC", "experiment_variant_id": "old", "fitness": 1, "fitness_bin": 0,
              "mutation_count": 99, "fitness_rank": 7, "split": "random"}]
        b = [{"sequence": "AC", "experiment_variant_id": "new", "fitness": 4, "fitness_bin": 1,
              "mutation_count": 11, "fitness_rank": 1, "split": "best"}]
        for name in ("count_only", "ridge", "random_forest"):
            model = FitnessPredictor(name, "AA")
            left, right = (model.encode([row["sequence"] for row in dataset]) for dataset in (a, b))
            if isspmatrix_csr(left):
                self.assertEqual((left != right).nnz, 0)
            else:
                np.testing.assert_array_equal(left, right)

    def test_sequence_overlap_rejected(self):
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            check_disjoint({"first": [{"sequence": "AA", "experiment_variant_id": "1"}],
                            "second": [{"sequence": "AA", "experiment_variant_id": "2"}]})


if __name__ == "__main__":
    unittest.main()
