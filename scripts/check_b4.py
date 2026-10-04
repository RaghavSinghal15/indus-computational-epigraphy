"""Run with .venv/bin/python scripts/check_b4.py from the B4 branch."""

from collections import Counter
import math
import unittest

import numpy as np
import pandas as pd

from indus.data import load_sequences
from indus.ngrams import count_ngrams, transition_matrices
from indus.entropy import entropy_from_counts, row_entropy_table, conditional_entropy, entropy_summary


class B4Checks(unittest.TestCase):
    def test_known_distributions_and_zero_cells(self):
        self.assertAlmostEqual(entropy_from_counts([5, 0]), 0)
        self.assertAlmostEqual(entropy_from_counts([1, 1]), 1)
        self.assertAlmostEqual(entropy_from_counts([1, 1, 1, 1]), 2)
        self.assertAlmostEqual(entropy_from_counts([0, 1, 1, 0]), 1)
        self.assertAlmostEqual(entropy_from_counts([20, 20, 60]), 1.3709505944546687)

    def test_invalid_distribution_inputs(self):
        for values in ([], [0, 0], [-1, 2], [np.nan], [np.inf], [[1, 1]]):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    entropy_from_counts(values)

    def test_occurrence_weighting_not_equal_row_weighting(self):
        counts = pd.DataFrame([[9, 0], [1, 1]], index=[10, 20], columns=[10, 20])
        self.assertAlmostEqual(conditional_entropy(counts), 2 / 11)
        rows = row_entropy_table(counts)
        self.assertAlmostEqual(rows["pair_weight"].sum(), 1)

    def test_undefined_row_has_zero_contribution(self):
        counts = pd.DataFrame([[0, 2], [0, 0]], index=[10, 20], columns=[10, 20])
        row = row_entropy_table(counts).set_index("current_wells_id").loc[20]
        self.assertTrue(np.isnan(row["entropy_bits"]))
        self.assertEqual(row["pair_weight"], 0)
        self.assertEqual(row["weighted_entropy_bits"], 0)
        self.assertFalse(row["entropy_defined"])
        self.assertAlmostEqual(conditional_entropy(counts), 0)

    def test_no_pairs_rejected(self):
        counts, _ = transition_matrices([[7], [9]])
        for function in (row_entropy_table, conditional_entropy, entropy_summary):
            with self.assertRaises(ValueError):
                function(counts)

    def test_invalid_count_matrices_rejected(self):
        for counts in (pd.DataFrame([[-1, 2]]), pd.DataFrame([[np.nan]]),
                       pd.DataFrame([[np.inf]]), pd.DataFrame([[1], [1]], index=[7, 7]),
                       pd.DataFrame([[1, 1]], columns=[7, 7])):
            with self.assertRaises(ValueError):
                row_entropy_table(counts)

    def test_hand_matrix_joint_marginal_and_direction_identities(self):
        counts = pd.DataFrame([[0, 2], [1, 1]], index=[1, 2], columns=[1, 2])
        summary = entropy_summary(counts)
        reverse = entropy_summary(counts.T)
        self.assertAlmostEqual(summary["conditional_entropy_bits"], 0.5)
        self.assertAlmostEqual(summary["current_marginal_entropy_bits"], 1)
        self.assertAlmostEqual(summary["joint_entropy_bits"], 1.5)
        self.assertAlmostEqual(summary["conditional_entropy_bits"],
                               summary["joint_entropy_bits"] - summary["current_marginal_entropy_bits"])
        self.assertAlmostEqual(summary["conditional_entropy_bits"] - reverse["conditional_entropy_bits"],
                               summary["next_marginal_entropy_bits"] - summary["current_marginal_entropy_bits"])
        self.assertAlmostEqual(summary["mutual_information_bits"], reverse["mutual_information_bits"])

    def test_relabel_order_and_scale_invariance(self):
        counts = pd.DataFrame([[0, 2], [1, 1]], index=[1, 2], columns=[1, 2])
        expected = conditional_entropy(counts)
        self.assertAlmostEqual(conditional_entropy(counts.loc[[2, 1], [2, 1]]), expected)
        self.assertAlmostEqual(conditional_entropy(counts * 100), expected)

    def test_double_reversal(self):
        sequences = [[1, 2, 2], [2, 1]]
        reversed_sequences = [s[::-1] for s in sequences]
        counts, _ = transition_matrices(sequences)
        reversed_counts, _ = transition_matrices(reversed_sequences)
        restored_counts, _ = transition_matrices([s[::-1] for s in reversed_sequences])
        np.testing.assert_array_equal(reversed_counts.to_numpy(), counts.to_numpy().T)
        pd.testing.assert_frame_equal(restored_counts, counts)
        self.assertAlmostEqual(conditional_entropy(restored_counts), conditional_entropy(counts))

    def test_real_corpus_independent_entropy_and_boundary_identity(self):
        sequences = load_sequences()
        counts, _ = transition_matrices(sequences)
        reverse_counts, _ = transition_matrices([s[::-1] for s in sequences])
        forward, backward = entropy_summary(counts), entropy_summary(reverse_counts)
        pairs = count_ngrams(sequences)
        outgoing = Counter()
        for (first, second), count in pairs.items():
            outgoing[first] += count
        independent = -sum(count / 6522 * math.log2(count / outgoing[first])
                           for (first, second), count in pairs.items())
        self.assertAlmostEqual(forward["conditional_entropy_bits"], independent, places=12)
        self.assertEqual(forward["pair_occurrences"], 6522)
        self.assertEqual(forward["undefined_rows"], 42)
        self.assertAlmostEqual(forward["conditional_entropy_bits"] - backward["conditional_entropy_bits"],
                               forward["next_marginal_entropy_bits"] - forward["current_marginal_entropy_bits"], places=12)
        self.assertAlmostEqual(forward["joint_entropy_bits"], backward["joint_entropy_bits"], places=12)
        self.assertAlmostEqual(forward["mutual_information_bits"], backward["mutual_information_bits"], places=12)
        tokens = Counter(sign for s in sequences for sign in s)
        starts = Counter(s[0] for s in sequences)
        ends = Counter(s[-1] for s in sequences)
        for sign in counts.index:
            self.assertEqual(int(counts.loc[sign].sum()), tokens[sign] - ends[sign])
            self.assertEqual(int(counts[sign].sum()), tokens[sign] - starts[sign])


if __name__ == "__main__":
    unittest.main(verbosity=2)
