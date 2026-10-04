"""Run with .venv/bin/python scripts/check_b3.py; no raw files are changed."""

from collections import Counter
import unittest

import numpy as np
import pandas as pd

from indus.data import load_sequences
from indus.ngrams import count_ngrams, bigram_table, transition_matrices, compare_im77_bigrams


class B3Checks(unittest.TestCase):
    def test_hand_counted_pairs_and_probabilities(self):
        toy = [[151, 105, 151, 740], [151, 740], [151, 151, 740]]
        expected = {(151, 105): 1, (105, 151): 1, (151, 740): 3, (151, 151): 1}
        self.assertEqual(count_ngrams(toy), expected)
        counts, probabilities = transition_matrices(toy)
        self.assertEqual(int(counts.to_numpy().sum()), 6)
        self.assertAlmostEqual(probabilities.loc[151, 740], 0.6)
        self.assertAlmostEqual(probabilities.loc[151, 105], 0.2)
        self.assertAlmostEqual(probabilities.loc[151, 151], 0.2)
        self.assertEqual(probabilities.loc[105, 151], 1)
        self.assertEqual(probabilities.loc[105, 740], 0)
        self.assertTrue(probabilities.loc[740].isna().all())

    def test_boundaries_repeats_and_overlapping_windows(self):
        self.assertEqual(count_ngrams([[1, 2], [3, 4]]), {(1, 2): 1, (3, 4): 1})
        self.assertEqual(count_ngrams([[1, 1, 1]]), {(1, 1): 2})
        self.assertEqual(count_ngrams([[1, 2, 3, 4]], n=3), {(1, 2, 3): 1, (2, 3, 4): 1})
        self.assertEqual(count_ngrams([[1, 2, 1]], n=1), {(1,): 2, (2,): 1})

    def test_no_pairs_and_unobserved_successors(self):
        self.assertEqual(count_ngrams([[7], [9]]), {})
        self.assertEqual(count_ngrams([[1, 2]], n=3), {})
        self.assertTrue(bigram_table([[7]]).empty)
        counts, probabilities = transition_matrices([[7], [9]])
        self.assertEqual(counts.shape, (2, 2))
        self.assertTrue(probabilities.isna().all().all())

    def test_invalid_inputs(self):
        for invalid in ([], [[]], [[1], []]):
            for function in (count_ngrams, bigram_table, transition_matrices):
                with self.subTest(invalid=invalid, function=function.__name__):
                    with self.assertRaises(ValueError):
                        function(invalid)
        for n in (0, -1, True, 2.5, "2"):
            with self.assertRaises(ValueError):
                count_ngrams([[1, 2]], n=n)

    def test_order_ties_and_reversal(self):
        sequences = [[3, 1], [2, 1], [2, 1], [1, 3]]
        self.assertEqual(bigram_table(sequences).values.tolist(), [[2, 1, 2], [1, 3, 1], [3, 1, 1]])
        counts, _ = transition_matrices(sequences)
        reverse_counts, _ = transition_matrices([s[::-1] for s in sequences])
        np.testing.assert_array_equal(reverse_counts.to_numpy(), counts.to_numpy().T)

    def test_successor_denominator_excludes_final_occurrences(self):
        _, probabilities = transition_matrices([[1, 2], [1]])
        self.assertEqual(probabilities.loc[1, 2], 1)  # One outgoing pair, not two sign tokens.

    def test_mapping_direction_pooling_and_coverage(self):
        pairs = Counter({(1, 2): 3, (5, 2): 4, (3, 2): 2, (4, 2): 1, (2, 1): 2})
        mapping = {1: {10}, 2: {20}, 3: {30, 31}, 5: {10}}
        im77 = pd.DataFrame([(20, 10, 8)], columns=["left_sign", "right_sign", "count"])
        compared, coverage = compare_im77_bigrams(pairs, mapping, im77)
        self.assertEqual(compared.iloc[0].tolist(), [10, 20, 7, 8, True])
        self.assertEqual(compared.iloc[1].tolist(), [20, 10, 2, 0, False])
        self.assertEqual(coverage["eligible_pair_tokens"], 9)
        self.assertEqual(coverage["ambiguous_pair_tokens"], 2)
        self.assertEqual(coverage["unmapped_pair_tokens"], 1)
        self.assertEqual(coverage["eligible_pair_tokens_present_in_im77"], 7)
        self.assertEqual(coverage["mapped_mahadevan_pair_types"], 2)

    def test_no_eligible_mappings(self):
        im77 = pd.DataFrame(columns=["left_sign", "right_sign", "count"])
        compared, coverage = compare_im77_bigrams(Counter({(1, 2): 1}), {}, im77)
        self.assertTrue(compared.empty)
        self.assertEqual(coverage["unmapped_pair_tokens"], 1)

    def test_real_corpus_independent_count_and_conservation(self):
        sequences = load_sequences()
        pairs = count_ngrams(sequences)
        independent = {}
        for sequence in sequences:
            for i in range(1, len(sequence)):
                pair = (sequence[i - 1], sequence[i])
                independent[pair] = independent.get(pair, 0) + 1
        self.assertEqual(dict(pairs), independent)
        self.assertEqual(sum(pairs.values()), 6522)
        counts, probabilities = transition_matrices(sequences)
        self.assertEqual(counts.shape, (568, 568))
        self.assertEqual(int(counts.to_numpy().sum()), 6522)
        tokens = Counter(sign for sequence in sequences for sign in sequence)
        starts = Counter(sequence[0] for sequence in sequences)
        ends = Counter(sequence[-1] for sequence in sequences)
        for sign in counts.index:
            self.assertEqual(int(counts.loc[sign].sum()), tokens[sign] - ends[sign])
            self.assertEqual(int(counts[sign].sum()), tokens[sign] - starts[sign])
        defined = counts.sum(axis=1) > 0
        np.testing.assert_allclose(probabilities.loc[defined].sum(axis=1), 1, atol=1e-12)
        self.assertTrue(probabilities.loc[~defined].isna().all().all())
        reverse_counts, _ = transition_matrices([s[::-1] for s in sequences])
        np.testing.assert_array_equal(reverse_counts.to_numpy(), counts.to_numpy().T)


if __name__ == "__main__":
    unittest.main(verbosity=2)
