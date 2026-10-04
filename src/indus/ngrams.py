"""Within-text n-gram counts and unsmoothed reading-order transitions."""

from collections import Counter

import numpy as np
import pandas as pd


def count_ngrams(sequences: list[list[int]], n: int = 2) -> Counter:
    """Return overlapping n-gram counts without joining separate inscriptions."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if not sequences or any(not sequence for sequence in sequences):
        raise ValueError("Expected a nonempty corpus of nonempty sequences")

    counts = Counter()
    for sequence in sequences:
        for start in range(len(sequence) - n + 1):
            window = tuple(sequence[start:start + n])
            counts[window] += 1
    return counts


def bigram_table(sequences: list[list[int]]) -> pd.DataFrame:
    """Return pairs ranked by descending count, with ascending IDs breaking ties."""
    counts = count_ngrams(sequences, n=2)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    rows = [(first, second, count) for (first, second), count in ranked]
    return pd.DataFrame(rows, columns=["current_wells_id", "next_wells_id", "count"])


def transition_matrices(sequences: list[list[int]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return count/probability matrices over all observed signs, sorted by ID.

    Rows are current signs and columns are next signs. Each defined probability
    row sums to one. A sign with no observed successor has an entirely NaN row
    (undefined); it is not assigned an invented distribution. Observed rows use
    zero for unseen pairs. There is no smoothing or boundary token in B3.
    """
    pairs = count_ngrams(sequences, n=2)
    signs = sorted({sign for sequence in sequences for sign in sequence})
    counts = pd.DataFrame(0, index=signs, columns=signs, dtype="int64")
    counts.index.name = "current_wells_id"
    counts.columns.name = "next_wells_id"
    for (first, second), count in pairs.items():
        counts.loc[first, second] = count

    outgoing = counts.sum(axis=1)
    # Replace zero denominators by NaN so undefined rows remain visibly undefined.
    probabilities = counts.div(outgoing.replace(0, np.nan), axis=0)
    return counts, probabilities


def compare_im77_bigrams(pairs: Counter, mapping: dict[int, set[int]],
                        im77_visual: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Compare single-target mappings with IM77, returning a table and coverage.

    Wells pairs are in reading order. IM77 left/right pairs are reversed to
    reading order. Pairs with missing or multi-target mappings are excluded and
    counted explicitly. Many Wells pairs mapping to one Mahadevan pair are pooled.
    ICIT counts include only eligible pairs; they are not full Mahadevan counts.
    Presence in the different IM77 corpus is a pattern check, not count equality.
    """
    mapped = Counter()
    coverage = {"observed_pair_types": len(pairs), "observed_pair_tokens": sum(pairs.values())}
    for status in ("eligible", "ambiguous", "unmapped"):
        coverage[status + "_pair_types"] = 0
        coverage[status + "_pair_tokens"] = 0

    for (first, second), count in pairs.items():
        first_ids, second_ids = mapping.get(first, set()), mapping.get(second, set())
        if not first_ids or not second_ids:
            status = "unmapped"
        elif len(first_ids) != 1 or len(second_ids) != 1:
            status = "ambiguous"
        else:
            status = "eligible"
            mapped[(next(iter(first_ids)), next(iter(second_ids)))] += count
        coverage[status + "_pair_types"] += 1
        coverage[status + "_pair_tokens"] += count

    im77_reading = Counter()
    for row in im77_visual.itertuples(index=False):
        im77_reading[(int(row.right_sign), int(row.left_sign))] += int(row.count)

    rows = []
    for (first, second), count in sorted(mapped.items(), key=lambda item: (-item[1], item[0])):
        rows.append((first, second, count, im77_reading[(first, second)],
                     (first, second) in im77_reading))
    columns = ["current_mahadevan_id", "next_mahadevan_id", "icit_unambiguous_count",
               "im77_count", "in_im77"]
    comparison = pd.DataFrame(rows, columns=columns)
    coverage["mapped_mahadevan_pair_types"] = len(mapped)
    coverage["mapped_types_present_in_im77"] = sum(pair in im77_reading for pair in mapped)
    coverage["eligible_pair_tokens_present_in_im77"] = sum(
        count for pair, count in mapped.items() if pair in im77_reading
    )
    return comparison, coverage
