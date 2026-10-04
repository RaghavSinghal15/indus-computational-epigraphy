"""Opening and closing sign counts for sequences already in reading order."""

from collections import Counter

import pandas as pd


def count_endpoints(sequences: list[list[int]]) -> tuple[Counter, Counter]:
    """Return start/end counts; each nonempty sequence contributes once to each."""
    if not sequences:
        raise ValueError("Expected a nonempty corpus")

    starts = Counter()
    ends = Counter()
    for sequence in sequences:
        if not sequence:
            raise ValueError("Expected every sequence to be nonempty")
        starts[sequence[0]] += 1
        ends[sequence[-1]] += 1
    return starts, ends


def endpoint_table(sequences: list[list[int]]) -> pd.DataFrame:
    """Return all observed types, counts and endpoint fractions, sorted by sign ID.

    *_probability uses the number of sequences as its denominator.
    *_share_of_sign uses all occurrences of that particular sign as its denominator.
    A one-sign sequence contributes its single token to both endpoint counts;
    these two shares are separate events, not mutually exclusive categories.
    """
    starts, ends = count_endpoints(sequences)
    totals = Counter()
    for sequence in sequences:
        totals.update(sequence)  # Include repeated signs, including interior tokens.

    n_sequences = len(sequences)
    rows = []
    for sign in sorted(totals):
        rows.append({
            "wells_id": sign,
            "total_count": totals[sign],
            "start_count": starts[sign],
            "end_count": ends[sign],
            "start_probability": starts[sign] / n_sequences,
            "end_probability": ends[sign] / n_sequences,
            "start_share_of_sign": starts[sign] / totals[sign],
            "end_share_of_sign": ends[sign] / totals[sign],
        })
    return pd.DataFrame(rows)
