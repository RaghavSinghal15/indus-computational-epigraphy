"""Basic descriptive counts for a corpus of sign sequences."""

from collections import Counter
from statistics import median


def count_signs(sequences: list[list[int]]) -> Counter:
    """Return the number of occurrences of each sign ID, including repeats."""
    counts = Counter()
    for sequence in sequences:
        for sign in sequence:
            counts[sign] += 1
    return counts


def count_lengths(sequences: list[list[int]]) -> Counter:
    """Return how many sequences have each length."""
    counts = Counter()
    for sequence in sequences:
        counts[len(sequence)] += 1
    return counts


def summarize_corpus(sequences: list[list[int]]) -> dict:
    """Return corpus size, token/type counts, singleton counts and length statistics."""
    if not sequences or any(len(sequence) == 0 for sequence in sequences):
        raise ValueError("Expected a nonempty corpus of nonempty sequences")

    signs = count_signs(sequences)
    lengths = [len(sequence) for sequence in sequences]
    tokens = sum(lengths)
    singletons = sum(count == 1 for count in signs.values())
    return {
        "n_sequences": len(sequences),
        "n_tokens": tokens,
        "n_types": len(signs),
        "n_singletons": singletons,
        "singleton_type_fraction": singletons / len(signs),
        "singleton_token_fraction": singletons / tokens,
        "min_length": min(lengths),
        "max_length": max(lengths),
        "mean_length": tokens / len(sequences),
        "median_length": median(lengths),
    }
