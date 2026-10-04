"""Empirical entropy in bits from unsmoothed adjacent-pair counts."""

import numpy as np
import pandas as pd


def entropy_from_counts(counts) -> float:
    """Return Shannon entropy of nonnegative frequencies; zero mass is undefined."""
    values = np.asarray(counts, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Expected a one-dimensional array of finite nonnegative counts")
    total = values.sum()
    if total <= 0 or not np.isfinite(total):
        raise ValueError("Entropy requires positive finite total mass")
    probabilities = values[values > 0] / total
    # Filter zero counts before logarithms; their 0*log(0) contribution is zero.
    return float(-np.sum(probabilities * np.log2(probabilities)))


def row_entropy_table(count_matrix: pd.DataFrame) -> pd.DataFrame:
    """Return each row's support, pair-occurrence weight, entropy and contribution.

    Rows without successors have NaN entropy and zero weight/contribution.
    A matrix with no pairs has undefined corpus entropy and is rejected.
    """
    if not count_matrix.index.is_unique or not count_matrix.columns.is_unique:
        raise ValueError("Matrix axis labels must be unique")
    values = count_matrix.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Matrix counts must be finite and nonnegative")
    outgoing = values.sum(axis=1)
    total = outgoing.sum()
    if total <= 0 or not np.isfinite(total):
        raise ValueError("Conditional entropy requires at least one observed pair")

    rows = []
    for sign, row, support in zip(count_matrix.index, values, outgoing):
        weight = float(support / total)
        if support > 0:
            entropy = entropy_from_counts(row)
            contribution = weight * entropy
        else:
            entropy = np.nan
            contribution = 0.0
        rows.append({
            "current_wells_id": sign,
            "outgoing_count": float(support),
            "pair_weight": weight,
            "entropy_bits": entropy,
            "weighted_entropy_bits": contribution,
            "entropy_defined": bool(support > 0),
        })
    return pd.DataFrame(rows)


def conditional_entropy(count_matrix: pd.DataFrame) -> float:
    """Return H(next | current), weighting row entropies by outgoing pair counts."""
    return float(row_entropy_table(count_matrix)["weighted_entropy_bits"].sum())


def entropy_summary(count_matrix: pd.DataFrame) -> dict:
    """Return conditional, pair-position marginal, joint and mutual information bits."""
    rows = row_entropy_table(count_matrix)
    values = count_matrix.to_numpy(dtype=float)
    current_entropy = entropy_from_counts(values.sum(axis=1))
    next_entropy = entropy_from_counts(values.sum(axis=0))
    joint_entropy = entropy_from_counts(values.ravel())
    conditional = float(rows["weighted_entropy_bits"].sum())
    return {
        "pair_occurrences": float(values.sum()),
        "defined_rows": int(rows["entropy_defined"].sum()),
        "undefined_rows": int((~rows["entropy_defined"]).sum()),
        "conditional_entropy_bits": conditional,
        "current_marginal_entropy_bits": current_entropy,
        "next_marginal_entropy_bits": next_entropy,
        "joint_entropy_bits": joint_entropy,
        "mutual_information_bits": next_entropy - conditional,
    }
