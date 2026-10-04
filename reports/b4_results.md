# B4: conditional entropy in both directions

B4 calculates unsmoothed empirical next-sign uncertainty using the full B3 pair
count matrix. It uses all 6,522 adjacent pair occurrences from 1,837 filtered
unique ICIT inscriptions, in Wells numbering. It depends on B3 and edits no raw
data, shared loader or existing tests.

## Deliverable: two values and one interpretation

| Direction | Conditional entropy | Defined rows | Undefined rows |
|---|---:|---:|---:|
| Adopted reading order (RTL) | 3.3081803036 bits | 526 | 42 |
| Reversed order (LTR) | 3.8230464891 bits | 436 | 132 |

**Interpretation:** Empirical conditional entropy is 3.3082 bits in the adopted reading order and 3.8230 bits in reversed order; the signed difference (-0.5149 bits) equals the pair-marginal entropy difference and does not independently establish reading direction.

The reading-order entropy is lower by 0.5148661854 bits under this estimator.
That is a corpus description, not an independent test identifying direction.
RTL/LTR labels come from the existing loader convention.

## What is computed

For each current sign X, normalise its outgoing pair counts to obtain P(Y|X).
Compute row entropy as `-sum P(Y|X) * log2 P(Y|X)`, omitting zero terms before
logarithms. A certain successor has entropy zero; two equally likely successors
have one bit; four equally likely successors have two bits.

Weight each row by `outgoing_count(X) / 6522`, then add its entropy contribution.
This is an average over observed pairs, not an equal average over sign types.
The calculation uses the full vocabulary, never the cropped B3 heatmap.

A row with no outgoing evidence has NaN entropy, zero weight and explicitly zero
contribution; we do not compute 0 times NaN or invent a successor distribution.
An entirely pairless corpus has undefined conditional entropy and is rejected.
Negative, nonfinite counts and duplicate matrix axis labels are rejected too.

## Direction identity and boundary interpretation

With X and Y the first and second positions of a randomly selected observed pair:

```text
H(Y|X) = H(X,Y) - H(X)
H(X|Y) = H(X,Y) - H(Y)
H(Y|X) - H(X|Y) = H(Y) - H(X)
```

| Quantity in adopted reading order | Bits |
|---|---:|
| Pair-first marginal H(X) | 7.0239056594 |
| Pair-second marginal H(Y) | 6.5090394740 |
| Joint pair entropy H(X,Y) | 10.3320859630 |
| Mutual information I(X;Y) | 3.2008591703 |
| Reading minus reversed conditional entropy | -0.5148661854 |
| H(Y) minus H(X) | -0.5148661854 |

The identity residual is -4.441e-16 bits, consistent
with floating-point rounding. Joint entropy and mutual information are unchanged
by reversing pairs. The code recomputes transitions from reversed sequences;
it does not incorrectly transpose the probability matrix.

In finite inscriptions, pair-first counts equal all-token counts minus end counts,
while pair-second counts equal all-token counts minus start counts. Consequently,
the entire difference between these conditional entropies is explained by the
endpoint-driven marginal difference. It does not independently demonstrate a
directional asymmetry in dependence, language, grammar or sign meanings.

## Files and reproduction

- `src/indus/entropy.py`: entropy from counts, row support/weights/entropy table,
  weighted conditional entropy and a marginal/joint/mutual-information summary.
- `notebooks/b4_direction_entropy.ipynb`: executed teaching notebook, examples,
  measured values, identity checks, figure and exports.
- `scripts/check_b4.py`: ten executable mathematical/regression checks.
- `reports/figures/b4_direction_entropy.png`: zero-baseline comparison chart with
  a caption stating the interpretation limit.
- `data/processed/b4/`: regenerated direction CSV, both per-row CSVs and summary
  JSON with estimator, source hash, checks and interpretation. Ignored by Git.

In VS Code select `.venv/bin/python` and run the notebook top to bottom. B3 must
be present; this branch is stacked on `p2-b3-transition-matrix`. Once B3 is merged,
the B4 PR can be retargeted to main as part of the normal team review workflow.

From the project root:

```sh
.venv/bin/python scripts/check_b4.py
.venv/bin/python scripts/check_b3.py
.venv/bin/python -m pytest -q
```

All **10 B4 checks, 9 B3 checks and 25 existing repository tests passed**.
Notebook format validated, all cells executed successfully, exports read back
and compared, and the figure visually inspected.

Checks cover known entropy values including zero-probability cells; invalid/no-mass
inputs; outgoing-frequency weighting rather than uniform row weighting; undefined
rows; no-pair corpora; duplicate/nonfinite matrix inputs; a hand-counted joint
matrix and its chain-rule identities; scale/label-order invariance; double
reversal; independent entropy from raw observed pair counts; real-corpus marginal
and boundary identities; and direction-invariant mutual information.

The independent reading-order calculation gives
3.3081803036 bits, agreeing with the
matrix/row calculation. It directly sums `-(pair_count/6522) *
log2(pair_count/outgoing_count(current))` over observed pairs.

## Limits

These are unsmoothed full-corpus empirical entropies, not held-out prediction
scores. Finite samples and many rare signs affect empirical estimates. There is
no randomness or evaluation split in this descriptive task. B5 smoothing is not
implemented here and can alter the estimator and its interpretation.

The corpus deduplicates inscriptions and excludes damaged, multiline and
single-sign texts. Results are not frequencies over all physical objects or
every Indus inscription. We do not directly compare the result with published
entropy values from different corpora or methods as an exact numerical target.
Passing tests establishes arithmetic and integration consistency, not source
transcription accuracy or archaeological conclusions.

Source: `data/raw/icit_1837_sequences.csv` via `indus.data.load_sequences()`.
SHA-256: `140a61e57dfaccd15867d9a00141f180a38080141b0ad18bb4262c782024f186`.
Local references: `docs/TASKS.md`, `data/README.md`, `src/indus/ngrams.py`.
