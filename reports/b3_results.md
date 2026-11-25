# B3: pair counts and transition probabilities

B3 estimates empirical next-sign probabilities from the existing reading-order
loader. It uses the ICIT corpus of 1,837 unique filtered inscriptions with Wells
IDs. It runs independently of B1/B2 code and edits no raw files.

## Results

| Quantity | Result |
|---|---:|
| Tokens / sign types | 8,359 / 568 |
| Adjacent pair occurrences | 6,522 |
| Distinct ordered pair types | 2,751 |
| Matrix size | 568 × 568 |
| Signs with observed successors | 526 |
| Signs without observed successors | 42 |

Each nonempty sequence of length L contributes L - 1 pairs, giving
8,359 - 1,837 = 6,522. Windows overlap, repeated signs count, and separate
inscriptions are never joined. No artificial start/end tokens are added.

| Frequent Wells pair, reading order | Count |
|---|---:|
| 861 → 2 | 114 |
| 817 → 2 | 91 |
| 740 → 400 | 88 |
| 820 → 2 | 68 |
| 760 → 740 | 66 |
| 100 → 740 | 65 |

For example, sign 861 has 141 observed outgoing pairs, so
P(next=2 | current=861) = 114 / 141 = 80.85%.
The jar sign (740) has 255 outgoing pairs, not 902: its 647 final-position
occurrences cannot contribute a successor. P(400 | 740) = 88 / 255 = 34.51%.

## API and probability semantics

`src/indus/ngrams.py` contains:

- `count_ngrams(sequences, n=2)`: Counter of overlapping tuple windows, kept
  inside inscription boundaries. Positive integer n only; supports later n=3.
- `bigram_table(sequences)`: descending pair counts, ties by ascending pair IDs.
- `transition_matrices(sequences)`: count and probability DataFrames over all
  observed signs, both axes sorted by Wells ID. Row=current; column=next.
- `compare_im77_bigrams(pairs, mapping, im77_visual)`: mapped pair comparison
  and explicit eligibility/coverage counts.

A probability is pair count divided by the row's outgoing pair total, not the
sign's total occurrences. Every defined row sums to one. **The 42 no-successor
rows are entirely NaN (undefined)**; they are not zero distributions or invented
uniform distributions. In defined rows, an unseen pair has zero probability.
There is no smoothing in B3; B5 will provide an explicit smoothing policy.

Empty corpora/sequences are rejected. Single-sign sequences and windows longer
than a text produce no n-grams; matrix axes still include the observed types.
Consumers must handle undefined rows before computing entropy or predictions.
This is a first-order Markov approximation, ignoring earlier context.

## Heatmap

`reports/figures/b3_transitions.png` shows the 15 most frequent sign types by
overall token count, with ties broken by ID. The complete matrix is normalised
before selecting the displayed rows/columns. Displayed row mass can be below
one because omitted successor signs retain their probability. No crop
renormalisation is performed.

Cell labels show probability and pair count in parentheses. Positive values
below 0.01 are labelled `<0.01`, blank cells are unobserved pairs, and grey denotes
undefined distributions if any occur in the displayed subset. Each row label
shows its full outgoing count so sparse support is visible.

## IM77 comparison: coverage and limits

The ICIT pairs are Wells IDs in reading order. The IM77 table is Mahadevan IDs
in visual order: `(left,right)` must become `(right,left)` before comparison.
For instance, its leading visual pair 99,267 becomes reading pair 267 → 99.
Conversion uses only `indus.data.wells_to_mahadevan()`.

| Original Wells pair eligibility | Distinct pair types | Pair occurrences |
|---|---:|---:|
| Both signs have exactly one mapped target | 1,517 | 3,802 |
| At least one multi-target mapping (neither missing) | 768 | 2,092 |
| At least one missing/empty mapping | 466 | 628 |
| Total | 2,751 | 6,522 |

Eligible occurrences represent 58.30%
of all pair occurrences. Eligible Wells pairs pool into 1,371 Mahadevan pairs;
1,042 are present in the supplied IM77 table. Of the 3,802 eligible occurrences,
3,440 (90.48%)
belong to a pair present in IM77. **This is occurrence-weighted support among
eligible pairs, not prediction accuracy or whole-corpus validation.**

| Reading-order Mahadevan pair | Eligible ICIT pooled count | IM77 count |
|---|---:|---:|
| 342 → 176 | 88 | 184 |
| 336 → 89 | 87 | 126 |
| 8 → 342 | 65 | 93 |
| 342 → 1 | 64 | 87 |
| 48 → 342 | 53 | 114 |

These frequent eligible pairs occur in both corpora, providing qualified pattern
agreement. Exact counts and ranks differ because the corpora and sign taxonomies
differ. The ICIT count includes only unambiguously mapped contributions; it is
not a complete count in Mahadevan taxonomy.

The highest-count original Wells pairs 861 → 2 and 817 → 2 cannot be converted
uniquely: their candidates are 267 → {99,100}. IM77 contains 267 → 99, which
shows compatibility, but does not establish which ID to choose. Top-pair candidate
compatibility is exposed separately; ambiguous counts are never distributed
among candidates or mixed into eligible pooled counts. The hand mapping is not
authoritative. Absence means absence from the supplied table, not impossibility.

## Reproduce and verify

Open `notebooks/b3_transitions.ipynb`, select `.venv/bin/python` and run all cells.
It includes a hand-counted corpus, actual tables, the chart, mapping evidence,
independent checks and exports. It also works from the notebooks directory
because paths come from `indus.data.REPO_ROOT`.

From the repo root:

```sh
.venv/bin/python scripts/check_b3.py
.venv/bin/python -m pytest -q
```

All **9 dedicated B3 checks and 25 existing repository tests passed**. The
notebook executed successfully and the chart was visually inspected. No shared
loader or existing test files were changed.

Checks cover hand-counted probabilities; repeats and boundaries; overlapping
trigrams/unigrams; no-pair/undefined rows; invalid inputs; stable sorting;
successor-only denominators; reversal; mapping direction, ambiguity, gaps and
many-to-one pooling; independent real-corpus pair counts; row totals equal to
token counts minus ends and column totals equal to tokens minus starts; defined
probability row sums; export round-trip; mapping coverage conservation.

Reversing sequences transposes the count matrix. The probability matrix is not
generally transposed, because row denominators change.

## Saved outputs and provenance

Regenerable files in `data/processed/b3/` (ignored by Git):

- `bigram_counts.csv`: complete ranked Wells pairs.
- `transition_counts.csv`: full integer count matrix.
- `transition_probabilities.csv`: full unsmoothed matrix; blank cells are NaN.
- `outgoing_counts.csv`: per-sign support and explicit probability-defined flag.
- `im77_comparison.csv`: pooled eligible Mahadevan pairs and IM77 presence/count.
- `top_pairs_mapping.csv`: top original Wells pairs, candidates and compatibility.
- `summary.json`: totals, undefined rows, mapping coverage and source SHA-256s.

The source file hashes are recorded for sequences, IM77 pairs and mapping.
Checksums identify the inputs; they do not independently certify transcriptions.

## Scope

The source deduplicates texts and excludes damaged, multiline and single-sign
inscriptions. Results describe this corpus, not frequencies of all physical
objects. Probabilities are estimated and inspected on the same corpus; no
held-out accuracy is claimed. There is no randomness or evaluation split in this
descriptive task. No sign meanings, grammar or independent reading direction
are inferred. B4 can reuse counts/probabilities; smoothing remains B5.

Local references: `data/README.md`, `src/indus/data.py`.
