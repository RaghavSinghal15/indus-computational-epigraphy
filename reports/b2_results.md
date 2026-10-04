# B2: opening and closing signs

B2 counts the first and last sign of every sequence returned by
`indus.data.load_sequences()`. These are 1,837 filtered unique ICIT inscriptions
in the project's reading order, using Wells IDs. Each sequence contributes one
start and one end, regardless of length. B2 is independent of B1 code.

## Results

The 568 observed sign types include 373 types appearing at starts and 199 at ends.
Start counts total 1,837; end counts independently total 1,837.

| Jar sign (Wells 740 / Mahadevan 342) quantity | Result |
|---|---:|
| All occurrences | 902 |
| Start occurrences | 3 |
| End occurrences | 647 |
| Fraction of texts starting with jar: 3 / 1,837 | 0.16% |
| Fraction of texts ending with jar: 647 / 1,837 | 35.22% |
| Fraction of jar occurrences at ends: 647 / 902 | 71.73% |
| End/start count ratio: 647 / 3 | 215.67 |

The jar is the most frequent closing sign and its end count exceeds ten times
its start count, matching the existing repository orientation sanity test.

| Top opening Wells ID | Start count | Share of texts |
|---|---:|---:|
| 820 | 109 | 5.93% |
| 817 | 106 | 5.77% |
| 861 | 102 | 5.55% |
| 920 | 55 | 2.99% |
| 32 | 50 | 2.72% |

| Top closing Wells ID | End count | Share of texts |
|---|---:|---:|
| 740 | 647 | 35.22% |
| 400 | 187 | 10.18% |
| 520 | 127 | 6.91% |
| 390 | 71 | 3.86% |
| 90 | 64 | 3.48% |

## Code and reproduction

- `src/indus/endpoints.py`: `count_endpoints()` returns start/end Counters;
  `endpoint_table()` returns counts and four fractions for all observed types.
- `notebooks/b2_endpoints.ipynb`: theory, hand examples, actual results, chart,
  independent checks and export. Select `.venv/bin/python` in VS Code and run all.
- `reports/figures/b2_endpoints.png`: paired start/end percentages for the union
  of the top five opening and top five closing IDs. Both series use 1,837 as the
  denominator. Selected IDs are ordered by larger endpoint count, then Wells ID.
- `data/processed/b2/`: full CSV table, top-ten CSV tables and provenance JSON.
  These are regenerable ignored files; rerunning the notebook recreates them.

Counts are integers. `*_probability` is endpoint count divided by number of texts;
`*_share_of_sign` is endpoint count divided by that sign's total tokens. The first
two columns each sum to one across types; the within-sign shares do not.
Types occurring only inside texts remain in the table with zero endpoint counts.
Count ties are broken by ascending Wells ID in ranked outputs.

Functions reject empty corpora or empty sequences because endpoints are undefined.
A one-sign sequence correctly contributes to both endpoints, though the current
corpus excludes such texts. The two within-sign shares are separate events and
need not be mutually exclusive for other corpora.

## Verification

- Executed the complete notebook with the exact project virtual environment.
- Two manually counted toy corpora check repeated signs, different endpoints,
  interior-only types, denominators and one-sign sequences; invalid inputs checked.
- Checked 1,837 sequences, 8,359 tokens and 568 types against reference totals.
- Start/end totals, probability sums/bounds and count bounds verified.
- Separate dictionary calculations agree with every table row.
- Reversing all sequences swaps endpoint counts, probabilities and within-sign
  fractions exactly.
- Confirmed Wells 740 maps to Mahadevan 342 through the approved mapping helper
  and passed the jar-sign expectation.
- Read exported CSV back and compared all columns to in-memory results.
- Inspected the chart visually and validated notebook format with no error outputs.
- All 25 existing repository tests passed. New B2 checks are in the executable
  notebook; shared data code and existing tests were not edited.

## Interpretation and limits

The jar sign is strongly associated with text endings under the loader's reading
order. The result does not decipher it, prove a grammatical function, or establish
direction independently of the adopted convention. Comparing positions is more
informative than B1's overall count alone, but rare types still have little evidence.

Unique texts are counted equally; these are not object-production frequencies.
The source excludes damaged, multiline and single-sign texts. Findings apply to
this corpus, not every Indus inscription. Counting checks do not independently
verify the original transcriptions. No Mahadevan dataset was mixed into the counts.

No training or randomness is involved, so a split or random seed is unnecessary.
No raw files were changed.

Sources: `docs/TASKS.md` (B2), `data/README.md` (provenance and exclusions),
`src/indus/data.py` (orientation/mapping), `tests/test_data.py` (jar expectation).
Source CSV SHA-256:
`140a61e57dfaccd15867d9a00141f180a38080141b0ad18bb4262c782024f186`.
