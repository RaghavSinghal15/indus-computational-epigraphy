# B1: descriptive sequence statistics

B1 is complete for the repository's **ICIT corpus of 1,837 unique filtered
inscriptions, using Wells sign IDs**. It does not analyse the Mahadevan/IM77
snapshot or predict anything from seal photographs.

## Results

| Quantity | Result |
|---|---:|
| Sequences | 1,837 |
| Sign occurrences (tokens) | 8,359 |
| Distinct sign types | 568 |
| Types appearing exactly once (singletons) | 219 |
| Singleton share of types | 38.56% |
| Singleton share of tokens | 2.62% |
| Shortest / longest sequence | 2 / 13 |
| Mean / median / mode length | 4.5504 / 4 / 3 |
| Most frequent type | Wells 740, 902 occurrences (10.79% of tokens) |

The totals of 8,359 tokens, 568 types and 219 singletons match the
reference counts. There are 361 length-three sequences, the largest length group.

## Files and how to run

- `notebooks/b1_descriptive.ipynb`: executed notebook, original student notes
  and image preserved, teaching example, explanations, tables and assertions.
- `src/indus/descriptive.py`: three reusable functions: `count_signs`,
  `count_lengths`, `summarize_corpus`.
- `reports/figures/b1_zipf.png`: frequency rank against occurrence count,
  both on logarithmic scales. Ties use ascending Wells ID.
- `reports/figures/b1_lengths.png`: number of sequences of each integer length.
- `data/processed/b1/`: regenerated frequency, length and singleton CSVs,
  provenance summary JSON, and a backup of the original notebook notes.

In VS Code, open the notebook, select the project's `.venv/bin/python` kernel,
and run all cells from top to bottom. The imports require the project to be
installed in that environment (`pip install -r requirements.txt` is the team
setup command). The notebook runs from either the project or notebooks directory
because output paths come from `indus.data.REPO_ROOT`.

## Checks performed

- Manually specified example: 9 tokens, 3 types, 1 singleton, lengths 4/2/3.
- Repeated-sign case and rejection of empty corpora/sequences.
- All six known corpus totals, plus a separate dictionary count of every sign.
- Sign counts sum to tokens; length counts sum to sequences; length times count
  sums to tokens. Reversing sequences preserves all B1 statistics.
- Descending frequency ranks, probabilities summing to one, CSV readback.
- Notebook executed using the project virtual environment; no error outputs.
- All **25 existing repository tests passed**. New B1-specific assertions live
  in the executed notebook; existing shared tests were not edited.
- Both figures visually inspected; original note sources and image attachments
  verified unchanged.

OpenCV and scikit-learn (including SciPy) were missing in the existing virtual
environment and installed from the already-declared requirements to run the
team's current vision tests. B1 itself uses no vision dependency.

## Interpretation and limits

A token is one occurrence; a type is one symbol category. For `[151, 151, 740]`,
there are three tokens and two types. Singleton means occurring exactly once
in the whole corpus, not once within a particular sequence.

Rank is an ordering by count, not the numerical value of a sign ID. Relative
frequency is count divided by 8,359. The Zipf plot displays common signs and a
long rare-sign tail; it does not fit or establish a power law or prove language.

The corpus excludes damaged, multiline and single-sign texts and removes
duplicate inscriptions. Frequencies describe these unique texts, not object
production frequencies or every known Indus inscription. Checks establish
counting consistency, not independent transcription or archaeological accuracy.
We neither edit raw data nor mix Wells IDs with Mahadevan IDs.

B1 uses the existing reading-order loader. Direction does not affect these
counts; it matters for B2 endpoints and B3 transitions. There is no training or
randomness, so no data split or random seed is needed here.

## Local sources

`data/README.md` documents
the corpus and its limitations. `src/indus/data.py` defines the loader.
Source CSV SHA-256:
`140a61e57dfaccd15867d9a00141f180a38080141b0ad18bb4262c782024f186`.
