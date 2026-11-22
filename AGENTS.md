# Context for AI coding assistants

Read this before writing any code in this repo. (Claude Code reads it through CLAUDE.md; Codex reads it directly.)

## Project

Course project (AI, BITS Pilani Goa): structural analysis of the undeciphered Indus script. We do not try to read the script. We analyse its structure.

- **M1** Vision pipeline: seal photo -> locate signs -> classify each sign -> ordered list of sign IDs
- **M2** Markov chains: sign frequencies, transition probabilities with smoothing, reading direction, start/end signs
- **M3** Masked prediction: hide a sign, predict it; entropy and perplexity; n-gram baseline first, neural model second
- **M4** Guardrails: Naive Bayes separating real inscriptions from shuffled ones; permutation null tests
- Extensions: error propagation (how M1 errors distort M2/M3 conclusions), null-model benchmark for decipherment claims

## Rules that matter

1. **Orientation.** Raw files store signs left to right as seen on the seal. The script reads right to left. Always load sequences with `indus.data.load_sequences()` (reading order by default). Never parse `data/raw/icit_1837_sequences.csv` directly.
2. **Numbering schemes are not interchangeable.** Sequences use Wells numbers. IM77 tables and sign templates use Mahadevan numbers (1 to 417). Convert only through `indus.data.wells_to_mahadevan()`. Never compare raw IDs across schemes. **From v2 on, M1 predicts Wells IDs** (see `docs/ROADMAP.md`); v1 (Mahadevan) results are kept for comparison only.
3. **Never modify `data/raw/`.** Write derived data to `data/processed/`. Files from Lipi (no licence) are downloaded by `scripts/fetch_lipi.py` into `data/external/` and are never committed.
4. **Run `pytest` after touching data code.** The tests encode known facts about the data. A failing test means the data or loader is wrong.
5. **Every result needs a sanity check** against something known: a published number, a hand count on a few rows, or the jar sign (Wells 740 / Mahadevan 342), which should appear mostly at the END of texts in reading order.
6. **Reusable code goes in `src/indus/`**, exploration in `notebooks/`. Notebooks import from `indus`, not the other way round.
7. Fix random seeds and use a single fixed train/test split for anything evaluated.

## Where things stand

`docs/ROADMAP.md` has the current plan, the decisions behind it and the issue order. Work is tracked as GitHub issues (labels P0/P1/P2). Read the issue you are working on before writing code.

## Reference numbers

- 1,837 sequences, 8,359 tokens, 568 distinct Wells signs, lengths 2 to 13
- IM77: 2,730 bigrams, 3,681 trigrams
- Published benchmarks (Yadav et al. 2010, Mahadevan corpus): bigram perplexity about 26.7, unigram about 68.8; about 75% accuracy restoring doubtful signs
- Rao et al. 2009: conditional entropy of about 2 to 3 bits

## Code style

Plain, readable Python. Short docstrings saying what a function returns. The team is learning this material, so prefer clear over clever, and explain any non-obvious library call in a comment.
