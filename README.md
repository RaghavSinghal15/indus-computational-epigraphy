# Computational Analysis of the Indus Script

AI course project, BITS Pilani K.K. Birla Goa Campus.

The Indus script (c. 2600 to 1900 BCE) is undeciphered. This project does not try to read it. It analyses the script's structure: a vision pipeline that turns seal photos into sign sequences, Markov models of sign order, masked sign prediction, and statistical guardrails against overclaimed decipherments.

## Final results (Nov 2026)

Fixed seal-level split: 466 train, 67 validation, 141 test seals, plus 181 Harappa seals never used for training. Every choice was made on validation; test numbers are reported once. Full write-up: [`report/`](report/) (course report source) and the files in `reports/`.

| Milestone | What | Result | Report |
| --- | --- | --- | --- |
| Data | 855 CISI seal photos with Wells sign sequences (via Lipi); team-drawn boxes on 834 seals (4,078 signs) | labels agree with our hand-linked IM77 labels on 96.5% of signs | `data/README.md`, `docs/ANNOTATION.md` |
| M1 | Seal photo to signs: forced cuts + CNN (renders + real crops) + ICIT bigram decoder, count given | 33.7% top-1, 55.1% top-5 on 141 test seals (frequency baseline 11.4%); v1 to v2 on the same signs: 29.7% to 35.5% | `reports/m1_v2.md` |
| M1 | Segmentation audit against hand boxes | same CNN: 33% on our cuts vs 56% on hand-box crops; mean IoU 0.455 | `reports/m1_segmentation.md` |
| M1 | Training on hand-box crops | same start and extra epochs, only the training crops differ: 73.9% top-1 / 90.1% top-5 on hand-box test crops (control 55.7%), 71.3% on Harappa; needs good boxes at test time | `reports/m1_handbox_cnn.md` |
| M1 | Whole-band CRNN + CTC (no count, no cutting) | sign error 56.3% (T4) / 55.4% (CPU rerun) vs 68.1% for cut-and-classify with the true count | `reports/m1_crnn.md` |
| M1 | Synthetic inscriptions from real sign crops (instructor's suggestion) | CRNN sign error with composites vs without (same seed): test 55.4% to 54.4%; Harappa 57.9% to 53.2%, correct counts 53 to 65 of 181 | `reports/m1_composites.md` |
| M1 | Sign detector (YOLOv8n on hand boxes + composites), then the hand-box CNN: fully automatic, no count | sign error 40.8%, 16.3% of inscriptions exact (forced cuts with the true count: 68.1%); exact count on 57.4% of seals | `reports/m1_sign_detector.md` |
| M1 | Cross-site: trained on Mohenjo-daro, tested on Harappa | 28.9% vs 29.9% in-site | `reports/m1_harappa.md` |
| M1 | Published 94.5% protocol (40 signs, crop split) on our data | 62.5% (majority 49.9%); random crop split did not inflate it | `reports/m1_dixit_protocol.md` |
| M2 | Frequencies, start/end signs, transitions, entropy | jar sign ends 35.2% of texts, starts 0.16%; entropy 3.31 bits read right to left vs 3.82 reversed | `reports/b1_results.md` to `b4_results.md` |
| M2-M4 | Full ICIT corpus and per site | all statistics within 1 point; jar-final 39.6% Mohenjo-daro vs 29.0% Harappa | `reports/m2m4_full_corpus.md` |
| M3 | Hide a sign, predict it | bigrams 35.0% / 60.0%; perplexity 37.8 (unigram 87.1); small transformer 29.9% (worse) | `reports/c1_masked_prediction.md`, `reports/c2_neural_m3.md` |
| M4 | Real vs shuffled texts | Naive Bayes 81.4% (control 50%); permutation p = 0.001 | `reports/d1_guardrails.md` |
| M4 | Same tests on non-language symbols | deity symbols also pass (84.6%), barn stars do not (50%) | `reports/m4_symbol_controls.md` |
| E1 | Error propagation from M1 into M2-M4 | pair statistics need about 60% reading accuracy; at the forced-cut CNN's 31% they mostly fail, at the hand-box CNN's 73.9% (its own confusions) every signal survives: real vs shuffled 73.7%, masked top-5 41.6% vs 21.3% | `reports/e1_error_propagation.md`, `reports/e1_cnn_confusions.md`, `reports/e1_handbox_confusions.md` |
| E2 | Null-model benchmark | a no-language bigram chain reproduces Zipf, the jar sign and entropy; only sign triples separate real texts (61%) | `reports/e2_null_benchmark.md` |

First-phase (Oct 6) results and the plan that led here: `docs/PLAN.md`, `docs/ROADMAP.md`.

## Setup

Requires Python 3.10 or newer.

```bash
git clone https://github.com/RaghavSinghal15/indus-computational-epigraphy.git
cd indus-computational-epigraphy

python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate

pip install -r requirements.txt
pytest
```

All tests should pass (some skip until `python scripts/fetch_lipi.py` has downloaded the Lipi data). If they do, the data is in place and loads correctly.

(Using conda instead? Create and activate an env, then run the same `pip install -r requirements.txt`.)

## Quick start

```python
from indus.data import load_sequences, template_path

seqs = load_sequences()    # 1,837 inscriptions, Wells numbers, reading order
print(seqs[0])
print(template_path(342))  # reference drawing of the jar sign
```

## Working together (read before your first change)

Five people work in this repo. These rules keep us from overwriting each other's work. They apply to everyone, including the repo owner.

### The rules

1. **Never push directly to `main`.** Every change goes in through a pull request (PR). `main` should always be in a working state.
2. **One branch per task**, named `p<your number>-<task id>-<short description>`, e.g. `p1-a1-segmentation` or `p3-d1-seal-photos`.
3. **Someone other than the author reviews and merges.** Nobody merges their own PR.
4. **Stay inside your own task's files and folders** (see [`docs/TASKS.md`](docs/TASKS.md)). If you need to change shared code (`src/indus/data.py`, `tests/`), say so in the team chat first.
5. **Never edit existing files in `data/raw/`.** Adding new files there is fine when that is your task (e.g. seal photos).
6. **Run `pytest` before opening a PR.** All tests must pass.
7. **Using an AI coding tool?** Make sure it has read `AGENTS.md`. You are responsible for everything it writes: read it before you commit it.

### Step by step

**1. Start a task** (always from an up-to-date `main`):

```bash
git checkout main
git pull
git checkout -b p3-d1-seal-photos
```

**2. Work and commit in small steps.** Commit whenever something works, not only at the end.

```bash
git status                       # see what changed; check nothing unexpected is there
git add path/to/your/file        # add specific files, not everything
git commit -m "D1: add first 10 seal photos"
git push -u origin p3-d1-seal-photos   # first push of this branch; after that, just: git push
```

Start commit messages with your task ID (`A1:`, `B3:`, `D4:` ...) so the history shows who did what.

**3. Open a pull request.** After pushing, GitHub shows a **Compare & pull request** button on the repo page. Use a title like `D1: seal photo dataset` and in the description write:
- what you did
- how you checked it works (tests passed, numbers you verified, a screenshot)
- anything unfinished

Then pick a reviewer on the right-hand side.

**4. Reviewing someone else's PR.** Open the **Files changed** tab and check:
- it does what its task in `docs/TASKS.md` says, and meets the "done when" check
- nothing unrelated sneaked in (stray files, edits to someone else's folder, changes to `data/raw/`)
- if it is code, pull the branch and run it: `git fetch`, `git checkout <branch-name>`, `pytest`

Then either **Approve** or **Request changes** with a comment saying what to fix. Once approved, merge with **Squash and merge** (keeps `main`'s history to one clean commit per task) and click **Delete branch**.

**5. When `main` moves on while you are still working**, bring its changes into your branch:

```bash
git checkout main
git pull
git checkout p3-d1-seal-photos
git merge main
```

If git reports a **conflict**, it means you and someone else changed the same lines. Don't guess: post in the team chat and sort it out with that person.

### Never

- `git push --force` on `main`, or on anyone else's branch
- commit passwords, API keys or tokens
- commit large generated files (models, the synthetic image set); those belong in `data/processed/`, which git ignores

## Layout

```
data/raw/          source data, read-only (see data/README.md)
data/processed/    anything we derive (not committed)
src/indus/         shared code; data.py loads every dataset
notebooks/         exploration, one per task
tests/             sanity checks on the data
reports/figures/   plots for slides and the report
docs/ROADMAP.md    current plan and decisions (final phase); work is in GitHub issues
docs/PLAN.md       first-phase direction (until Oct 6)
docs/TASKS.md      first-phase task list (until Oct 6)
AGENTS.md          context for AI coding tools (read by Claude Code and Codex)
```

## Two things to know before using the data

1. **Direction.** Files store signs left to right as seen on the seal; the script reads right to left. `load_sequences()` handles this.
2. **Numbering.** Sequences use Wells numbers; templates and IM77 tables use Mahadevan numbers. Convert with `wells_to_mahadevan()`.
