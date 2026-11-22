# Computational Analysis of the Indus Script

AI course project, BITS Pilani K.K. Birla Goa Campus.

The Indus script (c. 2600 to 1900 BCE) is undeciphered. This project does not try to read it. It analyses the script's structure: a vision pipeline that turns seal photos into sign sequences, Markov models of sign order, masked sign prediction, and statistical guardrails against overclaimed decipherments.

## Results so far (first evaluation, Oct 6 2026)

| Milestone | What | Result | Report |
| --- | --- | --- | --- |
| Data | 328 Mohenjo-daro seal photos (CISI Vol. 1) labelled with their IM77 sign sequences | 1,673 signs, 225 distinct; jar sign last in 116 texts, first in 0 | `data/raw/cisi_seals/README.md` |
| M1 | Seal photo to sign sequence: CNN + sign-pair statistics, sign count given | 28 to 30% top-1, 45 to 49% top-5 on 70 test seals (template matching: 0.3% / 1.4%) | `reports/a6_context_*.md` |
| M1 | Same, no labels at all (the model finds the sign count) | count right 19% (old pipeline 10%), character error rate 0.83 (1.48) | `reports/a7_unknown_count_gpu.md` |
| M2 | Frequencies, start/end signs, transitions, conditional entropy | 8,359 signs, 568 types; jar sign ends texts; 3.31 bits | `reports/b1_results.md` to `b4_results.md` |
| M3 | Hide a sign, predict it (smoothed bigrams) | 35% top-1, 60% top-5 from both neighbours; bigram perplexity 37.8 | `reports/c1_masked_prediction.md` |
| M4 | Real texts vs shuffled | Naive Bayes 81%; permutation tests p = 0.001 | `reports/d1_guardrails.md` |
| Extension | Error propagation: which conclusions survive M1's mistakes | positions survive at ~30% accuracy, pair statistics need ~60% | `reports/e1_error_propagation.md` |
| Extension | Null-model benchmark for claims | a no-language Markov chain reproduces Zipf, end signs and entropy; sign triples still separate real texts (61%, control 49%) | `reports/e2_null_benchmark.md` |

Every model choice was made on a fixed validation split; test numbers are reported once.

## Final phase (now)

We found the full ICIT corpus and about 570 more labelled seal photos (via Lipi), switched M1 to Wells sign IDs, and are adding hand-drawn sign boxes, synthetic inscriptions built from real signs, and a reader that needs no cutting (CRNN). **Plan, decisions and order of work: [`docs/ROADMAP.md`](docs/ROADMAP.md).** Open work is in the [issues](../../issues) (P0 first): claim one on WhatsApp and assign yourself.

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

All 48 tests should pass. If they do, the data is in place and loads correctly.

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
