# Computational Analysis of the Indus Script

AI course project, BITS Pilani K.K. Birla Goa Campus.

The Indus script (c. 2600 to 1900 BCE) is undeciphered. This project does not try to read it. It analyses the script's structure: a vision pipeline that turns seal photos into sign sequences, Markov models of sign order, masked sign prediction, and statistical guardrails against overclaimed decipherments.

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

All 6 tests should pass. If they do, the data is in place and loads correctly.

(Using conda instead? Create and activate an env, then run the same `pip install -r requirements.txt`.)

## Quick start

```python
from indus.data import load_sequences, template_path

seqs = load_sequences()    # 1,837 inscriptions, Wells numbers, reading order
print(seqs[0])
print(template_path(342))  # reference drawing of the jar sign
```

## Layout

```
data/raw/          source data, read-only (see data/README.md)
data/processed/    anything we derive (not committed)
src/indus/         shared code; data.py loads every dataset
notebooks/         exploration, one per task
tests/             sanity checks on the data
reports/figures/   plots for slides and the report
docs/PLAN.md       tracks, deadlines, what's next
AGENTS.md          context for AI coding tools (read by Claude Code and Codex)
```

## Two things to know before using the data

1. **Direction.** Files store signs left to right as seen on the seal; the script reads right to left. `load_sequences()` handles this.
2. **Numbering.** Sequences use Wells numbers; templates and IM77 tables use Mahadevan numbers. Convert with `wells_to_mahadevan()`.
