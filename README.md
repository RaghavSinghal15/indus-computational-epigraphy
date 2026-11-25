# Computational Analysis of the Indus Script

The Indus script (c. 2600 to 1900 BCE) survives on about 4,000 short inscriptions and has never been deciphered. This project does not try to read it. It measures the script's **structure** and tests how far each measurement can be trusted:

1. **Reading seals (M1).** A vision pipeline that turns a seal photograph into an ordered list of sign IDs.
2. **Sign order (M2).** Frequencies, start and end signs, transition probabilities, reading direction, entropy.
3. **Masked prediction (M3).** Hide one sign and predict it from its neighbours: n-grams first, then a small transformer.
4. **Guardrails (M4).** Can a classifier tell real texts from shuffled ones, and do non-linguistic symbol systems pass the same tests?
5. **Error propagation (E1)** from M1 into M2 to M4, and a **null-model benchmark (E2)** for decipherment-style claims.

Course project for CS F407 Artificial Intelligence, BITS Pilani K.K. Birla Goa Campus (2026). Report: [`report/project_report.pdf`](report/project_report.pdf).

## Results

Fixed seal-level split: 466 train, 67 validation and 141 test seals, plus 181 Harappa seals that are never used for training. Every choice was made on validation, and test numbers are reported once.

| Part | Question | Result | Details |
| --- | --- | --- | --- |
| Data | Labelled seal photos | 855 CISI seal photos with Wells sign sequences; hand-drawn boxes on 834 seals (4,078 signs); labels agree with independent IM77 labels on 96.5% of signs | [`data/README.md`](data/README.md), [`docs/ANNOTATION.md`](docs/ANNOTATION.md) |
| M1 | Equal cuts + CNN + bigram decoder (sign count given) | 33.7% top-1, 55.1% top-5 over 712 Wells signs (frequency baseline 11.4%) | [`reports/m1_v2.md`](reports/m1_v2.md) |
| M1 | Is cutting or recognising the bottleneck? | Same CNN: 33% on our cuts vs 56% on hand-box crops (mean IoU of cuts 0.455) | [`reports/m1_segmentation.md`](reports/m1_segmentation.md) |
| M1 | CNN trained on hand-box crops | 73.9% top-1 / 90.1% top-5 on test, 71.3% on Harappa | [`reports/m1_handbox_cnn.md`](reports/m1_handbox_cnn.md) |
| M1 | Fully automatic: sign detector (YOLOv8n) + that CNN, no count given | 40.8% sign error rate, 16.3% of inscriptions exact (equal cuts with the true count: 68.1%) | [`reports/m1_sign_detector.md`](reports/m1_sign_detector.md) |
| M1 | Synthetic inscriptions composed from real sign crops, for a whole-band CRNN + CTC reader | Harappa sign error 57.9% to 53.2%; correct sign count on 65 instead of 53 of 181 seals | [`reports/m1_composites.md`](reports/m1_composites.md) |
| M1 | Published 40-sign benchmark (Dixit et al. 2025: 94.5%) | 90.4% with hand boxes on unseen seals (equal cuts: 62.5%; always "Other": 49.9%) | [`reports/m1_dixit_handbox.md`](reports/m1_dixit_handbox.md), [`reports/m1_dixit_protocol.md`](reports/m1_dixit_protocol.md) |
| M2 | Start/end signs and direction | Jar sign ends 35.2% of texts and starts 0.16%; next-sign entropy 3.31 bits right to left vs 3.82 reversed | [`reports/b1_results.md`](reports/b1_results.md) to [`b4`](reports/b4_results.md) |
| M3 | Masked sign prediction | Bigrams 35.0% top-1, 60.0% top-5, perplexity 37.8 (unigram 87.1); small transformer 29.9% | [`reports/c1_masked_prediction.md`](reports/c1_masked_prediction.md), [`reports/c2_neural_m3.md`](reports/c2_neural_m3.md) |
| M4 | Real vs shuffled texts | Naive Bayes 81.4% (chance 50%), permutation p = 0.001; but Mesopotamian deity symbols (not writing) also pass (84.6%) | [`reports/d1_guardrails.md`](reports/d1_guardrails.md), [`reports/m4_symbol_controls.md`](reports/m4_symbol_controls.md) |
| E1 | Are machine readings good enough for M2 to M4? | Pair statistics need about 60% reading accuracy. At the hand-box CNN's 73.9%, with its real confusions, every signal survives (real vs shuffled 73.7%) | [`reports/e1_error_propagation.md`](reports/e1_error_propagation.md), [`reports/e1_handbox_confusions.md`](reports/e1_handbox_confusions.md) |
| E2 | What does a no-language model reproduce? | A bigram chain reproduces Zipf's law, the jar sign and the entropy; only sign triples separate real texts (61%) | [`reports/e2_null_benchmark.md`](reports/e2_null_benchmark.md) |

Each report in [`reports/`](reports/) comes from a script in [`scripts/`](scripts/) and says how to reproduce it.

## Setup

Python 3.10 or newer.

```bash
git clone https://github.com/RaghavSinghal15/indus-computational-epigraphy.git
cd indus-computational-epigraphy
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/fetch_lipi.py      # seal photos, corpus and sign font (downloaded on demand, not stored in this repo)
pytest
```

The tests encode known facts about the data (counts, orientation, the jar sign's position), so a passing run means the data is in place and loads correctly.

```python
from indus.data import load_sequences
seqs = load_sequences()    # 1,837 inscriptions as Wells sign numbers, in reading order
```

## Two things to know about the data

1. **Direction.** Files store signs left to right as seen on the seal impression; the script reads right to left. `indus.data.load_sequences()` returns reading order.
2. **Numbering.** Sequences and the M1 models use Wells numbers; IM77 tables and the sign templates use Mahadevan numbers. Convert only with `indus.data.wells_to_mahadevan()`.

Sources and licences for every dataset: [`data/README.md`](data/README.md).

## Layout

```
src/indus/        library code: data loaders, segmentation, CNN/CRNN, decoding, n-grams, guardrails, null models
scripts/          one script per experiment; each writes a report in reports/
reports/          results, one Markdown file per experiment
tests/            sanity checks on data and code
notebooks/        exploratory notebooks for M2
data/raw/         source data and our hand-drawn sign boxes (read-only)
data/processed/   everything derived (not committed)
docs/             annotation protocol and the CRNN design
report/           course report (LaTeX + PDF) and the video script
```

## Team

Team Artificially Intelligent: Raghav Singhal, Devam Jatin Bhavsar, Krrish Bagaria, Saatwik Tiwari, Agrim Agarwal. The individual contributions are listed in the report.

We thank Prof. Ashwin for the suggestion to compose synthetic inscriptions from real sign crops.
