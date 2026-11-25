# Roadmap: final phase (after the Oct 15 flash talk)

**Status (Nov 25):** done. All P0/P1 issues are merged; the composite GAN (#42) was dropped. Final numbers: the results table in the README; write-up: `report/`.

Short version of what we learned since the first evaluation, what we decided, and who can pick up what. Work is tracked as GitHub issues labelled **P0** (core), **P1** (strong addition), **P2** (stretch).

## What we found

- **Lipi** ([yajnadevam/lipi](https://github.com/yajnadevam/lipi)) is a digitised copy of Fuls's *Corpus of Indus Inscriptions* (the full ICIT corpus: 5,680 texts, Wells numbers) plus CISI images for 8,722 objects. These are the same CISI scans we already use.
  - 855 seals have an impression photo and a complete, legible, single-line text. 571 of them are new to us (577 Mohenjo-daro, 181 Harappa in total).
  - On our 328 seals its sequences agree with our IM77 labels on 96.5% of mappable signs, and are stored left to right as on the impression, like our raw files.
  - Its sign font has a clean drawing for every Wells sign (712, codepoint U+E000 + Wells code). These match Fuls's sign list (Table 2.8).
  - No licence, so we never commit Lipi files: `scripts/fetch_lipi.py` downloads them into `data/external/` (gitignored). We ignore its Sanskrit/translation columns (the author's proposed decipherment).
- **Fuls, *Corpus of Indus Inscriptions* (2022)** defines the Wells/ICIT sign list (709 signs) and indexes CISI and M77 numbers. We have a 35-page preview only; Lipi gives us the same content in machine-readable form.
- **The professor's suggestion (Oct 16 meeting):** use the real signs we have to generate realistic inscriptions of real sequences, as training data. This is copy-paste augmentation (as in SynthText for text recognition) and is exactly what a whole-band reader (CRNN) needs.

## Decisions

1. **M1 now predicts Wells (ICIT) sign IDs**, not Mahadevan. Most new seals have no clean Mahadevan conversion, and Wells is what M2-M4 use, so M1 output feeds them with no cross-scheme step. Mahadevan results stay in the reports as "v1".
2. **The 1,837-text benchmark stays** the reference for M2-M4 (comparable with published numbers). The full corpus and per-site splits are robustness checks, reported next to it.
3. **Hand-drawn sign boxes** on about 600 seals: clean crops for the compositor, ground truth for a detector, and a direct measure of segmentation. Boxes only, no typing of sign numbers: labels come from the known sequence.
4. **Old test and validation seals stay in test and validation**, so v1 and v2 are compared on the same seals. Harappa seals are held out for a cross-site test.

## What gets redone, what stays

| Part | Status |
| --- | --- |
| M1 (A5-A7) | Redone as v2: Wells labels, about 870 seals, ICIT-bigram decoder. Then the compositor, CRNN and detector on top. |
| M2 (B1-B4), M3 (C1), M4 (D1) | Kept as is on the benchmark. Rerun on the full corpus and per site as extra rows. Neural M3 is new. |
| E1 | Rerun at v2 accuracy and with real CNN confusions. |
| E2 | Kept; rerun on the full corpus as part of the M2-M4 robustness issue. |

## Order of work

```
#21 Lipi import ──► #22 Wells switch ──► #30 CNN v2 ──► #37 Harappa, #40 Dixit protocol, #41 E1 rerun
       │                                        │
       ├──► #23 box protocol ──► #24-29 boxes ──┼──► #31 segmentation eval
       │                                        └──► #32 compositor ──► #33 CRNN, #36 detector
       └──► #38 M2-M4 on full corpus
#39 neural M3, #35 PR #20, #34 public repo: no dependencies
#44-46 report, video, figures: last
```

## How we work

Claim an issue on WhatsApp, then assign yourself on GitHub so the board stays accurate. One branch and one PR per issue (`Closes #N` in the description), reviewed by someone else. The rules in the README still apply.
