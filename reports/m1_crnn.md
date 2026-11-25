# M1 · CRNN + CTC on whole seal bands

**Real-band baseline.** No #32 composites were available for this run.

Train: 466 seals. Validation: 67. Selected epoch 13 of 28 by lowest validation sign error rate then highest exact match. Seed 0, PyTorch 2.14.1+cu130, device Tesla T4, training 29.3 seconds.

| Split | Reader | Seals | Signs | Sign error rate ↓ | Entire inscription correct ↑ | Correct sign count ↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| test | Crop CNN, given count | 141 | 686 | 68.1% | 2.1% | 100.0% |
| test | Crop CNN, estimated count | 141 | 686 | 113.8% | 1.4% | 16.3% |
| test | Whole-band CRNN + CTC | 141 | 686 | 56.3% | 5.0% | 28.4% |
| harappa | Crop CNN, given count | 181 | 866 | 68.9% | 1.7% | 100.0% |
| harappa | Crop CNN, estimated count | 181 | 866 | 120.4% | 0.0% | 17.1% |
| harappa | Whole-band CRNN + CTC | 181 | 866 | 58.0% | 5.0% | 23.2% |

Sign error rate is the total insertions, deletions and substitutions divided by the number of reference signs. The CRNN uses an automatically located band, with a margin to reduce clipping. It receives no sign count, and reverses horizontal feature order without mirroring the glyphs. Adjacent repeats require a blank between them under CTC.

The estimated-count baseline counts boxes using the pixel-only segmentation heuristic and applies the same crop CNN. The given-count row has the true count. Neither baseline uses a language model. Harappa was held out from all real training and validation.

CNN initialization SHA-256: `d5c3821a90e08ed27f0861770306e854ca7a7f58149219db417cc5dfec158154`.

## Interpretation

This single-seed run measures the full reader on real bands. The CRNN beats both crop pipelines on sign error and exact matches on both splits. It still recovers just 7 of 141 test inscriptions and 9 of 181 Harappa inscriptions exactly. Sign error can exceed 100% when the count heuristic inserts enough extra signs, since the denominator is the number of reference signs.

The checkpoint warm-starts the visual weights and sign classifier. The recurrent layer and CTC alignment learn from the real bands during this run. The comparison measures each complete reader. It does not isolate the effect of pretrained weights. Issue #32 composites are pending.

Reproduce after fetching the pinned Lipi dataset:

```bash
python -u scripts/train_crnn.py --device cuda --init /path/to/wells_all.pt --epochs 65 --train-budget-minutes 20
```

Trained CRNN checkpoint SHA-256: `e5fe660053cdf9d9a58231b56cb81c5b0b33348600225ffb3aca6824e3973167`. The checkpoint is stored locally in an ignored directory.
