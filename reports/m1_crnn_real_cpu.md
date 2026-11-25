# M1 · CRNN + CTC on whole seal bands

**Real-band control for the composite run** (`m1_crnn_composites.md`): same CPU, seed and settings, no composites.

Train: 466 seals. Validation: 67. Selected epoch 13 of 28 by lowest validation sign error rate then highest exact match. Seed 0, PyTorch 2.15.0+cu132, device cpu, training 460.3 seconds.

| Split | Reader | Seals | Signs | Sign error rate ↓ | Entire inscription correct ↑ | Correct sign count ↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| test | Crop CNN, given count | 141 | 686 | 68.1% | 2.1% | 100.0% |
| test | Crop CNN, estimated count | 141 | 686 | 113.8% | 1.4% | 16.3% |
| test | Whole-band CRNN + CTC | 141 | 686 | 55.4% | 4.3% | 27.7% |
| harappa | Crop CNN, given count | 181 | 866 | 68.9% | 1.7% | 100.0% |
| harappa | Crop CNN, estimated count | 181 | 866 | 120.4% | 0.0% | 17.1% |
| harappa | Whole-band CRNN + CTC | 181 | 866 | 57.9% | 5.5% | 29.3% |

Sign error rate is the total insertions, deletions and substitutions divided by the number of reference signs. The CRNN uses an automatically located band, with a margin to reduce clipping. It receives no sign count, and reverses horizontal feature order without mirroring the glyphs. Adjacent repeats require a blank between them under CTC.

The estimated-count baseline counts boxes using the pixel-only segmentation heuristic and applies the same crop CNN. The given-count row has the true count. Neither baseline uses a language model. Harappa was held out from all real training and validation.

CNN initialization SHA-256: `d5c3821a90e08ed27f0861770306e854ca7a7f58149219db417cc5dfec158154`.
