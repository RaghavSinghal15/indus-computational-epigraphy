# M1 · Dixit et al. protocol audit

The paper [Dixit et al. (2025), § III.B](https://doi.org/10.5334/jcaa.175) reports 94.5% for MobileNet and describes a reserved, balanced set of 200 unaugmented crops. Its separate cross-validation result is 85.28% mean accuracy. The published list contains 38 named Mahadevan codes plus Other, though the text says 40 classes. This pilot uses the listed codes and leaves the unidentified class unresolved.

Both protocols use the same 2,212 eligible Mohenjo-daro crops (of 2,875 before excluding Wells codes without a single Mahadevan mapping). 1,112 eligible crops map to Other. Using the same pool for both tasks isolates the split and label grouping from the choice of images.

| Split | Target labels | Train / val / test crops | Test crops from train seals | Test classes | Test targets seen in train | Frequency top 1 | CNN top 1 | CNN top 5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed seal split | Full Wells IDs | 1531 / 212 / 469 | 0.0% | 118 | 95.1% | 14.1% | 16.4% | 33.0% |
| Fixed seal split | 38 listed codes + Other | 1531 / 212 / 469 | 0.0% | 26 | 100.0% | 49.9% | 49.9% | 76.3% |
| Random crop split | Full Wells IDs | 1531 / 212 / 469 | 94.5% | 109 | 95.7% | 12.6% | 13.6% | 30.5% |
| Random crop split | 38 listed codes + Other | 1531 / 212 / 469 | 94.5% | 27 | 100.0% | 49.7% | 52.0% | 76.1% |

Random crop splits deliberately allow different positions on one seal to appear in train and test. Both splits use the same partition sizes. Frequency top 1 always picks the most common training label. A change in these scores measures a protocol effect on our forced-cut crop pool. The published study used different images, class balance, cropping, and a different model, so matching its published accuracy is a separate question.

From-scratch CNN pilot, 2 epochs, seed 0 and device cpu.
Use --init with a render-only v2 checkpoint to repeat the comparison with pretrained features. Separate fits per split keep crop-level test examples out of their respective training runs.
