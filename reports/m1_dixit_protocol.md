# M1 · Dixit et al. protocol audit

The paper [Dixit et al. (2025), § III.B](https://doi.org/10.5334/jcaa.175) reports 94.5% for MobileNet and describes a reserved, balanced set of 200 unaugmented crops. Its separate cross-validation result is 85.28% mean accuracy. The published list contains 38 named Mahadevan codes plus Other, though the text says 40 classes. This pilot uses the listed codes and leaves the unidentified class unresolved.

Both protocols use the same 2,212 eligible Mohenjo-daro crops (of 2,875 before excluding Wells codes without a single Mahadevan mapping). 1,112 eligible crops map to Other. Using the same pool for both tasks isolates the split and label grouping from the choice of images.

| Split | Target labels | Train / val / test crops | Test crops from train seals | Test classes | Test targets seen in train | Frequency top 1 | CNN top 1 | CNN top 5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed seal split | Full Wells IDs | 1531 / 212 / 469 | 0.0% | 118 | 95.1% | 14.1% | 35.0% | 54.4% |
| Fixed seal split | 38 listed codes + Other | 1531 / 212 / 469 | 0.0% | 26 | 100.0% | 49.9% | 62.5% | 88.3% |
| Random crop split | Full Wells IDs | 1531 / 212 / 469 | 94.5% | 109 | 95.7% | 12.6% | 33.5% | 52.0% |
| Random crop split | 38 listed codes + Other | 1531 / 212 / 469 | 94.5% | 27 | 100.0% | 49.7% | 61.4% | 87.2% |

Random crop splits deliberately allow different positions on one seal to appear in train and test. Both splits use the same partition sizes. Frequency top 1 always picks the most common training label. A change in these scores measures a protocol effect on our forced-cut crop pool. The published study used different images, class balance, cropping, and a different model, so matching its published accuracy is a separate question.

Render-pretrained backbone with fresh classification heads, 15 epochs, seed 0 and device cuda.
Use --init with a render-only v2 checkpoint to repeat the comparison with pretrained features. Separate fits per split keep crop-level test examples out of their respective training runs.

Top-1 rose from 35.0% to 62.5% on the fixed seal split when the labels were collapsed, with a 49.9% frequency baseline in the collapsed task. The random crop split shared seals with training for 94.5% of test crops and did not improve top-1 in this run. The audit measures protocol sensitivity on our forced-cut pool. Differences in label balance, source images, annotations, and architecture remain unresolved in any comparison with the published score.
