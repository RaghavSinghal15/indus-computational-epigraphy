# M1 · Hand-box CNN on Dixit et al.'s 40-sign label set

Dixit et al. (2025) report 94.5% on a reserved, balanced set of 200 hand-cropped signs (38 named Mahadevan codes + Other); their cross-validation mean is 85.28%. `reports/m1_dixit_protocol.md` scored our forced-cut CNN on the same label set: 62.5%. Here the hand-box CNN (`reports/m1_handbox_cnn.md`), on hand-box crops of our fixed test seals, is scored on that label set. It was trained on full Wells IDs; its first guess is collapsed afterwards, so it was never tuned for the 39-class task.

| Test crops | Crops | Collapsed top-1 | Always 'Other' | Named-sign crops | Accuracy on them | Balanced over named classes | Full Wells top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mohenjo-daro test seals (as in the audit) | 469 | 90.4% | 49.9% | 235 | 87.2% | 78.3% (25) | 77.8% |
| All test seals | 526 | 89.2% | 50.0% | 263 | 86.3% | 75.2% (25) | 76.6% |

Sanity check: the Mohenjo-daro pool is the same 469 crops with the same 49.9% 'Other' share as the fixed-split row of the protocol audit.

With hand-drawn boxes our reader gets 90.4% on the paper's label set, against 94.5% published and 62.5% for our forced cuts. Half our test crops are 'Other'; on the named signs alone it gets 87.2%, and 78.3% averaged per class (closest to a balanced test set). The remaining gap may come from different images, class balance and their crop-level split; ours is by seal, so no test seal was seen in training.
