# A7: reading seals with no labels (sign count unknown), GPU model

Model: cnn_equal.pt (small CNN, single renders, aug 1.5; GPU run of Oct 5). Test: 70 seals, 358 signs (impression photos). Count chosen as best candidate score + 1.0 x n + 2.0 x log P(n); weights chosen on 32 validation seals (count right on 9/32). Sign statistics: pairs + ends, lambda 0.5 (chosen on validation in A6).

| Pipeline | Count exactly right | Count within 1 | Character error rate |
| --- | --- | --- | --- |
| Old: A1 boxes + A2 template matching | 10% | 36% | 1.48 |
| New: CNN-guided cutting + CNN | 19% | 51% | 0.84 |
| New + sign statistics | 19% | 51% | 0.83 |

Character error rate = edit distance / true length (0 = perfect). Chosen count minus true count: -7: 1, -5: 1, -4: 1, -3: 5, -2: 10, -1: 16, +0: 13, +1: 7, +2: 12, +3: 3, +5: 1
