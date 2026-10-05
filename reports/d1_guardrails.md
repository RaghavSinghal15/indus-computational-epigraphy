# D1 (M4 baseline): real texts vs the same signs in random order

Data: ICIT corpus, 1837 texts, reading order. Train/test split of texts as in C1 (1469 / 368); every text plus one within-text shuffle (seed 1).

## Naive Bayes: real or shuffled?

| Features | Accuracy on held-out texts (chance 50%) |
| --- | --- |
| pairs | 81.4% |
| single signs (control) | 50.0% |

The control has to be at chance: a shuffle keeps exactly the same signs, only their order changes. Sign pairs carry the order, so anything above 50% is order structure.

Pairs model by text length (texts of length 1 are identical to their shuffle):

| Length | Held-out texts | Accuracy |
| --- | --- | --- |
| 2 | 43 | 65.1% |
| 3-4 | 145 | 80.7% |
| 5-7 | 153 | 86.6% |
| 8+ | 27 | 81.5% |

## Permutation tests (1,000 within-text shuffles of the whole corpus)

| Statistic | Real corpus | Shuffled corpora (mean, 95% range) | p-value |
| --- | --- | --- | --- |
| Next-sign uncertainty H(next given current), bits | 3.685 | 4.715 (4.696 to 4.733) | 0.0010 |
| Texts ending with the jar sign | 35.2% | 11.1% (9.9% to 12.5%) | 0.0010 |

p = (1 + shuffles at least as extreme) / (1 + 1000); 0.001 is the smallest possible value, meaning no shuffle came close. Figure: reports/figures/d1_permutation.png.

Cross-check with B4: without the start/end markers the same estimator gives 3.3082 bits, B4's value (3.3082). With markers it is higher because it also counts the uncertainty about how texts begin and end.
