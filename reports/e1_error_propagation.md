# E1: error propagation, which conclusions survive reading mistakes?

Corpus: 1837 ICIT texts. Each sign kept with the given probability, otherwise replaced by a different sign drawn by frequency. Means over 3 seeds. Each cell: the statistic on the noisy readings vs on the same noisy readings with their order shuffled within each text (no-order level). A conclusion survives while the two stay apart. Our M1 reads about 29% of signs right (test seals, count given, CNN + statistics).

| Statistic | 100% right | 60% | 30% (about our M1) | 0% |
| --- | --- | --- | --- | --- |
| Texts ending with the jar sign | 35.2% vs 11.2% | 24.2% vs 10.7% | 14.4% vs 9.9% | 7.1% vs 9.6% |
| Next-sign uncertainty (bits) | 3.68 vs 4.72 | 4.62 vs 4.85 | 4.84 vs 4.87 | 4.86 vs 4.86 |
| Real vs shuffled, Naive Bayes accuracy | 82.7% vs 50.0% | 64.5% vs 50.0% | 53.5% vs 50.0% | 49.1% vs 50.0% |
| Hidden sign in top 5 (both neighbours) | 60.0% vs 26.1% | 25.8% vs 19.0% | 17.8% vs 16.8% | 17.3% vs 17.3% |

Naive Bayes is itself a real-vs-shuffled test, so its no-order level is 50%.

Permutation tests at 29% accuracy (one noisy corpus, 200 shuffles of it):

| Statistic | Noisy readings | Shuffled (mean) | p |
| --- | --- | --- | --- |
| Texts ending with the jar sign | 15.4% | 9.9% | 0.005 |
| Next-sign uncertainty (bits) | 4.815 | 4.847 | 0.005 |

Figure: reports/figures/e1_error_propagation.png (accuracy falls from left to right).

## What this means

- Where signs sit survives: at our M1's accuracy the jar sign still ends noisy texts more often than chance (p = 0.005 over the whole corpus), though the gap shrinks from 24 to about 5 points.
- Which sign follows which mostly does not: the entropy gap falls from 1.04 bits to 0.03, the real-vs-shuffled classifier drops to about 54%, and masked prediction is back near its no-order level.
- Sequence statistics (M2 pairs, M3) need readings that are roughly 60% right or better before they say much; below that they describe the reader's mistakes more than the script.

Caution: the simulation draws mistakes independently and by sign frequency. Real CNN mistakes are correlated (similar-looking signs get confused), which may preserve some structure the simulation destroys, or invent structure that is not there. A check with real M1 readings needs more read seals.
