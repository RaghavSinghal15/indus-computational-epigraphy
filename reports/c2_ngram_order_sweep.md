# M3 · How far to increase n-gram order?

On C1's ICIT benchmark, use the trigram to rank single-sign restorations. Use the bigram if minimizing masked log loss is the priority. The first drop in validation top-1 while training top-1 rises is at order four. More context beyond order seven has no material effect on rankings in this run.

## Method

Sweep orders 1 through 14, the longest inscription (13 physical signs) plus one boundary. The median inscription has four signs. Keep the fixed 368-text benchmark test partition aside while selecting orders by five-fold validation within the existing 1,469 training texts. Assign whole, distinct inscriptions to folds. Score all 6,666 validation sign positions once each. Training metrics score the same library of masks in the four folds in which each text was used for fitting.

Use recursive Witten–Bell interpolation with the same add-one unigram and boundary treatment as PR #55 for orders two and three. For a candidate sign, add the log probabilities of the affected factors on both sides of the blank, including an end boundary if affected. An unseen history falls back to its shorter suffix. The main sweep omits higher-order histories that merely add another start marker. The repeated-start check keeps those histories at all orders. Unseen target signs are misses for top-k and use the unknown bucket in masked log loss.

## Validation by order

| Order | Train top-1 | Validation top-1 | Validation top-5 | Train masked pseudo-perplexity | Validation masked pseudo-perplexity | Training histories seen once |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 10.67% | 10.67% | 23.99% | 116.16 | 125.13 | — |
| 2 | 56.10% | 34.65% | 57.92% | 4.76 | 32.26 | 39.2% |
| 3 | 86.10% | 36.15% | 58.18% | 1.55 | 37.49 | 69.9% |
| 4 | 92.01% | 34.98% | 57.98% | 1.27 | 39.07 | 86.6% |
| 5 | 92.94% | 35.13% | 58.03% | 1.22 | 38.67 | 94.0% |
| 6 | 93.03% | 35.27% | 58.01% | 1.21 | 38.32 | 98.0% |
| 7 | 93.04% | 35.25% | 58.01% | 1.21 | 38.28 | 99.1% |
| 8 | 93.04% | 35.25% | 58.01% | 1.21 | 38.30 | 99.5% |
| 9 | 93.04% | 35.25% | 58.01% | 1.21 | 38.31 | 99.5% |
| 10 | 93.04% | 35.25% | 58.01% | 1.21 | 38.32 | 99.3% |
| 11 | 93.04% | 35.25% | 58.01% | 1.21 | 38.34 | 99.3% |
| 12 | 93.04% | 35.25% | 58.01% | 1.21 | 38.34 | 100.0% |
| 13 | 93.04% | 35.25% | 58.01% | 1.21 | 38.34 | 100.0% |
| 14 | 93.04% | 35.25% | 58.01% | 1.21 | 38.34 | 100.0% |

Four-grams lowered validation top-1 and increased masked log loss on all five folds relative to trigrams. Singleton rates are means across folds, measured over distinct training histories at each order. At order four, 86.6% of those histories occurred once. Masked pseudo-perplexity is the exponential of mean masked log loss, so lower is better. The bigram led on this measure even though the trigram ranked more correct signs in its top one and top five.

| Fold | Top-1 change, 4 vs 3 (percentage points) | Masked log-loss change, 4 vs 3 |
| ---: | ---: | ---: |
| 1 | -1.74 | +0.059 |
| 2 | -1.09 | +0.025 |
| 3 | -1.14 | +0.074 |
| 4 | -1.34 | +0.033 |
| 5 | -0.59 | +0.019 |

Keeping repeated start markers at higher orders gave four-grams 34.80% validation top-1 and 57.77% top-5. Trigrams still led at 36.15% and 58.18%. The validation choice of order by each metric was the same in both runs.

## Existing held-out benchmark

Refit on all 1,469 training texts. Evaluate the selected orders and the pre-specified order-four comparison on 1,693 masks across the same 368 benchmark test texts, with 66 unseen target signs.

| Order | Top-1 | Top-5 | Masked pseudo-perplexity |
| ---: | ---: | ---: | ---: |
| 2 | 34.97% | 59.83% | 30.32 |
| 3 | 35.79% | 60.96% | 34.49 |
| 4 | 35.44% | 60.60% | 36.21 |

This is an empirical cutoff for this corpus, split protocol and smoothing. [Yadav et al.](https://doi.org/10.1371/journal.pone.0009506) tested orders one through five on the Mahadevan-derived EBUDS corpus and found held-out sequence perplexity gains flattened around four-grams. Their corpus, higher-order backoff and next-token evaluation differ from our masked task.

## Reproduce

```bash
python scripts/c2_ngram_order_sweep.py --output /tmp/c2_ngram_order_sweep.json
python scripts/c2_ngram_order_sweep.py --repeat-start --output /tmp/c2_ngram_repeat_start.json
```

The CPU runner needs NumPy and the local C1 CSV. It verifies its sparse scoring against direct candidate-sequence scoring before each run. During the audit, relative candidate scores for orders two and three matched the original PR #55 scorer within 1e-11 on all 1,693 benchmark masks.
