# C1 (M3 baseline): masked sign prediction with n-gram models

Data: ICIT corpus, 1837 texts in reading order (Wells numbers). Fixed split of texts, seed 0: 1469 for training, 368 held out (1693 signs, 66 of them never seen in training). Witten-Bell smoothing; start and end markers.

## Perplexity of held-out texts (lower is better)

| Model | Ours (ICIT, held out) | Published (Yadav et al. 2010, Mahadevan corpus) |
| --- | --- | --- |
| Unigram | 87.1 | about 68.8 |
| Bigram | 37.8 | about 26.7 |

Not the same corpus or sign list, so close but not identical numbers are expected.

## Hide one sign, predict it

| Model | Top-1 | Top-5 | Top-10 |
| --- | --- | --- | --- |
| Chance (568 signs) | 0.2% | 0.9% | 1.8% |
| Unigram (overall frequency) | 11.3% | 23.7% | 32.1% |
| Bigram, previous sign | 25.0% | 44.9% | 53.8% |
| Bigram, both neighbours | 35.0% | 60.0% | 67.8% |

## Direction check

Bigram perplexity reading right to left (the accepted direction): 37.8; read the other way: 37.9. Predicting a sign from the PREVIOUS one, top-1: 25.0% in reading order vs 23.6% reversed.

Caution: this is not a test of reading direction. With start and end markers, every sign occurrence is counted once as the left and once as the right member of a pair, so an unsmoothed bigram model gives its training corpus exactly the same total probability in both directions. Near-equal perplexities on held-out texts are therefore expected. Direction evidence comes from where signs sit (the jar sign ends texts, B2), not from bigram perplexity (B4 makes the same point with conditional entropy).
