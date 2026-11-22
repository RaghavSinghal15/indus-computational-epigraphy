# M3 · Masked transformer vs n-grams

ICIT benchmark, reading order, fixed C1 split. 1,469 training texts and 368 held-out texts, with 1,693 masked positions (66 unseen target signs). Every model sees the same inscription with one sign removed.

| Model | Top 1 | Top 5 | Masked pseudo-perplexity | Known-target pseudo-perplexity |
| --- | ---: | ---: | ---: | ---: |
| Witten–Bell bigram | 35.0% | 59.8% | 30.32 | 22.66 |
| Interpolated trigram | 35.8% | 61.0% | 34.49 | 24.82 |
| Masked transformer | 27.6% | 49.1% | 45.61 | 36.14 |

The transformer changed top-1 by -7.3 percentage points and top-5 by -10.7 points versus the bigram. This is a single-seed result on one small split.

Ranking uses known sign IDs and breaks ties by ascending ID. An unseen target counts as a miss. Pseudo-perplexity exponentiates average single-mask cross entropy and normalizes over known signs plus an unknown bucket. The known-target column omits unseen targets. These are masked probabilities and are distinct from the left-to-right sequence perplexity in C1.

The bigram scores both neighbouring factors; the trigram scores up to three factors affected by the mask and interpolates with the bigram. Both estimate all counts on training texts.

The transformer uses two layers, width 64, four heads, 128 feed-forward units, dropout 0.15 and 133,822 parameters. It masks one random sign per training text per epoch, drops 3% of context signs to UNKNOWN to handle unseen inputs, and uses AdamW, batch size 64, learning rate 0.001 and label smoothing 0.05.

Training selected 50 epochs from up to 50 using 147 validation texts drawn from C1's training partition, minimizing known-target pseudo-perplexity. The final fit starts from scratch on all training texts. Seed 0, device cpu, PyTorch 2.15.0+cpu.

Reproduce with `python scripts/c2_masked_transformer.py --epochs 50 --seed 0 --threads 2 --device cpu`.
