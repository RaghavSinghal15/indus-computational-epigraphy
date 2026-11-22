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

## Colab architecture comparison

Four bidirectional masked encoders ran with seeds 0, 1 and 2 on one Colab Tesla T4. The twelve trials took 254.2 seconds after corpus setup (PyTorch 2.14.1+cu130). The standalone runner downloads the [public PLOS supplement](https://doi.org/10.1371/journal.pone.0190735), verifies its archive and CSV hashes, and reconstructs the split above. The CSV matches the repository copy byte for byte.

For each seed and architecture, an inner validation split of 147 training texts chose the epoch by known-target masked pseudo-perplexity. The model then refit from scratch on all 1,469 training texts. The 368 test texts and all 1,693 masked positions were identical across trials. The same training recipe and epoch budget applied to each encoder. The GPU small model initializes each layer independently, so use its matched runs for the architecture comparison. The prior CPU result is recorded above.

Neural scores are means across seeds. The top-1 range reflects seed variation on this one split.

| Model | Parameters | Top 1 | Top-1 range | Top 5 | Pseudo-perplexity | Known-target pseudo-perplexity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Small, full attention (2 × 64) | 133,822 | 27.9% | 26.6%–28.6% | 49.8% | 45.51 | 35.94 |
| Wider and deeper, full attention (4 × 128) | 663,166 | 29.8% | 29.2%–30.5% | 52.0% | 41.75 | 33.13 |
| Shared block, full attention (4 passes × 128) | 265,726 | 29.9% | 29.1%–30.4% | 50.8% | 42.72 | 33.84 |
| Small, local attention (2 × 64) | 133,822 | 29.7% | 29.5%–29.9% | 51.2% | 42.62 | 33.75 |
| Witten–Bell bigram | — | 35.0% | — | 59.8% | 30.32 | 22.66 |
| Interpolated trigram | — | 35.8% | — | 61.0% | 34.49 | 24.82 |

The wider and deeper model improved mean top-1 by 1.9 percentage points and had the strongest neural top-5 and mean log-loss score. The shared-block model achieved the highest neural mean top-1 by about 0.08 percentage points over the larger model, using about 40% as many parameters. The small local model has the same parameter count as the small full-attention model and limits each layer to two neighbouring positions on either side. All three variants improved test top-1 over the small model on every seed. The n-grams still rank the held-out sign more accurately on this split.

The repeated block follows [ALBERT's weight-sharing idea](https://arxiv.org/abs/1909.11942). The short attention window tests a locality bias motivated by [sliding-window attention](https://arxiv.org/abs/2004.05150). All four encoders were trained from scratch on Wells sign IDs.

### Per-seed runs

| Model | Seed | Selected epoch | Top 1 | Top 5 | Pseudo-perplexity | Known-target pseudo-perplexity | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small | 0 | 50 | 28.65% | 50.38% | 45.33 | 35.69 | 28.7 |
| small | 1 | 50 | 28.29% | 50.03% | 44.66 | 35.51 | 20.7 |
| small | 2 | 45 | 26.64% | 48.91% | 46.54 | 36.62 | 20.1 |
| larger | 0 | 24 | 29.18% | 51.86% | 41.25 | 32.87 | 19.9 |
| larger | 1 | 22 | 29.77% | 51.80% | 40.90 | 32.23 | 17.8 |
| larger | 2 | 29 | 30.48% | 52.39% | 43.10 | 34.30 | 22.9 |
| shared | 0 | 30 | 30.18% | 51.21% | 43.97 | 34.80 | 22.1 |
| shared | 1 | 22 | 29.12% | 49.68% | 43.34 | 34.59 | 16.9 |
| shared | 2 | 29 | 30.36% | 51.39% | 40.84 | 32.15 | 22.4 |
| local | 0 | 50 | 29.89% | 50.80% | 43.15 | 33.86 | 21.0 |
| local | 1 | 50 | 29.71% | 51.33% | 42.05 | 33.54 | 21.1 |
| local | 2 | 43 | 29.47% | 51.39% | 42.65 | 33.85 | 20.6 |

Reproduce using the Colab CLI with `colab run --gpu T4 --timeout 900 scripts/c2_capacity_public.py`. The script emits a `RESULT_JSON` record with per-run metrics and device metadata.
