# M2–M4 on the full ICIT corpus and by site

Lipi revision `fc8813a52e183d19c441631dc26c33a4dccbea39`. Fetch the pinned CSV with `python scripts/fetch_lipi.py --corpus-only`, then run `python scripts/m2m4_full_corpus.py`.

The benchmark uses the original 1,837 texts. Lipi subsets keep complete, legible, single-line texts with known direction and at least two signs. Deduplication is by ordered Wells sequence, within each column after filtering. All sequences below are in reading order.

Exact sequence overlap: 1,694 benchmark texts occur in the full filtered corpus. 45 distinct sequences occur at both sites. The columns overlap and the site rows describe within-site fits, not transfer between sites.

| Metric | Benchmark | Full ICIT | Mohenjo-daro | Harappa |
| --- | ---: | ---: | ---: | ---: |
| B1 · Unique texts | 1,837 | 1,939 | 1,017 | 672 |
| B1 · Sign tokens / types | 8,359 / 568 | 8,807 / 585 | 4,918 / 450 | 2,699 / 349 |
| B1 · Singleton share (types) | 38.6% | 36.2% | 40.0% | 42.1% |
| B1 · Top 10 sign share (tokens) | 33.3% | 33.0% | 33.8% | 35.4% |
| B1 · Length min–max (mean) | 2–13 (4.55) | 2–13 (4.54) | 2–13 (4.84) | 2–13 (4.02) |
| B2 · Jar first / last | 0.16% / 35.22% | 0.15% / 34.91% | 0.00% / 39.63% | 0.15% / 29.02% |
| B3 · Adjacent pair tokens / types | 6,522 / 2,751 | 6,868 / 2,937 | 3,901 / 1,837 | 2,027 / 1,129 |
| B3 · Most frequent pair (count) | 861→2 (114) | 861→2 (116) | 861→2 (76) | 740→400 (69) |
| B3 · Pair tokens eligible for IM77 comparison | 58.3% | 57.9% | 57.1% | 62.9% |
| B3 · Eligible pair tokens also seen in IM77 | 90.5% | 88.5% | 93.0% | 87.8% |
| B4 · Next-sign entropy (reading / reverse, bits) | 3.308 / 3.823 | 3.373 / 3.875 | 3.015 / 3.461 | 2.557 / 3.238 |
| C1 · Train / test texts | 1,469 / 368 | 1,551 / 388 | 813 / 204 | 537 / 135 |
| C1 · Held-out perplexity (unigram / bigram) | 87.14 / 37.78 | 84.50 / 36.58 | 83.84 / 34.37 | 74.50 / 35.29 |
| C1 · unigram, masked top 1 / 5 / 10 | 11.3% / 23.7% / 32.1% | 10.9% / 24.0% / 34.9% | 11.3% / 23.4% / 33.7% | 9.9% / 25.0% / 32.7% |
| C1 · previous sign, masked top 1 / 5 / 10 | 25.0% / 44.9% / 53.8% | 25.4% / 46.3% / 54.6% | 27.0% / 46.2% / 51.8% | 25.4% / 44.9% / 51.3% |
| C1 · both neighbours, masked top 1 / 5 / 10 | 35.0% / 60.0% / 67.8% | 35.8% / 59.6% / 66.8% | 35.9% / 60.0% / 67.5% | 33.6% / 59.7% / 65.5% |
| D1 · Real vs shuffled (sign pairs) | 81.4% | 82.7% | 82.6% | 76.3% |
| D1 · Real vs shuffled (single signs) | 50.0% | 50.0% | 50.0% | 50.0% |
| D1 · jar last, real / shuffle mean (p) | 35.22% / 11.14% (0.0010) | 34.91% / 10.96% (0.0010) | 39.63% / 11.01% (0.0010) | 29.02% / 11.62% (0.0010) |
| D1 · entropy with boundaries (bits), real / shuffle mean (p) | 3.685 / 4.715 (0.0010) | 3.735 / 4.742 (0.0010) | 3.383 / 4.428 (0.0010) | 3.209 / 4.037 (0.0010) |
| E2 · Distinct signs (unigram) | 475 | 490 | 373 | 285 |
| E2 · Share of signs used only once (unigram) | 23.6% | 22.9% | 24.8% | 26.2% |
| E2 · Mean text length (unigram) | 4.55 | 4.54 | 4.84 | 4.02 |
| E2 · Zipf slope (top 100 signs) (unigram) | -0.93 | -0.92 | -0.91 | -0.91 |
| E2 · Next-sign entropy, bits (B4 estimator) (unigram) | 4.76 | 4.79 | 4.38 | 3.80 |
| E2 · Texts ending with the jar sign (unigram) | 11.1% | 10.7% | 10.5% | 12.1% |
| E2 · Distinct signs (Markov) | 476 | 493 | 374 | 291 |
| E2 · Share of signs used only once (Markov) | 23.2% | 23.0% | 26.3% | 25.4% |
| E2 · Mean text length (Markov) | 4.59 | 4.59 | 4.78 | 4.18 |
| E2 · Zipf slope (top 100 signs) (Markov) | -0.93 | -0.93 | -0.91 | -0.90 |
| E2 · Next-sign entropy, bits (B4 estimator) (Markov) | 3.00 | 3.06 | 2.69 | 2.23 |
| E2 · Texts ending with the jar sign (Markov) | 37.6% | 37.5% | 41.6% | 30.8% |
| E2 · real vs unigram, pairs / triples | 85.8% / 83.4% | 85.2% / 83.8% | 86.1% / 85.0% | 80.1% / 80.1% |
| E2 · real vs Markov, pairs / triples | 49.3% / 58.8% | 49.1% / 59.0% | 46.8% / 60.0% | 55.3% / 63.9% |
| E2 · Markov vs Markov, pairs / triples | 51.9% / 50.9% | 51.6% / 50.2% | 52.4% / 51.3% | 51.0% / 50.1% |

C1, D1 and E2 use the same seed-0 80/20 split procedure per corpus. C1 retains the published baseline script's tie handling and counts unknown signs as misses; its perplexity includes end markers.
D1 uses 1,000 within-text permutations (one-sided, plus-one correction). E2 descriptive rows average 20 draws per generator, with the Markov length bounds fitted to the corpus. E2 classifiers average five trials, with generated training and test samples drawn from training distributions only. They share samples across the pair and triple classifiers.

Compare site scores with their sample sizes and sign distributions in mind. The raw IM77 overlap is conditional on available, unambiguous mappings. Shuffled and generated controls measure departures from those specific nulls. The entropy difference under reversal follows from the pair marginals and gives no independent evidence of reading direction.
