# D1 and E2 on two non-language symbol corpora

Controls are Pennsylvania German barn-star motifs and the annotated deity symbols on Mesopotamian kudurrus. Their XML files include Apache 2.0 licence notices. [Sproat’s published archive](https://richardsproat.com/data/non-linguistic-symbols/)

Archive SHA-256: `fb50ff5acb5221ba75e134cc7a36619e1d990b5680ad8681bcde5c447b6b8f73`.

One sequence per nonempty XML docText. Symbols inside nested groups are flattened in document order by title. The analysis retains uncertain symbols as recorded, then removes single-symbol entries and exact duplicate sequences ahead of the split.

The pipeline uses the repository D1 classifiers and permutation test and E2 generators and classifiers. It omits the Indus-specific jar statistic. Bigram generation uses the observed maximum sequence length for each corpus or training partition. The 80/20 split uses seed 0. E2 descriptive simulations use the full control corpus and the held-out classifier experiment fits its generators on training records only.

## Pennsylvania German barn stars

The archive has 310 nonempty records, 963 symbols and 32 distinct labels. The filtered set has 103 distinct sequences (82 train, 21 test).

### D1 · Observed order versus shuffles

Held-out naïve Bayes accuracy was **50.0%** using individual signs and **50.0%** using adjacent pairs. Conditional entropy was 2.515 bits, with mean 2.457 over 1,000 within-sequence shuffles (one-sided p=0.9510).

### E2 · Corpus statistics

Generated columns show the mean and range over 20 corpora.

| Statistic | Observed | Unigram | Bigram |
| --- | ---: | ---: | ---: |
| Distinct signs | 32 | 29 (24–32) | 27 (21–31) |
| Share of signs used only once | 15.6% | 16.8% (6.5%–30.0%) | 12.7% (4.3%–25.9%) |
| Mean text length | 4.12 | 4.12 (4.12–4.12) | 4.45 (4.00–5.19) |
| Zipf slope (top 100 signs) | -1.49 | -1.49 (-1.62–-1.36) | -1.52 (-1.70–-1.34) |
| Next-sign entropy, bits (B4 estimator) | 1.86 | 3.07 (2.87–3.19) | 1.71 (1.51–1.91) |

### E2 · Held-out classifier

Five draws per row, with 50% chance accuracy.

| Comparison | Pairs | Triples |
| --- | ---: | ---: |
| Observed vs unigram | 58.1% | 65.2% |
| Observed vs bigram | 54.3% | 58.1% |
| Bigram vs bigram | 51.9% | 45.7% |

## Mesopotamian deity symbols

The archive has 69 nonempty records, 939 symbols and 64 distinct labels. The filtered set has 65 distinct sequences (52 train, 13 test).

### D1 · Observed order versus shuffles

Held-out naïve Bayes accuracy was **50.0%** using individual signs and **84.6%** using adjacent pairs. Conditional entropy was 3.138 bits, with mean 3.771 over 1,000 within-sequence shuffles (one-sided p=0.0010).

### E2 · Corpus statistics

Generated columns show the mean and range over 20 corpora.

| Statistic | Observed | Unigram | Bigram |
| --- | ---: | ---: | ---: |
| Distinct signs | 64 | 56 (53–60) | 56 (52–61) |
| Share of signs used only once | 23.4% | 15.0% (8.5%–22.8%) | 17.6% (8.8%–25.0%) |
| Mean text length | 14.38 | 14.38 (14.38–14.38) | 13.03 (11.26–15.86) |
| Zipf slope (top 100 signs) | -1.44 | -1.38 (-1.47–-1.31) | -1.36 (-1.44–-1.29) |
| Next-sign entropy, bits (B4 estimator) | 3.05 | 3.72 (3.65–3.83) | 2.68 (2.59–2.80) |

### E2 · Held-out classifier

Five draws per row, with 50% chance accuracy.

| Comparison | Pairs | Triples |
| --- | ---: | ---: |
| Observed vs unigram | 82.3% | 80.0% |
| Observed vs bigram | 49.2% | 50.8% |
| Bigram vs bigram | 50.0% | 49.2% |

## Interpretation

The tests measure whether recorded symbol order differs from the specified null process. The deity-symbol sample exhibits order relative to within-record shuffles, illustrating that a motif system can show such a result. This is one split over small, curated corpora. XML order may reflect the spatial layout or cataloguing convention. These controls do not establish the linguistic status of the Indus script.

## Reproduce

Download corpora.zip from the [data page](https://richardsproat.com/data/non-linguistic-symbols/). Run from the repository root.

```bash
python scripts/m4_symbol_controls.py --archive /path/to/corpora.zip
```
