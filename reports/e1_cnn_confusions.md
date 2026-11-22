# E1 · Error propagation with measured CNN confusions

Wells v2 CNN top-1 was 213/686 (31.0%) on the supplied predictions for 141 seals from the fixed test split. At least one error was observed for 167 distinct truth signs.

| Error draw | Realized sign accuracy | Fallback error share | Jar last | Entropy (bits) | NB real vs shuffle | Masked top 5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frequency-weighted | 31.0% | — | 15.6% / 10.3% | 4.807 / 4.855 | 53.8% / 50.0% | 19.3% / 18.1% |
| Measured CNN confusions | 31.0% | 15.5% | 25.9% / 17.6% | 4.280 / 4.424 | 61.0% / 50.0% | 43.2% / 41.0% |

Five draws on the 1,837-text C1 benchmark. Each metric cell shows noisy readings / within-text shuffle of the same noisy readings, with the NB chance control fixed at 50%. Both simulations share the target global accuracy. Only the distribution of wrong signs changes.

For a sign without measured wrong examples, draw from pooled measured errors, omitting the true sign. Frequency-weighted wrong labels are the final fallback if the pool is empty after omission. This conditions errors on the true sign and does not model errors correlated across positions, missing signs, or extra signs. Sparse confusion rows and coverage affect the comparison.

All 686 exported predictions came from the held-out v2 test split and were verified against the fixed crop labels. The 11.0-point Naive Bayes margin over chance is a result inside this substitution simulation. It does not establish reliable transcription from images or model missing, extra, and jointly mis-segmented signs.
