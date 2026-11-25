# E1 · Error propagation at the hand-box CNN's accuracy (measured confusions)

Rerun of `e1_cnn_confusions.md` with the CNN trained on hand-box crops (`wells_handbox.pt`, `m1_handbox_cnn.md`), scored on hand-box crops of the test seals. Predictions: `scripts/e1_handbox_predictions.py`.

Hand-box CNN top-1 was 507/686 (73.9%) on the supplied predictions for 141 seals from the fixed test split. At least one error was observed for 112 distinct truth signs.

| Error draw | Realized sign accuracy | Fallback error share | Jar last | Entropy (bits) | NB real vs shuffle | Masked top 5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frequency-weighted | 73.9% | — | 27.9% / 10.8% | 4.395 / 4.812 | 70.3% / 50.0% | 34.3% / 21.1% |
| Measured CNN confusions | 73.9% | 33.6% | 26.7% / 10.0% | 4.246 / 4.814 | 73.7% / 50.0% | 41.6% / 21.3% |

Five draws on the 1,837-text C1 benchmark. Each metric cell shows noisy readings / within-text shuffle of the same noisy readings, with the NB chance control fixed at 50%. Both simulations share the target global accuracy. Only the distribution of wrong signs changes.

For a sign without measured wrong examples, draw from pooled measured errors, omitting the true sign. Frequency-weighted wrong labels are the final fallback if the pool is empty after omission. This conditions errors on the true sign and does not model errors correlated across positions, missing signs, or extra signs. Sparse confusion rows and coverage affect the comparison.

**Reading.** At 31% accuracy (v2 CNN on forced cuts) the order signals were barely above their shuffled controls (masked top 5: 43.2% vs 41.0%). At 73.9% every signal is clearly separated: real vs shuffled 73.7% (clean corpus 82.7%), jar-final 26.7% vs 10.0%, next-sign entropy 4.25 vs 4.81 bits, masked top 5 41.6% vs 21.3%. A third of the errors here are drawn from the pooled error list because those signs had no measured mistakes on the test set. The simulation models substitutions only; the fully automatic detector pipeline also drops and inserts signs (40.8% sign error rate, `m1_sign_detector.md`).
