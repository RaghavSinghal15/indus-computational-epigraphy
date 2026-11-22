# M1 · Mohenjo-daro to Harappa

Frequency baseline only. To score the CNN, supply the source-site-only #30 checkpoint with class order and split provenance.

| Site | Method | Seals | Signs | Seen in source training | Top 1 | Top 5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Mohenjo-daro | Training frequency | 125 | 612 | 93.3% | 10.8% | 25.0% |
| Harappa | Training frequency | 181 | 866 | 92.0% | 11.2% | 23.2% |

Known sign count, forced-cut impressions, Wells categories, fixed object splits. The training frequency baseline ranks signs using Mohenjo-daro training crops only. Harappa is excluded from training and model selection. The cross-site gap can also reflect different sign frequencies or recognition difficulty. CRNN requires a separate whole-band evaluation.
