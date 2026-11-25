# M1 · Mohenjo-daro to Harappa

Checkpoint SHA-256: `680c5be73d12ef1e07d14e97bfc9f1959b51db6698c1d23d01eccb4a638a555d`. Training provenance lists 401 Mohenjo-daro training seals.

| Site | Method | Seals | Signs | Seen in source training | Top 1 | Top 5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Mohenjo-daro | Training frequency | 125 | 612 | 93.3% | 10.8% | 25.0% |
| Mohenjo-daro | v2 CNN | 125 | 612 | 93.3% | 29.9% | 51.8% |
| Harappa | Training frequency | 181 | 866 | 92.0% | 11.2% | 23.2% |
| Harappa | v2 CNN | 181 | 866 | 92.0% | 28.9% | 48.8% |

Known sign count, forced-cut impressions, Wells categories, fixed object splits. The training frequency baseline ranks signs using Mohenjo-daro training crops only. Harappa is excluded from training and model selection. The cross-site gap can also reflect different sign frequencies or recognition difficulty. CRNN requires a separate whole-band evaluation.

The two site rows use the same checkpoint and known sign counts. Pretraining used renders of the full Wells vocabulary. The site restriction applies to real training and validation seals. The gap is 1.0 percentage point on top-1 in this single-seed experiment. It does not establish equivalence between sites.
