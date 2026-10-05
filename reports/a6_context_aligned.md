# A6: reading the whole seal (CNN + sign-pair statistics)

Model: data/processed/models/cnn_aligned.pt. Cuts: aligned (sign count known, labels not used). Test: 352 scorable signs on 70 seals. Settings chosen on 32 validation seals: both photos no, test-time augmentation on, pairs + ends, lambda 0.75. IM77 pairs of all validation and test texts removed before building the bigram model.

| Method | Top-1 | Top-5 |
| --- | --- | --- |
| CNN alone | 24.1% | 36.6% |
| CNN + test-time augmentation | 23.9% | 38.1% |
| CNN + sign statistics (pairs + ends, lambda 0.75) | 25.6% | 40.9% |

Validation, all settings tried:

| Both photos | Test-time augmentation | Statistics | Lambda | Top-1 | Top-5 |
| --- | --- | --- | --- | --- | --- |
| no | off | pairs | 0.0 | 29.5% | 39.9% |
| no | off | pairs | 0.1 | 29.5% | 41.6% |
| no | off | pairs | 0.2 | 29.5% | 43.4% |
| no | off | pairs | 0.3 | 30.1% | 43.9% |
| no | off | pairs | 0.5 | 30.1% | 46.2% |
| no | off | pairs | 0.75 | 30.6% | 48.6% |
| no | off | pairs | 1.0 | 30.6% | 50.3% |
| no | off | pairs + ends | 0.0 | 29.5% | 39.9% |
| no | off | pairs + ends | 0.1 | 29.5% | 41.6% |
| no | off | pairs + ends | 0.2 | 29.5% | 44.5% |
| no | off | pairs + ends | 0.3 | 30.1% | 45.7% |
| no | off | pairs + ends | 0.5 | 30.6% | 47.4% |
| no | off | pairs + ends | 0.75 | 30.1% | 49.7% |
| no | off | pairs + ends | 1.0 | 28.9% | 52.0% |
| no | on | pairs | 0.0 | 29.5% | 41.6% |
| no | on | pairs | 0.1 | 29.5% | 42.2% |
| no | on | pairs | 0.2 | 29.5% | 43.4% |
| no | on | pairs | 0.3 | 29.5% | 43.9% |
| no | on | pairs | 0.5 | 30.1% | 46.2% |
| no | on | pairs | 0.75 | 31.2% | 46.2% |
| no | on | pairs | 1.0 | 30.6% | 49.1% |
| no | on | pairs + ends | 0.0 | 29.5% | 41.6% |
| no | on | pairs + ends | 0.1 | 29.5% | 42.2% |
| no | on | pairs + ends | 0.2 | 29.5% | 45.1% |
| no | on | pairs + ends | 0.3 | 29.5% | 45.7% |
| no | on | pairs + ends | 0.5 | 29.5% | 46.2% |
| no | on | pairs + ends | 0.75 | 31.8% | 47.4% |
| no | on | pairs + ends | 1.0 | 27.7% | 50.9% |
| yes | off | pairs | 0.0 | 23.7% | 41.6% |
| yes | off | pairs | 0.1 | 24.3% | 42.8% |
| yes | off | pairs | 0.2 | 26.0% | 42.8% |
| yes | off | pairs | 0.3 | 26.0% | 43.4% |
| yes | off | pairs | 0.5 | 26.6% | 43.4% |
| yes | off | pairs | 0.75 | 27.2% | 43.9% |
| yes | off | pairs | 1.0 | 27.2% | 46.2% |
| yes | off | pairs + ends | 0.0 | 23.7% | 41.6% |
| yes | off | pairs + ends | 0.1 | 24.9% | 43.4% |
| yes | off | pairs + ends | 0.2 | 26.6% | 43.4% |
| yes | off | pairs + ends | 0.3 | 26.6% | 44.5% |
| yes | off | pairs + ends | 0.5 | 27.2% | 46.2% |
| yes | off | pairs + ends | 0.75 | 27.2% | 45.7% |
| yes | off | pairs + ends | 1.0 | 27.7% | 47.4% |
| yes | on | pairs | 0.0 | 23.7% | 41.6% |
| yes | on | pairs | 0.1 | 23.7% | 42.2% |
| yes | on | pairs | 0.2 | 23.1% | 42.2% |
| yes | on | pairs | 0.3 | 25.4% | 42.2% |
| yes | on | pairs | 0.5 | 27.2% | 45.7% |
| yes | on | pairs | 0.75 | 26.0% | 44.5% |
| yes | on | pairs | 1.0 | 26.6% | 44.5% |
| yes | on | pairs + ends | 0.0 | 23.7% | 41.6% |
| yes | on | pairs + ends | 0.1 | 24.9% | 42.2% |
| yes | on | pairs + ends | 0.2 | 25.4% | 43.4% |
| yes | on | pairs + ends | 0.3 | 26.0% | 45.7% |
| yes | on | pairs + ends | 0.5 | 27.7% | 46.8% |
| yes | on | pairs + ends | 0.75 | 28.3% | 46.8% |
| yes | on | pairs + ends | 1.0 | 27.2% | 47.4% |
