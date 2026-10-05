# A6: reading the whole seal (CNN + sign-pair statistics)

Model: data/processed/models/cnn_equal.pt. Cuts: equal (sign count known, labels not used). Test: 352 scorable signs on 70 seals. Settings chosen on 32 validation seals: both photos no, test-time augmentation off, pairs + ends, lambda 0.5. IM77 pairs of all validation and test texts removed before building the bigram model.

| Method | Top-1 | Top-5 |
| --- | --- | --- |
| CNN alone | 25.6% | 42.9% |
| CNN + sign statistics (pairs + ends, lambda 0.5) | 29.0% | 45.5% |

Validation, all settings tried:

| Both photos | Test-time augmentation | Statistics | Lambda | Top-1 | Top-5 |
| --- | --- | --- | --- | --- | --- |
| no | off | pairs | 0.0 | 28.3% | 46.8% |
| no | off | pairs | 0.1 | 27.7% | 46.8% |
| no | off | pairs | 0.2 | 27.7% | 48.6% |
| no | off | pairs | 0.3 | 28.3% | 48.6% |
| no | off | pairs | 0.5 | 28.3% | 48.0% |
| no | off | pairs | 0.75 | 27.7% | 46.8% |
| no | off | pairs | 1.0 | 28.3% | 46.2% |
| no | off | pairs + ends | 0.0 | 28.3% | 46.8% |
| no | off | pairs + ends | 0.1 | 27.7% | 47.4% |
| no | off | pairs + ends | 0.2 | 28.3% | 48.0% |
| no | off | pairs + ends | 0.3 | 30.1% | 49.1% |
| no | off | pairs + ends | 0.5 | 31.2% | 49.1% |
| no | off | pairs + ends | 0.75 | 30.6% | 48.0% |
| no | off | pairs + ends | 1.0 | 28.9% | 45.7% |
| no | on | pairs | 0.0 | 25.4% | 45.7% |
| no | on | pairs | 0.1 | 27.7% | 46.2% |
| no | on | pairs | 0.2 | 27.2% | 48.0% |
| no | on | pairs | 0.3 | 28.9% | 48.6% |
| no | on | pairs | 0.5 | 28.3% | 48.6% |
| no | on | pairs | 0.75 | 26.6% | 46.8% |
| no | on | pairs | 1.0 | 26.0% | 44.5% |
| no | on | pairs + ends | 0.0 | 25.4% | 45.7% |
| no | on | pairs + ends | 0.1 | 27.7% | 45.7% |
| no | on | pairs + ends | 0.2 | 27.2% | 48.6% |
| no | on | pairs + ends | 0.3 | 29.5% | 49.1% |
| no | on | pairs + ends | 0.5 | 29.5% | 49.1% |
| no | on | pairs + ends | 0.75 | 28.3% | 48.0% |
| no | on | pairs + ends | 1.0 | 27.2% | 47.4% |
| yes | off | pairs | 0.0 | 25.4% | 42.8% |
| yes | off | pairs | 0.1 | 24.9% | 42.8% |
| yes | off | pairs | 0.2 | 23.1% | 43.9% |
| yes | off | pairs | 0.3 | 23.7% | 44.5% |
| yes | off | pairs | 0.5 | 24.9% | 42.8% |
| yes | off | pairs | 0.75 | 23.7% | 42.2% |
| yes | off | pairs | 1.0 | 22.5% | 42.2% |
| yes | off | pairs + ends | 0.0 | 25.4% | 42.8% |
| yes | off | pairs + ends | 0.1 | 26.6% | 43.4% |
| yes | off | pairs + ends | 0.2 | 23.7% | 43.9% |
| yes | off | pairs + ends | 0.3 | 24.3% | 45.1% |
| yes | off | pairs + ends | 0.5 | 24.3% | 45.7% |
| yes | off | pairs + ends | 0.75 | 22.5% | 45.1% |
| yes | off | pairs + ends | 1.0 | 20.8% | 45.7% |
| yes | on | pairs | 0.0 | 26.0% | 42.2% |
| yes | on | pairs | 0.1 | 25.4% | 41.6% |
| yes | on | pairs | 0.2 | 24.3% | 43.4% |
| yes | on | pairs | 0.3 | 23.7% | 44.5% |
| yes | on | pairs | 0.5 | 24.3% | 43.9% |
| yes | on | pairs | 0.75 | 25.4% | 42.2% |
| yes | on | pairs | 1.0 | 23.7% | 43.4% |
| yes | on | pairs + ends | 0.0 | 26.0% | 42.2% |
| yes | on | pairs + ends | 0.1 | 26.0% | 42.8% |
| yes | on | pairs + ends | 0.2 | 25.4% | 43.9% |
| yes | on | pairs + ends | 0.3 | 25.4% | 45.7% |
| yes | on | pairs + ends | 0.5 | 24.9% | 46.2% |
| yes | on | pairs + ends | 0.75 | 23.1% | 46.2% |
| yes | on | pairs + ends | 1.0 | 22.0% | 46.8% |
