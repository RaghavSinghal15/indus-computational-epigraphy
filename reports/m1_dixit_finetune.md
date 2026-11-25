# M1 · Aiming the hand-box CNN at Dixit et al.'s 40-sign task

Label set: the paper's 38 named Mahadevan codes + Other. Validation and test crops are hand-box crops of Mohenjo-daro seals in our fixed seal split (210 val, 469 test); training uses the 1712 eligible hand-box training crops from all sites. The method is chosen on validation; every test score is shown. Published: 94.5% (balanced 200-crop set, crops of the same seals can be in train and test), 85.28% cross-validation.

| Method | Validation | Test |
| --- | ---: | ---: |
| A first guess, collapsed | 85.2% | 90.4% |
| B group probabilities summed | 86.2% | 88.3% |
| A first guess, collapsed + TTA | 85.7% | 90.2% |
| B group probabilities summed + TTA | 87.6% | 87.4% |
| C fine-tuned 39-way head **(chosen on validation)** | 91.0% | 88.1% |
| C fine-tuned 39-way head + TTA | 90.0% | 89.1% |

Chosen on validation: C fine-tuned 39-way head, test 88.1%. Always guessing Other: 49.9% of test crops.
