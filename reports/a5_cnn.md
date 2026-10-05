# A5: sign CNN on real seal crops

Test set: 352 sign crops from 70 verified seals (data/raw/cisi_seals/test_seals.txt), impression photos, cut with the known sign count (real_crops.py). Doubtful and illegible signs excluded. Never used for training or for choosing settings.

Training: 20850 carved renders (all 417 signs), 2580 real crops from 258 seals (impression + flipped seal photo). 8 pretrain + 15 fine-tune epochs on cpu, 42 min, seed 0.

| Method | Top-1 | Top-5 |
| --- | --- | --- |
| Template matching (A2) | 0.3% | 1.4% |
| CNN, renders only | 4.8% | 11.6% |
| CNN, renders + real fine-tune | 30.7% | 47.7% |
| Chance | 0.2% | 1.2% |
| Always guess the commonest training signs (no image used) | 10.2% | 25.6% |

Fine-tuned CNN by how common the sign is in the real training crops:

| Sign seen in training | Test crops | Top-1 | Top-5 |
| --- | --- | --- | --- |
| never in real training crops | 25 | 8.0% | 16.0% |
| 1-4 real training crops | 34 | 2.9% | 11.8% |
| 5-19 | 79 | 8.9% | 21.5% |
| 20 or more | 214 | 45.8% | 66.8% |
