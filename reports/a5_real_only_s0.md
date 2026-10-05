# A5: sign CNN on real seal crops

Crops: crops.npz. Test set: 352 sign crops from 70 verified seals (data/raw/cisi_seals/test_seals.txt), impression photos, cut with the known sign count only. Doubtful and illegible signs excluded. Never used for training or for choosing settings.

Training: 0 synthetic crops (none; all 417 signs), 2232 real crops from 226 seals (impression + flipped seal photo). 0 pretrain + 80 fine-tune epochs on cpu, 26 min, seed 0; small network, freeze 0, weight decay 0.0001, fine-tune lr 0.0005, augmentation 1.5. Kept fine-tune epoch 50, chosen on 173 validation crops (top-1 23.7%, top-5 42.8%).

| Method | Top-1 | Top-5 |
| --- | --- | --- |
| Template matching (A2) | 0.3% | 1.4% |
| CNN before fine-tuning (random weights) | 0.0% | 0.9% |
| CNN, real crops only | 22.4% | 38.4% |
| Chance | 0.2% | 1.2% |
| Always guess the commonest training signs (no image used) | 10.2% | 24.4% |

With 352 test crops, a result has a margin of error of about 4% either way (95%).

Fine-tuned CNN by how common the sign is in the real training crops:

| Sign seen in training | Test crops | Top-1 | Top-5 |
| --- | --- | --- | --- |
| never in real training crops | 30 | 0.0% | 0.0% |
| 1-4 real training crops | 39 | 0.0% | 0.0% |
| 5-19 | 73 | 2.7% | 12.3% |
| 20 or more | 210 | 36.7% | 60.0% |
