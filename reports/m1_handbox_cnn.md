# M1 · Training the sign CNN on hand-drawn box crops

Same start (v2 `wells_all.pt`), same extra fine-tune; only the real training crops differ. Scored on test and Harappa seals that have reviewed boxes, with each kind of crop. Sign count given in every row (crops come from known boxes or forced cuts). Seed 0, CPU, 15 epochs, best validation epoch kept.

Training crops: forced cuts 2279, hand boxes 2240 (from 459 training seals).

| Model | Test, forced-cut crops | Test, hand-box crops | Harappa, forced-cut crops | Harappa, hand-box crops |
| --- | ---: | ---: | ---: | ---: |
| v2 CNN (forced cuts), as released | 31.0% / 52.3% (n=686) | 52.8% / 77.7% (n=686) | 30.6% / 51.1% (n=805) | 47.8% / 75.2% (n=805) |
| +15 epochs on forced cuts (control) | 31.6% / 53.2% (n=686) | 55.7% / 77.8% (n=686) | 32.7% / 52.2% (n=805) | 50.3% / 75.9% (n=805) |
| +15 epochs on hand-box crops | 28.6% / 44.8% (n=686) | 73.9% / 90.1% (n=686) | 27.5% / 44.7% (n=805) | 71.3% / 91.2% (n=805) |

Cells: top-1 / top-5. Hand-box crops of test and Harappa seals are used only for scoring.

Reproduce: `python scripts/m1_handbox_cnn.py` after `scripts/make_real_crops_v2.py` and with `wells_all.pt` from the `wells-v2-cnn-pr60` release in `data/processed/models/`.
