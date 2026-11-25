# M1 · Single-class sign detector (#36)

YOLOv8n, one class (`sign`), trained at 320 px for 20 epochs on CPU (seed 0) on the reviewed boxes of 456 training seals plus 1,000 composites (`data/processed/composites/manifest_1000.jsonl`, training seals only); validation on 65 seals (val precision 90.4%, recall 87.7%, AP50 95.4%). Test seals were used only here.

141 hand-boxed seals from the fixed v2 test split. Detector checkpoint `sign_detector_yolov8n.pt`.

| Box precision | Box recall | AP50 | AP50–95 | Count exact | Count MAE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 91.5% | 87.9% | 94.0% | 58.2% | 57.4% | 0.57 |

Box metrics use Ultralytics' test validator. Count metrics use a fixed score threshold of 0.25.

Detect then classify with `wells_handbox.pt`: sign error rate 40.8%, exact sequence 16.3%. Boxes are sorted right to left before classification.

| Detect, then classify with | Sign error rate ↓ | Exact sequence ↑ | Sign count given? |
| --- | ---: | ---: | --- |
| `wells_all.pt` (v2 CNN, trained on forced cuts) | 59.2% | 7.8% | no |
| `wells_handbox.pt` (trained on hand-box crops, `m1_handbox_cnn.md`) | **40.8%** | **16.3%** | no |

For comparison on the same 141 test seals: forced cuts + v2 CNN with the true count 68.1% SER; CRNN + composites 54.4% (`m1_composites.md`). The detector finds the exact sign count on 57.4% of seals (label-free count search: 16.3%, CRNN: 29.1%).

Reproduce:
```bash
python scripts/m1_sign_detector.py prepare --composites data/processed/composites/manifest_1000.jsonl --dataset data/processed/sign_detector_1k
python scripts/m1_sign_detector.py train --dataset data/processed/sign_detector_1k --epochs 20 --imgsz 320
python scripts/m1_sign_detector.py evaluate <runs/.../best.pt> --dataset data/processed/sign_detector_1k --imgsz 320 --cnn-checkpoint data/processed/models/wells_handbox.pt
```
