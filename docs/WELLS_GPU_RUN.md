# Wells CNN GPU runbook

The Wells classifier unlocks the v2 recognition checks in #37, #40 and #41. It reuses the small A5 architecture, pretrains once on carved Wells renders, then starts two fine-tunes from that same checkpoint. The all-site model uses the fixed training and validation seals. The Mohenjo-daro model uses only that site for both partitions. Neither fine-tune uses Harappa or the test seals.

## Before booking the GPU

Use a checkout of this branch. Set up a Python environment with lab-compatible CUDA PyTorch. The data setup additionally needs NumPy, pandas, OpenCV, Pillow, fonttools and brotli. Install the repo with `python -m pip install -e .`. The whole workflow can run on a lab GPU or in a private notebook with a GPU runtime.

Build the two NPZ archives once on a CPU or before the GPU session:

```bash
python scripts/fetch_lipi.py
python scripts/make_wells_renders.py
python scripts/make_real_crops_v2.py
```

Expect 35,600 synthetic crops and 4,181 real crops from 855 seals when using the pinned data. Keep the Lipi images, font and derived archives outside Git. A private notebook can fetch the pinned source itself if the team uses an authorised cloud environment for those materials. Avoid adding image packs, tokens or model checkpoints to commits.

## First minutes on the GPU

```bash
nvidia-smi
python -u scripts/train_wells_cnn.py time
```

The timing command runs a complete render epoch and a complete all-site fine-tune epoch, including its validation pass. It never scores the held-out test and saves no weights. Record the GPU model, batch size and both epoch times. Repeat with `--batch 64` before the subcommand if the default batch of 128 exceeds memory. Use the same batch setting for the main run.

For a planning budget, multiply the render time by 8 and the fine-tune time by 30. The latter is conservative for the two fine-tunes, since the Mohenjo-daro subset is smaller. The #40 follow-up refits four configurations. Budget another 60 fine-tune epoch equivalents for four 15-epoch comparisons as a rough upper estimate, plus 25–40 minutes for setup, exports and retries. For example, render/fine-tune epochs of 2 min/30 sec put all seven training runs at about 61 minutes before buffer. Epochs of 5 min/60 sec put them at about 130 minutes before buffer. These are conditional planning examples, not measured GPU times. If the session is tight, prioritise pretraining and the two fine-tunes.

## Train and save each stage

```bash
python -u scripts/train_wells_cnn.py pretrain --epochs 8
python -u scripts/train_wells_cnn.py finetune --scope all --epochs 15
python -u scripts/train_wells_cnn.py finetune --scope mohenjo --epochs 15
python -u scripts/train_wells_cnn.py export data/processed/wells_runs/wells_all.pt --split test
```

Copy the contents of `data/processed/wells_runs/` to approved persistent storage between stages. The atomic stage files keep the completed work if a later step fails. A new session can start at fine-tune using `--init /path/to/wells_renders.pt`; an interrupted individual stage restarts. Fine-tuning keeps the best validation epoch. Use a fresh `--run-dir` before the subcommand for a separate run, such as a second seed.

## Use the outputs on the P1 branches

| Issue | Input and next command |
| --- | --- |
| #37 Harappa | On the #37 branch, run `python scripts/m1_harappa.py --crops /path/to/crops_v2.npz --checkpoint /path/to/wells_mohenjo.pt --device cuda`. It reports the comparable in-site test subset and all held-out Harappa seals. |
| #41 CNN errors | On the #41 branch, run `python scripts/e1_cnn_confusions.py /path/to/predictions_all_test.csv --crops /path/to/crops_v2.npz`. This part works on CPU once the predictions exist. |
| #40 crop protocol | On the #40 branch, run `python -u scripts/m1_dixit_protocol.py /path/to/crops_v2.npz --init /path/to/wells_renders.pt --epochs 15 --device cuda`. Each protocol needs its own fit, so use the render-only weights here. |
| #36 detector | The detector has a separate training job once the reviewed boxes and compositor outputs are available. Reuse `wells_all.pt` for its detect-then-classify evaluation. |

Use the same pinned Lipi source in each checkout so the Wells vocabulary is available. The exported test JSON includes top-1, top-5, and frequency bins. For the v1 comparison, report the actual shared seal coverage shown in that JSON; an intersection of seal IDs can still have different sign annotations. The CNN runner supplies the weights and diagnostics for #30. Evaluate context decoding after the all-site fit:

```bash
python scripts/eval_wells_context.py data/processed/wells_runs/wells_all.pt \
  --output reports/m1_v2.md
```

The decoder fits the ICIT bigram prior after excluding held-out seal IDs and their exact sequences, then selects its weight on validation. CPU inference is sufficient for this step. A strictly matched v1 comparison also needs v1 predictions for the overlapping seal IDs and a check that the labels align under the two sign inventories.
