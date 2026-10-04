"""A5: train the sign CNN and measure it on the fixed test seals.

Usage:
    python scripts/make_carved_synthetic.py      # once: carved renders (about 1 minute)
    python scripts/make_real_crops.py            # once: real crops + fixed split
    python scripts/train_sign_cnn.py             # GPU if available, else CPU
    python scripts/train_sign_cnn.py --quick     # 1+1 epochs on a subset, to check it runs

Stage 1 (pretrain): carved renders only, all 417 signs.
Stage 2 (fine-tune): every batch is half real training crops, half renders
(so signs that never occur in the real training seals are not forgotten).
No model selection on the test set: fixed epochs, test measured once at the end.

Writes the model to data/processed/models/sign_cnn.pt and the numbers to
reports/a5_cnn.md: top-1 / top-5 on test crops, overall and by how often the
sign occurs in the real training crops, next to template matching (A2) on the
same crops and to the pretrained-only model.
"""

import argparse
import time
from collections import Counter

import cv2
import numpy as np
import torch
import torch.nn as nn

from indus.classify import classify_crop
from indus.cnn import SignCNN, augment, load_carved_renders, predict, to_input
from indus.data import REPO_ROOT

PROCESSED = REPO_ROOT / "data" / "processed"
MODEL_OUT = PROCESSED / "models" / "sign_cnn.pt"
REPORT = REPO_ROOT / "reports" / "a5_cnn.md"
SEED = 0
BATCH = 128
BUCKETS = [(0, 0, "never in real training crops"), (1, 4, "1-4 real training crops"),
           (5, 19, "5-19"), (20, 10**9, "20 or more")]


def run_epochs(model, sample_batch, steps_per_epoch, epochs, lr, device, name):
    """Train with AdamW and a cosine learning-rate schedule (high at first, smoothly to 0)."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps_per_epoch)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)   # softens targets a little: less over-confident
    for epoch in range(epochs):
        model.train()
        t0, total, correct, seen = time.time(), 0.0, 0, 0
        for _ in range(steps_per_epoch):
            x, y = sample_batch()
            x, y = augment(x.to(device)), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad()          # clear gradients from the previous step
            loss.backward()          # backpropagation: gradient of the loss for every weight
            opt.step()               # move every weight a small step downhill
            sched.step()
            total += loss.item() * len(y); seen += len(y)
            correct += (logits.argmax(1) == y).sum().item()
        print(f"  {name} epoch {epoch + 1}/{epochs}: loss {total / seen:.3f}, "
              f"train acc {correct / seen:.1%}, {time.time() - t0:.0f}s")


def sampler(x, y, gen, size=BATCH):
    """Function returning a random batch (with replacement) from arrays x, y."""
    xt, yt = torch.from_numpy(x), torch.from_numpy(y - 1)      # sign number -> class index

    def sample():
        i = torch.randint(len(xt), (size,), generator=gen)
        return xt[i], yt[i]
    return sample


def scores(top_ids, labels):
    return (top_ids[:, 0] == labels), (top_ids == labels[:, None]).any(axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrain-epochs", type=int, default=30)
    ap.add_argument("--finetune-epochs", type=int, default=40)
    ap.add_argument("--crops", default="crops.npz", help="file in data/processed/real_crops/")
    ap.add_argument("--init", help="start from this trained model and skip pretraining")
    ap.add_argument("--quick", action="store_true", help="tiny run to check everything works")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.quick:
        args.pretrain_epochs = args.finetune_epochs = 1

    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)
    device = torch.device(args.device)
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    syn_img, syn_lab = load_carved_renders(PROCESSED / "synthetic_carved")
    real = np.load(PROCESSED / "real_crops" / args.crops)
    train = real["split"] == "train"
    test = ~train
    syn_x, real_x = to_input(syn_img), to_input(real["images"])
    tr_x, tr_y = real_x[train], real["labels"][train]
    te_x, te_y = real_x[test], real["labels"][test]
    print(f"renders: {len(syn_x)}, real train crops: {len(tr_x)}, real test crops: {len(te_x)}")

    steps = len(syn_x) // BATCH
    if args.quick:
        steps = 20

    # ---- stage 1: renders only
    model = SignCNN().to(device)
    t_start = time.time()
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device))
        args.pretrain_epochs = 0
    else:
        run_epochs(model, sampler(syn_x, syn_lab, gen), steps, args.pretrain_epochs, 2e-3, device, "pretrain")
    pre_ids, _ = predict(model, te_x)

    # ---- stage 2: half real, half renders in every batch
    real_s, syn_s = sampler(tr_x, tr_y, gen, BATCH // 2), sampler(syn_x, syn_lab, gen, BATCH // 2)

    def mixed():
        (a, b), (c, d) = real_s(), syn_s()
        return torch.cat([a, c]), torch.cat([b, d])
    run_epochs(model, mixed, max(1, len(tr_x) // (BATCH // 2)) if not args.quick else 20,
               args.finetune_epochs, 5e-4, device, "fine-tune")
    minutes = (time.time() - t_start) / 60
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_OUT)
    ids, _ = predict(model, te_x)

    # ---- A2 template matching on the very same test crops, for comparison
    a2_ids = np.array([[m.mahadevan_id for m in classify_crop(cv2.cvtColor(c, cv2.COLOR_GRAY2BGR))]
                       for c in real["images"][test]])

    # ---- report
    counts = Counter(tr_y.tolist())
    freq = np.array([counts.get(int(s), 0) for s in te_y])
    rows = []
    for name, top in [("Template matching (A2)", a2_ids), ("CNN before fine-tuning" + (" (= --init model)" if args.init else ", renders only"), pre_ids),
                      ("CNN, renders + real fine-tune", ids)]:
        t1, t5 = scores(top, te_y)
        rows.append(f"| {name} | {t1.mean():.1%} | {t5.mean():.1%} |")
    t1, t5 = scores(ids, te_y)
    bucket_rows = []
    for lo, hi, label in BUCKETS:
        m = (freq >= lo) & (freq <= hi)
        if m.any():
            bucket_rows.append(f"| {label} | {m.sum()} | {t1[m].mean():.1%} | {t5[m].mean():.1%} |")
    lines = [
        "# A5: sign CNN on real seal crops",
        "",
        f"Crops: {args.crops}" + (f", started from {args.init}" if args.init else "") + ". "
        f"Test set: {len(te_y)} sign crops from {len(set(real['seal'][test]))} verified seals "
        "(data/raw/cisi_seals/test_seals.txt), impression photos, cut with the known sign count "
        "(equal slices, real_crops.py; or CNN-aligned WITHOUT labels, align.py). Doubtful and illegible signs excluded. "
        "Never used for training or for choosing settings.",
        "",
        f"Training: {len(syn_x)} carved renders (all 417 signs), {len(tr_x)} real crops from "
        f"{len(set(real['seal'][train]))} seals (impression + flipped seal photo). "
        f"{args.pretrain_epochs} pretrain + {args.finetune_epochs} fine-tune epochs on {device}, "
        f"{minutes:.0f} min, seed {SEED}.",
        "",
        "| Method | Top-1 | Top-5 |",
        "| --- | --- | --- |",
        *rows,
        f"| Chance | {1 / 417:.1%} | {5 / 417:.1%} |",
        "| Always guess the commonest training signs (no image used) | "
        f"{np.mean(te_y == Counter(tr_y.tolist()).most_common(1)[0][0]):.1%} | "
        f"{np.isin(te_y, [s for s, _ in Counter(tr_y.tolist()).most_common(5)]).mean():.1%} |",
        "",
        "Fine-tuned CNN by how common the sign is in the real training crops:",
        "",
        "| Sign seen in training | Test crops | Top-1 | Top-5 |",
        "| --- | --- | --- | --- |",
        *bucket_rows,
    ]
    if args.quick:
        lines.insert(2, "**QUICK SMOKE RUN, numbers meaningless.**\n")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
