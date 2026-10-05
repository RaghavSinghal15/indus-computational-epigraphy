"""A5: train the sign CNN, choose settings on the validation seals, test once.

Usage:
    python scripts/make_carved_synthetic.py      # once: carved renders (about 1 minute)
    python scripts/make_synthetic_rows.py        # once: carved ROWS cut like real seals (about 3 minutes)
    python scripts/make_real_crops.py            # once: real crops + fixed val/test split
    python scripts/train_sign_cnn.py             # GPU if available, else CPU
    python scripts/train_sign_cnn.py --quick     # 1+1 epochs on a subset, to check it runs

Synthetic data (--synthetic): "single" = one centred render per image
(make_carved_synthetic.py), "rows" = crops cut from rendered rows with the
real-seal cutter (make_synthetic_rows.py), "both" = all of them. Default "single":
in the GPU sweep, adding rows LOWERED validation top-1 by about 5 points on all
9 settings (rows cut with the equal-slice cutter carry its labelling mistakes).

Stage 1 (pretrain): synthetic data only, all 417 signs. The result is saved to
data/processed/models/sign_cnn_pretrained_<synthetic>.pt, so fine-tuning
experiments can start from it with --init and skip this stage.
Stage 2 (fine-tune): every batch is half real training crops, half renders
(so signs that never occur in the real training seals are not forgotten).

Overfitting watch: after every epoch the model is scored on the VALIDATION
seals (never trained on). The epoch with the best validation top-1 is kept,
not the last one ("early stopping"). If training accuracy keeps rising while
validation accuracy stalls or falls, the model is memorising the training crops.

Settings worth trying (all chosen on validation, never on test):
  --freeze N        keep the first N of the 4 conv blocks fixed while fine-tuning
                    (fewer weights to fit 2,000 crops = less memorising)
  --weight-decay    pull weights towards 0 (default 1e-4; try 1e-3, 5e-3)
  --finetune-lr     step size while fine-tuning (default 5e-4)
  --aug             augmentation strength for fine-tuning (default 1.0; try 1.5)
  --no-test         don't score the test seals (use this for every experiment)

Writes the model to data/processed/models/sign_cnn.pt, one line per run to
reports/a5_runs.csv, and (unless --no-test) the numbers to reports/a5_cnn.md.
"""

import argparse
import copy
import csv
import time
from collections import Counter

import cv2
import numpy as np
import torch
import torch.nn as nn

from indus.classify import classify_crop
from indus.cnn import (ARCHS, augment, build_model, load_carved_renders, predict, read_checkpoint, save_model,
                       to_input)
from indus.data import REPO_ROOT

PROCESSED = REPO_ROOT / "data" / "processed"
MODELS = PROCESSED / "models"
REPORT = REPO_ROOT / "reports" / "a5_cnn.md"
RUNS = REPO_ROOT / "reports" / "a5_runs.csv"
SEED = 0
BATCH = 128
BUCKETS = [(0, 0, "never in real training crops"), (1, 4, "1-4 real training crops"),
           (5, 19, "5-19"), (20, 10**9, "20 or more")]


def scores(top_ids, labels):
    """Per-crop booleans: (top-1 right, right answer among the top 5)."""
    return (top_ids[:, 0] == labels), (top_ids == labels[:, None]).any(axis=1)


def run_epochs(model, sample_batch, steps_per_epoch, epochs, lr, device, name,
               val=None, weight_decay=1e-4, aug=1.0, frozen=()):
    """Train with AdamW and a one-cycle schedule (learning rate warms up, then
    falls smoothly to 0). If val = (x, y) is given, score it after every epoch
    and return the model at the epoch with the best val top-1.
    frozen: blocks of model.features whose weights must not change."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps_per_epoch)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)   # softens targets a little: less over-confident
    best = (-1.0, 0, None)                               # (val top-1, epoch, weights)
    history = []
    for epoch in range(epochs):
        model.train()
        for block in frozen:
            block.eval()         # frozen blocks: keep batch-norm statistics fixed too
        t0, total, correct, seen = time.time(), 0.0, 0, 0
        for _ in range(steps_per_epoch):
            x, y = sample_batch()
            x, y = augment(x.to(device), aug), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad()          # clear gradients from the previous step
            loss.backward()          # backpropagation: gradient of the loss for every weight
            opt.step()               # move every weight a small step downhill
            sched.step()
            total += loss.item() * len(y); seen += len(y)
            correct += (logits.argmax(1) == y).sum().item()
        line = f"  {name} epoch {epoch + 1}/{epochs}: loss {total / seen:.3f}, train acc {correct / seen:.1%}"
        if val is not None:
            t1, t5 = scores(predict(model, val[0])[0], val[1])
            history.append((epoch + 1, correct / seen, t1.mean(), t5.mean()))
            line += f", VAL top-1 {t1.mean():.1%} top-5 {t5.mean():.1%}"
            if t1.mean() > best[0]:
                best = (t1.mean(), epoch + 1, copy.deepcopy(model.state_dict()))
        print(line + f", {time.time() - t0:.0f}s")
    if best[2] is not None:
        model.load_state_dict(best[2])
        print(f"  {name}: keeping epoch {best[1]} (best validation top-1 {best[0]:.1%})")
    return best[1], history


def sampler(x, y, gen, size=BATCH):
    """Function returning a random batch (with replacement) from arrays x, y."""
    xt, yt = torch.from_numpy(x), torch.from_numpy(y - 1)      # sign number -> class index

    def sample():
        i = torch.randint(len(xt), (size,), generator=gen)
        return xt[i], yt[i]
    return sample


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=ARCHS, default="small",
                    help="small = our 4-block CNN from scratch; resnet18 = ImageNet-pretrained ResNet-18")
    ap.add_argument("--pretrain-epochs", type=int, default=30)
    ap.add_argument("--finetune-epochs", type=int, default=40)
    ap.add_argument("--crops", default="crops.npz", help="file in data/processed/real_crops/")
    ap.add_argument("--init", help="start from this trained model and skip pretraining")
    ap.add_argument("--freeze", type=int, default=0, help="conv blocks (0-4) kept fixed while fine-tuning")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--finetune-lr", type=float, default=5e-4)
    ap.add_argument("--aug", type=float, default=1.0, help="augmentation strength while fine-tuning")
    ap.add_argument("--synthetic", choices=["single", "rows", "both"], default="single")
    ap.add_argument("--out", default="sign_cnn.pt", help="file name for the trained model in data/processed/models/")
    ap.add_argument("--seed", type=int, default=SEED, help="random seed (fine-tuning order, augmentation)")
    ap.add_argument("--no-test", action="store_true", help="experiment: don't look at the test seals")
    ap.add_argument("--quick", action="store_true", help="tiny run to check everything works")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.quick:
        args.pretrain_epochs = args.finetune_epochs = 1

    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    device = torch.device(args.device)
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    parts = []
    if args.synthetic in ("single", "both"):
        parts.append(load_carved_renders(PROCESSED / "synthetic_carved"))
    if args.synthetic in ("rows", "both"):
        d = np.load(PROCESSED / "synthetic_rows_64.npz")
        parts.append((d["images"], d["labels"]))
    syn_img = np.concatenate([p[0] for p in parts])
    syn_lab = np.concatenate([p[1] for p in parts])
    real = np.load(PROCESSED / "real_crops" / args.crops)
    split = real["split"]
    syn_x, real_x = to_input(syn_img), to_input(real["images"])
    tr_x, tr_y = real_x[split == "train"], real["labels"][split == "train"]
    va_x, va_y = real_x[split == "val"], real["labels"][split == "val"]
    te_x, te_y = real_x[split == "test"], real["labels"][split == "test"]
    print(f"synthetic ({args.synthetic}): {len(syn_x)}, real crops: {len(tr_x)} train, {len(va_x)} val, {len(te_x)} test")

    # ---- stage 1: renders only
    t_start = time.time()
    if args.init:
        args.arch, weights = read_checkpoint(args.init, device)       # the init file decides the architecture
        model = build_model(args.arch, imagenet=False).to(device)
        model.load_state_dict(weights)
        args.pretrain_epochs = 0
    else:
        model = build_model(args.arch).to(device)
        steps = 20 if args.quick else len(syn_x) // BATCH
        run_epochs(model, sampler(syn_x, syn_lab, gen), steps, args.pretrain_epochs, 2e-3, device, "pretrain")
        MODELS.mkdir(parents=True, exist_ok=True)
        prefix = "sign_cnn" if args.arch == "small" else args.arch
        save_model(model, args.arch, MODELS / f"{prefix}_pretrained_{args.synthetic}.pt")
        print(f"pretrained model -> data/processed/models/{prefix}_pretrained_{args.synthetic}.pt")
    start_state = copy.deepcopy(model.state_dict())

    # ---- stage 2: half real, half renders in every batch
    frozen = list(model.features[:args.freeze])
    for block in frozen:
        for p in block.parameters():
            p.requires_grad = False      # no gradient: these weights stay as they are
    real_s, syn_s = sampler(tr_x, tr_y, gen, BATCH // 2), sampler(syn_x, syn_lab, gen, BATCH // 2)

    def mixed():
        (a, b), (c, d) = real_s(), syn_s()
        return torch.cat([a, c]), torch.cat([b, d])
    steps = 20 if args.quick else max(1, len(tr_x) // (BATCH // 2))
    best_epoch, history = run_epochs(model, mixed, steps, args.finetune_epochs, args.finetune_lr, device,
                                     "fine-tune", val=(va_x, va_y), weight_decay=args.weight_decay,
                                     aug=args.aug, frozen=frozen)
    minutes = (time.time() - t_start) / 60
    MODELS.mkdir(parents=True, exist_ok=True)
    save_model(model, args.arch, MODELS / args.out)
    print(f"model -> data/processed/models/{args.out}")

    # ---- one line per run, so experiments can be compared on VALIDATION only
    _, train_acc, val_t1, val_t5 = history[best_epoch - 1]
    if RUNS.exists() and "arch" not in RUNS.read_text().splitlines()[0]:
        # older column layout: keep it aside under the first free name (rename never overwrites on Windows)
        k = 1
        while RUNS.with_name(f"a5_runs_old{k}.csv").exists():
            k += 1
        RUNS.rename(RUNS.with_name(f"a5_runs_old{k}.csv"))
    new = not RUNS.exists()
    RUNS.parent.mkdir(parents=True, exist_ok=True)
    with open(RUNS, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["arch", "seed", "crops", "synthetic", "init", "freeze", "weight_decay", "finetune_lr", "aug", "finetune_epochs",
                        "best_epoch", "train_acc_at_best", "val_top1", "val_top5"])
        w.writerow([args.arch, args.seed, args.crops, args.synthetic, args.init or "", args.freeze, args.weight_decay, args.finetune_lr, args.aug,
                    args.finetune_epochs, best_epoch, f"{train_acc:.3f}", f"{val_t1:.3f}", f"{val_t5:.3f}"])
    print(f"validation: top-1 {val_t1:.1%}, top-5 {val_t5:.1%} at epoch {best_epoch} -> {RUNS.relative_to(REPO_ROOT)}")
    if args.no_test:
        return

    # ---- the test seals, once
    ids, _ = predict(model, te_x)
    start = build_model(args.arch, imagenet=False).to(device)
    start.load_state_dict(start_state)
    pre_ids, _ = predict(start, te_x)
    a2_ids = np.array([[m.mahadevan_id for m in classify_crop(cv2.cvtColor(c, cv2.COLOR_GRAY2BGR))]
                       for c in real["images"][split == "test"]])
    counts = Counter(tr_y.tolist())
    freq = np.array([counts.get(int(s), 0) for s in te_y])
    rows = []
    for name, top in [("Template matching (A2)", a2_ids),
                      ("CNN before fine-tuning" + (" (= --init model)" if args.init else ", renders only"), pre_ids),
                      ("CNN, renders + real fine-tune", ids)]:
        t1, t5 = scores(top, te_y)
        rows.append(f"| {name} | {t1.mean():.1%} | {t5.mean():.1%} |")
    common = [s for s, _ in counts.most_common(5)]
    t1, t5 = scores(ids, te_y)
    bucket_rows = []
    for lo, hi, label in BUCKETS:
        m = (freq >= lo) & (freq <= hi)
        if m.any():
            bucket_rows.append(f"| {label} | {m.sum()} | {t1[m].mean():.1%} | {t5[m].mean():.1%} |")
    settings = (f"{args.arch} network, freeze {args.freeze}, weight decay {args.weight_decay}, fine-tune lr {args.finetune_lr}, "
                f"augmentation {args.aug}")
    lines = [
        "# A5: sign CNN on real seal crops",
        "",
        f"Crops: {args.crops}" + (f", started from {args.init}" if args.init else "") + ". "
        f"Test set: {len(te_y)} sign crops from {len(set(real['seal'][split == 'test']))} verified seals "
        "(data/raw/cisi_seals/test_seals.txt), impression photos, cut with the known sign count only. "
        "Doubtful and illegible signs excluded. Never used for training or for choosing settings.",
        "",
        f"Training: {len(syn_x)} synthetic crops ({args.synthetic}; all 417 signs), {len(tr_x)} real crops from "
        f"{len(set(real['seal'][split == 'train']))} seals (impression + flipped seal photo). "
        f"{args.pretrain_epochs} pretrain + {args.finetune_epochs} fine-tune epochs on {device}, {minutes:.0f} min, "
        f"seed {args.seed}; {settings}. Kept fine-tune epoch {best_epoch}, chosen on {len(va_y)} validation crops "
        f"(top-1 {val_t1:.1%}, top-5 {val_t5:.1%}).",
        "",
        "| Method | Top-1 | Top-5 |",
        "| --- | --- | --- |",
        *rows,
        f"| Chance | {1 / 417:.1%} | {5 / 417:.1%} |",
        "| Always guess the commonest training signs (no image used) | "
        f"{np.mean(te_y == common[0]):.1%} | {np.isin(te_y, common).mean():.1%} |",
        "",
        f"With {len(te_y)} test crops, a result has a margin of error of about "
        f"{1.96 * np.sqrt(t1.mean() * (1 - t1.mean()) / len(te_y)):.0%} either way (95%).",
        "",
        "Fine-tuned CNN by how common the sign is in the real training crops:",
        "",
        "| Sign seen in training | Test crops | Top-1 | Top-5 |",
        "| --- | --- | --- | --- |",
        *bucket_rows,
    ]
    if args.quick:
        lines.insert(2, "**QUICK SMOKE RUN, numbers meaningless.**\n")
    REPORT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
