"""Train and evaluate the whole-band Wells CRNN described by issue #33."""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import time

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn

from indus.cnn import SignCNN, predict, to_input
from indus.crnn import HORIZONTAL_STRIDE, SealCRNN, collapse_ctc, required_frames, sequence_metrics
from indus.data import LIPI_COMMIT, REPO_ROOT
from indus.pipeline import find_text_row
from indus.real_crops import sign_crops
from indus.segment import find_sign_boxes, load_image, trim_caption
from indus.wells import SPLIT_FILE, load_v2_seals, wells_codes

HEIGHT = 64
MIN_WIDTH = 192
MAX_WIDTH = 768
OUT = REPO_ROOT / "reports" / "m1_crnn.md"


@dataclass
class Band:
    name: str
    pixels: np.ndarray
    labels: tuple[int, ...]


def normalise_band(image):
    """Keep the aspect ratio where possible and pad short bands with a neutral margin."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    width = max(8, min(MAX_WIDTH, round(HEIGHT * gray.shape[1] / gray.shape[0])))
    cut = cv2.resize(gray, (width, HEIGHT), interpolation=cv2.INTER_AREA).astype(np.float32)
    cut = (cut - cut.mean()) / (cut.std() + 0.001)
    aligned = max(MIN_WIDTH, ((width + HORIZONTAL_STRIDE - 1) // HORIZONTAL_STRIDE) * HORIZONTAL_STRIDE)
    left = (aligned - width) // 2
    out = np.zeros((HEIGHT, aligned), dtype=np.float32)
    out[:, left:left + width] = cut
    return out


def seal_band(image):
    """Locate the inscription band from pixels alone, with room below short detections."""
    bounds = find_text_row(image)
    h, w = image.shape[:2]
    margin = max(1, round(h * 0.04))
    y0, x0 = max(0, bounds.y0 - margin), max(0, bounds.x0 - margin)
    y1 = min(h, max(bounds.y1 + margin, round(h * 0.32)))
    x1 = min(w, bounds.x1 + margin)
    if y0 >= y1 or x0 >= x1:
        raise ValueError("Text band finder returned an empty region")
    return image[y0:y1, x0:x1]


def build_real(plan, seals, indices):
    """Load all fixed train, validation, test and Harappa seals."""
    location = {name: [] for name in ("train", "val", "test", "harappa")}
    for row in seals.itertuples():
        if row.cisi not in plan.index:
            raise ValueError(f"Unknown seal in data: {row.cisi}")
        image = trim_caption(load_image(row.image))
        pixels = normalise_band(seal_band(image))
        labels = tuple(indices[sign] for sign in row.signs_reading)
        location[plan.at[row.cisi, "split"]].append(Band(row.cisi, pixels, labels))
    if {item.name for group in location.values() for item in group} != set(plan.index):
        raise ValueError("The downloaded seal images do not cover the fixed split")
    return location


def build_composites(manifest, plan, indices):
    """Accept composite bands with explicit training-seal provenance."""
    if manifest is None:
        return []
    allowed = set(plan.index[plan["split"] == "train"])
    items = []
    with manifest.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            row = json.loads(line)
            sources = row["source_seals"]
            if not isinstance(sources, list) or not sources or not set(sources) <= allowed:
                raise ValueError(f"Composite {i} has missing or held-out source seals")
            image = load_image(manifest.parent / row["image"])
            labels = tuple(indices[sign] for sign in row["signs_reading"])
            items.append(Band(f"composite-{i}", normalise_band(image), labels))
    return items


def check_alignment(records):
    for entry in records:
        frames = entry.pixels.shape[1] // HORIZONTAL_STRIDE
        if not entry.labels or required_frames(entry.labels) > frames:
            raise ValueError(f"CTC target is too long for its image frames: {entry.name}")


def minibatches(records, size, rng=None):
    """Group similarly sized bands before padding, shuffle batches during training."""
    ordered = list(records)
    if rng is not None:
        rng.shuffle(ordered)
    ordered.sort(key=lambda item: item.pixels.shape[1] // 64)
    batches = [ordered[i:i + size] for i in range(0, len(ordered), size)]
    if rng is not None:
        rng.shuffle(batches)
    for batch in batches:
        widths = torch.tensor([item.pixels.shape[1] for item in batch])
        x = np.zeros((len(batch), 1, HEIGHT, int(widths.max())), dtype=np.float32)
        for i, item in enumerate(batch):
            x[i, 0, :, :item.pixels.shape[1]] = item.pixels
        targets = torch.tensor([label for item in batch for label in item.labels], dtype=torch.long)
        lengths = torch.tensor([len(item.labels) for item in batch])
        yield batch, torch.from_numpy(x), widths, targets, lengths


@torch.inference_mode()
def evaluate(model, records, size, device):
    model.eval()
    references, guesses = [], []
    for batch, x, widths, _, _ in minibatches(records, size):
        logits, input_lengths = model(x.to(device), widths)
        emissions = logits.argmax(dim=-1).cpu().tolist()
        for entry, tokens, width in zip(batch, emissions, input_lengths.tolist()):
            references.append(entry.labels)
            guesses.append(collapse_ctc(tokens[:width]))
    return sequence_metrics(references, guesses)


def load_cnn(path, codes, plan, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (checkpoint.get("arch") != "small" or checkpoint.get("training_phase") != "real_finetuned"
            or tuple(checkpoint.get("wells_codes", ())) != codes
            or checkpoint.get("lipi_commit") != LIPI_COMMIT or checkpoint.get("scope") != "all"):
        raise ValueError("Use the all-site real-fine-tuned Wells v2 checkpoint")
    for partition, key in (("train", "train_seals"), ("val", "val_seals")):
        expected = set(plan.index[plan["split"] == partition])
        if expected != set(checkpoint[key]):
            raise ValueError(f"CNN checkpoint does not match the fixed {partition} split")
    cnn = SignCNN(len(codes))
    cnn.load_state_dict(checkpoint["state_dict"])
    return cnn.to(device).eval()


def train(model, real, composites, validation, args):
    rng = random.Random(args.seed)
    opt = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": 0.0001},
        {"params": list(model.recurrent.parameters()) + list(model.classify.parameters()), "lr": 0.001},
    ], weight_decay=0.01)
    criterion = nn.CTCLoss()
    best, best_epoch, state = (float("inf"), 0.0), 0, None
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        sampled = rng.sample(composites, min(len(composites), len(real)))
        losses = []
        for _, images, widths, target, lengths in minibatches(real + sampled, args.batch, rng):
            x = images.to(args.device)
            # Photometric jitter preserves the order and identity of every target sign.
            gain = torch.rand((len(images), 1, 1, 1), device=args.device) * 0.25 + 0.875
            x = x * gain
            emit, steps = model(x, widths)
            loss = criterion(emit.log_softmax(-1).transpose(0, 1), target.to(args.device), steps, lengths)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite CTC loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            losses.append(float(loss.item()))
        metrics = evaluate(model, validation, args.batch, args.device)
        score = metrics["sign_error_rate"], -metrics["exact"]
        if score < best:
            best, best_epoch = score, epoch
            state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(f"Epoch {epoch:2d}: CTC loss {np.mean(losses):.3f}, validation sign error "
              f"{metrics['sign_error_rate']:.1%}, exact {metrics['exact']:.1%}, "
              f"elapsed {time.perf_counter() - start:.1f}s", flush=True)
        if epoch - best_epoch >= args.patience:
            break
        if time.perf_counter() - start >= args.train_budget_minutes * 60:
            break
    model.load_state_dict(state)
    return best_epoch, epoch, time.perf_counter() - start


def cnn_baselines(cnn, frames, indices, codes, test_ids):
    """Compare with the same crop CNN given the true count or a pixel-only estimate."""
    references, known, estimated = [], [], []
    code_to_index = np.array([indices[code] for code in codes])
    for seal in frames[frames.cisi.isin(test_ids)].itertuples():
        image = trim_caption(load_image(seal.image))
        truth = tuple(indices[sign] for sign in seal.signs_reading)
        references.append(truth)
        band = find_text_row(image).crop(image)
        estimate = len(find_sign_boxes(band))
        for count, destination in ((len(truth), known), (estimate, estimated)):
            if count == 0:
                destination.append([])
                continue
            crops, _, _ = sign_crops(image, count)
            top, _ = predict(cnn, to_input(np.stack(crops)), k=1)
            destination.append(code_to_index[top[:, 0] - 1].tolist())
    return sequence_metrics(references, known), sequence_metrics(references, estimated)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", type=Path)
    parser.add_argument("--composites", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=65)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-budget-minutes", type=int, default=25)
    parser.add_argument("--check-data", action="store_true", help="verify dimensions and splits without training")
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "data/processed/wells_crnn.pt")
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch, args.train_budget_minutes) < 1:
        parser.error("epochs, batch, patience and time budget must be positive")
    if not args.check_data and args.init is None:
        parser.error("--init is required for training")
    if not args.check_data and args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Use a GPU runtime for the experiment.")
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    plan = pd.read_csv(SPLIT_FILE).set_index("cisi")
    if plan.index.has_duplicates:
        raise ValueError("Duplicate seal IDs in split")
    codes = wells_codes()
    indices = {code: i for i, code in enumerate(codes, start=1)}
    seals = load_v2_seals()
    real = build_real(plan, seals, indices)
    composites = build_composites(args.composites, plan, indices)
    for part, records in real.items():
        print(f"{part}: {len(records)} seals, {sum(len(item.labels) for item in records)} signs", flush=True)
        check_alignment(records)
    check_alignment(composites)
    print(f"Composites: {len(composites)} training-only examples", flush=True)
    if args.check_data:
        return

    cnn = load_cnn(args.init, codes, plan, args.device)
    crnn = SealCRNN(len(codes)).to(args.device)
    crnn.load_sign_cnn(cnn)
    best, trained, seconds = train(crnn, real["train"], composites, real["val"], args)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": crnn.state_dict(), "wells_codes": list(codes), "lipi_commit": LIPI_COMMIT,
                "horizontal_stride": HORIZONTAL_STRIDE, "epoch": best, "seed": args.seed,
                "composites": len(composites)}, args.checkpoint)

    results = {}
    for name in ("test", "harappa"):
        ids = {item.name for item in real[name]}
        known, estimated = cnn_baselines(cnn, seals, indices, codes, ids)
        whole = evaluate(crnn, real[name], args.batch, args.device)
        results[name] = {"Crop CNN, given count": known, "Crop CNN, estimated count": estimated,
                         "Whole-band CRNN + CTC": whole}
    layout = ["# M1 · CRNN + CTC on whole seal bands", "",
              (f"Trained on real bands and {len(composites):,} reviewed-crop composites."
               if composites else "**Real-band baseline.** No #32 composites were available for this run."), "",
              f"Train: {len(real['train'])} seals. Validation: {len(real['val'])}. "
              f"Selected epoch {best} of {trained} by lowest validation sign error rate "
              f"then highest exact match. Seed {args.seed}, PyTorch {torch.__version__}, "
              f"device {torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else args.device}, "
              f"training {seconds:.1f} seconds.", "",
              "| Split | Reader | Seals | Signs | Sign error rate ↓ | Entire inscription correct ↑ | Correct sign count ↑ |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for split, methods in results.items():
        for method, m in methods.items():
            layout.append(f"| {split} | {method} | {m['seals']} | {m['signs']} | {m['sign_error_rate']:.1%} | "
                          f"{m['exact']:.1%} | {m['count_accuracy']:.1%} |")
    layout += ["", "Sign error rate is the total insertions, deletions and substitutions divided by the "
               "number of reference signs. The CRNN uses an automatically located band, with a margin to "
               "reduce clipping. It receives no sign count, and reverses horizontal feature order without "
               "mirroring the glyphs. Adjacent repeats require a blank between them under CTC.",
               "", "The estimated-count baseline counts boxes using the pixel-only segmentation heuristic "
               "and applies the same crop CNN. The given-count row has the true count. Neither baseline "
               "uses a language model. Harappa was held out from all real training and validation.",
               "", f"CNN initialization SHA-256: `{hashlib.file_digest(args.init.open('rb'), 'sha256').hexdigest()}`."]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(layout) + "\n", encoding="utf-8")
    args.output.with_suffix(".json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(args.output.read_text(), flush=True)


if __name__ == "__main__":
    main()
