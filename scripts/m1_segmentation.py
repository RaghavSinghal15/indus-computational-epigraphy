"""Measure forced cuts and A7 against reviewed Wells sign rectangles."""

import argparse
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import torch

from a7_unknown_count import N_MAX, choose_n
from indus.align import count_candidates
from indus.cnn import SignCNN, predict, to_input
from indus.data import BOXES_DIR, LIPI_COMMIT, REPO_ROOT, read_yolo_boxes
from indus.real_crops import sign_crops, square_crop
from indus.segment import load_image, trim_caption
from indus.wells import SPLIT_FILE, load_v2_seals, wells_codes

SPLITS = {"val": "Validation", "test": "Test"}
# Historical A7 settings from reports/a7_unknown_count_gpu.md.
COUNT_REWARD = 1.0
LENGTH_WEIGHT = 2.0


class Geometry(NamedTuple):
    iou: float
    matched_at_half: int
    clean: int


@dataclass(frozen=True)
class SealResult:
    code: str
    signs: int
    a7_count: int
    forced: Geometry
    a7: Geometry
    forced_hits: tuple[int, int]
    reviewed_hits: tuple[int, int]


def as_rectangles(band, intervals):
    return np.array(
        [(band.x0 + int(a), band.y0, band.x0 + int(b), band.y1) for a, b in intervals],
        dtype=int,
    ).reshape(-1, 4)


def measure_geometry(reference, prediction):
    """Match boxes one to one by IoU. Unmatched references contribute zero."""
    if not len(prediction):
        return Geometry(0.0, 0, 0)
    start = np.maximum(reference[:, None, :2], prediction[None, :, :2])
    end = np.minimum(reference[:, None, 2:], prediction[None, :, 2:])
    intersection = np.maximum(end - start, 0).prod(axis=2)
    reference_area = (reference[:, 2:] - reference[:, :2]).prod(axis=1)
    predicted_area = (prediction[:, 2:] - prediction[:, :2]).prod(axis=1)
    if (reference_area <= 0).any() or (predicted_area <= 0).any():
        raise ValueError("Expected positive-area rectangles")
    iou = intersection / (reference_area[:, None] + predicted_area[None, :] - intersection)
    i, j = linear_sum_assignment(iou, maximize=True)
    coverage = intersection / reference_area[:, None]
    neighbours = coverage[:, j].copy()
    neighbours[i, np.arange(len(j))] = 0
    clean = (coverage[i, j] >= 0.95) & (neighbours.max(axis=0) < 0.10)
    return Geometry(float(iou[i, j].sum()), int((iou[i, j] >= 0.50).sum()), int(clean.sum()))


def score(model, crops, truth, codes):
    ranked, _ = predict(model, to_input(np.stack(crops)), k=5)
    guesses = codes[ranked - 1]
    return int((guesses[:, 0] == truth).sum()), int((guesses == truth[:, None]).any(axis=1).sum())


def submitted_labels(boxes_dir, plan):
    manifest = pd.read_csv(BOXES_DIR / "batches.csv")
    inventory = {
        (f"batch_{row.batch:02d}", Path(row.image_file).with_suffix(".txt").name): row.cisi
        for row in manifest.itertuples(index=False)
    }
    for label in sorted(boxes_dir.glob("batch_*/*.txt")):
        if label.name == "labels.txt":
            continue
        code = inventory.get((label.parent.name, label.name))
        if code is None:
            raise ValueError(f"Label is outside the committed batch manifest: {label}")
        if plan.at[code, "split"] in SPLITS:
            yield code, label


def audit_seal(code, label, row, cnn, codes, length_prior):
    image = load_image(row.image)
    height, width = image.shape[:2]
    reviewed = np.array(read_yolo_boxes(label, width, height))
    truth = np.array(row.signs_reading)
    if len(reviewed) != len(truth):
        raise ValueError(f"{code}: {len(reviewed)} reviewed boxes for {len(truth)} signs")
    if (reviewed[:, :2] < 0).any() or (reviewed[:, 2:] > [width, height]).any():
        raise ValueError(f"{code}: reviewed box lies outside the original image")

    photo = trim_caption(image)
    forced_crops, band, edges = sign_crops(photo, len(truth))
    forced_boxes = as_rectangles(band, zip(edges, edges[1:]))
    a7_band, _, candidates = count_candidates(cnn, photo, N_MAX)
    count = choose_n(candidates, COUNT_REWARD, LENGTH_WEIGHT, length_prior) if candidates else 0
    a7_boxes = as_rectangles(a7_band, candidates[count][1] if count else ())

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hand_crops = [square_crop(gray[y0:y1], x0, x1) for x0, y0, x1, y1 in reviewed[::-1]]
    return SealResult(
        code, len(truth), count, measure_geometry(reviewed, forced_boxes),
        measure_geometry(reviewed, a7_boxes), score(cnn, forced_crops, truth, codes),
        score(cnn, hand_crops, truth, codes),
    )


def split_report(name, seals):
    n = sum(item.signs for item in seals)
    exact = sum(item.a7_count == item.signs for item in seals)
    close = sum(abs(item.a7_count - item.signs) <= 1 for item in seals)
    count_error = np.mean([abs(item.a7_count - item.signs) for item in seals])
    lines = [
        f"## {name} results", "",
        "| Method | Mean IoU | IoU ≥ 0.50 | Cleanly covered signs | Exact count | Within one sign |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, key, right, near in (("Forced cuts, true count supplied", "forced", len(seals), len(seals)),
                                    ("A7 count search, same Wells CNN", "a7", exact, close)):
        iou, matched, clean = np.sum([getattr(item, key) for item in seals], axis=0)
        lines.append(f"| {label} | {iou / n:.3f} | {int(matched)}/{n} | {int(clean)}/{n} "
                     f"| {right}/{len(seals)} | {near}/{len(seals)} |")
    lines += [
        "", f"A7's mean absolute count error was {count_error:.2f} signs per seal.", "",
        "| Crop source | Correct first guesses | Correct in first five |",
        "| --- | ---: | ---: |",
    ]
    for label, key in (("Forced cuts", "forced_hits"), ("Reviewed boxes", "reviewed_hits")):
        top1, top5 = np.sum([getattr(item, key) for item in seals], axis=0)
        lines.append(f"| {label} | {top1}/{n} | {top5}/{n} |")
    lines += ["", "### Counts by seal", "",
              "| Seal | True count | A7 count | Forced mean IoU | A7 mean IoU |",
              "| --- | ---: | ---: | ---: | ---: |"]
    lines += [f"| {s.code} | {s.signs} | {s.a7_count} | {s.forced.iou / s.signs:.3f} | {s.a7.iou / s.signs:.3f} |"
              for s in seals]
    return lines + [""]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--boxes-dir", type=Path, default=BOXES_DIR, help="directory containing batch folders")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "reports/m1_segmentation.md")
    args = parser.parse_args()
    torch.set_num_threads(4)
    plan = pd.read_csv(SPLIT_FILE).set_index("cisi")
    seals = load_v2_seals().set_index("cisi")
    labels = list(submitted_labels(args.boxes_dir, plan))
    if not labels:
        parser.error("no reviewed validation or test labels in the supplied batch folders")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    vocab = wells_codes()
    if (checkpoint.get("arch") != "small" or checkpoint.get("scope") != "all"
            or checkpoint.get("training_phase") != "real_finetuned"
            or checkpoint.get("lipi_commit") != LIPI_COMMIT or tuple(checkpoint.get("wells_codes", [])) != vocab):
        raise ValueError("Expected the all-site fine-tuned Wells v2 checkpoint")
    train = list(plan.index[plan.split == "train"])
    if (set(checkpoint["train_seals"]) != set(train)
            or set(checkpoint["val_seals"]) != set(plan.index[plan.split == "val"])):
        raise ValueError("Checkpoint train or validation partition does not match the fixed split")
    counts = np.bincount(seals.loc[train].n_signs.astype(int), minlength=N_MAX + 1)[: N_MAX + 1] + 1
    length_prior = np.log(counts / counts.sum())
    cnn = SignCNN(len(vocab))
    cnn.load_state_dict(checkpoint["state_dict"])
    cnn.eval()
    codes = np.array(vocab)

    by_split = {part: [] for part in SPLITS}
    with torch.inference_mode():
        for index, (code, label) in enumerate(labels, 1):
            part = plan.at[code, "split"]
            by_split[part].append(audit_seal(code, label, seals.loc[code], cnn, codes, length_prior))
            if index % 7 == 0 or index == len(labels):
                print(f"Measured {index}/{len(labels)} reviewed seals", flush=True)

    with args.checkpoint.open("rb") as checkpoint_stream:
        digest = hashlib.file_digest(checkpoint_stream, "sha256").hexdigest()
    lines = [
        "# M1 segmentation against reviewed sign boxes", "",
        "The coverage table shows which parts of the fixed splits have reviewed labels. "
        "Metrics for validation and test seals are reported separately.", "",
        "| Split | Reviewed seals | Fixed split seals | Reviewed signs |",
        "| --- | ---: | ---: | ---: |",
    ]
    for part, label in SPLITS.items():
        rows = by_split[part]
        lines.append(f"| {label} | {len(rows)} | {(plan.split == part).sum()} | {sum(r.signs for r in rows)} |")
    lines += [
        "", "## Method", "",
        "Both automatic methods trim the catalogue caption and use the existing inscription-band finder. "
        "Forced cuts receive the true count. A7 searches counts 1 through 13, using the same Wells CNN and an "
        "add-one length prior fit on the 466 training seals. It uses the historical A7 count weights 1 and 2 "
        "without retuning them for Wells.", "",
        "One-to-one box matches maximize total intersection-over-union (IoU), with unmatched references scoring zero. "
        "A clean box covers at least 95% of its matched reference and less than 10% of each other reference. "
        "This strict rectangle-based measure can penalize intentional overlap in the hand labels.", "",
        "The frozen CNN scores both crop sets using the same 64×64 preprocessing, with signs in reading order.", "",
        f"Wells checkpoint SHA-256: `{digest}`.", "",
    ]
    for part, label in SPLITS.items():
        if by_split[part]:
            lines += split_report(label, by_split[part])
    lines += [
        "## Reproduce", "",
        "Fetch Lipi and supply the reviewed batch folders. For the 28-seal validation run, the labels are in "
        "[PR #62](https://github.com/RaghavSinghal15/indus-computational-epigraphy/pull/62).",
        "", "```bash",
        "python scripts/m1_segmentation.py --checkpoint /path/to/wells_all.pt --boxes-dir /path/to/data/raw/sign_boxes",
        "```", "",
    ]
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
