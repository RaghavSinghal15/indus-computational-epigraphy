"""Prepare, train, and evaluate a one-class YOLOv8n sign detector (#36)."""

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from indus.data import BOXES_DIR, REPO_ROOT
from indus.wells import SPLIT_FILE, load_v2_seals

DEFAULT = REPO_ROOT / "data" / "processed" / "sign_detector"
COMPOSITE_FIELDS = ("image", "label", "source_seals")
VALID_SPLITS = ("train", "val", "test")


def checked_labels(path, expected=None):
    """Reject missing, malformed, non-finite or out-of-image YOLO boxes."""
    rows = path.read_text(encoding="utf-8").splitlines()
    if not rows or (expected is not None and len(rows) != expected):
        raise ValueError(f"Empty or incorrect box count in {path}")
    for line in rows:
        parts = line.split()
        if len(parts) != 5 or parts[0] != "0":
            raise ValueError(f"Expected YOLO class 0 and 4 normalized coordinates in {path}")
        cx, cy, w, h = map(float, parts[1:])
        if not (w > 0 and h > 0 and -1e-5 <= cx - w / 2 < cx + w / 2 <= 1 + 1e-5
                and -1e-5 <= cy - h / 2 < cy + h / 2 <= 1 + 1e-5):
            raise ValueError(f"Out-of-bounds box in {path}")
    return len(rows)


def composite_rows(path, allowed_sources):
    """Load manifest records and require train-only origin for crops and background."""
    if path is None:
        return
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        row = json.loads(line)
        if not all(field in row for field in COMPOSITE_FIELDS):
            raise ValueError(f"Composite record {i} needs {COMPOSITE_FIELDS}")
        sources = row["source_seals"]
        if not isinstance(sources, list) or not sources or not set(sources) <= allowed_sources:
            raise ValueError(f"Composite record {i} contains an unknown or non-training source seal")
        yield (path.parent / row["image"]).resolve(), (path.parent / row["label"]).resolve()


def prepare(output, composites):
    if output.exists():
        raise FileExistsError(f"Use a fresh output directory: {output}")
    batches = pd.read_csv(BOXES_DIR / "batches.csv").set_index("cisi", verify_integrity=True)
    plan = pd.read_csv(SPLIT_FILE).set_index("cisi", verify_integrity=True)
    exported = []
    for cisi, batch in batches.iterrows():
        label = BOXES_DIR / f"batch_{int(batch.batch):02d}" / (Path(batch.image_file).stem + ".txt")
        if label.is_file() and cisi in plan.index and plan.loc[cisi, "split"] in VALID_SPLITS:
            exported.append((cisi, batch, label))
    if not exported:
        raise ValueError("No reviewed hand-box exports found for fixed train/val/test seals")
    seals = load_v2_seals().set_index("cisi", verify_integrity=True)
    examples = []
    for cisi, batch, label in exported:
        if cisi not in seals.index:
            raise ValueError(f"Missing pinned impression for {cisi}")
        seal = seals.loc[cisi]
        if Path(seal.image).name != batch.image_file:
            raise ValueError(f"Image filename mismatch for {cisi}")
        count = checked_labels(label, seal.n_signs)
        examples.append((seal.split, cisi, Path(seal.image), label, count))
    genuine = pd.Series([row[0] for row in examples]).value_counts()
    if any(not genuine.get(split, 0) for split in VALID_SPLITS):
        raise ValueError(f"Need reviewed hand boxes in all train/val/test partitions, found {dict(genuine)}")
    allowed = set(plan.index[plan.split == "train"])
    for i, (image, label) in enumerate(composite_rows(composites, allowed), start=1):
        examples.append(("train", f"composite-{i:05}", image, label, checked_labels(label)))

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp) / "dataset"
        for split, item, image, label, count in examples:
            image_out = stage / "images" / split / f"{item}{image.suffix.lower()}"
            label_out = stage / "labels" / split / f"{item}.txt"
            image_out.parent.mkdir(parents=True, exist_ok=True)
            label_out.parent.mkdir(parents=True, exist_ok=True)
            if not image.is_file():
                raise FileNotFoundError(image)
            image_out.symlink_to(image.resolve())
            shutil.copyfile(label, label_out)
        spec = {"path": str(output), **{split: f"images/{split}" for split in VALID_SPLITS},
                "names": {0: "sign"}}
        # JSON syntax is also valid YAML and avoids a separate YAML dependency.
        (stage / "signs.yaml").write_text(json.dumps(spec, indent=2), encoding="utf-8")
        stage.rename(output)
    print(f"{output}: real counts {dict(genuine)}, train composites {len(examples) - int(genuine.sum())}")


def classify_detection(image, boxes, classifier, codes):
    """Predict Wells signs in right-to-left reading order from detector boxes."""
    import cv2
    from indus.cnn import predict, to_input
    from indus.real_crops import square_crop

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    crops = []
    for x0, y0, x1, y1 in sorted(boxes, key=lambda box: box[0] + box[2], reverse=True):
        x0, x1 = np.clip([np.floor(x0), np.ceil(x1)], 0, width).astype(int)
        y0, y1 = np.clip([np.floor(y0), np.ceil(y1)], 0, height).astype(int)
        if x1 > x0 and y1 > y0:
            crops.append(square_crop(gray[y0:y1, x0:x1], 0, x1 - x0))
    if not crops:
        return []
    indices, _ = predict(classifier, to_input(np.stack(crops)), k=1)
    return codes[indices[:, 0] - 1].tolist()


def load_cnn(path, device):
    """Load a v2 CNN with an explicit, validated Wells vocabulary."""
    import torch
    from indus.cnn import ResNetSign, SignCNN
    from indus.wells import wells_codes

    data = torch.load(path, map_location=device, weights_only=True)
    codes = tuple(data.get("wells_codes", ()))
    if codes != wells_codes():
        raise ValueError("CNN checkpoint must identify the pinned Wells class order")
    if data["arch"] == "small":
        model = SignCNN(n_classes=len(codes))
    elif data["arch"] == "resnet18":
        model = ResNetSign(n_classes=len(codes), imagenet=False)
    else:
        raise ValueError(f"Unknown CNN architecture: {data['arch']}")
    model.load_state_dict(data["state_dict"])
    return model.to(device).eval(), np.array(codes)


def evaluate(model, root, device, confidence, cnn=None):
    result = model.val(data=str(root / "signs.yaml"), split="test", device=device, plots=False, workers=0)
    images = sorted((root / "images" / "test").iterdir())
    stream = model.predict(source=[str(p) for p in images], stream=True, conf=confidence, device=device,
                           verbose=False)
    true, guessed = [], []
    edits = exact = 0
    if cnn is not None:
        from a7_unknown_count import edit_distance
        signs = load_v2_seals().set_index("cisi").signs_reading
    for image, img in zip(images, stream, strict=True):
        if image.resolve() != Path(img.path).resolve():
            raise ValueError("Test image and prediction order do not match")
        label = root / "labels" / "test" / (image.stem + ".txt")
        true.append(checked_labels(label))
        guessed.append(len(img.boxes))
        if cnn is not None:
            truth = signs.loc[image.stem]
            prediction = classify_detection(img.orig_img, img.boxes.xyxy.cpu().numpy(), *cnn)
            edits += edit_distance(prediction, truth)
            exact += prediction == truth
    if not true:
        raise ValueError("The test set is empty")
    difference = np.abs(np.asarray(true) - guessed)
    reading = (edits / sum(true), exact / len(true)) if cnn is not None else None
    return result.box, float(np.mean(difference == 0)), float(difference.mean()), len(images), reading


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--composites", type=Path, help="Training JSONL with image, label and source_seals")
    train = sub.add_parser("train")
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--device", default="cpu")
    train.add_argument("--initial", default="yolov8n.pt")
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("checkpoint", type=Path)
    evaluate_parser.add_argument("--device", default="cpu")
    evaluate_parser.add_argument("--cnn-checkpoint", type=Path, help="optional v2 Wells CNN for detect-then-classify")
    evaluate_parser.add_argument("--confidence", type=float, default=0.25)
    evaluate_parser.add_argument("--output", type=Path, default=REPO_ROOT / "reports" / "m1_sign_detector.md")
    for action in (prep, train, evaluate_parser):
        action.add_argument("--dataset", type=Path, default=DEFAULT)
    args = parser.parse_args()
    root = args.dataset.resolve()
    if args.command == "prepare":
        prepare(root, args.composites)
        return

    from ultralytics import YOLO                 # optional dependency only for training or inference

    if args.command == "train":
        model = YOLO(args.initial)
        model.train(data=str(root / "signs.yaml"), epochs=args.epochs, imgsz=640, seed=0, fliplr=0.0,
                    device=args.device, workers=0, project=str(root / "runs"), name="yolov8n")
    else:
        if not 0 <= args.confidence <= 1:
            parser.error("--confidence must be between 0 and 1")
        cnn = load_cnn(args.cnn_checkpoint, args.device) if args.cnn_checkpoint else None
        box, accuracy, mae, count, reading = evaluate(YOLO(str(args.checkpoint)), root, args.device,
                                                     args.confidence, cnn)
        lines = [
            "# M1 · Single-class sign detector", "",
            f"{count} hand-boxed seals from the fixed v2 test split. "
            f"Detector checkpoint `{args.checkpoint.name}`.",
            "",
            "| Box precision | Box recall | AP50 | AP50–95 | Count exact | Count MAE |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
            f"| {box.mp:.1%} | {box.mr:.1%} | {box.map50:.1%} | {box.map:.1%} | {accuracy:.1%} | {mae:.2f} |",
            "",
            f"Box metrics use Ultralytics' test validator. Count metrics use a fixed score threshold of "
            f"{args.confidence:g}.",
        ]
        if reading is not None:
            cer, exact = reading
            lines += ["", f"Detect then classify with `{args.cnn_checkpoint.name}`: "
                      f"sign error rate {cer:.1%}, exact sequence {exact:.1%}. "
                      "Boxes are sorted right to left before classification."]
        args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(args.output)


if __name__ == "__main__":
    main()
