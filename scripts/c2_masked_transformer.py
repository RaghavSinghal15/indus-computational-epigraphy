"""Compare a small masked transformer and n-grams on the fixed C1 split."""

import argparse
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

from c1_masked_prediction import split
from indus.data import REPO_ROOT, load_sequences
from indus.masked import NGramModel
from indus.masked_transformer import MaskedNGram, MaskedTransformer, metrics

OUT = REPO_ROOT / "reports" / "c2_neural_m3.md"
BATCH_SIZE = 64


def encode_examples(texts, lookup, unknown, rng=None, dropout=0):
    for text in texts:
        ids = [lookup.get(sign, unknown) for sign in text]
        positions = range(len(ids)) if rng is None else [rng.randrange(len(ids))]
        for pos in positions:
            context = ids if rng is None else [
                unknown if i != pos and rng.random() < dropout else sign
                for i, sign in enumerate(ids)
            ]
            yield context, pos, ids[pos]


def batches(examples, model):
    device = model.tokens.weight.device
    for start in range(0, len(examples), BATCH_SIZE):
        group = examples[start:start + BATCH_SIZE]
        width = max(len(ids) for ids, _, _ in group)
        padded, positions, targets = [], [], []
        for ids, pos, target in group:
            padded.append([*ids[:pos], model.mask, *ids[pos + 1:], *([model.pad] * (width - len(ids)))])
            positions.append(pos)
            targets.append(target)
        yield tuple(torch.tensor(x, device=device) for x in (padded, positions, targets))


@torch.inference_mode()
def predict(model, examples):
    model.eval()
    return np.concatenate([model(x, pos).cpu().numpy() for x, pos, _ in batches(examples, model)])


def train_model(texts, lookup, max_length, epochs, seed, device, validation=None):
    unknown = len(lookup)
    rng = random.Random(seed)
    torch.manual_seed(seed)
    model = MaskedTransformer(unknown, max_length).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    best_epoch, best_loss, stale = epochs, float("inf"), 0
    val_examples = list(encode_examples(validation, lookup, unknown)) if validation is not None else []
    val_targets = [target for _, _, target in val_examples]
    for epoch in range(1, epochs + 1):
        model.train()
        examples = list(encode_examples(texts, lookup, unknown, rng, dropout=0.03))
        rng.shuffle(examples)
        for x, pos, target in batches(examples, model):
            optimizer.zero_grad()
            loss = criterion(model(x, pos), target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        if validation is not None:
            val_loss = metrics(predict(model, val_examples), val_targets, unknown)["known_pseudo_ppl"]
            if val_loss < best_loss:
                best_loss, best_epoch, stale = val_loss, epoch, 0
            else:
                stale += 1
            if epoch == 1 or epoch % 5 == 0:
                print(f"Epoch {epoch}: validation known-sign pseudo-perplexity {val_loss:.2f}", flush=True)
            if stale >= 10:
                break
        elif epoch == 1 or epoch % 5 == 0:
            print(f"Full fit epoch {epoch}/{epochs}", flush=True)
    return model, best_epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if args.epochs < 1 or args.threads < 1:
        parser.error("--epochs and --threads must be positive")
    torch.set_num_threads(args.threads)

    # C1's held-out texts stay sealed until the epoch is selected and the model is refitted.
    train, test = split(load_sequences())
    ngram = NGramModel().fit(train)
    max_length = max(map(len, train))
    tuning_indices = list(range(len(train)))
    random.Random(args.seed).shuffle(tuning_indices)
    split_at = int(0.90 * len(train))
    tuning = [train[i] for i in tuning_indices[:split_at]]
    validation = [train[i] for i in tuning_indices[split_at:]]
    tune_lookup = NGramModel().fit(tuning).index
    _, epochs = train_model(tuning, tune_lookup, max_length, args.epochs, args.seed, args.device, validation)
    model, _ = train_model(train, ngram.index, max_length, epochs, args.seed, args.device)

    assert max(map(len, test)) <= max_length
    examples = list(encode_examples(test, ngram.index, ngram.UNK))
    targets = [target for _, _, target in examples]
    baseline = MaskedNGram(ngram, train)
    predictions = {}
    for order, label in ((2, "Witten–Bell bigram"), (3, "Interpolated trigram")):
        predictions[label] = [baseline.scores(text, pos, order)
                              for text in test for pos in range(len(text))]
    predictions["Masked transformer"] = predict(model, examples)
    results = {name: metrics(values, targets, ngram.UNK) for name, values in predictions.items()}
    lines = [
        "# M3 · Masked transformer vs n-grams", "",
        f"ICIT benchmark, reading order, fixed C1 split. {len(train):,} training texts and {len(test):,} held-out "
        f"texts, with {len(examples):,} masked positions ({targets.count(ngram.UNK)} unseen target signs). "
        "Every model sees the same inscription with one sign removed.",
        "",
        "| Model | Top 1 | Top 5 | Masked pseudo-perplexity | Known-target pseudo-perplexity |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, row in results.items():
        lines.append(f'| {name} | {row["top1"]:.1%} | {row["top5"]:.1%} | {row["pseudo_ppl"]:.2f} | {row["known_pseudo_ppl"]:.2f} |')
    neural, bigram = results["Masked transformer"], results["Witten–Bell bigram"]
    lines += [
        "",
        f"The transformer changed top-1 by {(neural['top1'] - bigram['top1']) * 100:+.1f} percentage points and top-5 "
        f"by {(neural['top5'] - bigram['top5']) * 100:+.1f} points versus the bigram. "
        "This is a single-seed result on one small split.",
        "",
        "Ranking uses known sign IDs and breaks ties by ascending ID. An unseen target counts as a miss. "
        "Pseudo-perplexity exponentiates average single-mask cross entropy and normalizes over known signs plus "
        "an unknown bucket. The known-target column omits unseen targets. These are masked probabilities "
        "and are distinct from the left-to-right sequence perplexity in C1.",
        "",
        "The bigram scores both neighbouring factors; the trigram scores up to three factors affected by the mask "
        "and interpolates with the bigram. Both estimate all counts on training texts.",
        "",
        f"The transformer uses two layers, width 64, four heads, 128 feed-forward units, dropout 0.15 and "
        f"{sum(p.numel() for p in model.parameters()):,} parameters. It masks one random sign per training text "
        "per epoch, drops 3% of context signs to UNKNOWN to handle unseen inputs, and uses AdamW, batch size 64, "
        "learning rate 0.001 and label smoothing 0.05.",
        "",
        f"Training selected {epochs} epochs from up to {args.epochs} using {len(validation)} validation texts "
        "drawn from C1's training partition, minimizing known-target pseudo-perplexity. The final fit starts "
        f"from scratch on all training texts. Seed {args.seed}, device {args.device}, PyTorch {torch.__version__}.",
        "",
        f"Reproduce with `python scripts/c2_masked_transformer.py --epochs {args.epochs} --seed {args.seed} "
        f"--threads {args.threads} --device {args.device}`.",
    ]
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
