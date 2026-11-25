"""Compare four masked encoders on Ashraf and Sinha's publicly released Indus corpus."""

import csv
import hashlib
import io
import json
import random
import time
from urllib.request import urlopen
import zipfile

import numpy as np
import torch
from torch import nn

ARCHIVE = "https://ndownloader.figshare.com/files/10242114"
MEMBER = "CSV_files_all_Databases/25_Undeciphered(Indus).csv"
ARCHIVE_SHA256 = "7b5e7d5243ef34fe3a5990be879c02f24664bf8b07fe427b2942e6611005f688"

SETTINGS = {
    "small": (64, 2, 4, False, None),
    "larger": (128, 4, 8, False, None),
    "shared": (128, 4, 8, True, None),
    "local": (64, 2, 4, False, 2),
}
# Match the published file byte for byte before reconstructing the fixed split.
CSV_SHA256 = "140a61e57dfaccd15867d9a00141f180a38080141b0ad18bb4262c782024f186"


def load_public_corpus():
    for attempt in range(3):
        try:
            with urlopen(ARCHIVE, timeout=35) as response:
                data = response.read()
            break
        except OSError:
            if attempt == 2:
                raise
            time.sleep(3)
    if hashlib.sha256(data).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("The publisher archive hash changed")
    with zipfile.ZipFile(io.BytesIO(data)) as files:
        raw = files.read(MEMBER)
    if hashlib.sha256(raw).hexdigest() != CSV_SHA256:
        raise ValueError("The Indus CSV does not match the pinned benchmark")
    csv_rows = csv.reader(io.StringIO(raw.decode("ascii")))
    return [tuple(int(code) for code in row if code.strip())[::-1] for row in csv_rows]


class Encoder(nn.Module):
    def __init__(self, vocab, length, settings):
        super().__init__()
        width, depth, heads, shared, radius = settings
        self.mask, self.pad = vocab + 1, vocab + 2
        self.heads, self.depth, self.shared, self.radius = heads, depth, shared, radius
        self.symbol = nn.Embedding(vocab + 3, width, padding_idx=self.pad)
        self.position = nn.Embedding(length, width)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(width, heads, 2 * width, dropout=0.15,
                                       batch_first=True, norm_first=True)
            for _ in range(1 if shared else depth)
        ])
        self.norm = nn.LayerNorm(width)
        self.classifier = nn.Linear(width, vocab + 1)

    def forward(self, sequence, hidden_pos):
        position = torch.arange(sequence.shape[1], device=sequence.device)
        padding = sequence == self.pad
        x = self.symbol(sequence) + self.position(position)
        mask = None
        if self.radius is not None:
            # A padded query can see the real tokens, so no query has every key masked.
            excluded = (position[:, None] - position[None, :]).abs() > self.radius
            local = excluded[None, :, :] & ~padding[:, :, None]
            mask = local.repeat_interleave(self.heads, dim=0)
        for block in range(self.depth):
            x = self.blocks[0 if self.shared else block](x, src_mask=mask, src_key_padding_mask=padding)
        rows = torch.arange(len(sequence), device=sequence.device)
        return self.classifier(self.norm(x[rows, hidden_pos]))


def vocabulary(texts):
    return {sign: i for i, sign in enumerate(sorted({s for row in texts for s in row}))}


def examples(texts, index, rng=None):
    unknown = len(index)
    out = []
    for text in texts:
        tokens = [index.get(s, unknown) for s in text]
        positions = range(len(tokens)) if rng is None else [rng.randrange(len(tokens))]
        for pos in positions:
            view = list(tokens)
            if rng is not None:
                for i in range(len(view)):
                    if i != pos and rng.random() < 0.03:
                        view[i] = unknown
            view[pos] = unknown + 1
            out.append((view, pos, tokens[pos]))
    if rng is not None:
        rng.shuffle(out)
    return out


def batches(values, pad):
    for offset in range(0, len(values), 64):
        chunk = values[offset:offset + 64]
        size = max(len(ids) for ids, _, _ in chunk)
        rows = [ids + [pad] * (size - len(ids)) for ids, _, _ in chunk]
        yield (torch.tensor(rows, device="cuda"),
               torch.tensor([p for _, p, _ in chunk], device="cuda"),
               torch.tensor([y for _, _, y in chunk], device="cuda"))


@torch.inference_mode()
def measure(model, values, unknown):
    model.eval()
    outputs = np.concatenate([model(x, p).cpu().numpy() for x, p, _ in batches(values, model.pad)])
    target = np.array([y for _, _, y in values])
    genuine = target != unknown
    scores = outputs.astype(np.float64)
    losses = np.logaddexp.reduce(scores, axis=1) - scores[np.arange(len(target)), target]
    known_rank = np.argsort(-scores[:, :unknown], axis=1, kind="stable")
    return {
        "top1": float((known_rank[:, 0] == target).mean()),
        "top5": float((known_rank[:, :5] == target[:, None]).any(axis=1).mean()),
        "pseudo_ppl": float(np.exp(losses.mean())),
        "known_pseudo_ppl": float(np.exp(losses[genuine].mean())),
        "positions": len(target), "unseen": int((~genuine).sum()),
    }


def fit(texts, mapping, config, max_length, seed, epochs, deadline, validation=None):
    torch.manual_seed(seed)
    gen = random.Random(seed)
    model = Encoder(len(mapping), max_length, config).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    best, selected, stale = float("inf"), epochs, 0
    val = examples(validation, mapping) if validation is not None else None
    for epoch in range(1, epochs + 1):
        if time.monotonic() > deadline:
            raise TimeoutError("Budget reached")
        model.train()
        for x, p, y in batches(examples(texts, mapping, gen), model.pad):
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x, p), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
        if val is not None:
            loss = measure(model, val, len(mapping))["known_pseudo_ppl"]
            if loss < best:
                best, selected, stale = loss, epoch, 0
            else:
                stale += 1
            if stale == 10:
                break
    return model, selected


def trial(name, settings, train, test, seed, deadline):
    start = time.monotonic()
    order = list(range(len(train)))
    random.Random(seed).shuffle(order)
    cut = int(0.90 * len(train))
    inner = [train[i] for i in order[:cut]]
    val = [train[i] for i in order[cut:]]
    span = max(map(len, train))
    tuner, epochs = fit(inner, vocabulary(inner), settings, span, seed, 50, deadline, val)
    del tuner
    index = vocabulary(train)
    model, _ = fit(train, index, settings, span, seed, epochs, deadline)
    score = measure(model, examples(test, index), len(index))
    assert (score["positions"], score["unseen"]) == (1693, 66)
    score.update(model=name, seed=seed, parameters=sum(p.numel() for p in model.parameters()),
                 epochs=epochs, seconds=round(time.monotonic()-start, 1))
    return score


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This study requires a cloud GPU")
    torch.set_num_threads(2)
    rows = load_public_corpus()
    assert len(rows) == len(set(rows)) == 1837
    indices = list(range(len(rows)))
    random.Random(0).shuffle(indices)
    pivot = int(0.80 * len(rows))
    train, test = ([rows[i] for i in part] for part in (indices[:pivot], indices[pivot:]))
    print(f"Public benchmark matched; {len(train)} training texts, {len(test)} test texts", flush=True)
    results = {"gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__), "runs": [],
               "dataset": ARCHIVE, "csv_sha256": CSV_SHA256, "matched_seeds": [0, 1, 2]}
    print(f"GPU {results['gpu']}; matched small, larger, shared-weight and local variants", flush=True)
    started = time.monotonic()
    deadline = started + 22 * 60
    try:
        for seed in results["matched_seeds"]:
            for name, config in SETTINGS.items():
                print(f"Training {name}, seed {seed}", flush=True)
                outcome = trial(name, config, train, test, seed, deadline)
                results["runs"].append(outcome)
                print("TRIAL " + json.dumps(outcome), flush=True)
    except TimeoutError as error:
        results["stop"] = str(error)
    finally:
        results["wall_seconds"] = round(time.monotonic() - started, 1)
        print("RESULT_JSON " + json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
