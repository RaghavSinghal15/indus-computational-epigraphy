"""Can the hand-box CNN do better on Dixit et al.'s 40-sign task if we aim it at that task?

Three ways to turn the hand-box CNN into a 39-way classifier (38 named Mahadevan codes + Other):
  A  first full-Wells guess, collapsed afterwards (what m1_dixit_handbox.py reports)
  B  add up the full-Wells probabilities inside each of the 39 groups, take the largest group
  C  fine-tune the hand-box CNN with a new 39-way head on hand-box training crops
Each with and without test-time shifts (TTA). The method is chosen on the validation crops
(Mohenjo-daro, as in the paper); test is scored once for every method and reported in full.

    python scripts/m1_dixit_finetune.py          # about 5 minutes on CPU
Writes reports/m1_dixit_finetune.md.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from m1_dixit_protocol import NAMED, OTHER                         # noqa: E402
from train_sign_cnn import run_epochs                               # noqa: E402
from train_wells_cnn import load_weights                            # noqa: E402
from indus.cnn import SignCNN, log_probs, to_input                  # noqa: E402
from indus.data import REPO_ROOT, wells_to_mahadevan                # noqa: E402
from indus.wells import SPLIT_FILE, wells_codes                     # noqa: E402

HAND = REPO_ROOT / "data" / "processed" / "real_crops" / "crops_handbox.npz"
MODEL = REPO_ROOT / "data" / "processed" / "models" / "wells_handbox.pt"
OUT = REPO_ROOT / "reports" / "m1_dixit_finetune.md"
GROUPS = (*NAMED, OTHER)                                            # class k+1 = GROUPS[k]


def main():
    torch.set_num_threads(2)
    torch.manual_seed(0)
    m = wells_to_mahadevan()
    codes = wells_codes()
    site = pd.read_csv(SPLIT_FILE).set_index("cisi").site
    with np.load(HAND) as d:
        crops = {k: d[k] for k in d.files}
    single = np.array([len(m.get(int(w), ())) == 1 for w in crops["labels"]])
    group_of_wells = {w: GROUPS.index(next(iter(m[w])) if next(iter(m[w])) in NAMED else OTHER) + 1
                      for w in codes if len(m.get(w, ())) == 1}
    y = np.array([group_of_wells.get(int(w), 0) for w in crops["labels"]])          # 1..39, 0 = not eligible
    md = np.isin(crops["seal"], site.index[site == "Mohenjo-daro"])
    x = to_input(crops["images"])
    rows = {s: np.flatnonzero(single & (crops["split"] == s) & (md if s != "train" else True))
            for s in ("train", "val", "test")}
    print({s: len(r) for s, r in rows.items()})

    # full-Wells class index -> group index (0-based), Other for Wells codes without one Mahadevan code
    to_group = np.array([group_of_wells.get(w, len(GROUPS)) - 1 for w in codes])
    full = SignCNN(len(codes))
    load_weights(MODEL, full, codes, "real_finetuned")
    results = {}
    for tta in (False, True):
        for split in ("val", "test"):
            lp = log_probs(full, x[rows[split]], tta=tta)
            p = np.exp(lp)
            first = to_group[p.argmax(1)] + 1
            summed = np.zeros((len(p), len(GROUPS)))
            np.add.at(summed.T, to_group, p.T)                      # add each Wells probability into its group
            results.setdefault(f"A first guess, collapsed{' + TTA' if tta else ''}", {})[split] = (first == y[rows[split]]).mean()
            results.setdefault(f"B group probabilities summed{' + TTA' if tta else ''}", {})[split] = \
                (summed.argmax(1) + 1 == y[rows[split]]).mean()

    # C: new 39-way head on the hand-box CNN, fine-tuned on hand-box training crops (all sites)
    model = SignCNN(len(codes))
    load_weights(MODEL, model, codes, "real_finetuned")
    model.head[-1] = nn.Linear(256, len(GROUPS))
    gen = torch.Generator().manual_seed(0)
    tx, ty = torch.from_numpy(x[rows["train"]]), torch.from_numpy(y[rows["train"]] - 1)

    def batch():
        ix = torch.randint(len(tx), (64,), generator=gen)
        return tx[ix], ty[ix]
    run_epochs(model, batch, (len(tx) + 63) // 64, 25, 0.0005, "cpu", "39-way head",
               val=(x[rows["val"]], y[rows["val"]]))
    for tta in (False, True):
        for split in ("val", "test"):
            pred = log_probs(model, x[rows[split]], tta=tta).argmax(1) + 1
            results.setdefault(f"C fine-tuned 39-way head{' + TTA' if tta else ''}", {})[split] = (pred == y[rows[split]]).mean()

    best = max(results, key=lambda k: results[k]["val"])
    lines = ["# M1 · Aiming the hand-box CNN at Dixit et al.'s 40-sign task", "",
             "Label set: the paper's 38 named Mahadevan codes + Other. Validation and test crops are hand-box crops "
             f"of Mohenjo-daro seals in our fixed seal split ({len(rows['val'])} val, {len(rows['test'])} test); "
             f"training uses the {len(rows['train'])} eligible hand-box training crops from all sites. "
             "The method is chosen on validation; every test score is shown. Published: 94.5% (balanced 200-crop "
             "set, crops of the same seals can be in train and test), 85.28% cross-validation.", "",
             "| Method | Validation | Test |", "| --- | ---: | ---: |"]
    for k, r in results.items():
        mark = " **(chosen on validation)**" if k == best else ""
        lines.append(f"| {k}{mark} | {r['val']:.1%} | {r['test']:.1%} |")
    lines += ["", f"Chosen on validation: {best}, test {results[best]['test']:.1%}. Always guessing Other: "
              f"{(y[rows['test']] == len(GROUPS)).mean():.1%} of test crops."]
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
