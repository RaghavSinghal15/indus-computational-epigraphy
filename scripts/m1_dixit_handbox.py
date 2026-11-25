"""Score the hand-box CNN's test predictions on Dixit et al.'s 40-sign label set.

Reads data/processed/e1/handbox_test_predictions.csv (from e1_handbox_predictions.py), maps
target and prediction to the paper's 38 named Mahadevan codes + Other, and writes
reports/m1_dixit_handbox.md. Same crop pool rules as m1_dixit_protocol.py.
"""
import pandas as pd

from m1_dixit_protocol import NAMED, OTHER
from indus.data import REPO_ROOT, wells_to_mahadevan
from indus.wells import SPLIT_FILE

PRED = REPO_ROOT / "data" / "processed" / "e1" / "handbox_test_predictions.csv"
OUT = REPO_ROOT / "reports" / "m1_dixit_handbox.md"


def collapse(wells, m):
    """Mahadevan code if it is one of the 38 named, Other if not, None if Wells has no single Mahadevan code."""
    codes = m.get(int(wells), ())
    if len(codes) != 1:
        return None
    code = next(iter(codes))
    return code if code in NAMED else OTHER


def score(q, m):
    """Collapsed top-1, majority baseline, named-only accuracy, balanced accuracy over named classes."""
    q = q[[len(m.get(int(w), ())) == 1 for w in q.target]].copy()       # same eligibility as the protocol audit
    q["t"] = q.target.map(lambda w: collapse(w, m))
    q["p"] = q.predicted.map(lambda w: collapse(w, m)).fillna(OTHER)   # unmappable guess counts as Other
    q["ok"] = q.t == q.p
    named = q[q.t != OTHER]
    per_class = named.groupby("t").ok.mean()
    return dict(crops=len(q), top1=q.ok.mean(), majority=(q.t == OTHER).mean(), named=len(named),
                named_acc=named.ok.mean(), classes=len(per_class), balanced=per_class.mean(),
                wells=(q.target == q.predicted).mean())


def main():
    m = wells_to_mahadevan()
    plan = pd.read_csv(SPLIT_FILE).set_index("cisi")
    p = pd.read_csv(PRED)
    p["site"] = p.cisi.map(plan.site)
    rows = [("Mohenjo-daro test seals (as in the audit)", score(p[p.site == "Mohenjo-daro"], m)),
            ("All test seals", score(p, m))]
    lines = ["# M1 · Hand-box CNN on Dixit et al.'s 40-sign label set", "",
             "Dixit et al. (2025) report 94.5% on a reserved, balanced set of 200 hand-cropped signs (38 named "
             "Mahadevan codes + Other); their cross-validation mean is 85.28%. `reports/m1_dixit_protocol.md` scored "
             "our forced-cut CNN on the same label set: 62.5%. Here the hand-box CNN (`reports/m1_handbox_cnn.md`), "
             "on hand-box crops of our fixed test seals, is scored on that label set. It was trained on full Wells "
             "IDs; its first guess is collapsed afterwards, so it was never tuned for the 39-class task.", "",
             "| Test crops | Crops | Collapsed top-1 | Always 'Other' | Named-sign crops | Accuracy on them | "
             "Balanced over named classes | Full Wells top-1 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, r in rows:
        lines.append(f"| {name} | {r['crops']} | {r['top1']:.1%} | {r['majority']:.1%} | {r['named']} | "
                     f"{r['named_acc']:.1%} | {r['balanced']:.1%} ({r['classes']}) | {r['wells']:.1%} |")
    r = rows[0][1]
    lines += ["", f"Sanity check: the Mohenjo-daro pool is the same {r['crops']} crops with the same {r['majority']:.1%} "
              "'Other' share as the fixed-split row of the protocol audit.", "",
              f"With hand-drawn boxes our reader gets {r['top1']:.1%} on the paper's label set, against 94.5% published "
              "and 62.5% for our forced cuts. Half our test crops are 'Other'; on the named signs alone it gets "
              f"{r['named_acc']:.1%}, and {r['balanced']:.1%} averaged per class (closest to a balanced test set). "
              "The remaining gap may come from different images, class balance and their crop-level split; "
              "ours is by seal, so no test seal was seen in training."]
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
