"""Run the D1 and E2 guardrails on Sproat's public motif corpora."""

import argparse
import hashlib
from pathlib import Path
import random
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import numpy as np

from d1_guardrails import labelled
from e2_null_benchmark import N_CORPORA, STATS, discriminate, split
from indus.data import REPO_ROOT
from indus.guardrails import NaiveBayes, conditional_entropy, pair_features, permutation_test, sign_features
from indus.nullmodels import BigramProcess, triple_features, unigram_corpus

SOURCE = "https://richardsproat.com/data/non-linguistic-symbols/"
ARCHIVE_SHA256 = "fb50ff5acb5221ba75e134cc7a36619e1d990b5680ad8681bcde5c447b6b8f73"
CORPORA = {
    "Pennsylvania German barn stars": "BarnStars.xml",
    "Mesopotamian deity symbols": "Kudurrus.xml",
}


def load_control(xml):
    """Flatten each catalogue record in XML order and deduplicate multi-sign rows."""
    root = ET.fromstring(xml)
    records = []
    for section in root.iter("docText"):
        signs = tuple(symbol.findtext("title", "").strip() for symbol in section.iter("symbol"))
        if not all(signs):
            raise ValueError("Symbol without a title in the published corpus")
        if signs:
            records.append(signs)
    vocabulary = {sign: i for i, sign in enumerate(sorted({s for row in records for s in row}), 1)}
    unique = list(dict.fromkeys(row for row in records if len(row) >= 2))
    return [[vocabulary[s] for s in row] for row in unique], len(records), sum(map(len, records)), len(vocabulary)


def compare(title, xml):
    texts, n_raw, n_tokens, n_labels = load_control(xml)
    train, test = split(texts)
    rng = random.Random(1)
    tr_x, tr_y = labelled(train, rng)
    te_x, te_y = labelled(test, rng)
    scores = []
    for feat in (sign_features, pair_features):
        classifier = NaiveBayes().fit([feat(t) for t in tr_x], tr_y)
        guesses = classifier.predict([feat(t) for t in te_x])
        scores.append(float((guesses == np.array(te_y)).mean()))
    real, null, p = permutation_test(texts, conditional_entropy, n_shuffles=1000, larger_is_extreme=False)
    rng = random.Random(0)
    full_markov = BigramProcess(texts, max_len=max(map(len, texts)))
    train_markov = BigramProcess(train, max_len=max(map(len, train)))
    generated = (
        [unigram_corpus(texts, rng) for _ in range(N_CORPORA)],
        [full_markov.corpus(len(texts), rng) for _ in range(N_CORPORA)],
    )

    lines = [
        f"## {title}", "",
        f"The archive has {n_raw} nonempty records, {n_tokens} symbols and {n_labels} distinct labels. "
        f"The filtered set has {len(texts)} distinct sequences ({len(train)} train, {len(test)} test).", "",
        "### D1 · Observed order versus shuffles", "",
        f"Held-out naïve Bayes accuracy was **{scores[0]:.1%}** using individual signs and "
        f"**{scores[1]:.1%}** using adjacent pairs. Conditional entropy was {real:.3f} bits, "
        f"with mean {np.mean(null):.3f} over 1,000 within-sequence shuffles (one-sided p={p:.4f}).", "",
        "### E2 · Corpus statistics", "",
        "Generated columns show the mean and range over 20 corpora.", "",
        "| Statistic | Observed | Unigram | Bigram |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, metric, fmt in STATS[:-1]:
        cells = [fmt.format(metric(texts))]
        for corpora in generated:
            values = [metric(corpus) for corpus in corpora]
            cells.append(f"{fmt.format(np.mean(values))} ({fmt.format(min(values))}–{fmt.format(max(values))})")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines += ["", "### E2 · Held-out classifier", "", "Five draws per row, with 50% chance accuracy.", "",
              "| Comparison | Pairs | Triples |", "| --- | ---: | ---: |"]
    for name, make in (
        ("Observed vs unigram", lambda n: unigram_corpus(rng.sample(train, n), rng)),
        ("Observed vs bigram", lambda n: train_markov.corpus(n, rng)),
        ("Bigram vs bigram", lambda n: train_markov.corpus(n, rng)),
    ):
        scores = []
        for feat in (pair_features, triple_features):
            accuracies = []
            for _ in range(5):
                observed_train, observed_test = (
                    (make(len(train)), make(len(test))) if name == "Bigram vs bigram" else (train, test)
                )
                accuracies.append(discriminate(observed_train, observed_test, make(len(train)), make(len(test)), feat))
            scores.append(float(np.mean(accuracies)))
        lines.append(f"| {name} | {scores[0]:.1%} | {scores[1]:.1%} |")
    return lines + [""]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path, help="corpora.zip from Sproat's data page")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "reports/m4_symbol_controls.md")
    args = parser.parse_args()
    with args.archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != ARCHIVE_SHA256:
        raise ValueError("The archive does not match the pinned Sproat release")

    lines = [
        "# D1 and E2 on two non-language symbol corpora", "",
        "Controls are Pennsylvania German barn-star motifs and the annotated deity symbols on "
        f"Mesopotamian kudurrus. Their XML files include Apache 2.0 licence notices. "
        f"[Sproat’s published archive]({SOURCE})", "",
        f"Archive SHA-256: `{digest}`.", "",
        "One sequence per nonempty XML docText. Symbols inside nested groups are flattened in document "
        "order by title. The analysis retains uncertain symbols as recorded, then removes single-symbol "
        "entries and exact duplicate sequences ahead of the split.", "",
        "The pipeline uses the repository D1 classifiers and permutation test and E2 generators "
        "and classifiers. It omits the Indus-specific jar statistic. Bigram generation uses the "
        "observed maximum sequence length for each corpus or training partition. The 80/20 split "
        "uses seed 0. E2 descriptive simulations use the full control corpus and the held-out "
        "classifier experiment fits its generators on training records only.", "",
    ]
    with ZipFile(args.archive) as archive:
        for title, file in CORPORA.items():
            lines += compare(title, archive.read(f"corpora/{file}"))
    lines += [
        "## Interpretation", "",
        "The tests measure whether recorded symbol order differs from the specified null process. "
        "The deity-symbol sample exhibits order relative to within-record shuffles, illustrating "
        "that a motif system can show such a result. This is one split over small, curated corpora. "
        "XML order may reflect the spatial layout or cataloguing convention. These controls do not "
        "establish the linguistic status of the Indus script.", "",
        "## Reproduce", "",
        f"Download corpora.zip from the [data page]({SOURCE}). Run from the repository root.", "",
        "```bash",
        "python scripts/m4_symbol_controls.py --archive /path/to/corpora.zip",
        "```", "",
    ]
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
