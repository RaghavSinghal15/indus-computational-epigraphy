"""Run the B1–B4, C1, D1 and E2 comparison on the benchmark and Lipi subsets."""

import argparse
import random

import numpy as np

from c1_masked_prediction import split
from d1_guardrails import labelled
from e2_null_benchmark import N_CORPORA, STATS, discriminate
from indus.data import LIPI_COMMIT, REPO_ROOT, load_icit_full, load_im77_ngrams, load_sequences, wells_to_mahadevan
from indus.descriptive import count_signs, summarize_corpus
from indus.endpoints import count_endpoints
from indus.entropy import entropy_summary
from indus.guardrails import NaiveBayes, conditional_entropy, last_sign_share, pair_features, permutation_test, sign_features
from indus.masked import NGramModel, masked_accuracy
from indus.ngrams import compare_im77_bigrams, count_ngrams, transition_matrices
from indus.nullmodels import BigramProcess, triple_features, unigram_corpus

JAR = 740
SITES = ("Mohenjo-daro", "Harappa")
OUT = REPO_ROOT / "reports" / "m2m4_full_corpus.md"


def discrimination_rows(train, test, make, generated_control=False):
    """Mean balanced accuracy over five draws, using shared samples for both feature sets."""
    rows = []
    for _ in range(5):
        fake_train, fake_test = make(len(train)), make(len(test))
        left_train, left_test = (make(len(train)), make(len(test))) if generated_control else (train, test)
        rows.append([discriminate(left_train, left_test, fake_train, fake_test, feat)
                     for feat in (pair_features, triple_features)])
    return np.mean(rows, axis=0)


def e2_rows(texts, train, test):
    rng = random.Random(0)
    corpus_chain = BigramProcess(texts, min_len=min(map(len, texts)), max_len=max(map(len, texts)))
    train_chain = BigramProcess(train, min_len=min(map(len, train)), max_len=max(map(len, train)))

    def draw_unigrams(n):
        # Match sampled training lengths. Both token frequencies and lengths come from train.
        return unigram_corpus(rng.sample(train, n), rng)

    rows = {}
    for label, make in (
        ("unigram", lambda: unigram_corpus(texts, rng)),
        ("Markov", lambda: corpus_chain.corpus(len(texts), rng)),
    ):
        draws = [make() for _ in range(N_CORPORA)]
        for name, measure, fmt in STATS:
            values = [measure(draw) for draw in draws]
            rows[f"E2 · {name} ({label})"] = fmt.format(np.mean(values))
    for label, make, control in (
        ("real vs unigram", draw_unigrams, False),
        ("real vs Markov", lambda n: train_chain.corpus(n, rng), False),
        ("Markov vs Markov", lambda n: train_chain.corpus(n, rng), True),
    ):
        pairs, triples = discrimination_rows(train, test, make, control)
        rows[f"E2 · {label}, pairs / triples"] = f"{pairs:.1%} / {triples:.1%}"
    return rows


def comparison(texts, n_shuffles):
    info = summarize_corpus(texts)
    signs = count_signs(texts)
    first, last = count_endpoints(texts)
    pairs = count_ngrams(texts)
    counts, probabilities = transition_matrices(texts)
    forward = entropy_summary(counts)
    reverse = entropy_summary(counts.T)
    _, mapped = compare_im77_bigrams(pairs, wells_to_mahadevan(), load_im77_ngrams(2))
    eligible = mapped["eligible_pair_tokens"]
    pair, count = sorted(pairs.items(), key=lambda item: (-item[1], item[0]))[0]
    assert sum(pairs.values()) == info["n_tokens"] - info["n_sequences"]
    assert last[JAR] > first[JAR], "Check reading order: jar should favour the last position"
    assert np.allclose(probabilities.dropna(how="all").sum(axis=1), 1)
    assert np.isclose(
        forward["conditional_entropy_bits"] - reverse["conditional_entropy_bits"],
        forward["next_marginal_entropy_bits"] - forward["current_marginal_entropy_bits"],
    )

    train, test = split(texts)
    model = NGramModel().fit(train)
    rows = {
        "B1 · Unique texts": f'{info["n_sequences"]:,}',
        "B1 · Sign tokens / types": f'{info["n_tokens"]:,} / {info["n_types"]:,}',
        "B1 · Singleton share (types)": f'{info["singleton_type_fraction"]:.1%}',
        "B1 · Top 10 sign share (tokens)": f'{sum(n for _, n in signs.most_common(10)) / info["n_tokens"]:.1%}',
        "B1 · Length min–max (mean)": f'{info["min_length"]}–{info["max_length"]} ({info["mean_length"]:.2f})',
        "B2 · Jar first / last": f'{first[JAR] / len(texts):.2%} / {last[JAR] / len(texts):.2%}',
        "B3 · Adjacent pair tokens / types": f'{sum(pairs.values()):,} / {len(pairs):,}',
        "B3 · Most frequent pair (count)": f"{pair[0]}→{pair[1]} ({count})",
        "B3 · Pair tokens eligible for IM77 comparison": f"{eligible / sum(pairs.values()):.1%}",
        "B3 · Eligible pair tokens also seen in IM77": f'{mapped["eligible_pair_tokens_present_in_im77"] / eligible:.1%}' if eligible else "—",
        "B4 · Next-sign entropy (reading / reverse, bits)": f'{forward["conditional_entropy_bits"]:.3f} / {reverse["conditional_entropy_bits"]:.3f}',
        "C1 · Train / test texts": f"{len(train):,} / {len(test):,}",
        "C1 · Held-out perplexity (unigram / bigram)": f'{model.perplexity(test, "unigram"):.2f} / {model.perplexity(test):.2f}',
    }
    for kind, label in (("unigram", "unigram"), ("left", "previous sign"), ("both", "both neighbours")):
        accuracy = masked_accuracy(model, test, kind)
        rows[f"C1 · {label}, masked top 1 / 5 / 10"] = f"{accuracy[1]:.1%} / {accuracy[5]:.1%} / {accuracy[10]:.1%}"

    rng = random.Random(1)
    train_x, train_y = labelled(train, rng)
    test_x, test_y = labelled(test, rng)
    for feature, label in ((pair_features, "sign pairs"), (sign_features, "single signs")):
        classifier = NaiveBayes().fit([feature(t) for t in train_x], train_y)
        accuracy = np.mean(classifier.predict([feature(t) for t in test_x]) == np.array(test_y))
        rows[f"D1 · Real vs shuffled ({label})"] = f"{accuracy:.1%}"
    for label, measure, larger, fmt in (
        ("jar last", lambda t: last_sign_share(t, JAR), True, "{:.2%}"),
        ("entropy with boundaries (bits)", conditional_entropy, False, "{:.3f}"),
    ):
        observed, null, p = permutation_test(texts, measure, n_shuffles, larger_is_extreme=larger)
        rows[f"D1 · {label}, real / shuffle mean (p)"] = f"{fmt.format(observed)} / {fmt.format(null.mean())} ({p:.4f})"
    return rows | e2_rows(texts, train, test)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shuffles", type=int, default=1000)
    parser.add_argument("--output", type=type(OUT), default=OUT)
    args = parser.parse_args()
    if args.shuffles < 1:
        parser.error("--shuffles must be positive")

    corpora = {"Benchmark": load_sequences(), "Full ICIT": load_icit_full()}
    corpora.update({site: load_icit_full(site=site) for site in SITES})
    benchmark, full, mohenjo, harappa = (set(map(tuple, texts)) for texts in corpora.values())
    columns = []
    for label, texts in corpora.items():
        print(f"{label}: {len(texts):,} texts", flush=True)
        columns.append(comparison(texts, args.shuffles))

    lines = [
        "# M2–M4 on the full ICIT corpus and by site", "",
        f"Lipi revision `{LIPI_COMMIT}`. Fetch the pinned CSV with "
        "`python scripts/fetch_lipi.py --corpus-only`, then run `python scripts/m2m4_full_corpus.py`.",
        "",
        "The benchmark uses the original 1,837 texts. Lipi subsets keep complete, legible, single-line texts "
        "with known direction and at least two signs. Deduplication is by ordered Wells sequence, within each "
        "column after filtering. All sequences below are in reading order.",
        "",
        f"Exact sequence overlap: {len(benchmark & full):,} benchmark texts occur in the full filtered corpus. "
        f"{len(mohenjo & harappa):,} distinct sequences occur at both sites. "
        "The columns overlap and the site rows describe within-site fits, not transfer between sites.",
        "",
        "| Metric | " + " | ".join(corpora) + " |",
        "| --- | " + " | ".join("---:" for _ in corpora) + " |",
        *["| " + " | ".join([metric, *[c[metric] for c in columns]]) + " |" for metric in columns[0]],
        "",
        "C1, D1 and E2 use the same seed-0 80/20 split procedure per corpus. C1 retains the published baseline "
        "script's tie handling and counts unknown signs as misses; its perplexity includes end markers.",
        f"D1 uses {args.shuffles:,} within-text permutations (one-sided, plus-one correction). E2 descriptive "
        f"rows average {N_CORPORA} draws per generator, with the Markov length bounds fitted to the corpus. "
        "E2 classifiers average five trials, with generated training and test samples drawn from training "
        "distributions only. They share samples across the pair and triple classifiers.",
        "",
        "Compare site scores with their sample sizes and sign distributions in mind. "
        "The raw IM77 overlap is conditional on available, unambiguous mappings. Shuffled and generated controls "
        "measure departures from those specific nulls. The entropy difference under reversal follows from "
        "the pair marginals and gives no independent evidence of reading direction.",
    ]
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
