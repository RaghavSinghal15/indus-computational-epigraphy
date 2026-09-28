# Tasks until the first evaluation (Oct 6)

Five people, numbered P1 to P5. Decide who is who among yourselves.

| Person | Workstream | Tasks |
|---|---|---|
| P1 | Vision pipeline (M1) | A1, A2, A3, A4, A5 |
| P2 | Sequence statistics (M2) | B1, B2, B3, B4, B5 |
| P3 | Seal photo dataset and ground truth | D1, D2 |
| P4 | Synthetic training data | D3 |
| P5 | Literature review and presentation | D4, D5 |

Everyone contributes to the slides (C1) and the rehearsal (C2).

Each task is tagged with when it pays off:
- **Oct 6:** the presentation uses it directly.
- **Stretch:** strengthens Oct 6 if there is time.
- **Next phase:** sets up the work right after Oct 6.

---

## P1: Vision pipeline (M1)

### A1. Find the signs on a seal photo · Oct 6
- **What:** given a seal photo, draw a box around each sign in the text row and cut it out. Classical image processing (grayscale, separate dark carving from light stone, find connected shapes). No training data needed.
- **Depends on:** a few photos. Start with 3 to 5 sample photos from github.com/tpsatish95/OCR-on-Indus-Seals on day one, then switch to the D1 set when it lands.
- **Deliverable:** `src/indus/segment.py`, plus a notebook showing boxes drawn on photos.
- **Done when:** on 3 clear photos, most signs get their own box, ordered right to left.

### A2. Identify each cut-out sign · Oct 6
- **What:** compare each cut-out against the 417 template drawings and return the closest matches. A direct comparison, no neural network yet. This is the baseline.
- **Depends on:** only `data/raw/sign_templates/`. Does not need A1: build and test on distorted copies of the templates.
- **Deliverable:** `src/indus/classify.py`, returning the top 5 guesses per cut-out.
- **Done when:** given a slightly rotated or blurred template, the right sign appears in the top 5.

### A3. Join into one pipeline · Oct 6
- **What:** photo in, A1 finds signs, A2 identifies them, ordered list of Mahadevan IDs out.
- **Depends on:** A1, A2.
- **Deliverable:** `src/indus/pipeline.py`, plus a demo notebook.
- **Done when:** one command turns a photo into a sign sequence.

### A4. Measure accuracy · Oct 6
- **What:** run the pipeline on the D2 ground-truth photos, report top-1 and top-5 sign accuracy.
- **Depends on:** A3, D2.
- **Deliverable:** one results table in `reports/`.
- **Note:** a low number is fine; it is the baseline we improve on.

### A5. Record the demo · Oct 6
- **What:** 30 to 45 second screen recording: photo in, boxes appear, IDs come out.
- **Depends on:** A3.

**If A1 runs into trouble:** draw boxes by hand on 2 or 3 demo photos, run A2 on those, and say on the slide that automatic detection is in progress.

---

## P2: Sequence statistics (M2)

### B1. Basic statistics · Oct 6
- **What:** sign frequencies ranked (Zipf plot), sequence lengths, how many signs appear only once.
- **Depends on:** `indus.data.load_sequences()` (done).
- **Deliverable:** `notebooks/b1_descriptive.ipynb`, 2 plots in `reports/figures/`.
- **Done when:** numbers match known facts: 8,359 signs, 568 distinct, 219 appearing once.

### B2. Start and end signs · Oct 6
- **What:** which signs open texts and which close them. Show the jar sign (Wells 740) is overwhelmingly a closing sign.
- **Depends on:** the loader only. Can run alongside B1.
- **Deliverable:** notebook section, one chart.

### B3. Pair counts and transition matrix · Oct 6
- **What:** count how often each sign follows each other sign, convert to probabilities ("given sign X, how likely is Y next").
- **Depends on:** B1.
- **Deliverable:** `src/indus/ngrams.py` (reusable, M3 needs it), plus a heatmap of the top signs.
- **Done when:** every row of probabilities sums to 1, and top pairs are consistent with the IM77 table.

### B4. Reading direction from the data · Oct 6
- **What:** compare how predictable the next sign is reading right to left versus left to right (conditional entropy both ways).
- **Depends on:** B3.
- **Deliverable:** one number per direction plus one sentence of interpretation.

### B5. Smoothing · Stretch
- **What:** Witten-Bell smoothing so unseen pairs don't get probability zero.
- **Depends on:** B3.
- **Note:** M3 needs it soon after Oct 6.

---

## P3: Seal photo dataset and ground truth

### D1. Build the seal photo dataset · Oct 6
- **What:** 30 to 50 photos of real seals with a clear, readable row of signs. Sources: sample images in the 2017 OCR repo, harappa.com, indusscript.in, museum websites.
- **Deliverable:** photos in `data/raw/seal_photos/`, plus `data/raw/seal_photos/sources.csv` with columns `filename, source_url, site, im77_text_number` (leave unknowns blank).
- **Why it matters:** this is the real-world data the vision pipeline runs on. The M1 demo and the accuracy result both depend on it.
- **Target:** Oct 3.

### D2. Ground truth for 10 to 15 photos · Oct 6
- **What:** for each photo, the correct sign sequence, found by comparing each sign to the images in `data/raw/sign_templates/`. Where the IM77 text number is known, indusscript.in shows the official sequence.
- **Depends on:** D1.
- **Deliverable:** `data/raw/seal_photos/ground_truth.csv` with columns `filename, mahadevan_ids_reading_order` (IDs separated by spaces, first sign read first, i.e. the right-most sign on the seal).
- **Why it matters:** this is what makes an accuracy number possible. Without it the pipeline can only be shown running, not measured.
- **Target:** Oct 3.

---

## P4: Synthetic training data

### D3. Synthetic training images · Next phase
- **What:** a script that makes about 50 distorted copies of each of the 417 templates (rotation, blur, thicker or thinner strokes, noise) to imitate worn carvings.
- **Deliverable:** `scripts/make_synthetic.py`, writing to `data/processed/synthetic/<sign_id>/`, plus one grid image of examples in `reports/figures/` (usable on the Oct 6 slides).
- **Done when:** running the script once regenerates the full set, and the example grid looks like plausible worn signs.
- **Why it matters:** this is the training data for M1's learned classifier, the main step after Oct 6. It addresses the project's biggest data gap: we have only one drawing per sign.
- **Target:** Oct 3.

---

## P5: Literature review and presentation

### D4. Literature review · Oct 6
- **What:** a half-page card each for: Rao et al. 2009 (Science), Yadav et al. 2010 (PLOS ONE), Farmer, Sproat and Witzel 2004, Palaniappan and Adhikari 2017 (arXiv), Ashraf and Sinha 2018 (PLOS ONE). For each: what they did, what data, and **the exact numbers they report**.
- **Deliverable:** `docs/literature.md`.
- **Why it matters:** gives every result we present a published number to compare against, and becomes the related-work section of the final report.
- **Target:** Oct 3.

### D5. Presentation design · Oct 6
- **What:** the slide template, and slide 1 (problem and data). Results slides get filled in together on Oct 5.
- **Deliverable:** link to the shared slides in `docs/PLAN.md`.
- **Target:** Oct 3.

---

## Together

- **C1. Slides · Oct 6:** results from A3 or A5, B2, B4 dropped into the D5 template. Oct 5.
- **C2. Rehearsal · Oct 6:** one full run-through out loud before presenting. Everyone.

---

## Dependencies

```
P1   A1 (find signs) ───┐
                        ├──► A3 (pipeline) ──► A5 (demo) ─────────┐
     A2 (identify) ─────┘        │                               │
                                 └──► A4 (accuracy) ◄── D2 (P3)   │
                                                                  ├──► C1 slides ──► C2 rehearse
P2   B1 (basic stats) ──► B3 (transitions) ──► B4 (direction) ───┤
     B2 (start/end) ─────────────────────────────────────────────┤
                              └──► B5 (smoothing)                │
                                                                  │
P3   D1 (photos) ──► D2 (ground truth) ──► A4                     │
P4   D3 (synthetic data) ──► M1 training after Oct 6              │
P5   D4 (literature) ──► slides and report                       │
     D5 (presentation design) ───────────────────────────────────┘
```

The longest chain is **A1 → A3 → A5**, which is why A1 starts on day one.

## Schedule

| Day | P1 | P2 | P3 | P4 | P5 |
|---|---|---|---|---|---|
| Tue 29 Sep | Setup, A1 | Setup, B1 | Setup, D1 | Setup, D3 | Setup, D4 |
| Wed 30 | A1 | B2, B3 | D1 | D3 | D4 |
| Thu 1 Oct | A2 | B3 | D2 | D3 | D4, D5 |
| Fri 2 | A3 | B4 | D2 | D3 | D5 |
| Sat 3 | Pipeline on D1 photos | B5 | **D1, D2 done** | **D3 done** | **D4, D5 done** |
| Sun 4 | A4 | Check numbers vs D4 | Help with fixes | Help with fixes | Help with fixes |
| Mon 5 | A5, C1 | C1 | C1 | C1 | C1 |
| Tue 6 | **C2, then present** | | | | |

## How to work without colliding

1. **Never push straight to `main`.** Make a branch named after your person number and task, e.g. `p3-d1-seal-photos`.
2. Push your branch and open a pull request on GitHub. Someone other than the author reviews and merges it.
3. Write only inside your own task's files and folders.
4. Before using any AI coding tool on this repo, make sure it has read `AGENTS.md`.
5. Run `pytest` before opening a pull request. All tests must still pass.

```bash
git checkout main
git pull
git checkout -b p3-d1-seal-photos
# ... do the work ...
git add <your files>
git commit -m "D1: add 32 seal photos and sources.csv"
git push -u origin p3-d1-seal-photos
# then open the pull request on GitHub
```
