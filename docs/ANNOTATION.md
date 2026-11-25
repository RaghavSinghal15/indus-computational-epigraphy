# Sign boxes: how they were drawn

The team drew a tight box around every sign on 834 CISI seal photos (4,078 signs), in batches of about 100 photos, using [makesense.ai](https://www.makesense.ai) (Object Detection, one label `sign`, YOLO export). Nobody typed sign numbers: each seal's sequence is already known from the corpus, so boxes are labelled automatically. The boxes are in `data/raw/sign_boxes/batch_NN/`, one YOLO `.txt` per photo.

## Rules

- **Tight**: touch the outermost strokes, with no big margins. Include strokes that belong to the sign, like the "ears" of the jar sign.
- **One box per sign in the sequence.** Groups of short strokes (`||`, `|||`) count as one sign.
- **Only the text**, never the animal, the manger or the CISI caption.
- A sign that is half broken off still gets a box around what is visible.
- Photos that were unreadable, or whose sequence did not match the photo, were skipped.

`python scripts/make_box_batch.py N` builds a batch with a guide image per seal (the expected signs, left to right); `python scripts/check_boxes.py N --show` checks box counts and draws each assigned sign under its box for review.

## How the labels are assigned

Boxes are sorted by their horizontal centre, left to right, and get the seal's known sequence in left-to-right order (the impression shows the text mirrored back into reading direction, so the first sign read is the right-most one). `indus.data.load_sign_boxes()` returns one row per boxed sign with its Wells number. Seals whose box count does not match are left out; `scripts/check_boxes.py` lists them.

## Alternate photos and mirrored views

A batch may name a better view of the same inscription in
`data/raw/sign_boxes/batch_NN/photo_overrides.csv`. Run
`python scripts/fetch_box_photos.py NN` after fetching Lipi. The K-39 archival
scan requires Poppler's `pdftoppm`. The other alternate views use the pinned Lipi
source or the linked published colour plate. Photo files stay in the ignored
external cache.

Boxes on an alternate view use that photo's filename stem. The box loader, checker
and guide generator use the selected photo's dimensions. A photograph whose visible sign sequence is reversed,
such as the selected L-16, L-58 or M-657 view, gets a filename-specific order
correction. The catalogue transcription is kept as recorded.
