# Drawing sign boxes (issues #24 to #29)

Goal: a tight box around every sign on about 600 seal photos. You only draw boxes; nobody types sign numbers, because each seal's sequence is already known and boxes are labelled automatically from left to right. About an hour per batch of 100 seals.

## Once

```bash
pip install -r requirements.txt
python scripts/fetch_lipi.py            # downloads the photos (about 20 MB)
```

## For your batch (say batch 3)

1. `python scripts/make_box_batch.py 3` copies the 100 photos to `data/processed/box_batches/batch_03/`, plus a `guide/` folder showing each photo with its expected signs underneath, left to right.
2. Open **makesense.ai** in a browser, then Get Started. Drop in the 100 photos (not the `guide` folder) and choose **Object Detection**. Create one label: `sign`.
3. For each photo, draw one rectangle per sign (the Rect tool or the R key):
   - **Tight**: touch the outermost strokes, with no big margins. Include strokes that belong to the sign, like the "ears" of the jar sign.
   - **One box per sign in the guide.** Groups of short strokes (`||`, `|||`) count as one sign. If the guide shows 5 signs, draw 5 boxes.
   - **Only the text**, never the animal, the manger or the CISI caption.
   - A sign that is half broken off still gets a box around what is visible.
   - If a photo is unreadable or the guide clearly does not match the photo, skip it and write its CISI number in your PR.
4. Actions, Export Annotations, **YOLO format**. Unzip the `.txt` files into `data/raw/sign_boxes/batch_03/` (one file per photo, same name as the photo).
5. `python scripts/check_boxes.py 3 --show`. Fix every seal it lists, which usually means a missing or extra box. Then skim a few pictures in `data/processed/box_check/batch_03/`: the sign drawn under each box should match the sign in the box.
6. Commit only `data/raw/sign_boxes/batch_03/` and open a PR with `Closes #<issue>`.

Tip: makesense.ai keeps work only in the open browser tab. Export every 20 or 30 photos so you never lose more than that.

## How the labels are assigned

Boxes are sorted by their horizontal centre, left to right, and get the seal's known sequence in left-to-right order (the impression shows the text mirrored back into reading direction, so the first sign read is the right-most one). `indus.data.load_sign_boxes()` returns one row per boxed sign with its Wells number. Seals whose box count does not match are left out, which is why `check_boxes.py` must pass.
