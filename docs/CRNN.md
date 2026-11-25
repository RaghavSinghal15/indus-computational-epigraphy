# Whole-band reader (CRNN + CTC)

The CRNN starts from the trained all-site Wells v2 sign CNN. For each seal it
locates one inscription band from pixels, then reads the entire band. It
uses no supplied sign count or individual sign boxes at inference. The
automatic band finder still matters because it can leave part of a sign
outside the band.

The CNN makes a row of visual features. The model reverses that row into
right-to-left reading order without mirroring individual glyphs, then sends
it through a bidirectional LSTM. CTC uses the reference sign sequence to
learn an alignment with the image. Blank frames let the decoder keep two
consecutive instances of the same sign separate.

## Run

Fetch the pinned Lipi data using the repository script. Check the fixed split
and CTC frame lengths before using the GPU:

```bash
python scripts/train_crnn.py --check-data
python -u scripts/train_crnn.py --device cuda --init /path/to/wells_all.pt
```

The first run uses only real inscription bands. It takes the epoch with the
lowest validation sign error rate, using exact sequence accuracy as a
tiebreaker. It reports sign error rate, exact sequence accuracy and count
accuracy on the 141 test seals and the 181 held-out Harappa seals. The same
test seals are evaluated with the crop CNN using the true sign count and an
image-based estimate. Harappa stays outside training and model selection.

When #32 provides composites, pass `--composites /path/to/manifest.jsonl`.
Each row needs `image` relative to the manifest, `signs_reading` as a
list of Wells sign IDs in right-to-left reading order, and `source_seals`
listing every seal used for a real crop or background. The loader rejects
held-out sources. It samples at most one composite per real training band
each epoch. Record the real-only and composite runs separately.

Training outputs one checkpoint in ignored storage and the report
`reports/m1_crnn.md`. `--checkpoint` and `--output` can select other
destinations. The fixture tests cover repeated signs, direction, padding
length and edit distance.
