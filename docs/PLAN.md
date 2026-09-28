# Plan

## Team workflow

- Two parallel tracks. Each person owns one track and explains it to the other at the end of each working day.
- AI tools write the code. Our job is understanding it and checking it. For every result: know what the output should look like before running it, then verify it against something known.
- Pull before you start, push when you stop. Don't both edit the same file at once. For anything larger than a quick fix, work on a branch and open a pull request.

## Tracks until the first evaluation (October 6)

| Track | Owner | Goal for Oct 6 |
|---|---|---|
| A: M1 vision | TBD | Seal photo in, boxes around signs, predicted Mahadevan IDs out. Accuracy can be rough; it must run end to end. |
| B: data + M2 | TBD | Sign frequency distribution, start vs end signs, reading direction from the data, first transition matrix. |

## Oct 6 evaluation (4 to 5 minutes, 2 to 3 slides)

1. **Data:** what we assembled and verified, including the orientation check using the jar sign.
2. **M1 demo:** short screen recording of the pipeline running on a real seal.
3. **First M2 results + next steps.**

## M1 data situation

- Have: 417 clean reference drawings (`data/raw/sign_templates/`).
- Need: a few dozen seal photos to prototype on, and about 200 hand-labelled real sign crops as a test set later.
- Photo sources to try: sample images in github.com/tpsatish95/OCR-on-Indus-Seals, harappa.com, indusscript.in, museum collections. Dataset requested from D. Mitra (Florida Tech); no reply yet.
- Training data options: synthetic augmentation of the templates; crowdsourced hand-drawn signs (Saini, Shibata, Takama method).

## Later milestones

M3 and M4 follow after the Oct 6 evaluation. Extensions (error propagation, null-model benchmark) after that.
