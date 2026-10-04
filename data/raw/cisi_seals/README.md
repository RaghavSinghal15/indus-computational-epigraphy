# Labelled seal photos (CISI Vol. 1 + IM77 sequences)

Real photographs of Indus seals, each with its sign sequence, for training and testing M1.
Currently 328 Mohenjo-daro seals (281 with verified sequences, 324 with an impression photo), 1,673 sign occurrences, 225 distinct signs. Rebuild with `python scripts/cisi/build_dataset.py <photo dir>`.

## Files

| File | What it is |
|---|---|
| `labelled_seals_M.csv` | One row per seal: `cisi_label`, `im77`, `impression_photo`, `seal_photo`, `signs_impression_left_to_right`, `signs_reading_order`, `n_signs`, `source`, `verified` |
| `sequences/*.csv` | The sequences per source (hover, reader, manual, chatgpt_a/b) and `multiline.txt`; see the build script for the order of trust |
| `photos/M-<n>_a.jpg` | Photo of the **impression** (reads correctly, right to left). Use this one. |
| `photos/M-<n>_A.jpg` | Photo of the **seal itself** (mirrored). |
| `cisi_concordance.csv` | All 1,542 CISI Vol. 1 objects and the number in column 2 of the book's data table (pp. 366-375 of the book, PDF pages 402-411) |
| `site_glyphs/<n>.png`, `meta.json` | indusscript.in's own drawing of each of the 417 signs, plus `0.png` (the site's mark for an illegible sign); used to read sequences off the site |

## Sources

- **Photos:** Joshi & Parpola (eds.), *Corpus of Indus Seals and Inscriptions*, Vol. 1 (1987), scanned PDF from IGNCA (ignca.gov.in/Asi_data/76884.pdf). Photos cut out automatically and matched to their printed captions (`scripts/cisi/photo_extraction/`).
- **Sequences:** Mahadevan & RMRL, *The Indus Script Web Application (IM77/IDF80)*, indusscript.in (2021). Read from the site's Browse page, one text at a time, with the user's login.
- **Copyright:** both belong to their publishers. This repo is private; ask RMRL and the CISI editors before any public release.

## How the sequences were read

1. Look up the IM77 number on the Browse page and save a screenshot of the text card (`scripts/cisi/CAPTURE.md`).
2. `scripts/cisi/read_card2.py` reads the signs from the screenshot. It matches each drawn sign against the site's own 417 drawings on shape, size and height on the line, using exactly the number of signs the card states ("Length").
3. Checks:
   - **Hover check:** 14 texts were read by hovering every sign (the site shows "Sign: N"), and the reader agrees on all 14.
   - **Random audit:** 10 random cards were compared by eye, all correct.
   - **ChatGPT cross-check:** 3 texts were read independently by ChatGPT's agent, and all agree.
   - `source` says which method gave each row.
4. **Manual and ChatGPT readings:** Raghav read 50 texts by hand and two ChatGPT agent runs read the rest. Where they overlapped (24 texts) they disagreed on 3; the reader settled all 3 in favour of the manual reading. ChatGPT rows therefore have `verified = 0` and are kept out of test sets.
5. **Sanity check (AGENTS.md rule 5):** in reading order the jar sign (342) is the last sign 116 times and the first sign 0 times.

## Conventions and caveats

- **Sign order:** `signs_impression_left_to_right` is the order seen on the impression photo, which is also the site's card order. `signs_reading_order` is the reverse (right to left, the order the script is read).
- **Doubtful and illegible signs:** `*342` means the site marks the sign as doubtful (drawn grey). `0` means illegible. Leave both out of accuracy scoring.
- **CISI and IM77 numbers:** for Mohenjo-daro, the CISI table's number equals the IM77 text number. This was checked by eye on M-8, M-15 and M-16. Numbers below 1001 are objects that are not in IM77, so they have no sequence here. Other sites may use a different numbering (Lothal is 6xxx in CISI and 7xxx in IM77) and need checking before use.
- **Left out:** multi-line texts (the site splits them over several cards) are excluded for now.
- **Uncertain captions:** `caption_confident = 0` means the photo's caption needed heavy OCR correction, so check those by eye.
