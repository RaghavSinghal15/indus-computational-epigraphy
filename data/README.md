# Data

Everything in `raw/` is treated as read-only. Derived files go in `processed/`, which is not committed.

**Orientation:** every sequence file stores signs in visual order (left to right on the seal). The script reads right to left. Use `indus.data.load_sequences()`, which returns reading order by default. See `src/indus/data.py` for the full note.

## Files

| File | What it is | Numbering | Source | Licence |
|---|---|---|---|---|
| `raw/icit_1837_sequences.csv` | 1,837 unique inscriptions, one per row, 8,359 sign occurrences, 568 distinct signs. Excludes damaged, multi-line and single-sign texts. Left-to-right texts were flipped so all rows share one direction. | Wells | Supplementary S2 of Ashraf & Sinha (IMSc Chennai), "The 'handedness' of language: Directional symmetry breaking of sign usage in words", PLOS ONE 2018 (doi:10.1371/journal.pone.0190735), built from the ICIT database (Wells, Fuls) | CC BY 4.0 |
| `raw/im77_bigrams.csv` | 2,730 adjacent sign pairs with counts | Mahadevan (1 to 417) | IM77 web app, indusscript.in (Indus Research Centre, RMRL), Tables tab, parsed from its PDF export | RMRL. Keep repo private; check terms before any public release |
| `raw/im77_trigrams.csv` | 3,681 sign triples with counts | Mahadevan | Same as above | Same as above |
| `raw/sign_mapping_parpola_wells_mahadevan.csv` | 397 Parpola signs, each with its Wells and Mahadevan codes. Covers 359 of 417 Mahadevan signs. Many-to-many in places. | All three | `features/` folder of github.com/mayig/indus-valley-script-corpus. One developer's hand mapping, not an authoritative concordance. | MIT |
| `raw/sign_templates/N.png` | 417 reference drawings, one per Mahadevan sign, 150x150 grayscale | Mahadevan | Kaggle, storesource/im-417-150, from Mahadevan (1977) pp. 32-34 | Apache 2.0 |

## Known limitations

- **The two corpora are different.** The 1,837 sequences (ICIT, Wells) and the IM77 tables (Mahadevan) describe overlapping but not identical sets of inscriptions. Compare their statistics at the level of patterns, not exact counts.
- **Unique texts only.** Duplicate inscriptions were removed from the 1,837, so frequencies describe distinct texts, not how often texts occur on real objects.
- **No object metadata.** The 1,837 file has no site, object type or text ID. Rows are sorted by their first number.
- **Mapping gaps.** 58 Mahadevan signs have no mapping, and some Wells signs map to more than one Mahadevan sign. This only matters where M1 output (Mahadevan) meets the sequences (Wells).

## Not in the repo, but useful

- Wells sign key (what each Wells number looks like): Supplementary file `Keylist_Udeciphered_Indus.pdf` in the same PLOS S2 zip.
- Seal photos for M1: see `docs/PLAN.md`.
