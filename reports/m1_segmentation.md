# M1 segmentation against reviewed sign boxes

The coverage table shows which parts of the fixed splits have reviewed labels. Metrics for validation and test seals are reported separately.

| Split | Reviewed seals | Fixed split seals | Reviewed signs |
| --- | ---: | ---: | ---: |
| Validation | 28 | 67 | 163 |
| Test | 0 | 141 | 0 |

## Method

Both automatic methods trim the catalogue caption and use the existing inscription-band finder. Forced cuts receive the true count. A7 searches counts 1 through 13, using the same Wells CNN and an add-one length prior fit on the 466 training seals. It uses the historical A7 count weights 1 and 2 without retuning them for Wells.

One-to-one box matches maximize total intersection-over-union (IoU), with unmatched references scoring zero. A clean box covers at least 95% of its matched reference and less than 10% of each other reference. This strict rectangle-based measure can penalize intentional overlap in the hand labels.

The frozen CNN scores both crop sets using the same 64×64 preprocessing, with signs in reading order.

Wells checkpoint SHA-256: `d5c3821a90e08ed27f0861770306e854ca7a7f58149219db417cc5dfec158154`.

## Validation results

| Method | Mean IoU | IoU ≥ 0.50 | Cleanly covered signs | Exact count | Within one sign |
| --- | ---: | ---: | ---: | ---: | ---: |
| Forced cuts, true count supplied | 0.450 | 75/163 | 9/163 | 28/28 | 28/28 |
| A7 count search, same Wells CNN | 0.293 | 48/163 | 2/163 | 2/28 | 10/28 |

A7's mean absolute count error was 2.25 signs per seal.

| Crop source | Correct first guesses | Correct in first five |
| --- | ---: | ---: |
| Forced cuts | 53/163 | 73/163 |
| Reviewed boxes | 87/163 | 129/163 |

### Counts by seal

| Seal | True count | A7 count | Forced mean IoU | A7 mean IoU |
| --- | ---: | ---: | ---: | ---: |
| M-104 | 5 | 4 | 0.520 | 0.412 |
| M-107 | 6 | 4 | 0.504 | 0.423 |
| M-10 | 11 | 5 | 0.461 | 0.249 |
| M-118 | 8 | 5 | 0.460 | 0.297 |
| M-129 | 6 | 4 | 0.420 | 0.240 |
| M-130 | 6 | 3 | 0.489 | 0.248 |
| M-138 | 3 | 6 | 0.215 | 0.154 |
| M-143 | 6 | 3 | 0.430 | 0.150 |
| M-171 | 8 | 5 | 0.572 | 0.387 |
| M-187 | 2 | 5 | 0.536 | 0.291 |
| M-236 | 6 | 5 | 0.578 | 0.533 |
| M-24 | 7 | 5 | 0.125 | 0.096 |
| M-251 | 5 | 3 | 0.482 | 0.252 |
| M-260 | 6 | 3 | 0.557 | 0.159 |
| M-261 | 4 | 4 | 0.279 | 0.249 |
| M-269 | 3 | 3 | 0.338 | 0.392 |
| M-292 | 4 | 3 | 0.542 | 0.324 |
| M-31 | 6 | 5 | 0.440 | 0.411 |
| M-386 | 4 | 3 | 0.560 | 0.452 |
| M-38 | 13 | 6 | 0.286 | 0.141 |
| M-390 | 5 | 3 | 0.737 | 0.338 |
| M-404 | 2 | 3 | 0.700 | 0.745 |
| M-43 | 6 | 4 | 0.662 | 0.412 |
| M-50 | 9 | 4 | 0.324 | 0.141 |
| M-70 | 5 | 4 | 0.381 | 0.464 |
| M-72 | 7 | 5 | 0.193 | 0.132 |
| M-86 | 7 | 5 | 0.759 | 0.483 |
| M-9 | 3 | 2 | 0.401 | 0.288 |

## Reproduce

Fetch Lipi and supply the reviewed batch folders. For the 28-seal validation run, the labels are in [PR #62](https://github.com/RaghavSinghal15/indus-computational-epigraphy/pull/62).

```bash
python scripts/m1_segmentation.py --checkpoint /path/to/wells_all.pt --boxes-dir /path/to/data/raw/sign_boxes
```
