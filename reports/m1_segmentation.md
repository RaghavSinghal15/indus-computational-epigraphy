# M1 segmentation against reviewed sign boxes

The coverage table shows which parts of the fixed splits have reviewed labels. Metrics for validation and test seals are reported separately.

| Split | Reviewed seals | Fixed split seals | Reviewed signs |
| --- | ---: | ---: | ---: |
| Validation | 28 | 67 | 163 |
| Test | 64 | 141 | 322 |

## Method

Both automatic methods trim the catalogue caption and use the existing inscription-band finder. Forced cuts receive the true count. A7 searches counts 1 through 13, using the same Wells CNN and an add-one length prior fit on the 466 training seals. It uses the historical A7 count weights 1 and 2 without retuning them for Wells.

One-to-one box matches maximize total intersection-over-union (IoU), with unmatched references scoring zero. A clean box covers at least 95% of its matched reference and less than 10% of each other reference. This strict rectangle-based measure can penalize intentional overlap in the hand labels.

The frozen CNN scores both crop sets using the same 64×64 preprocessing, with signs in reading order.

Wells checkpoint SHA-256: `d5c3821a90e08ed27f0861770306e854ca7a7f58149219db417cc5dfec158154`.

## Validation results

| Method | Mean IoU | IoU ≥ 0.50 | Cleanly covered signs | Exact count | Within one sign |
| --- | ---: | ---: | ---: | ---: | ---: |
| Forced cuts, true count supplied | 0.445 | 75/163 | 9/163 | 28/28 | 28/28 |
| A7 count search, same Wells CNN | 0.288 | 47/163 | 2/163 | 2/28 | 10/28 |

A7's mean absolute count error was 2.25 signs per seal.

| Crop source | Correct first guesses | Correct in first five |
| --- | ---: | ---: |
| Forced cuts | 53/163 | 73/163 |
| Reviewed boxes | 86/163 | 128/163 |

### Counts by seal

| Seal | True count | A7 count | Forced mean IoU | A7 mean IoU |
| --- | ---: | ---: | ---: | ---: |
| M-104 | 5 | 4 | 0.520 | 0.412 |
| M-107 | 6 | 4 | 0.455 | 0.365 |
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
| M-43 | 6 | 4 | 0.549 | 0.339 |
| M-50 | 9 | 4 | 0.324 | 0.141 |
| M-70 | 5 | 4 | 0.381 | 0.464 |
| M-72 | 7 | 5 | 0.193 | 0.132 |
| M-86 | 7 | 5 | 0.759 | 0.483 |
| M-9 | 3 | 2 | 0.401 | 0.288 |

## Test results

| Method | Mean IoU | IoU ≥ 0.50 | Cleanly covered signs | Exact count | Within one sign |
| --- | ---: | ---: | ---: | ---: | ---: |
| Forced cuts, true count supplied | 0.455 | 150/322 | 21/322 | 64/64 | 64/64 |
| A7 count search, same Wells CNN | 0.314 | 90/322 | 7/322 | 14/64 | 32/64 |

A7's mean absolute count error was 1.83 signs per seal.

| Crop source | Correct first guesses | Correct in first five |
| --- | ---: | ---: |
| Forced cuts | 107/322 | 181/322 |
| Reviewed boxes | 180/322 | 252/322 |

### Counts by seal

| Seal | True count | A7 count | Forced mean IoU | A7 mean IoU |
| --- | ---: | ---: | ---: | ---: |
| M-114 | 6 | 5 | 0.599 | 0.589 |
| M-119 | 10 | 3 | 0.359 | 0.113 |
| M-131 | 6 | 4 | 0.680 | 0.397 |
| M-132 | 3 | 3 | 0.718 | 0.631 |
| M-135 | 4 | 3 | 0.497 | 0.443 |
| M-140 | 7 | 4 | 0.338 | 0.217 |
| M-145 | 4 | 4 | 0.392 | 0.431 |
| M-146 | 5 | 5 | 0.185 | 0.246 |
| M-148 | 3 | 4 | 0.450 | 0.505 |
| M-149 | 5 | 5 | 0.431 | 0.359 |
| M-150 | 2 | 4 | 0.494 | 0.491 |
| M-151 | 3 | 6 | 0.183 | 0.227 |
| M-157 | 7 | 4 | 0.490 | 0.271 |
| M-164 | 5 | 10 | 0.127 | 0.106 |
| M-172 | 6 | 4 | 0.607 | 0.379 |
| M-173 | 5 | 4 | 0.527 | 0.575 |
| M-176 | 2 | 3 | 0.104 | 0.100 |
| M-195 | 6 | 5 | 0.402 | 0.254 |
| M-196 | 5 | 4 | 0.369 | 0.357 |
| M-197 | 2 | 3 | 0.492 | 0.375 |
| M-198 | 6 | 4 | 0.494 | 0.304 |
| M-1 | 5 | 3 | 0.595 | 0.254 |
| M-204 | 4 | 4 | 0.577 | 0.552 |
| M-211 | 7 | 3 | 0.519 | 0.194 |
| M-214 | 3 | 5 | 0.338 | 0.196 |
| M-228 | 3 | 3 | 0.360 | 0.273 |
| M-230 | 4 | 4 | 0.179 | 0.158 |
| M-238 | 3 | 5 | 0.295 | 0.198 |
| M-241 | 7 | 4 | 0.587 | 0.323 |
| M-242 | 5 | 5 | 0.450 | 0.320 |
| M-252 | 7 | 5 | 0.490 | 0.311 |
| M-255 | 3 | 5 | 0.194 | 0.100 |
| M-257 | 4 | 3 | 0.430 | 0.340 |
| M-259 | 4 | 3 | 0.304 | 0.252 |
| M-263 | 6 | 4 | 0.378 | 0.274 |
| M-264 | 2 | 5 | 0.176 | 0.304 |
| M-267 | 5 | 4 | 0.594 | 0.391 |
| M-274 | 7 | 4 | 0.692 | 0.288 |
| M-275 | 5 | 4 | 0.733 | 0.572 |
| M-280 | 7 | 5 | 0.649 | 0.415 |
| M-367 | 5 | 3 | 0.465 | 0.424 |
| M-369 | 11 | 3 | 0.517 | 0.078 |
| M-383 | 2 | 4 | 0.938 | 0.472 |
| M-393 | 8 | 5 | 0.577 | 0.323 |
| M-3 | 3 | 4 | 0.458 | 0.710 |
| M-401 | 3 | 3 | 0.436 | 0.388 |
| M-408 | 4 | 3 | 0.629 | 0.368 |
| M-411 | 7 | 3 | 0.458 | 0.161 |
| M-412 | 5 | 2 | 0.495 | 0.157 |
| M-51 | 8 | 6 | 0.330 | 0.289 |
| M-53 | 9 | 3 | 0.051 | 0.072 |
| M-54 | 7 | 5 | 0.339 | 0.206 |
| M-58 | 5 | 5 | 0.550 | 0.494 |
| M-64 | 6 | 5 | 0.374 | 0.254 |
| M-6 | 4 | 5 | 0.662 | 0.507 |
| M-73 | 3 | 3 | 0.738 | 0.716 |
| M-75 | 5 | 5 | 0.551 | 0.386 |
| M-832 | 3 | 3 | 0.376 | 0.462 |
| M-84 | 6 | 5 | 0.539 | 0.479 |
| M-8 | 3 | 5 | 0.344 | 0.651 |
| M-91 | 9 | 3 | 0.471 | 0.143 |
| M-95 | 5 | 4 | 0.514 | 0.323 |
| M-97 | 3 | 6 | 0.104 | 0.071 |
| M-99 | 5 | 5 | 0.376 | 0.450 |

## Reproduce

Fetch Lipi and supply the reviewed batch folders. The Batch 1 labels are in [PR #62](https://github.com/RaghavSinghal15/indus-computational-epigraphy/pull/62).

```bash
python scripts/m1_segmentation.py --checkpoint /path/to/wells_all.pt --boxes-dir /path/to/data/raw/sign_boxes
```
