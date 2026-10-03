# A4: M1 pipeline on real seal impressions

Seals: 242 Mohenjo-daro impressions with verified sequences (data/raw/cisi_seals).

| Measure | Result |
| --- | --- |
| A1 finds the right number of signs | 20/242 (8%) |
| A1 finds too few (merges) | 55 |
| A1 finds too many (splits / junk) | 167 |
| Full pipeline, top-1 per sign (seals with right count) | 1/110 (0.9%) |
| Full pipeline, top-5 per sign | 7/110 (6.4%) |
| A2 with equal-width boxes, top-1 | 4/1190 (0.3%) |
| A2 with equal-width boxes, top-5 | 15/1190 (1.3%) |
| Chance level, top-1 / top-5 | 0.2% / 1.2% |

Box count minus true count: -5: 1, -4: 3, -3: 6, -2: 19, -1: 26, +0: 20, +1: 26, +2: 24, +3: 20, +4: 23, +5: 17, +6: 22, +7: 14, +8: 9, +9: 5, +10: 3, +11: 1, +13: 1, +16: 1, +22: 1
