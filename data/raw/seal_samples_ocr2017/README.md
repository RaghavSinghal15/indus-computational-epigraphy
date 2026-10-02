# Sample text rows (OCR-on-Indus-Seals, 2017)

22 photos of the **text row only** of Indus seals (not the full seal), cropped by the pipeline in Palaniappan and Adhikari, "Deep Learning the Indus Script" (arXiv:1702.00523, 2017).

- **Source:** `code/Test/Regions/<id>/*_text.png` in github.com/tpsatish95/OCR-on-Indus-Seals. Renamed here to `text_row_<id>.png`; `<id>` is that repo's image number, not a CISI or IM77 text number.
- **Licence:** Apache 2.0. Original seal images credited by the authors to the Indus Research Centre and Roja Muthiah Research Library, with one image from harappa.com.
- **Used for:** developing and demonstrating A1 (splitting a text row into individual signs) until the D1 seal photo dataset is ready.

## Things to know

- **No ground truth.** We do not know which signs these are or which inscription each photo shows, so these can show the segmentation working but cannot give an accuracy number. That needs D2.
- **Seal or impression is unknown.** A seal is carved in reverse; its impression (sealing) reads correctly, right to left. Photos of the seal itself show the text mirrored, which would flip both the sign shapes and the reading order. Treat reading-order output on these photos as provisional.
- Lighting, colour and resolution vary a lot between photos (white, gold, grey stone), which is useful for testing robustness.
