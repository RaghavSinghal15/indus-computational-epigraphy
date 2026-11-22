"""Cut every v2 seal (Lipi photos, Wells labels) into its signs (issue #22).

Same forced cuts as v1 (sign count given; src/indus/real_crops.py), after trimming
the caption strip under each CISI photo. Split from data/raw/splits/v2_seals.csv.
Writes data/processed/real_crops/crops_v2.npz:
    images  uint8 (N, 64, 64)    labels  int (N,)  Wells code
    seal    str   (N,)           split   str (N,)  train / val / test / harappa
    pos     int   (N,)           position in reading order (0 = read first)
"""
import numpy as np

from indus.data import REPO_ROOT
from indus.real_crops import sign_crops
from indus.segment import load_image, trim_caption
from indus.wells import load_v2_seals

OUT = REPO_ROOT / "data" / "processed" / "real_crops" / "crops_v2.npz"


def main() -> None:
    images, labels, seals, splits, pos = [], [], [], [], []
    v2 = load_v2_seals()
    for _, s in v2.iterrows():
        crops, _, _ = sign_crops(trim_caption(load_image(s.image)), s.n_signs)   # reading order
        for i, (crop, sign) in enumerate(zip(crops, s.signs_reading)):
            images.append(crop); labels.append(sign); seals.append(s.cisi); splits.append(s.split); pos.append(i)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, images=np.stack(images), labels=np.array(labels), seal=np.array(seals),
                        split=np.array(splits), pos=np.array(pos))
    split = np.array(splits)
    print(f"{len(images)} crops from {len(v2)} seals -> {OUT.relative_to(REPO_ROOT)}")
    for name in ("train", "val", "test", "harappa"):
        print(f"  {name}: {(split == name).sum()} crops, {len(set(np.array(labels)[split == name]))} distinct signs")


if __name__ == "__main__":
    main()
