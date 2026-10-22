"""Generate a tiny fake COCO-style filament dataset for smoke-testing the notebooks.

Layout mirrors the real competition data (names are fake):

    <out>/SYNTHETIC_Kaggle/train/SYNTHETIC_Annotations_train.json
    <out>/SYNTHETIC_Kaggle/train/train_images/YYYYMMDDhhmmssXx.jpeg

Images are 256 x 256 grayscale JPEGs: a bright disc with a few dark elongated blobs, each annotated
with a polygon. The images contain no real data. The output directory must be outside the repository.

Usage:
    python tests/make_synthetic_coco.py              # writes to a new temp dir, prints DATA_DIR
    python tests/make_synthetic_coco.py --out DIR    # DIR must not be inside the repo
"""
import argparse
import json
import os
import tempfile

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIZE = 256
DAYS = ["20140609", "20140610", "20150301", "20150302", "20160510", "20160511", "20170815", "20180102",
        "20190320", "20200704", "20211111", "20220803"]
CATEGORIES = [{"supercategory": "synthetic", "id": i, "name": f"synthetic_{i}"} for i in (1, 2, 3, 4)]


def _blob_polygon(rng):
    """A thin, slightly bent strip as a polygon (flat [x1, y1, x2, y2, ...] list)."""
    cx, cy = rng.uniform(70, SIZE - 70, size=2)
    angle = rng.uniform(0, np.pi)
    length, width = rng.uniform(25, 60), rng.uniform(3, 7)
    t = np.linspace(-0.5, 0.5, 8) * length
    bend = rng.uniform(-0.02, 0.02) * t ** 2
    d = np.array([np.cos(angle), np.sin(angle)])
    n = np.array([-d[1], d[0]])
    centre = np.stack([cx + t * d[0] + bend * n[0], cy + t * d[1] + bend * n[1]], 1)
    pts = np.concatenate([centre + width / 2 * n, (centre - width / 2 * n)[::-1]])
    return [round(float(v), 2) for v in pts.reshape(-1)]


def make(out_dir, seed=0):
    out_dir = os.path.abspath(out_dir)
    if os.path.commonpath([out_dir, REPO_ROOT]) == REPO_ROOT:
        raise SystemExit(f"refusing to write inside the repository: {out_dir}")
    rng = np.random.default_rng(seed)
    train_dir = os.path.join(out_dir, "SYNTHETIC_Kaggle", "train")
    img_dir = os.path.join(train_dir, "train_images")
    os.makedirs(img_dir, exist_ok=True)

    yy, xx = np.mgrid[:SIZE, :SIZE]
    disc = (xx - SIZE / 2) ** 2 + (yy - SIZE / 2) ** 2 <= (0.45 * SIZE) ** 2
    images, annotations = [], []
    for day in DAYS:
        for _ in range(int(rng.integers(2, 5))):  # 2-4 frames per day
            hhmmss = f"{rng.integers(0, 24):02d}{rng.integers(0, 60):02d}{rng.integers(0, 60):02d}"
            file_name = f"{day}{hhmmss}{rng.choice(['Mh', 'Bh'])}.jpeg"
            if any(r["file_name"] == file_name for r in images):
                continue
            image_id = len(images) + 1
            canvas = Image.fromarray(np.where(disc, 170, 10).astype(np.uint8))
            draw = ImageDraw.Draw(canvas)
            for _ in range(int(rng.integers(1, 4))):
                poly = _blob_polygon(rng)
                draw.polygon(list(zip(poly[::2], poly[1::2])), fill=70)
                xs, ys = poly[::2], poly[1::2]
                seg = [poly] if rng.random() < 0.5 else poly  # both nested and flat polygon formats
                annotations.append({
                    "id": len(annotations) + 1, "image_id": image_id,
                    "category_id": int(rng.integers(1, 4)), "segmentation": seg, "iscrowd": 0,
                    "bbox": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)], "area": 0.0})
            arr = np.asarray(canvas, dtype=np.float32) + rng.normal(0, 6, (SIZE, SIZE))
            Image.fromarray(arr.clip(0, 255).astype(np.uint8)).save(os.path.join(img_dir, file_name), quality=90)
            images.append({"id": image_id, "file_name": file_name, "height": SIZE, "width": SIZE})

    # The real JSON has several image records that share one file name; add two such records here.
    for rec in images[:2]:
        image_id = len(images) + 1
        poly = _blob_polygon(rng)
        annotations.append({"id": len(annotations) + 1, "image_id": image_id, "category_id": 1,
                            "segmentation": [poly], "iscrowd": 0, "bbox": [0, 0, 0, 0], "area": 0.0})
        images.append({**rec, "id": image_id})

    with open(os.path.join(train_dir, "SYNTHETIC_Annotations_train.json"), "w") as f:
        json.dump({"images": images, "annotations": annotations, "categories": CATEGORIES}, f)
    return out_dir, len(images), len(annotations)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="output directory (default: a new temp directory)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out, n_img, n_ann = make(args.out or tempfile.mkdtemp(prefix="synthetic_coco_"), args.seed)
    print(f"{n_img} images, {n_ann} annotations, {len(DAYS)} days")
    print(f"DATA_DIR={out}")
