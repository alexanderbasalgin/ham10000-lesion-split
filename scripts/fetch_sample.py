"""Download a small HAM10000 sample (metadata + images) for local development.

The full dataset (~3 GB) is only used on the GPU box; this pulls a few hundred
rows through the Hugging Face datasets-server API and writes them in the same
layout as the Kaggle copy: HAM10000_metadata.csv + <image_id>.jpg files.
"""
import argparse
import csv
import io
import time
from pathlib import Path

import requests
from PIL import Image

API = "https://datasets-server.huggingface.co/rows"
COLS = ["lesion_id", "image_id", "dx", "dx_type", "age", "sex", "localization"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sample")
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    rows = []
    # rows are sorted by class in the HF export, so sample from spread-out offsets
    for offset in range(0, 9000, 9000 // (args.n // 20)):
        params = {"dataset": "marmal88/skin_cancer", "config": "default",
                  "split": "train", "offset": offset, "length": 20}
        for attempt in range(5):
            r = requests.get(API, params=params, timeout=60)
            if r.ok:
                break
            time.sleep(5 * (attempt + 1))
        r.raise_for_status()
        for item in r.json()["rows"]:
            row = item["row"]
            img = requests.get(row["image"]["src"], timeout=60).content
            Image.open(io.BytesIO(img)).convert("RGB").save(out / "images" / f"{row['image_id']}.jpg", quality=92)
            rows.append({c: row[c] for c in COLS})
        print(f"offset {offset}: {len(rows)} rows")

    with open(out / "HAM10000_metadata.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
