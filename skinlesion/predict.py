"""Classify one dermoscopic image.

python -m skinlesion.predict path/to/image.jpg --age 60 --sex male --site back
"""
import argparse
import io
import json
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
from PIL import Image

from .data import CLASS_NAMES, CLASSES, MALIGNANT, META_DIM, encode_meta, load_image
from .model import SkinNet
from .train import predict_proba

DEFAULT_CKPT = "artifacts/finetune.pt"


@lru_cache(maxsize=2)
def load_model(ckpt_path=DEFAULT_CKPT):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = SkinNet(ckpt["backbone"], len(ckpt["classes"]), META_DIM if ckpt["use_meta"] else 0, pretrained=False)
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ckpt["state_dict"].items()})
    return model.eval(), ckpt["img_size"]


def predict(image, age=None, sex=None, site=None, model=None, img_size=224) -> dict:
    """image: path, bytes or PIL image. Returns class probabilities sorted high to low."""
    if model is None:
        model, img_size = load_model()
    if isinstance(image, bytes):
        image = io.BytesIO(image)
    arr = load_image(image, 256)[None]
    row = pd.DataFrame([{"age": np.nan if age is None else float(age),
                         "sex": sex or "unknown", "localization": site or "unknown"}])
    proba = predict_proba(model, arr, encode_meta(row), torch.device("cpu"), img_size)[0]
    order = np.argsort(-proba)
    return {
        "prediction": CLASSES[order[0]],
        "prediction_name": CLASS_NAMES[CLASSES[order[0]]],
        "malignant_probability": round(float(sum(proba[CLASSES.index(c)] for c in MALIGNANT)), 4),
        "probabilities": {CLASSES[i]: round(float(proba[i]), 4) for i in order},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--age", type=float)
    ap.add_argument("--sex", choices=["male", "female"])
    ap.add_argument("--site", help="body site, e.g. back, face, lower extremity")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    args = ap.parse_args()
    model, size = load_model(args.ckpt)
    print(json.dumps(predict(Image.open(args.image), args.age, args.sex, args.site, model, size), indent=2))


if __name__ == "__main__":
    main()
