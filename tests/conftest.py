import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from skinlesion.data import CLASSES, META_DIM
from skinlesion.model import SkinNet


@pytest.fixture
def tiny_dataset(tmp_path):
    """60 synthetic 'lesions' (some with 2-3 photos) in the Kaggle HAM10000 layout."""
    rng = np.random.default_rng(0)
    rows, n_img = [], 0
    (tmp_path / "part_1").mkdir()
    for les in range(60):
        dx = CLASSES[les % len(CLASSES)]
        for _ in range(1 + les % 3):
            iid = f"ISIC_{n_img:07d}"
            arr = rng.integers(0, 255, (45, 60, 3), dtype=np.uint8)
            Image.fromarray(arr).save(tmp_path / "part_1" / f"{iid}.jpg")
            rows.append({"lesion_id": f"HAM_{les:07d}", "image_id": iid, "dx": dx, "dx_type": "histo",
                         "age": float(30 + les % 50), "sex": ["male", "female"][les % 2], "localization": "back"})
            n_img += 1
    pd.DataFrame(rows).to_csv(tmp_path / "HAM10000_metadata.csv", index=False)
    return tmp_path


@pytest.fixture
def tiny_ckpt(tmp_path):
    torch.manual_seed(0)
    model = SkinNet("resnet18", len(CLASSES), META_DIM, pretrained=False)
    path = tmp_path / "model.pt"
    torch.save({"state_dict": model.state_dict(), "backbone": "resnet18", "use_meta": True,
                "img_size": 64, "classes": CLASSES}, path)
    return path
