import numpy as np
import torch

from skinlesion.baselines import color_features
from skinlesion.data import CLASSES, META_DIM
from skinlesion.model import SkinNet
from skinlesion.predict import load_model, predict
from skinlesion.train import FinetuneConfig, finetune


def test_forward_with_meta():
    model = SkinNet("resnet18", len(CLASSES), META_DIM, pretrained=False).eval()
    x = torch.randint(0, 255, (2, 3, 64, 64), dtype=torch.uint8)
    assert model(x, torch.zeros(2, META_DIM)).shape == (2, len(CLASSES))


def test_color_features_shape():
    imgs = np.random.default_rng(0).integers(0, 255, (3, 64, 64, 3), dtype=np.uint8)
    assert color_features(imgs).shape == (3, 36)


def test_finetune_one_epoch_runs():
    rng = np.random.default_rng(0)
    images = rng.integers(0, 255, (32, 72, 72, 3), dtype=np.uint8)
    y = np.arange(32) % len(CLASSES)
    meta = np.zeros((32, META_DIM), np.float32)
    cfg = FinetuneConfig(backbone="resnet18", pretrained=False, epochs=1, batch_size=8, img_size=64, workers=0)
    model, hist = finetune(cfg, images, y, meta, np.arange(24), np.arange(24, 32), torch.device("cpu"), log=lambda s: None)
    assert len(hist) == 1 and np.isfinite(hist[0]["loss"])


def test_predict_returns_distribution(tiny_ckpt, tiny_dataset):
    model, size = load_model(str(tiny_ckpt))
    img = next((tiny_dataset / "part_1").glob("*.jpg"))
    out = predict(str(img), age=55, sex="female", site="back", model=model, img_size=size)
    assert out["prediction"] in CLASSES
    assert abs(sum(out["probabilities"].values()) - 1) < 1e-3
    assert 0 <= out["malignant_probability"] <= 1
