import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_NAMES = {
    "akiec": "actinic keratosis / Bowen's",
    "bcc": "basal cell carcinoma",
    "bkl": "benign keratosis",
    "df": "dermatofibroma",
    "mel": "melanoma",
    "nv": "melanocytic nevus",
    "vasc": "vascular lesion",
}
MALIGNANT = {"akiec", "bcc", "mel"}

# the Hugging Face mirror spells labels out
_ALIASES = {
    "actinic_keratoses": "akiec",
    "basal_cell_carcinoma": "bcc",
    "benign_keratosis-like_lesions": "bkl",
    "dermatofibroma": "df",
    "melanoma": "mel",
    "melanocytic_nevi": "nv",
    "vascular_lesions": "vasc",
}

SEXES = ["male", "female"]
SITES = ["abdomen", "acral", "back", "chest", "ear", "face", "foot", "genital", "hand",
         "lower extremity", "neck", "scalp", "trunk", "upper extremity"]


def normalize_dx(dx: str) -> str:
    dx = str(dx).strip()
    key = dx.lower()
    if key in CLASSES:
        return key
    if key in _ALIASES:
        return _ALIASES[key]
    raise ValueError(f"unknown diagnosis label: {dx}")


def find_dataset(root) -> tuple[Path, dict[str, Path]]:
    """Locate HAM10000_metadata.csv and all ISIC_*.jpg files anywhere under root."""
    meta, images = None, {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f == "HAM10000_metadata.csv" and meta is None:
                meta = Path(dirpath) / f
            elif f.lower().endswith(".jpg") and f.startswith("ISIC_"):
                images.setdefault(f[:-4], Path(dirpath) / f)
    if meta is None:
        raise FileNotFoundError(f"HAM10000_metadata.csv not found under {root}")
    return meta, images


def load_metadata(root) -> pd.DataFrame:
    meta_path, images = find_dataset(root)
    df = pd.read_csv(meta_path)
    df["dx"] = df["dx"].map(normalize_dx)
    df["label"] = df["dx"].map(CLASSES.index)
    df["path"] = df["image_id"].map(lambda i: str(images[i]) if i in images else None)
    missing = df["path"].isna().sum()
    if missing:
        print(f"warning: {missing} images listed in metadata are missing, dropping them")
        df = df[df["path"].notna()]
    df["sex"] = df["sex"].where(df["sex"].isin(SEXES), "unknown")
    df["localization"] = df["localization"].where(df["localization"].isin(SITES), "unknown")
    return df.reset_index(drop=True)


def lesion_split(df: pd.DataFrame, seed: int = 0, n_folds: int = 7) -> pd.Series:
    """Train/val/test assignment with all images of a lesion kept in one part.

    HAM10000 has several photos of the same lesion (~1500 lesions have 2-6
    images). Splitting by image puts near-duplicates on both sides.
    Uses 1/7 for test, 1/7 for val, the rest for train, stratified by class.
    """
    n_folds = min(n_folds, df.groupby("dx")["lesion_id"].nunique().min())
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold = np.zeros(len(df), dtype=int)
    for k, (_, idx) in enumerate(sgkf.split(df, df["label"], groups=df["lesion_id"])):
        fold[idx] = k
    return pd.Series(np.select([fold == 0, fold == 1], ["test", "val"], "train"), index=df.index)


def random_split(df: pd.DataFrame, seed: int = 0, n_folds: int = 7) -> pd.Series:
    """Naive image-level split, used only to measure how much leakage inflates scores."""
    n_folds = min(n_folds, df["dx"].value_counts().min())
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold = np.zeros(len(df), dtype=int)
    for k, (_, idx) in enumerate(skf.split(df, df["label"])):
        fold[idx] = k
    return pd.Series(np.select([fold == 0, fold == 1], ["test", "val"], "train"), index=df.index)


def load_image(path, size: int) -> np.ndarray:
    # HAM10000 images are 600x450; center-crop to square then resize
    img = (path if isinstance(path, Image.Image) else Image.open(path)).convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    return np.asarray(img.resize((size, size), Image.BICUBIC), dtype=np.uint8)


def load_images(paths, size: int = 256, workers: int = 8) -> np.ndarray:
    with ThreadPoolExecutor(workers) as ex:
        return np.stack(list(ex.map(lambda p: load_image(p, size), paths)))


def encode_meta(df: pd.DataFrame) -> np.ndarray:
    """Age (scaled, with missing flag) + one-hot sex + one-hot body site."""
    age = df["age"].astype(float)
    cols = [
        (age.fillna(50.0) / 100.0).to_numpy()[:, None],
        age.isna().to_numpy(dtype=float)[:, None],
        np.stack([(df["sex"] == s).to_numpy(dtype=float) for s in SEXES], 1),
        np.stack([(df["localization"] == s).to_numpy(dtype=float) for s in SITES], 1),
    ]
    return np.concatenate(cols, 1).astype(np.float32)


META_DIM = 2 + len(SEXES) + len(SITES)
