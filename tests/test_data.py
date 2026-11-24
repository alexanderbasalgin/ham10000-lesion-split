import numpy as np

from skinlesion.data import META_DIM, encode_meta, lesion_split, load_metadata, normalize_dx, random_split


def test_normalize_labels():
    assert normalize_dx("mel") == "mel"
    assert normalize_dx("melanocytic_Nevi") == "nv"
    assert normalize_dx("benign_keratosis-like_lesions") == "bkl"


def test_lesion_split_has_no_shared_lesions(tiny_dataset):
    df = load_metadata(tiny_dataset)
    split = lesion_split(df)
    assert set(split) == {"train", "val", "test"}
    parts = {k: set(df.loc[split == k, "lesion_id"]) for k in ("train", "val", "test")}
    assert not parts["train"] & parts["test"]
    assert not parts["train"] & parts["val"]
    assert not parts["val"] & parts["test"]


def test_random_split_leaks_lesions(tiny_dataset):
    # the naive split is expected to put photos of one lesion on both sides
    df = load_metadata(tiny_dataset)
    split = random_split(df)
    assert set(df.loc[split == "train", "lesion_id"]) & set(df.loc[split == "test", "lesion_id"])


def test_encode_meta_handles_missing(tiny_dataset):
    df = load_metadata(tiny_dataset).head(3).copy()
    df.loc[0, "age"] = np.nan
    df.loc[1, "sex"] = "unknown"
    m = encode_meta(df)
    assert m.shape == (3, META_DIM)
    assert m[0, 1] == 1.0 and np.isfinite(m).all()
