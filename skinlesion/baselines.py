"""Cheap baselines: class prior and hand-crafted color features."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import CLASSES


def prior_proba(y_train, n):
    counts = np.bincount(y_train, minlength=len(CLASSES)).astype(float)
    return np.tile(counts / counts.sum(), (n, 1))


def color_features(images: np.ndarray, bins: int = 8) -> np.ndarray:
    """HSV histograms of the central region plus center-vs-border color contrast."""
    import matplotlib.colors as mcolors

    feats = []
    for img in images:
        hsv = mcolors.rgb_to_hsv(img[::2, ::2] / 255.0)
        h, w = hsv.shape[:2]
        center = hsv[h // 4: 3 * h // 4, w // 4: 3 * w // 4].reshape(-1, 3)
        border = np.concatenate([hsv[: h // 8].reshape(-1, 3), hsv[-h // 8:].reshape(-1, 3)])
        hists = [np.histogram(center[:, c], bins=bins, range=(0, 1))[0] / len(center) for c in range(3)]
        stats = [center.mean(0), center.std(0), center.mean(0) - border.mean(0),
                 np.percentile(center[:, 2], [5, 25, 50])]
        feats.append(np.concatenate(hists + stats))
    return np.asarray(feats, dtype=np.float32)


def linear_head(C=1.0):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))
