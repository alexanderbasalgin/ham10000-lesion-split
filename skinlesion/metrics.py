import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score

from .data import CLASSES, MALIGNANT

MEL = CLASSES.index("mel")
MALIGNANT_IDX = [CLASSES.index(c) for c in MALIGNANT]


def sens_at_spec(y_true, score, spec=0.9):
    """Sensitivity at the threshold that keeps the given specificity."""
    neg = score[y_true == 0]
    thr = np.quantile(neg, spec)
    return float((score[y_true == 1] > thr).mean())


def summarize(y, proba) -> dict:
    y = np.asarray(y)
    proba = np.asarray(proba)
    pred = proba.argmax(1)
    labels = list(range(len(CLASSES)))
    is_mel = (y == MEL).astype(int)
    is_malig = np.isin(y, MALIGNANT_IDX).astype(int)
    malig_score = proba[:, MALIGNANT_IDX].sum(1)
    out = {
        "n": int(len(y)),
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "per_class_recall": dict(zip(CLASSES, map(float, recall_score(y, pred, labels=labels, average=None, zero_division=0)))),
        "malignant_recall": float(np.isin(pred, MALIGNANT_IDX)[is_malig == 1].mean()) if is_malig.any() else float("nan"),
    }
    if 0 < is_mel.sum() < len(y):
        out["mel_auc"] = float(roc_auc_score(is_mel, proba[:, MEL]))
        out["mel_sens_at_90spec"] = sens_at_spec(is_mel, proba[:, MEL], 0.9)
    if 0 < is_malig.sum() < len(y):
        out["malignant_auc"] = float(roc_auc_score(is_malig, malig_score))
    return out


def bootstrap_ci(y, proba, metric="balanced_accuracy", n=1000, seed=0, groups=None):
    """Percentile CI; resamples whole lesions when groups are given."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    proba = np.asarray(proba)
    if groups is None:
        groups = np.arange(len(y))
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[g] for g in pick])
        pred = proba[idx].argmax(1)
        if metric == "balanced_accuracy":
            vals.append(balanced_accuracy_score(y[idx], pred))
        elif metric == "macro_f1":
            vals.append(f1_score(y[idx], pred, labels=list(range(len(CLASSES))), average="macro", zero_division=0))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
