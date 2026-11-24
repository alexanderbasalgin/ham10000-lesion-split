import numpy as np

from skinlesion.data import CLASSES
from skinlesion.metrics import bootstrap_ci, summarize


def test_perfect_predictions():
    y = np.repeat(np.arange(len(CLASSES)), 5)
    proba = np.eye(len(CLASSES))[y]
    m = summarize(y, proba)
    assert m["balanced_accuracy"] == 1.0
    assert m["mel_auc"] == 1.0 and m["malignant_recall"] == 1.0
    lo, hi = bootstrap_ci(y, proba, n=50)
    assert lo == hi == 1.0


def test_majority_class_has_low_balanced_accuracy():
    y = np.array([5] * 90 + [4] * 10)
    proba = np.tile(np.eye(len(CLASSES))[5], (100, 1))
    m = summarize(y, proba)
    assert m["accuracy"] == 0.9
    assert m["balanced_accuracy"] == 0.5
