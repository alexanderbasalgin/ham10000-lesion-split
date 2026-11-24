"""Turn a run directory (metrics.json, test predictions, thumbs) into README figures and tables."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NAMES = {
    "prior": "class prior",
    "color_lr": "color histograms + LR",
    "probe": "DINOv2-S frozen + LR",
    "probe_meta": "DINOv2-S frozen + metadata + LR",
    "finetune_noweight": "ConvNeXt-T fine-tuned, plain CE",
    "finetune": "ConvNeXt-T fine-tuned",
    "finetune_meta": "ConvNeXt-T fine-tuned + metadata",
}

plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})


def results_figure(m, path):
    keys = [k for k in NAMES if f"{k}/lesion" in m]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.35, 1]})
    vals = [m[f"{k}/lesion"]["balanced_accuracy"] for k in keys]
    ci = np.array([m[f"{k}/lesion"]["bal_acc_ci"] for k in keys])
    ypos = np.arange(len(keys))
    a1.barh(ypos, vals, height=0.6, color=BLUE, xerr=[np.array(vals) - ci[:, 0], ci[:, 1] - np.array(vals)],
            error_kw={"ecolor": INK2, "lw": 1, "capsize": 2})
    for yy, v, hi in zip(ypos, vals, ci[:, 1]):
        a1.text(hi + 0.015, yy, f"{v:.3f}", va="center", color=INK, fontsize=9)
    a1.set_yticks(ypos, [NAMES[k] for k in keys])
    a1.invert_yaxis()
    a1.set_xlim(0, 1)
    a1.set_xlabel("balanced accuracy, lesion-level test set (95% CI)")
    a1.grid(axis="x", color=GRID, lw=0.8)
    a1.set_axisbelow(True)
    a1.set_title("Test balanced accuracy over 7 classes", loc="left", color=INK)

    pairs = [k for k in ("color_lr", "probe", "finetune") if f"{k}/image" in m]
    x = np.arange(len(pairs))
    w = 0.36
    les = [m[f"{k}/lesion"]["balanced_accuracy"] for k in pairs]
    img = [m[f"{k}/image"]["balanced_accuracy"] for k in pairs]
    a2.bar(x - w / 2 - 0.01, les, w, color=BLUE, label="split by lesion")
    a2.bar(x + w / 2 + 0.01, img, w, color=ORANGE, label="split by image (leaky)")
    for xx, a, b in zip(x, les, img):
        a2.text(xx - w / 2, a + 0.015, f"{a:.2f}", ha="center", fontsize=9, color=INK)
        a2.text(xx + w / 2, b + 0.015, f"{b:.2f}", ha="center", fontsize=9, color=INK)
    a2.set_xticks(x, [NAMES[k].replace(" + ", "\n+ ").replace(" fine-tuned", "\nfine-tuned") for k in pairs], fontsize=9)
    a2.set_ylim(0, 1)
    a2.set_ylabel("test balanced accuracy")
    a2.grid(axis="y", color=GRID, lw=0.8)
    a2.set_axisbelow(True)
    a2.legend(frameon=False, loc="upper left", fontsize=9)
    a2.set_title("Same model, different split", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def confusion_figure(pred, col, path):
    cm = confusion_matrix(pred["dx"], pred[col], labels=CLASSES)
    norm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if cm[i, j]:
                ax.text(j, i, f"{norm[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_yticks(range(len(CLASSES)), [f"{c} ({n})" for c, n in zip(CLASSES, cm.sum(1))])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true (test images)")
    ax.set_title("Row-normalised confusion, best model", loc="left", color=INK, fontsize=10)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def error_gallery(pred, col, thumbs, path, n=6):
    wrong = pred[pred[col] != pred["dx"]].sort_values(f"{col.removesuffix('_pred')}_conf", ascending=False)
    wrong = wrong[[Path(thumbs, f"{i}.jpg").exists() for i in wrong["image_id"]]].head(n)
    fig, axes = plt.subplots(1, len(wrong), figsize=(2.1 * len(wrong), 2.6))
    for ax, (_, r) in zip(np.atleast_1d(axes), wrong.iterrows()):
        ax.imshow(Image.open(Path(thumbs, f"{r['image_id']}.jpg")))
        ax.set_title(f"true {r['dx']}\npred {r[col]} ({r[col.removesuffix('_pred') + '_conf']:.2f})", fontsize=8.5, color=INK)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return wrong


def table(m):
    rows = []
    for key, r in m.items():
        if key.startswith("_"):
            continue
        name, split = key.split("/")
        rows.append({"model": NAMES[name], "split": split, "bal_acc": f"{r['balanced_accuracy']:.3f}",
                     "95% CI": f"{r['bal_acc_ci'][0]:.3f}-{r['bal_acc_ci'][1]:.3f}",
                     "macro F1": f"{r['macro_f1']:.3f}", "accuracy": f"{r['accuracy']:.3f}",
                     "mel recall": f"{r['per_class_recall']['mel']:.2f}",
                     "mel sens @90% spec": f"{r.get('mel_sens_at_90spec', float('nan')):.2f}",
                     "malignant AUC": f"{r.get('malignant_auc', float('nan')):.3f}"})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()
    run, out = Path(args.run_dir), Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    m = json.load(open(run / "metrics.json"))
    pred = pd.read_csv(run / "test_predictions_lesion.csv")
    best = "finetune_meta_pred"
    results_figure(m, out / "figures" / "results.png")
    confusion_figure(pred, best, out / "figures" / "confusion.png")
    wrong = error_gallery(pred, best, run / "thumbs", out / "figures" / "errors.png")
    t = table(m)
    md = ["| " + " | ".join(t.columns) + " |", "|" + "---|" * len(t.columns)]
    md += ["| " + " | ".join(map(str, r)) + " |" for r in t.itertuples(index=False)]
    (out / "results.md").write_text("\n".join(md) + "\n")
    print(t.to_string(index=False))
    print(wrong[["image_id", "dx", "age", "sex", "localization", best, "finetune_meta_conf"]].to_string(index=False))
    for k in ("color_lr/image", "probe/image", "finetune/image"):
        if k in m:
            r = m[k]
            print(k, "seen share", round(r["test_share_seen_lesion"], 3), "acc seen", round(r.get("accuracy_seen_lesion", float("nan")), 3),
                  "acc unseen", round(r.get("accuracy_unseen_lesion", float("nan")), 3))


if __name__ == "__main__":
    main()
