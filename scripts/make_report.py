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


def seed_values(m, rep, model, split):
    vals = [m[f"{model}/{split}"]["balanced_accuracy"]]
    vals += [v["balanced_accuracy"] for k, v in rep.items() if k.startswith(f"{model}/{split}/s")]
    return vals


def seed_table(m, rep):
    rows = []
    for model in ("probe", "probe_meta", "finetune", "finetune_meta"):
        for split in ("lesion", "image"):
            if f"{model}/{split}" not in m:
                continue
            keys = [f"{model}/{split}"] + [k for k in rep if k.startswith(f"{model}/{split}/s")]
            runs = [m.get(k) or rep[k] for k in keys]
            cell = {"model": NAMES[model], "split": split, "seeds": len(runs)}
            for metric in ("balanced_accuracy", "macro_f1", "accuracy", "mel_sens_at_90spec", "malignant_auc"):
                v = [r[metric] for r in runs]
                cell[metric] = f"{np.mean(v):.3f} ± {np.std(v):.3f}"
            rows.append(cell)
    return pd.DataFrame(rows)


def results_figure(m, rep, path):
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
    a1.set_xlabel("balanced accuracy, lesion-split test, seed 0 (95% CI)")
    a1.grid(axis="x", color=GRID, lw=0.8)
    a1.set_axisbelow(True)
    a1.set_title("Test balanced accuracy over 7 classes", loc="left", color=INK)

    pairs = ["probe", "finetune"]
    x = np.arange(len(pairs))
    w = 0.36
    for off, split, color, label in ((-w / 2 - 0.01, "lesion", BLUE, "split by lesion"),
                                     (w / 2 + 0.01, "image", ORANGE, "split by image (leaky)")):
        per_seed = [seed_values(m, rep, k, split) for k in pairs]
        means = [np.mean(v) for v in per_seed]
        a2.bar(x + off, means, w, color=color, label=label)
        for xx, v, mu in zip(x, per_seed, means):
            a2.scatter(np.full(len(v), xx + off), v, s=12, color=INK, zorder=3)
            a2.text(xx + off, max(v) + 0.02, f"{mu:.3f}", ha="center", fontsize=9, color=INK)
    a2.set_xticks(x, ["DINOv2-S frozen + LR", "ConvNeXt-T fine-tuned"], fontsize=9)
    a2.set_ylim(0, 1)
    a2.set_ylabel("test balanced accuracy (mean, dots = seeds)")
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
    ax.set_title("ConvNeXt-T, lesion-split test, row-normalised", loc="left", color=INK, fontsize=10)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def error_gallery(pred, col, thumbs, path, n=6):
    wrong = pred[pred[col] != pred["dx"]].sort_values(f"{col.removesuffix('_pred')}_conf", ascending=False)
    wrong = wrong[[Path(thumbs, f"{i}.jpg").exists() for i in wrong["image_id"]]]
    wrong = wrong.drop_duplicates("lesion_id").head(n)
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
    ap.add_argument("--repeats", help="metrics_repeats.json from the second round")
    args = ap.parse_args()
    run, out = Path(args.run_dir), Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    m = json.load(open(run / "metrics.json"))
    pred = pd.read_csv(run / "test_predictions_lesion.csv")
    best = "finetune_pred"
    rep = json.load(open(args.repeats)) if args.repeats else {}
    results_figure(m, rep, out / "figures" / "results.png")
    confusion_figure(pred, best, out / "figures" / "confusion.png")
    wrong = error_gallery(pred, best, run / "thumbs", out / "figures" / "errors.png")
    t = table(m)
    md = ["| " + " | ".join(t.columns) + " |", "|" + "---|" * len(t.columns)]
    md += ["| " + " | ".join(map(str, r)) + " |" for r in t.itertuples(index=False)]
    if rep:
        st = seed_table(m, rep)
        md += ["", "Mean ± std over split seeds:", "", "| " + " | ".join(st.columns) + " |", "|" + "---|" * len(st.columns)]
        md += ["| " + " | ".join(map(str, r)) + " |" for r in st.itertuples(index=False)]
        print(st.to_string(index=False))
    (out / "results.md").write_text("\n".join(md) + "\n")
    print(t.to_string(index=False))
    print(wrong[["image_id", "dx", "age", "sex", "localization", best, "finetune_conf"]].to_string(index=False))
    # how the ground truth was established matters a lot: follow-up nevi are a very easy subgroup
    pred = pred.merge(pd.read_csv(run / "splits.csv")[["image_id", "dx_type"]], on="image_id")
    pred["error"] = pred[best] != pred["dx"]
    print(pred.groupby("dx_type")["error"].agg(["mean", "size"]).round(3).to_string())
    histo = pred[pred["dx_type"] == "histo"]
    print("balanced accuracy on histology-confirmed only:",
          round(float(np.mean([(histo.loc[histo.dx == c, best] == c).mean() for c in CLASSES if (histo.dx == c).sum()])), 3))
    for k in ("color_lr/image", "probe/image", "finetune/image"):
        if k in m:
            r = m[k]
            print(k, "seen share", round(r["test_share_seen_lesion"], 3), "acc seen", round(r.get("accuracy_seen_lesion", float("nan")), 3),
                  "acc unseen", round(r.get("accuracy_unseen_lesion", float("nan")), 3))


if __name__ == "__main__":
    main()
