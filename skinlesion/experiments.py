"""Runs every model on the lesion-level split and the naive image-level split."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import baselines
from .data import CLASSES, encode_meta, lesion_split, load_images, load_metadata, random_split
from .metrics import bootstrap_ci, summarize
from .model import extract_features
from .train import FinetuneConfig, finetune, predict_proba


def run(data_root, out_dir, quick=False, log=print):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_start = time.time()

    df = load_metadata(data_root)
    log(f"{len(df)} images, {df['lesion_id'].nunique()} lesions, device {device}")
    t0 = time.time()
    images = load_images(df["path"].tolist(), size=96 if quick else 256)
    log(f"decoded images {images.shape} in {time.time() - t0:.0f}s")
    y = df["label"].to_numpy()
    meta = encode_meta(df)
    groups = df["lesion_id"].to_numpy()

    splits = {"lesion": lesion_split(df), "image": random_split(df)}
    for name, s in splits.items():
        df[f"split_{name}"] = s
        log(f"{name} split sizes: {s.value_counts().to_dict()}")
    df.drop(columns=["path"]).to_csv(out / "splits.csv", index=False)

    probe_backbone = "resnet18.a1_in1k" if quick else "vit_small_patch14_dinov2.lvd142m"
    ft_base = FinetuneConfig(epochs=1, batch_size=16, img_size=64, workers=0,
                             backbone="resnet18.a1_in1k") if quick else FinetuneConfig()

    t0 = time.time()
    color = baselines.color_features(images)
    log(f"color features {color.shape} in {time.time() - t0:.0f}s")
    t0 = time.time()
    deep = extract_features(probe_backbone, images, size=64 if quick else 224, device=str(device))
    log(f"{probe_backbone} features {deep.shape} in {time.time() - t0:.0f}s")

    results, histories, test_probs = {}, {}, {}

    def record(key, split_name, proba, extra=None):
        s = splits[split_name]
        te = np.flatnonzero(s == "test")
        m = summarize(y[te], proba)
        m["bal_acc_ci"] = bootstrap_ci(y[te], proba, groups=groups[te], n=200 if quick else 1000)
        if split_name == "image":
            # how much of the score comes from lesions that also appear in train
            seen = np.isin(groups[te], groups[s == "train"])
            m["test_share_seen_lesion"] = float(seen.mean())
            for flag, mask in (("seen", seen), ("unseen", ~seen)):
                if mask.sum():
                    m[f"accuracy_{flag}_lesion"] = float((proba[mask].argmax(1) == y[te][mask]).mean())
        m.update(extra or {})
        results[key] = m
        test_probs[key] = proba
        log(f"[{key}] bal_acc {m['balanced_accuracy']:.3f} {tuple(round(v, 3) for v in m['bal_acc_ci'])} "
            f"macro_f1 {m['macro_f1']:.3f} acc {m['accuracy']:.3f} mel_sens@90spec {m.get('mel_sens_at_90spec', float('nan')):.3f}")
        json.dump(results, open(out / "metrics.json", "w"), indent=2)

    def probe(feats, split_name, C_grid=(0.01, 0.03, 0.1, 0.3, 1.0)):
        s = splits[split_name]
        tr, va, te = (np.flatnonzero(s == k) for k in ("train", "val", "test"))
        best = None
        for C in C_grid:
            clf = baselines.linear_head(C).fit(feats[tr], y[tr])
            score = summarize(y[va], clf.predict_proba(feats[va]))["balanced_accuracy"]
            if best is None or score > best[0]:
                best = (score, C)
        clf = baselines.linear_head(best[1]).fit(feats[np.concatenate([tr, va])], y[np.concatenate([tr, va])])
        return clf.predict_proba(feats[te]), {"C": best[1], "val_bal_acc": best[0]}

    for split_name in ("lesion", "image"):
        s = splits[split_name]
        tr, va, te = (np.flatnonzero(s == k) for k in ("train", "val", "test"))
        if split_name == "lesion":
            record("prior/lesion", "lesion", baselines.prior_proba(y[tr], len(te)))
        record(f"color_lr/{split_name}", split_name, *probe(color, split_name))
        record(f"probe/{split_name}", split_name, *probe(deep, split_name))
        if split_name == "lesion":
            record("probe_meta/lesion", "lesion", *probe(np.concatenate([deep, meta * 3], 1), "lesion"))

    ft_runs = [
        ("finetune/lesion", "lesion", {}),
        ("finetune_meta/lesion", "lesion", {"use_meta": True}),
        ("finetune_noweight/lesion", "lesion", {"class_weight_power": 0.0}),
        ("finetune/image", "image", {}),
    ]
    for key, split_name, overrides in ft_runs:
        s = splits[split_name]
        tr, va, te = (np.flatnonzero(s == k) for k in ("train", "val", "test"))
        cfg = FinetuneConfig(**{**ft_base.__dict__, **overrides})
        log(f"== {key}: {cfg}")
        t0 = time.time()
        model, hist = finetune(cfg, images, y, meta, tr, va, device, log=log)
        minutes = (time.time() - t0) / 60
        histories[key] = hist
        proba = predict_proba(model, images[te], meta[te], device, cfg.img_size)
        record(key, split_name, proba, {"train_minutes": round(minutes, 1)})
        if split_name == "lesion" and key in ("finetune/lesion", "finetune_meta/lesion"):
            state = {k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}
            torch.save({"state_dict": state, "backbone": cfg.backbone, "use_meta": cfg.use_meta,
                        "img_size": cfg.img_size, "classes": CLASSES},
                       out / f"{key.split('/')[0]}.pt")
        json.dump(histories, open(out / "histories.json", "w"), indent=2)
        del model
        torch.cuda.empty_cache()

    # test-set predictions of every model on the lesion split, for error analysis
    te = np.flatnonzero(splits["lesion"] == "test")
    pred_df = df.iloc[te][["image_id", "lesion_id", "dx", "age", "sex", "localization"]].reset_index(drop=True)
    for key, p in test_probs.items():
        if key.endswith("/lesion"):
            name = key.split("/")[0]
            pred_df[f"{name}_pred"] = [CLASSES[i] for i in p.argmax(1)]
            pred_df[f"{name}_conf"] = p.max(1).round(4)
            if name in ("finetune", "finetune_meta"):
                for j, c in enumerate(CLASSES):
                    pred_df[f"{name}_p_{c}"] = p[:, j].round(4)
    pred_df.to_csv(out / "test_predictions_lesion.csv", index=False)

    # thumbnails of the test set for the README error analysis (small JPEGs)
    from PIL import Image
    thumbs = out / "thumbs"
    thumbs.mkdir(exist_ok=True)
    fp = pred_df["finetune_meta_pred"] != pred_df["dx"]
    worst = pred_df[fp].sort_values("finetune_meta_conf", ascending=False).head(24)
    right = pred_df[~fp].groupby("dx").head(2)
    for iid in pd.concat([worst, right])["image_id"]:
        Image.fromarray(images[df.index[df["image_id"] == iid][0]]).resize((192, 192)).save(thumbs / f"{iid}.jpg", quality=88)

    results["_total_minutes"] = round((time.time() - t_start) / 60, 1)
    json.dump(results, open(out / "metrics.json", "w"), indent=2)
    log(f"done in {results['_total_minutes']} min")
    return results
