import math
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from .data import CLASSES
from .metrics import summarize
from .model import SkinNet


@dataclass
class FinetuneConfig:
    backbone: str = "convnext_tiny.fb_in22k_ft_in1k"
    use_meta: bool = False
    class_weight_power: float = 0.5  # 0 = plain CE, 1 = inverse frequency
    epochs: int = 15
    batch_size: int = 48
    lr: float = 2e-4
    weight_decay: float = 0.05
    label_smoothing: float = 0.05
    img_size: int = 224
    pretrained: bool = True
    workers: int = 4
    seed: int = 0


def train_transform(size):
    return v2.Compose([
        v2.RandomResizedCrop(size, scale=(0.4, 1.0), ratio=(0.8, 1.25), antialias=True),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.RandomApply([v2.RandomRotation(90)], p=0.5),
        v2.ColorJitter(0.2, 0.2, 0.15, 0.02),
    ])


class LesionDataset(Dataset):
    def __init__(self, images, labels, meta, transform=None):
        self.images, self.labels, self.meta, self.transform = images, labels, meta, transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        x = torch.from_numpy(self.images[i]).permute(2, 0, 1)
        if self.transform is not None:
            x = self.transform(x)
        return x, self.labels[i], self.meta[i]


def class_weights(y, power):
    counts = np.bincount(y, minlength=len(CLASSES)).astype(float)
    w = (counts.sum() / np.maximum(counts, 1)) ** power
    return torch.tensor(w / w[counts > 0].mean(), dtype=torch.float32)


@torch.no_grad()
def predict_proba(model, images, meta, device, size=224, batch_size=128, tta=True):
    model.eval()
    out = []
    for i in range(0, len(images), batch_size):
        x = torch.from_numpy(images[i:i + batch_size]).to(device).permute(0, 3, 1, 2)
        if x.shape[-1] != size:
            x = F.interpolate(x.float(), size=(size, size), mode="bilinear", antialias=True).round().clamp(0, 255)
        m = torch.from_numpy(meta[i:i + batch_size]).to(device)
        views = [x, x.flip(3), x.flip(2), x.flip(2).flip(3)] if tta else [x]
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            p = sum(model(v, m).float().softmax(1) for v in views) / len(views)
        out.append(p.cpu().numpy())
    return np.concatenate(out)


def finetune(cfg: FinetuneConfig, images, y, meta, train_idx, val_idx, device, log=print):
    """Train on train_idx, keep the epoch with the best val balanced accuracy."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    meta_dim = meta.shape[1] if cfg.use_meta else 0
    model = SkinNet(cfg.backbone, len(CLASSES), meta_dim, pretrained=cfg.pretrained).to(device)

    ds = LesionDataset(images[train_idx], y[train_idx], meta[train_idx], train_transform(cfg.img_size))
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                    num_workers=cfg.workers, persistent_workers=cfg.workers > 0)
    weights = class_weights(y[train_idx], cfg.class_weight_power).to(device)

    params_bb = list(model.backbone.parameters())
    params_head = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": params_bb, "lr": cfg.lr},
                             {"params": params_head, "lr": cfg.lr * 5}], weight_decay=cfg.weight_decay)
    total = cfg.epochs * len(dl)
    warmup = len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    best, best_state, history = -1.0, None, []
    for epoch in range(cfg.epochs):
        model.train()
        t0, losses = time.time(), []
        for x, yb, mb in dl:
            x, yb, mb = x.to(device, non_blocking=True), yb.to(device), mb.to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(x, mb)
            loss = F.cross_entropy(logits.float(), yb, weight=weights, label_smoothing=cfg.label_smoothing)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
        val_p = predict_proba(model, images[val_idx], meta[val_idx], device, cfg.img_size, tta=False)
        m = summarize(y[val_idx], val_p)
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)),
                        "val_bal_acc": m["balanced_accuracy"], "val_macro_f1": m["macro_f1"]})
        log(f"  ep {epoch + 1:2d} loss {np.mean(losses):.3f} val bal_acc {m['balanced_accuracy']:.3f} "
            f"f1 {m['macro_f1']:.3f} ({time.time() - t0:.0f}s)")
        if m["balanced_accuracy"] > best:
            best = m["balanced_accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, history
