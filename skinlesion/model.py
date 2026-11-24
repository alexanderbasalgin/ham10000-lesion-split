import timm
import torch
from torch import nn

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class SkinNet(nn.Module):
    """timm backbone, optionally fused with an MLP over age/sex/site."""

    def __init__(self, backbone: str, n_classes: int, meta_dim: int = 0, pretrained: bool = True, drop: float = 0.2):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        feat = self.backbone.num_features
        self.meta_dim = meta_dim
        if meta_dim:
            self.meta = nn.Sequential(nn.Linear(meta_dim, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
                                      nn.Linear(64, 64), nn.ReLU())
            feat += 64
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(feat, n_classes))
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1) * 255)

    def forward(self, x_uint8, meta=None):
        x = (x_uint8.float() - self.mean) / self.std
        f = self.backbone(x)
        if self.meta_dim:
            f = torch.cat([f, self.meta(meta)], 1)
        return self.head(f)


@torch.no_grad()
def extract_features(backbone: str, images, batch_size=64, size=224, device="cpu"):
    """Frozen pooled features for a linear probe. images: uint8 NHWC numpy."""
    import numpy as np
    import torch.nn.functional as F

    kwargs = {"img_size": size} if "vit" in backbone else {}
    m = timm.create_model(backbone, pretrained=True, num_classes=0, **kwargs).to(device).eval()
    cfg = m.pretrained_cfg
    mean = torch.tensor(cfg.get("mean", MEAN), device=device).view(1, 3, 1, 1)
    std = torch.tensor(cfg.get("std", STD), device=device).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(images), batch_size):
        x = torch.from_numpy(images[i:i + batch_size]).to(device).permute(0, 3, 1, 2).float() / 255
        x = F.interpolate(x, size=(size, size), mode="bilinear", antialias=True, align_corners=False)
        x = (x - mean) / std
        with torch.autocast(device_type=device.split(":")[0], enabled=device != "cpu"):
            f = m(x)
        # horizontal flip averaged in, cheap TTA
        with torch.autocast(device_type=device.split(":")[0], enabled=device != "cpu"):
            f = (f + m(x.flip(3))) / 2
        out.append(f.float().cpu().numpy())
    return np.concatenate(out)
