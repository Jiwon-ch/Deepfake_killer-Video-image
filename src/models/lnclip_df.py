from __future__ import annotations
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .clip_backbone import ClipBackbone


def _slerp(z_i: torch.Tensor, z_j: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    z_i, z_j: (..., D) unit vectors, t in [0,1]
    Returns spherical linear interpolation result with numerical safeguards.
    """
    # clamp dot to avoid nan from arccos
    cos_theta = torch.clamp((z_i * z_j).sum(dim=-1), -1.0 + 1e-6, 1.0 - 1e-6)
    theta = torch.acos(cos_theta)
    sin_theta = torch.sin(theta)
    # when vectors are extremely close, fall back to linear interpolation
    mask_small = sin_theta < 1e-6
    coeff_i = torch.where(mask_small, 1.0 - t, torch.sin((1.0 - t) * theta) / sin_theta)
    coeff_j = torch.where(mask_small, t, torch.sin(t * theta) / sin_theta)
    return coeff_i.unsqueeze(-1) * z_i + coeff_j.unsqueeze(-1) * z_j


def _alignment_loss(z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Alignment loss: mean squared distance within the same class.
    """
    losses = []
    for cls in labels.unique():
        idx = (labels == cls).nonzero(as_tuple=False).flatten()
        if idx.numel() < 2:
            continue
        z_c = z[idx]
        dists = torch.pdist(z_c, p=2)
        losses.append((dists ** 2).mean())
    if not losses:
        return z.new_tensor(0.0)
    return torch.stack(losses).mean()


def _uniformity_loss(z: torch.Tensor) -> torch.Tensor:
    """
    Uniformity loss: log mean exp of pairwise squared distance (SimCLR-style).
    """
    if z.size(0) < 2:
        return z.new_tensor(0.0)
    dists = torch.pdist(z, p=2)  # (N*(N-1)/2,)
    return torch.log(torch.mean(torch.exp(-2.0 * (dists ** 2))))


class LNCLIPDF(nn.Module):
    """
    LN-CLIP-DF: CLIP ViT-L/14 CLS 토큰만 사용, L2 정규화 + slerp 증강 + CE/align/uniform loss.
    영상 입력 시 프레임별 확률 평균.
    """
    def __init__(
        self,
        backbone: ClipBackbone,
        num_classes: int = 2,
        align_weight: float = 0.1,
        uniform_weight: float = 0.5,
        slerp_target: int = 1024,
        classifier_hidden: int = 0,
        class_weights: Optional[torch.Tensor] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_classes = int(num_classes)
        self.align_weight = float(align_weight)
        self.uniform_weight = float(uniform_weight)
        self.slerp_target = int(slerp_target)
        self.dropout = float(dropout)

        hidden = int(classifier_hidden)
        if hidden and hidden > 0:
            self.classifier = nn.Sequential(
                nn.Linear(backbone.embed_dim, hidden),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(hidden, self.num_classes),
            )
        else:
            self.classifier = nn.Linear(backbone.embed_dim, self.num_classes)
        # Init all linear layers
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if class_weights is not None:
            cw = torch.as_tensor(class_weights, dtype=torch.float32)
            self.register_buffer("class_weights", cw)
        else:
            self.class_weights = None

    def _ensure_4d(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[Tuple[int, int]]]:
        """
        입력을 (B*,3,H,W)로 만듦. 비디오면 (B*T,3,H,W)로 펴고 원래 (B,T) 반환.
        """
        if x.ndim == 4:
            return x, None
        if x.ndim == 5:
            B, T, C, H, W = x.shape
            return x.view(B * T, C, H, W), (B, T)
        raise ValueError(f"Expected 4D or 5D pixel_values, got {tuple(x.shape)}")

    def _aggregate_video(
        self, logits_flat: torch.Tensor, feats_flat: torch.Tensor, shape: Optional[Tuple[int, int]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        logits_flat: (B*T,2), feats_flat: (B*T,D)
        shape: (B,T) or None
        returns: (logits_out, feats_out, logits_for_loss)
        """
        if shape is None:
            return logits_flat, feats_flat, logits_flat
        B, T = shape
        logits_bt = logits_flat.view(B, T, -1)
        feats_bt = feats_flat.view(B, T, -1)
        # 프레임별 softmax 후 평균(논문: 확률 평균)
        probs = torch.softmax(logits_bt, dim=-1).mean(dim=1)
        feats_mean = feats_bt.mean(dim=1)
        logits_mean = logits_bt.mean(dim=1)  # CE용: 프레임별 logits 평균
        return probs, feats_mean, logits_mean

    def _slerp_augment(
        self, z: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        z: (B,D) unit vectors
        labels: (B,)
        """
        B, D = z.shape
        if B < 2 or self.slerp_target <= B:
            return z.new_zeros((0, D)), labels.new_zeros((0,), dtype=labels.dtype)

        repeats = max(1, self.slerp_target // B)
        num_aug = min(self.slerp_target - B, B * (repeats - 1))

        aug_feats = []
        aug_labels = []
        for _ in range(num_aug):
            i = torch.randint(0, B, (1,), device=z.device).item()
            same_cls = (labels == labels[i]).nonzero(as_tuple=False).flatten()
            same_cls = same_cls[same_cls != i]
            if same_cls.numel() == 0:
                continue
            j = same_cls[torch.randint(0, same_cls.numel(), (1,), device=z.device)].item()
            t = torch.rand(1, device=z.device)
            z_ij = _slerp(z[i:i+1], z[j:j+1], t)[0]
            z_ij = F.normalize(z_ij, p=2, dim=-1)
            aug_feats.append(z_ij)
            aug_labels.append(labels[i].unsqueeze(0))

        if not aug_feats:
            return z.new_zeros((0, D)), labels.new_zeros((0,), dtype=labels.dtype)

        return torch.stack(aug_feats, dim=0), torch.cat(aug_labels, dim=0)

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        *,
        apply_slerp: bool = False,
        return_dict: bool = False,
    ):
        x_flat, vid_shape = self._ensure_4d(pixel_values)

        feats_flat = self.backbone.forward_images(x_flat)  # (B*,D)
        feats_flat = F.normalize(feats_flat, p=2, dim=-1)

        logits_flat = self.classifier(feats_flat)
        logits, feats, logits_for_loss = self._aggregate_video(logits_flat, feats_flat, vid_shape)

        # slerp는 학습 중에만 사용(논문 설정)
        labels_out = labels
        feats_for_loss = feats
        if self.training and apply_slerp and labels is not None:
            aug_feats, aug_labels = self._slerp_augment(feats, labels)
            if aug_feats.numel() > 0:
                feats_for_loss = torch.cat([feats, aug_feats], dim=0)
                labels_out = torch.cat([labels, aug_labels], dim=0)
                logits_for_loss = self.classifier(feats_for_loss)

        if labels is None and not return_dict:
            return logits

        out = {
            "logits": logits,
            "features": feats,
            "logits_for_loss": logits_for_loss,
            "features_for_loss": feats_for_loss,
            "labels_for_loss": labels_out,
        }
        if labels is not None:
            ce = F.cross_entropy(
                logits_for_loss,
                labels_out.long(),
                weight=self.class_weights if getattr(self, "class_weights", None) is not None else None,
            )
            align = _alignment_loss(feats_for_loss, labels_out.long())
            uniform = _uniformity_loss(feats_for_loss)
            loss = ce + self.align_weight * align + self.uniform_weight * uniform
            out["loss"] = loss
            out["loss_ce"] = ce
            out["loss_align"] = align
            out["loss_uniform"] = uniform
        if not return_dict:
            return out["loss"] if "loss" in out else out["logits"]
        return out


__all__ = ["LNCLIPDF"]
