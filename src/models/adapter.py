from __future__ import annotations
from typing import Literal, Optional, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Temporal Adapter (Depthwise 1DConv + PW Conv)
# ----------------------------
class TemporalConvAdapter(nn.Module):
    """
    (B, T, D) -> (B, T, D)
    - 채널별(depthwise) 1D conv로 시간축 로컬 문맥 학습
    - pointwise(1x1) conv로 채널 상호작용
    - residual 연결
    """
    def __init__(self, d: int, k: int = 3):
        super().__init__()
        assert k % 2 == 1, "kernel size k는 홀수여야 padding이 균등합니다."
        self.dw = nn.Conv1d(d, d, kernel_size=k, padding=k // 2, groups=d, bias=False)
        self.pw = nn.Conv1d(d, d, kernel_size=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.norm = nn.LayerNorm(d)

        # 초기화
        nn.init.kaiming_normal_(self.dw.weight, nonlinearity="relu")
        nn.init.xavier_uniform_(self.pw.weight)
        if self.pw.bias is not None:
            nn.init.zeros_(self.pw.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        idt = x
        x = self.norm(x)
        y = x.transpose(1, 2)            # (B, D, T)
        y = self.dw(y)
        y = self.act(y)
        y = self.pw(y)
        y = y.transpose(1, 2)            # (B, T, D)
        return idt + y                    # residual


# ----------------------------
# Attention Pool (선택)
# ----------------------------
class TemporalAttentionPool(nn.Module):
    """
    간단한 점수 기반 temporal attention pooling
    (B, T, D) -> (B, D)
    """
    def __init__(self, d: int):
        super().__init__()
        self.score = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        w = self.score(x)                 # (B, T, 1)
        w = torch.softmax(w, dim=1)       # (B, T, 1)
        out = torch.sum(w * x, dim=1)     # (B, D)
        return out


# ----------------------------
# MLP Head
# ----------------------------
class MLPHead(nn.Module):
    def __init__(self, d_in: int, hidden: int = 1024, num_classes: int = 2, p: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(hidden, num_classes),
        )
        # 초기화
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ----------------------------
# Unified Image/Video Model
# ----------------------------
class UnifiedAdapterModel(nn.Module):
    """
    이미지/비디오 겸용 분류 모델.
      입력:
        - 이미지: (B, 3, H, W)
        - 비디오: (B, T, 3, H, W)
      동작:
        - 이미지면 num_frames로 복제해 (B, T, 3, H, W)로 변환
        - backbone.forward_images((B*T,3,H,W)) -> (B*T, D) 프레임 임베딩
        - (B,T,D)로 reshape 후 temporal adapter 적용
        - temporal pooling(mean/attn)으로 (B,D) 요약
        - MLP head로 (B, num_classes) logits 출력
    """
    def __init__(
        self,
        backbone: nn.Module,                         # forward_images((N,3,H,W))->(N,D), .embed_dim 필요
        num_frames: int = 12,
        adapter_type: Literal["tconv", "none"] = "tconv",
        temporal_pool: Literal["mean", "attn"] = "mean",
        head_hidden: int = 1024,
        num_classes: int = 2,
        chunk_size: Optional[int] = None,            # (B*T) 인코딩 시 슬라이스 크기 (VRAM 절약)
        id2label: Optional[Dict[int, str]] = None,   # 선택: 라벨 맵 저장용
        label2id: Optional[Dict[str, int]] = None,   # 선택: 라벨 맵 저장용
    ):
        super().__init__()
        # --- sanity checks ---
        assert hasattr(backbone, "forward_images") and callable(backbone.forward_images), \
            "backbone은 forward_images((N,3,H,W))->(N,D) 를 제공해야 함"
        assert hasattr(backbone, "embed_dim"), "backbone은 embed_dim 속성을 제공해야 함"
        assert isinstance(num_frames, int) and num_frames > 0
        if chunk_size is not None:
            assert isinstance(chunk_size, int) and chunk_size > 0, "chunk_size는 양의 정수여야 함"

        self.backbone = backbone
        self.num_frames = num_frames
        self.chunk_size = chunk_size
        self.num_classes = int(num_classes)

        d = int(backbone.embed_dim)

        # --- adapter 선택 ---
        if adapter_type == "tconv":
            self.adapter = TemporalConvAdapter(d, k=3)
        elif adapter_type == "none":
            self.adapter = nn.Identity()
        else:
            raise ValueError(f"지원하지 않는 adapter_type: {adapter_type}")

        # --- temporal pooling 선택 ---
        if temporal_pool == "mean":
            self.temporal_pool = None
            self.pool_mode = "mean"
        elif temporal_pool == "attn":
            self.temporal_pool = TemporalAttentionPool(d)
            self.pool_mode = "attn"
        else:
            raise ValueError(f"지원하지 않는 temporal_pool: {temporal_pool}")

        # --- head ---
        self.head = MLPHead(d_in=d, hidden=head_hidden, num_classes=self.num_classes)

        # --- 라벨 맵(옵션) ---
        self.id2label = id2label
        self.label2id = label2id

    # ----------------------------
    # 유틸
    # ----------------------------
    def num_parameters(self, only_trainable: bool = False) -> int:
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def set_label_maps(self, id2label: Dict[int, str], label2id: Dict[str, int]) -> None:
        self.id2label = id2label
        self.label2id = label2id

    # ----------------------------
    # 내부 헬퍼
    # ----------------------------
    def _ensure_5d(self, x: torch.Tensor) -> torch.Tensor:
        """이미지(4D)->비디오(5D)로 확장. 비디오는 그대로."""
        if x.ndim == 4:
            # (B,3,H,W)->(B,T,3,H,W), 동일 프레임 복제
            x = x.unsqueeze(1).repeat(1, self.num_frames, 1, 1, 1)
        elif x.ndim != 5:
            raise ValueError(f"입력 텐서 차원 오류: (B,3,H,W) 또는 (B,T,3,H,W) 필요, 현재={tuple(x.shape)}")
        return x

    def _encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames: (B, T, 3, H, W)
        return: (B, T, D)
        """
        B, T, C, H, W = frames.shape
        flat = frames.view(B * T, C, H, W)

        feats = []
        if self.chunk_size is None:
            feat = self.backbone.forward_images(flat)  # (B*T, D)
            feats.append(feat)
        else:
            cs = self.chunk_size
            for i in range(0, flat.size(0), cs):
                feat = self.backbone.forward_images(flat[i:i+cs])  # (cs, D)
                feats.append(feat)

        feat = torch.cat(feats, dim=0).view(B, T, -1)  # (B,T,D)
        return feat

    # ----------------------------
    # forward (HF Trainer + 커스텀 Trainer 안전)
    # ----------------------------
    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        x: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_dict: bool = False,
        **kwargs,
    ):
        """
        - 커스텀 Trainer 경로: labels=None & return_dict=False 인 경우 **Tensor(logits)** 반환
        - HF Trainer 경로: labels가 있거나 return_dict=True면 dict 반환 (loss/logits 포함)
        """
        if x is None:
            x = pixel_values
        if x is None:
            raise ValueError("forward() expects 'x' or 'pixel_values'.")

        x = self._ensure_5d(x)          # (B,T,3,H,W)
        feat = self._encode_frames(x)   # (B,T,D)
        feat = self.adapter(feat)       # (B,T,D)

        if hasattr(self, "pool_mode") and self.pool_mode == "attn":
            vid = self.temporal_pool(feat)  # (B,D)
        else:
            vid = feat.mean(dim=1)          # (B,D)

        logits = self.head(vid)             # (B,num_classes)

        # === 커스텀 Trainer (e.g., AdapterTrainer.compute_loss에서 model(x) 호출) ===
        if (labels is None) and (return_dict is False):
            return logits  # <<<<<<<< Tensor만 반환

        # === HF Trainer 호환 (loss/logits dict 반환) ===
        out = {"logits": logits}
        if labels is not None:
            loss = F.cross_entropy(logits, labels.long(), label_smoothing=0.1)
            out["loss"] = loss
        return out

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor, pool: Literal["mean", "attn"] = "mean") -> torch.Tensor:
        """
        분류기 이전 비디오 임베딩 (B,D) 추출.
        """
        x = self._ensure_5d(x)
        feat = self._encode_frames(x)   # (B,T,D)
        feat = self.adapter(feat)       # (B,T,D)
        if pool == "mean":
            return feat.mean(dim=1)     # (B,D)
        elif pool == "attn":
            if self.temporal_pool is None:
                self.temporal_pool = TemporalAttentionPool(feat.size(-1)).to(feat.device)
            return self.temporal_pool(feat)
        else:
            raise ValueError(f"unknown pool: {pool}")
