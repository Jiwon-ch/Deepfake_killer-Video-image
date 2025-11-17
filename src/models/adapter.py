from __future__ import annotations
from typing import Literal, Optional, Dict, Tuple, Any, Sequence
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
class TemporalTransformerBlock(nn.Module):
    """
    Transformer encoder block along temporal axis (similar to TimeSformer/Vivit temporal branch).
    """
    def __init__(
        self,
        d: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(
            d, num_heads=num_heads, dropout=attn_dropout, batch_first=True
        )
        self.drop1 = nn.Dropout(dropout)
        hidden = int(d * mlp_ratio)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d),
        )
        self.drop2 = nn.Dropout(dropout)

        for m in self.ffn:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attn(
            x, x, x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = residual + self.drop1(attn_out)

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + self.drop2(x)
        return x


class TemporalTransformerAdapter(nn.Module):
    """
    Stacked temporal self-attention blocks to capture long-range cues across frames.
    """
    def __init__(
        self,
        d: int,
        num_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TemporalTransformerBlock(
                d,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        pad_mask = None
        if mask is not None:
            mask = mask.to(dtype=torch.bool)
            pad_mask = ~mask        # MultiheadAttention expects True for padding locations.
        for blk in self.layers:
            x = blk(x, key_padding_mask=pad_mask)
        return x


class TemporalAttentionPool(nn.Module):
    """
    간단한 점수 기반 temporal attention pooling
    (B, T, D) -> (B, D)
    """
    def __init__(self, d: int):
        super().__init__()
        self.score = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, T, D)
        w = self.score(x).squeeze(-1)     # (B, T)
        if mask is not None:
            mask = mask.to(dtype=torch.bool, device=w.device)
            w = w.masked_fill(~mask, float("-inf"))
        w = torch.softmax(w, dim=1).unsqueeze(-1)  # (B,T,1)
        out = torch.sum(w * x, dim=1)     # (B, D)
        return out


# ----------------------------
# MLP Head
# ----------------------------
class MLPHead(nn.Module):
    def __init__(
        self,
        d_in: int,
        hidden: int | Sequence[int] | None = 1024,
        num_classes: int = 2,
        p: float = 0.1,
    ):
        super().__init__()

        # hidden이 int/시퀀스/None 모두 허용되도록 정규화
        if hidden is None:
            hidden_dims: Tuple[int, ...] = ()
        elif isinstance(hidden, Sequence) and not isinstance(hidden, (str, bytes)):
            hidden_dims = tuple(int(h) for h in hidden if int(h) > 0)
        else:
            h = int(hidden)
            hidden_dims = (h,) if h > 0 else ()

        layers: list[nn.Module] = []
        in_dim = d_in
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p))
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

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
        - backbone.forward_images((B*T,3,H,W)) -> (B*T, D) 프레임 임베딩
        - 이미지 시퀀스(T=1 또는 동일 프레임)는 곧바로 Image Classifier 통과
        - 비디오 시퀀스(T>1, 시간 변화)는 temporal adapter+pool 적용 후 Video Classifier 통과
        - batch 단위로 이미지/비디오를 자동 분리 (필요 시 media_type/temporal_lengths로 override)
    """
    def __init__(
        self,
        backbone: nn.Module,                         # forward_images((N,3,H,W))->(N,D), .embed_dim 필요
        num_frames: int = 12,
        adapter_type: Literal["tconv", "transformer", "none"] = "tconv",
        temporal_pool: Literal["mean", "attn"] = "mean",
        head_hidden: int | Sequence[int] | None = 1024,
        num_classes: int = 2,
        chunk_size: Optional[int] = None,            # (B*T) 인코딩 시 슬라이스 크기 (VRAM 절약)
        id2label: Optional[Dict[int, str]] = None,   # 선택: 라벨 맵 저장용
        label2id: Optional[Dict[str, int]] = None,   # 선택: 라벨 맵 저장용
        *,
        temporal_diff_threshold: float = 1e-6,       # 이미지/비디오 구분 허용 오차
        adapter_kwargs: Optional[Dict[str, Any]] = None,
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
        self.temporal_diff_threshold = float(temporal_diff_threshold)

        d = int(backbone.embed_dim)
        
        self._keys_to_ignore_on_save = None

        # --- adapter 선택 ---
        if adapter_kwargs is None:
            adapter_kwargs = {}

        if adapter_type == "tconv":
            k = int(adapter_kwargs.get("kernel_size", 3))
            self.adapter = TemporalConvAdapter(d, k=k)
        elif adapter_type == "transformer":
            self.adapter = TemporalTransformerAdapter(
                d,
                num_layers=adapter_kwargs.get("num_layers", 2),
                num_heads=adapter_kwargs.get("num_heads", 8),
                mlp_ratio=adapter_kwargs.get("mlp_ratio", 4.0),
                dropout=adapter_kwargs.get("dropout", 0.1),
                attn_dropout=adapter_kwargs.get("attn_dropout", 0.1),
            )
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
        self.image_head = MLPHead(d_in=d, hidden=head_hidden, num_classes=self.num_classes)
        self.video_head = MLPHead(d_in=d, hidden=head_hidden, num_classes=self.num_classes)
        # ✅ 기존 체크포인트 호환용 alias
        self.head = self.video_head

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

    def _normalize_temporal_lengths(
        self,
        lengths: Optional[torch.Tensor | Tuple[int, ...] | list],
        batch_size: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if lengths is None:
            return None
        lens = torch.as_tensor(lengths, device=device)
        if lens.ndim == 0:
            lens = lens.unsqueeze(0).repeat(batch_size)
        if lens.numel() != batch_size:
            raise ValueError(f"temporal_lengths 크기({lens.numel()})와 배치({batch_size})가 다릅니다.")
        return lens.long()

    def _detect_video_mask(
        self,
        frames: torch.Tensor,
        *,
        media_type: Optional[torch.Tensor] = None,
        temporal_lengths: Optional[torch.Tensor | Tuple[int, ...] | list] = None,
    ) -> torch.Tensor:
        """
        media_type/temporal_lengths가 있으면 우선 사용,
        없으면 프레임 차이를 이용해 이미지 여부를 추정한다.
        반환: (B,) bool tensor (True → video)
        """
        B = frames.size(0)
        device = frames.device

        if media_type is not None:
            mask = torch.as_tensor(media_type, device=device, dtype=torch.bool)
            if mask.ndim == 0:
                mask = mask.unsqueeze(0).repeat(B)
            if mask.numel() != B:
                raise ValueError(f"media_type 길이({mask.numel()})와 배치({B})가 다릅니다.")
            return mask

        lengths = self._normalize_temporal_lengths(temporal_lengths, B, device)
        if lengths is not None:
            return lengths > 1

        if frames.size(1) <= 1:
            return torch.zeros(B, dtype=torch.bool, device=device)

        diff = frames[:, 1:] - frames[:, :-1]              # (B, T-1, 3, H, W)
        motion = diff.abs().amax(dim=(2, 3, 4))            # (B, T-1)
        motion_mask = motion > self.temporal_diff_threshold
        return motion_mask.any(dim=1)

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

    def _classify_sequences(
        self,
        seq: torch.Tensor,
        video_mask: torch.Tensor,
        temporal_lengths: Optional[torch.Tensor | Tuple[int, ...] | list] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        seq: (B,T,D)
        video_mask: (B,) bool
        반환: (logits, representations)
        """
        B, _, D = seq.shape
        logits = None
        reps = seq.new_zeros((B, D))
        lengths = self._normalize_temporal_lengths(temporal_lengths, B, seq.device)

        if (~video_mask).any():
            idx = torch.nonzero(~video_mask, as_tuple=False).squeeze(1)
            img_feat = seq[idx, 0, :]                     # 이미지 → 첫 프레임
            img_logits = self.image_head(img_feat)
            if logits is None:
                logits = seq.new_zeros((B, self.num_classes))
            reps[idx] = img_feat.to(reps.dtype)
            logits[idx] = img_logits.to(logits.dtype)

        if video_mask.any():
            idx = torch.nonzero(video_mask, as_tuple=False).squeeze(1)
            vid_seq = seq[idx]
            # lengths specific to video subset
            vid_lengths = None
            frame_mask = None
            if lengths is not None:
                vid_lengths = lengths[idx].clamp(min=1)
                T = vid_seq.size(1)
                frame_mask = (
                    torch.arange(T, device=vid_seq.device).unsqueeze(0)
                    < vid_lengths.unsqueeze(1)
                )
            if isinstance(self.adapter, TemporalTransformerAdapter):
                vid_seq = self.adapter(vid_seq, mask=frame_mask)
            else:
                vid_seq = self.adapter(vid_seq)
            if getattr(self, "pool_mode", "mean") == "attn":
                vid_feat = self.temporal_pool(vid_seq, mask=frame_mask)
            else:
                if frame_mask is None:
                    vid_feat = vid_seq.mean(dim=1)
                else:
                    denom = vid_lengths.unsqueeze(-1).to(dtype=vid_seq.dtype)
                    vid_feat = (vid_seq * frame_mask.unsqueeze(-1).to(dtype=vid_seq.dtype)).sum(dim=1) / denom
            vid_logits = self.video_head(vid_feat)
            if logits is None:
                logits = seq.new_zeros((B, self.num_classes))
            reps[idx] = vid_feat.to(reps.dtype)
            logits[idx] = vid_logits.to(logits.dtype)

        if logits is None:
            logits = seq.new_zeros((B, self.num_classes))
        return logits, reps

    # ----------------------------
    # forward (HF Trainer + 커스텀 Trainer 안전)
    # ----------------------------
    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        x: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_dict: bool = False,
        media_type: Optional[torch.Tensor] = None,
        temporal_lengths: Optional[torch.Tensor | Tuple[int, ...] | list] = None,
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

        x = self._ensure_5d(x)  # (B,T,3,H,W)
        video_mask = self._detect_video_mask(
            x, media_type=media_type, temporal_lengths=temporal_lengths
        )
        feat = self._encode_frames(x)  # (B,T,D)
        logits, _ = self._classify_sequences(
            feat,
            video_mask,
            temporal_lengths=temporal_lengths,
        )

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
    def extract_features(
        self,
        x: torch.Tensor,
        pool: Literal["mean", "attn"] = "mean",
        media_type: Optional[torch.Tensor] = None,
        temporal_lengths: Optional[torch.Tensor | Tuple[int, ...] | list] = None,
    ) -> torch.Tensor:
        """
        분류기 이전 임베딩 (B,D) 추출.
        pool 파라미터는 video branch에만 적용.
        """
        x = self._ensure_5d(x)
        video_mask = self._detect_video_mask(
            x, media_type=media_type, temporal_lengths=temporal_lengths
        )
        feat = self._encode_frames(x)   # (B,T,D)

        reps = feat.new_zeros((feat.size(0), feat.size(-1)))
        if (~video_mask).any():
            idx = torch.nonzero(~video_mask, as_tuple=False).squeeze(1)
            reps[idx] = feat[idx, 0, :].to(reps.dtype)

        if video_mask.any():
            idx = torch.nonzero(video_mask, as_tuple=False).squeeze(1)
            vid_seq = feat[idx]
            lengths = self._normalize_temporal_lengths(
                temporal_lengths, feat.size(0), feat.device
            )
            vid_lengths = None
            frame_mask = None
            if lengths is not None:
                vid_lengths = lengths[idx].clamp(min=1)
                T = vid_seq.size(1)
                frame_mask = (
                    torch.arange(T, device=vid_seq.device).unsqueeze(0)
                    < vid_lengths.unsqueeze(1)
                )
            if isinstance(self.adapter, TemporalTransformerAdapter):
                vid_seq = self.adapter(vid_seq, mask=frame_mask)
            else:
                vid_seq = self.adapter(vid_seq)
            if pool == "attn":
                if self.temporal_pool is None:
                    self.temporal_pool = TemporalAttentionPool(vid_seq.size(-1)).to(vid_seq.device)
                reps[idx] = self.temporal_pool(vid_seq, mask=frame_mask).to(reps.dtype)
            elif pool == "mean":
                if frame_mask is None:
                    reps[idx] = vid_seq.mean(dim=1).to(reps.dtype)
                else:
                    denom = vid_lengths.unsqueeze(-1).to(dtype=vid_seq.dtype)
                    reps[idx] = ((vid_seq * frame_mask.unsqueeze(-1).to(dtype=vid_seq.dtype)).sum(dim=1) / denom).to(reps.dtype)
            else:
                raise ValueError(f"unknown pool: {pool}")
        return reps
