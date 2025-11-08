from typing import Literal, Optional
import torch
import torch.nn as nn
from transformers import CLIPModel

class ClipBackbone(nn.Module):
    """
    CLIP ViT-L/14 백본 래퍼
      - forward_images: (B,3,H,W) → (B,D)
      - freeze / unfreeze_last_n_blocks 지원
    """
    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        dtype: Literal["fp32","bf16","fp16"] = "fp32",
        freeze_backbone: bool = True,
        unfreeze_last_n_blocks: int = 0,  # 0이면 완전 동결
    ):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_name)
        self.vision = self.clip.vision_model   # ViT encoder
        self.embed_dim = getattr(self.clip.config, "projection_dim", None)
        if self.embed_dim is None:
            # 일부 버전은 projection_dim 대신 vision_model.config.hidden_size 사용
            self.embed_dim = self.vision.config.hidden_size

        # dtype 설정
        if dtype == "bf16":
            self.clip = self.clip.to(dtype=torch.bfloat16)
        elif dtype == "fp16":
            self.clip = self.clip.to(dtype=torch.float16)
        else:
            self.clip = self.clip.to(dtype=torch.float32)

        # 동결/언프리즈
        if freeze_backbone:
            for p in self.parameters():
                p.requires_grad = False
        if unfreeze_last_n_blocks > 0:
            blocks = list(self.vision.encoder.layers)
            for blk in blocks[-unfreeze_last_n_blocks:]:
                for p in blk.parameters():
                    p.requires_grad = True

    @torch.no_grad()
    def forward_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.vision(pixel_values=pixel_values)
        pooled = (
            out.pooler_output
            if hasattr(out, "pooler_output") and out.pooler_output is not None
            else out.last_hidden_state[:, 0, :]
        )  # (B, 1024)

        # ✅ 여기서 projection 통과시켜 1024 -> 768
        emb = self.clip.visual_projection(pooled)  # (B, 768)
        return emb
