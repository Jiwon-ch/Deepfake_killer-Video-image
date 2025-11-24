from typing import Literal
import os
import torch
import torch.nn as nn
from transformers import CLIPModel


class ClipBackbone(nn.Module):
    """
    CLIP ViT-L/14 백본 래퍼
      - forward_images: (B,3,H,W) → (B,D)
      - CLS 토큰만 반환할지 여부(use_cls_token) 선택
      - freeze / unfreeze_last_n_blocks / LayerNorm-only 학습 지원
      - strict 로더 호환: text/vision의 position_ids를 강제 등록 + state_dict에 alias 키 복제
    """
    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        dtype: Literal["fp32", "bf16", "fp16"] = "fp32",
        freeze_backbone: bool = True,
        unfreeze_last_n_blocks: int = 0,  # 0이면 완전 동결
        use_cls_token: bool = False,      # True면 projection 없이 CLS 토큰(1024차원) 반환
        train_layer_norm_only: bool = False,  # True면 LayerNorm 파라미터만 학습
    ):
        super().__init__()

        # 로컬 디렉토리면 오프라인 로드, 아니면 사전에 TRANSFORMERS_OFFLINE=1 권장
        local_only = os.path.isdir(model_name)
        self.clip: CLIPModel = CLIPModel.from_pretrained(
            model_name, local_files_only=local_only
        )

        # 편의 alias (일부 코드/체크포인트가 vision.* 경로를 기대하는 경우가 있어 둘 다 보유)
        self.vision = self.clip.vision_model
        self.visual_projection = self.clip.visual_projection

        self.use_cls_token = bool(use_cls_token)

        # 임베딩 차원
        if self.use_cls_token:
            self.embed_dim = self.vision.config.hidden_size  # CLS 토큰 차원(1024)
        else:
            self.embed_dim = getattr(self.clip.config, "projection_dim", None)
            if self.embed_dim is None:
                self.embed_dim = self.vision.config.hidden_size  # fallback

        # 버전 차이 대비: position_ids 강제 등록(텍스트/비전)
        self._ensure_position_ids()

        # dtype 지정: 텍스트는 사용 안 하므로 vision/projection만 캐스팅
        if dtype == "bf16":
            cast_dtype = torch.bfloat16
        elif dtype == "fp16":
            cast_dtype = torch.float16
        else:
            cast_dtype = torch.float32

        self.vision.to(dtype=cast_dtype)
        self.visual_projection.to(dtype=cast_dtype)

        # 동결/언프리즈
        if freeze_backbone:
            for p in self.clip.parameters():
                p.requires_grad = False
        if unfreeze_last_n_blocks > 0:
            blocks = list(self.vision.encoder.layers)
            for blk in blocks[-unfreeze_last_n_blocks:]:
                for p in blk.parameters():
                    p.requires_grad = True

        # LayerNorm만 학습하도록 설정(논문 설정: 나머지는 동결)
        if train_layer_norm_only:
            for m in self.clip.modules():
                if isinstance(m, nn.LayerNorm):
                    for p in m.parameters():
                        p.requires_grad = True

    def _ensure_position_ids(self) -> None:
        """HF/환경 버전에 따라 state_dict에 없을 수 있는 position_ids를 강제로 등록."""
        # 텍스트 position_ids (보통 77)
        try:
            txt_conf = self.clip.text_model.config
            txt_len = int(getattr(txt_conf, "max_position_embeddings", 77))
        except Exception:
            txt_len = 77
        txt_pos = torch.arange(0, txt_len, dtype=torch.long).unsqueeze(0)
        try:
            txt_emb = self.clip.text_model.embeddings
            if not hasattr(txt_emb, "position_ids"):
                txt_emb.register_buffer("position_ids", txt_pos, persistent=True)
        except Exception:
            pass  # 일부 변형 모델에서 text_model이 없을 수도

        # 비전 position_ids (ViT-L/14: 224/14=16 → 16*16+1=257)
        try:
            vconf = self.clip.vision_model.config
            v_len = (vconf.image_size // vconf.patch_size) ** 2 + 1
        except Exception:
            v_len = 257
        vis_pos = torch.arange(0, v_len, dtype=torch.long).unsqueeze(0)
        try:
            vemb = self.clip.vision_model.embeddings
            if not hasattr(vemb, "position_ids"):
                vemb.register_buffer("position_ids", vis_pos, persistent=True)
        except Exception:
            pass

        # 주의: self.vision은 self.clip.vision_model과 같은 객체라
        #       위에서 등록하면 둘 다에 경로상 존재하지만,
        #       state_dict 저장 시 중복 키가 스킵될 수 있으므로 아래 state_dict 오버라이드에서 보강.

    def state_dict(self, destination=None, prefix:str='', keep_vars:bool=False):
        """
        저장 시 'clip.vision_model.embeddings.position_ids' 뿐 아니라
        'vision.embeddings.position_ids' 키도 강제로 포함시켜 strict 로더 호환성 확보.
        """
        sd = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

        src_key = prefix + "clip.vision_model.embeddings.position_ids"
        dst_key = prefix + "vision.embeddings.position_ids"
        if src_key in sd and dst_key not in sd:
            # 동일 텐서를 복제 저장(메모리 오버헤드는 미미)
            sd[dst_key] = sd[src_key]

        # 텍스트/비전 쪽 키가 없으면 여기서도 한 번 더 보강
        # (이 경우는 거의 없지만 제출 환경이 strict=True면 안전망이 필요)
        txt_key = prefix + "clip.text_model.embeddings.position_ids"
        if txt_key not in sd:
            sd[txt_key] = torch.arange(0, 77, dtype=torch.long).unsqueeze(0)
        vis_key = src_key
        if vis_key not in sd:
            sd[vis_key] = torch.arange(0, 257, dtype=torch.long).unsqueeze(0)

        return sd

    @torch.no_grad()
    def forward_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # 입력을 vision 모듈의 디바이스/ dtype에 맞춤
        dev = next(self.vision.parameters()).device
        dt = next(self.vision.parameters()).dtype
        pixel_values = pixel_values.to(device=dev, dtype=dt)

        out = self.vision(pixel_values=pixel_values)
        cls = out.last_hidden_state[:, 0, :]  # (B, hidden)
        if self.use_cls_token:
            return cls

        pooled = (
            out.pooler_output
            if hasattr(out, "pooler_output") and out.pooler_output is not None
            else cls
        )
        emb = self.visual_projection(pooled)
        return emb
