from __future__ import annotations
import torch

def collate_fn(examples, expand_image_to_t: int | None = None):
    """
    examples[i]["pixel_values"]:
      - 이미지: (3,H,W)
      - 비디오: (T,3,H,W)
    반환:
      - 모두 이미지뿐이면 -> (B,3,H,W)  (expand_image_to_t가 주어지면 (B,T,3,H,W)로 확장)
      - 비디오가 하나라도 있으면 -> (B,T,3,H,W) (이미지는 T=1로 확장 후(필요 시) pad/replicate)
    """
    xs = [ex["pixel_values"] for ex in examples]
    ys = torch.tensor([int(ex["label"]) for ex in examples], dtype=torch.long)

    # 3D/4D 섞임 여부 확인
    ndims = [x.ndim for x in xs]
    has_video = any(n == 4 for n in ndims)
    has_image = any(n == 3 for n in ndims)

    if not has_video:
        x = torch.stack(xs, dim=0)  # (B,3,H,W)
        if expand_image_to_t:
            B, C, H, W = x.shape
            x = x.unsqueeze(1).repeat(1, expand_image_to_t, 1, 1, 1)  # (B,T,3,H,W)
        return {"pixel_values": x, "labels": ys}

    # 비디오가 있는 배치 → 전부 (T,3,H,W)화
    proc = []
    T_max = 0
    for t in xs:
        if t.ndim == 3:  # 이미지
            t = t.unsqueeze(0)  # (1,3,H,W)
        T_max = max(T_max, t.shape[0])
        proc.append(t)

    # 길이 다르면 마지막 프레임으로 replicate pad
    padded = []
    for t in proc:
        if t.shape[0] == T_max:
            padded.append(t)
        else:
            pad = t[-1:].repeat(T_max - t.shape[0], 1, 1, 1)  # (ΔT,3,H,W)
            padded.append(torch.cat([t, pad], dim=0))

    x = torch.stack(padded, dim=0)  # (B,T,3,H,W)
    return {"pixel_values": x, "labels": ys}