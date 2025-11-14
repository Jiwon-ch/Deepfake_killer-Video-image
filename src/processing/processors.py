from __future__ import annotations
from typing import Callable, Tuple, Optional, List
import os, io
import numpy as np
from PIL import Image

import torch
from torchvision.transforms import (
    Compose, Resize, RandomRotation, RandomAdjustSharpness, ToTensor, Normalize
)
from transformers import CLIPImageProcessor

# --------------------------
# (옵션) 얼굴 크롭 - dlib 있으면 사용
# --------------------------
try:
    import dlib
    _HAVE_DLIB = True
    _face_detector = dlib.get_frontal_face_detector()
except Exception:
    _HAVE_DLIB = False
    _face_detector = None


# --------------------------
# 유틸: 이미지 로딩
# --------------------------
def _to_pil(x) -> Image.Image:
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    if isinstance(x, dict) and "bytes" in x:
        return Image.open(io.BytesIO(x["bytes"])).convert("RGB")
    if isinstance(x, str):
        return Image.open(x).convert("RGB")
    if isinstance(x, np.ndarray):
        return Image.fromarray(x).convert("RGB")
    raise ValueError(f"Unsupported image payload: {type(x)}")


def _get_boundingbox(face, width: int, height: int):
    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * 1.3)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    x1 = max(int(cx - size_bb // 2), 0)
    y1 = max(int(cy - size_bb // 2), 0)
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)
    return x1, y1, size_bb


def _maybe_face_crop(img: Image.Image, enable: bool) -> Image.Image:
    if not enable or not _HAVE_DLIB:
        return img
    np_img = np.array(img.convert("RGB"))
    h, w = np_img.shape[:2]
    faces = _face_detector(np_img, 1)
    if not faces:
        return img
    face = max(faces, key=lambda r: r.width() * r.height())
    x, y, size = _get_boundingbox(face, w, h)
    cropped = np_img[y:y + size, x:x + size]
    return Image.fromarray(cropped)


# --------------------------
# 비디오 프레임 샘플링
# --------------------------
def sample_video_frames_to_pil(
    video_path: str,
    num_frames: int = 12,
    strategy: str = "uniform",  # "uniform" | "center-jitter" | "window"
) -> Optional[List[Image.Image]]:
    """
    비디오에서 정확히 num_frames개 PIL 이미지를 뽑아 리스트로 반환. 실패 시 None.
    """
    if not isinstance(video_path, str) or not os.path.exists(video_path):
        return None

    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        return None

    # 인덱스 선택
    if strategy == "uniform":
        idxs = np.linspace(0, max(0, total - 1), num_frames).round().astype(int).tolist()
    elif strategy == "center-jitter":
        center = total // 2
        half = max(1, total // 4)
        span = np.linspace(center - half, center + half, num_frames)
        noise = np.random.uniform(-0.5, 0.5, size=num_frames)
        idxs = np.clip(np.round(span + noise), 0, total - 1).astype(int).tolist()
    elif strategy == "window":
        if total >= num_frames:
            start = max(0, (total - num_frames) // 2)
            idxs = list(range(start, start + num_frames))
        else:
            base = list(range(total))
            idxs = base + [total - 1] * (num_frames - total)
    else:
        cap.release()
        raise ValueError(f"unknown strategy: {strategy}")

    frames: List[Image.Image] = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            frames.append(None)
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))

    cap.release()

    # 실패 프레임 보정
    if any(f is None for f in frames):
        for i in range(len(frames)):
            if frames[i] is None:
                j = i - 1
                while j >= 0 and frames[j] is None:
                    j -= 1
                if j >= 0:
                    frames[i] = frames[j]
                else:
                    k = i + 1
                    while k < len(frames) and frames[k] is None:
                        k += 1
                    if k < len(frames):
                        frames[i] = frames[k]
                    else:
                        return None
    return frames


# --------------------------
# CLIP Processor
# --------------------------
def get_clip_processor(name: str = "openai/clip-vit-large-patch14") -> CLIPImageProcessor:
    return CLIPImageProcessor.from_pretrained(name)


# --------------------------
# 이미지/비디오 겸용 변환 (set_transform에 연결) — ★배치 단위 버전★
# --------------------------
def build_media_transforms(
    processor: CLIPImageProcessor,
    *,
    image_key: Optional[str] = "image",
    video_key: Optional[str] = "video_path",
    do_face_crop: bool = False,
    rotation_deg: int = 15,
    num_frames: int = 12,
    video_strategy: str = "uniform",
) -> Tuple[Callable, Callable]:
    """
    반환: (train_transform, val_transform)
    모든 샘플을 **고정된 T**로 맞춤:
      - 비디오: pixel_values = (T,3,H,W)
      - 이미지: pixel_values = (T,3,H,W)  ← 단일 프레임을 num_frames만큼 복제
    """
    size = processor.size.get("shortest_edge", processor.size.get("height", 224))
    mean, std = processor.image_mean, processor.image_std

    train_img_tfm = Compose([
        Resize((size, size)),
        RandomRotation(rotation_deg),
        RandomAdjustSharpness(2.0),
        ToTensor(),
        Normalize(mean=mean, std=std),
    ])
    val_img_tfm = Compose([
        Resize((size, size)),
        ToTensor(),
        Normalize(mean=mean, std=std),
    ])

    def _prep_one_image(im: Image.Image, train: bool) -> torch.Tensor:
        tfm = train_img_tfm if train else val_img_tfm
        im = _maybe_face_crop(im, do_face_crop)
        t = tfm(im)                         # (3,H,W)
        t = t.unsqueeze(0).repeat(num_frames, 1, 1, 1)  # (T,3,H,W)로 복제
        return t

    def _prep_one_video(path: str, train: bool) -> Optional[torch.Tensor]:
        frames = sample_video_frames_to_pil(
            path, num_frames=num_frames, strategy=video_strategy
        )
        if frames is None:
            return None
        tfm = train_img_tfm if train else val_img_tfm
        tens = [tfm(_maybe_face_crop(f, do_face_crop)) for f in frames]
        return torch.stack(tens, dim=0)  # (T,3,H,W)

    def _prep_batch(examples, train: bool):
        n = None
        for k, v in examples.items():
            if isinstance(v, list):
                n = len(v)
                break
        if n is None:
            # 빈 배치 방어
            examples["pixel_values"] = []
            return examples

        pix_list: List[torch.Tensor] = []
        for i in range(n):
            ten = None
            # 1) 비디오 우선
            if video_key and (video_key in examples):
                v = examples[video_key][i]
                if isinstance(v, str) and v:
                    ten = _prep_one_video(v, train=train)

            # 2) 이미지 폴백
            if (ten is None) and image_key and (image_key in examples):
                img_payload = examples[image_key][i]
                if img_payload is not None:
                    try:
                        im = _to_pil(img_payload)
                        ten = _prep_one_image(im, train=train)
                    except Exception:
                        ten = None

            # 3) 마지막 방어: 실패 시 스킵 대신 최소 더미(검정 프레임) 생성
            if ten is None:
                # (T,3,H,W) 제로 텐서 — 배치 스택을 깨지 않기 위함
                H = W = size
                ten = torch.zeros((num_frames, 3, H, W), dtype=torch.float32)

            pix_list.append(ten)

        examples["pixel_values"] = pix_list
        return examples

    def train_transform(examples): return _prep_batch(examples, train=True)
    def val_transform(examples):   return _prep_batch(examples, train=False)

    return train_transform, val_transform


def attach_media_transforms(
    train_data, test_data,
    *,
    image_key: Optional[str],
    video_key: Optional[str],
    clip_model_name: str = "openai/clip-vit-large-patch14",
    do_face_crop: bool = False,
    rotation_deg: int = 15,
    num_frames: int = 12,
    video_strategy: str = "uniform",
):

    processor = get_clip_processor(clip_model_name)
    train_tfm, val_tfm = build_media_transforms(
        processor,
        image_key=image_key,
        video_key=video_key,
        do_face_crop=do_face_crop,
        rotation_deg=rotation_deg,
        num_frames=num_frames,
        video_strategy=video_strategy,
    )
    train_data.set_transform(train_tfm)
    test_data.set_transform(val_tfm)
    return processor


__all__ = [
    "get_clip_processor",
    "build_media_transforms",
    "attach_media_transforms",
    "sample_video_frames_to_pil",
]