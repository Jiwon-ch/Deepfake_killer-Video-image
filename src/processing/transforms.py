from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

# dlib은 선택 사용: 설치 안 되어도 전체 파이프라인이 죽지 않도록 try/except
try:
    import dlib
    _HAVE_DLIB = True
    _face_detector = dlib.get_frontal_face_detector()
except Exception:
    _HAVE_DLIB = False
    _face_detector = None

import numpy as np
from typing import Callable, Tuple
from PIL import Image
from torchvision.transforms import (
    Compose, Resize, RandomRotation, RandomAdjustSharpness,
    ToTensor, Normalize
)
from transformers import CLIPImageProcessor


# ----------------------------
# 얼굴 검출 & 크롭 (선택)
# ----------------------------
def _get_boundingbox(face, width: int, height: int) -> Tuple[int, int, int]:
    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * 1.3)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    x1 = max(int(cx - size_bb // 2), 0)
    y1 = max(int(cy - size_bb // 2), 0)
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)
    return x1, y1, size_bb


def crop_face_if_possible(image: Image.Image) -> Image.Image:
    """
    dlib이 있으면 얼굴 검출 후 크롭, 없으면 원본 반환.
    얼굴 못 찾으면 원본 반환.
    """
    if not _HAVE_DLIB:
        return image

    if image.mode != "RGB":
        image = image.convert("RGB")

    np_img = np.array(image)
    h, w = np_img.shape[:2]
    faces = _face_detector(np_img, 1)
    if not faces:
        return image

    # 가장 큰 얼굴
    face = max(faces, key=lambda r: r.width() * r.height())
    x, y, size = _get_boundingbox(face, w, h)
    cropped = np_img[y:y + size, x:x + size]
    return Image.fromarray(cropped)


# ----------------------------
# CLIP Processor & Transforms
# ----------------------------
def get_clip_processor(clip_model_name: str = "openai/clip-vit-large-patch14") -> CLIPImageProcessor:
    """
    CLIP 전처리 파라미터(평균/표준편차/사이즈)를 얻기 위한 processor.
    """
    return CLIPImageProcessor.from_pretrained(clip_model_name)


def build_transforms(
    processor: CLIPImageProcessor,
    do_face_crop: bool = True,
    rotation_deg: int = 15,
) -> Tuple[Callable, Callable]:
    """
    HF Dataset.set_transform 에 바로 넣어 쓸 수 있는 transform 함수 두 개를 생성.
    - train_transform(examples)
    - val_transform(examples)
    """
    image_mean = processor.image_mean       # [0.481, 0.457, 0.408]
    image_std = processor.image_std         # [0.268, 0.261, 0.275]
    size = processor.size.get("shortest_edge", processor.size.get("height", 224))

    train_tfm = Compose([
        Resize((size, size)),
        RandomRotation(rotation_deg),
        RandomAdjustSharpness(2.0),
        ToTensor(),
        Normalize(mean=image_mean, std=image_std),
    ])

    val_tfm = Compose([
        Resize((size, size)),
        ToTensor(),
        Normalize(mean=image_mean, std=image_std),
    ])

    def _maybe_crop(img: Image.Image) -> Image.Image:
        if do_face_crop:
            return crop_face_if_possible(img.convert("RGB"))
        return img.convert("RGB")

    def train_transform(examples):
        imgs = [_maybe_crop(img) for img in examples["image"]]
        examples["pixel_values"] = [train_tfm(img) for img in imgs]
        return examples

    def val_transform(examples):
        imgs = [_maybe_crop(img) for img in examples["image"]]
        examples["pixel_values"] = [val_tfm(img) for img in imgs]
        return examples

    return train_transform, val_transform


# ----------------------------
# collate_fn (Trainer용)
# ----------------------------
import torch

def collate_fn(examples):
    """
    set_transform 로 examples['pixel_values'] 에 (3,H,W) 텐서가 들어있다고 가정.
    """
    pixel_values = torch.stack([ex["pixel_values"] for ex in examples])
    labels = torch.tensor([int(ex["label"]) for ex in examples], dtype=torch.long)
    return {"pixel_values": pixel_values, "labels": labels}