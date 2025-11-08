# src/processing/dataloader.py
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from typing import Tuple, Dict, Optional
from datasets import load_dataset, Dataset, ClassLabel, DatasetDict

from .transforms import get_clip_processor, build_transforms, collate_fn


def prepare_deepfake_dataset(
    hf_path: str = "Hemg/deepfake-and-real-images",
    split: str = "train",
    test_size: float = 0.4,
    seed: int = 42,
    label_names: Optional[list] = None,
) -> Tuple[Dataset, Dataset, Dict[str, int], Dict[int, str], ClassLabel]:
    """
    - HF dataset 로드
    - 라벨을 ClassLabel 로 보장
    - stratified split
    - label2id / id2label dict 반환
    """
    ds: Dataset = load_dataset(hf_path, split=split)

    # 라벨 스키마
    if isinstance(ds.features.get("label"), ClassLabel) and label_names is None:
        class_labels: ClassLabel = ds.features["label"]
    else:
        if label_names is None:
            uniques = sorted(set(ds["label"]))
            label_names = list(uniques)
        class_labels = ClassLabel(num_classes=len(label_names), names=label_names)

    id2label = {i: name for i, name in enumerate(class_labels.names)}
    label2id = {name: i for i, name in id2label.items()}

    # 라벨을 int로 통일
    def _map_label2id(example):
        val = example["label"]
        if isinstance(val, int):
            return {"label": val}
        return {"label": class_labels.str2int(val)}

    ds = ds.map(_map_label2id, batched=False)
    ds = ds.cast_column("label", class_labels)

    # stratified split
    dsd: DatasetDict = ds.train_test_split(
        test_size=test_size, shuffle=True, stratify_by_column="label", seed=seed
    )
    train_data: Dataset = dsd["train"]
    test_data: Dataset = dsd["test"]

    return train_data, test_data, label2id, id2label, class_labels


def attach_clip_transforms(
    train_data: Dataset,
    test_data: Dataset,
    clip_model_name: str = "openai/clip-vit-large-patch14",
    do_face_crop: bool = True,
    rotation_deg: int = 15,
):
    """
    CLIP 전처리 + (옵션) 얼굴 크롭 transform 을 HF Dataset 에 연결.
    """
    processor = get_clip_processor(clip_model_name)
    train_tfm, val_tfm = build_transforms(processor, do_face_crop=do_face_crop, rotation_deg=rotation_deg)
    train_data.set_transform(train_tfm)
    test_data.set_transform(val_tfm)
    return processor  # 필요하면 밖에서 image_mean/std 등 참고 가능


__all__ = [
    "prepare_deepfake_dataset",
    "attach_clip_transforms",
    "collate_fn",
]