from __future__ import annotations
import warnings, os
warnings.filterwarnings("ignore")

from typing import Tuple, Dict, Optional, List
from datasets import load_dataset, Dataset, ClassLabel, DatasetDict

# ✅ 단수 파일명 + 함수명 수정
from .processors import get_clip_processor, build_media_transforms
from .transforms import collate_fn  # 여기에 '스마트' collate가 있어야 함

VIDEO_COL_CANDIDATES: List[str] = ["video_path", "video", "filepath", "path", "file"]
IMAGE_COL_CANDIDATES: List[str] = ["image", "img", "pixel_values"]
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _pick_first_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in cols:
            return c
    return None


def _derive_label(example) -> Dict:
    if "label_id" in example:
        try:
            return {"label": int(example["label_id"])}
        except Exception:
            pass
    if "answer" in example and isinstance(example["answer"], str):
        tok = example["answer"].strip().split()[0].upper()
        if tok in ["REAL", "FAKE"]:
            return {"label": 0 if tok == "REAL" else 1}
    if "label" in example:
        v = example["label"]
        if isinstance(v, int):
            return {"label": v}
        if isinstance(v, str):
            vv = v.strip().lower()
            if vv in ["real", "fake"]:
                return {"label": 0 if vv == "real" else 1}
    return {"label": None}


def _split_path_by_modality(example: Dict) -> Dict:
    """
    FFPP metadata처럼 path 하나만 있고 modality로 이미지/비디오를 구분하는 경우,
    명시적인 image / video_path 컬럼을 만들어준다.
    """
    path = None
    for k in ["path", "filepath", "file"]:
        if k in example:
            path = example.get(k)
            if path:
                break
    modality = str(example.get("modality", "")).lower()
    ext = os.path.splitext(path or "")[1].lower()

    is_image = modality.startswith("image") or ext in IMAGE_EXTS
    is_video = modality.startswith("video") or ext in VIDEO_EXTS

    img_path = path if is_image else ""
    vid_path = path if is_video else ""
    # 빈 문자열로 넣어 Arrow가 string 타입으로 고정되도록 한다(None -> null 타입 방지)
    return {
        "image": img_path,
        "video_path": vid_path,
    }


def _has_valid_media_factory(image_key: Optional[str], video_key: Optional[str]):
    def _fn(ex):
        ok_img = False
        ok_vid = False
        if image_key is not None and ex.get(image_key) is not None:
            img_path = ex.get(image_key)
            ok_img = True if not isinstance(img_path, str) else os.path.exists(img_path)
        if video_key is not None:
            vp = ex.get(video_key)
            ok_vid = isinstance(vp, str) and len(vp) > 0 and os.path.exists(vp)
        return ok_img or ok_vid
    return _fn


def prepare_deepfake_dataset(
    data_path: Optional[str] = None,
    split: str = "train",
    test_size: float = 0.2,
    seed: int = 42,
    label_names: Optional[list] = None,
    *,
    data_files: Optional[str | dict] = None,
    delimiter: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Tuple[Dataset, Dataset, Dict[str, int], Dict[int, str], ClassLabel, Optional[str], Optional[str]]:
    if data_files is not None:
        loader_name = "csv"
        kwargs = {"data_files": data_files, "split": "train"}
        if delimiter:
            kwargs["delimiter"] = delimiter
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        ds: Dataset = load_dataset(loader_name, **kwargs)
    else:
        if data_path is None:
            raise ValueError("data_path 또는 data_files 중 하나는 필요해.")
        ds: Dataset = load_dataset(data_path, split=split, cache_dir=cache_dir)

    ds = ds.map(_derive_label, batched=False)
    ds = ds.filter(lambda ex: ex["label"] is not None)

    cols = ds.column_names

    # path + modality 조합이면 명시적인 image/video_path 컬럼 생성
    if ("path" in cols or "filepath" in cols or "file" in cols) and ("modality" in cols):
        if "image" not in cols and "video_path" not in cols:
            ds = ds.map(_split_path_by_modality, batched=False)
            cols = ds.column_names

    image_key = _pick_first_col(cols, IMAGE_COL_CANDIDATES)
    video_key = _pick_first_col(cols, VIDEO_COL_CANDIDATES)
    if image_key is None and video_key is None:
        raise ValueError(f"'image'나 'video_path' 계열 컬럼이 없어. columns={cols}")

    ds = ds.filter(_has_valid_media_factory(image_key, video_key))

    if isinstance(ds.features.get("label"), ClassLabel) and label_names is None:
        class_labels: ClassLabel = ds.features["label"]
    else:
        if label_names is None:
            label_names = ["real", "fake"]
        class_labels = ClassLabel(num_classes=len(label_names), names=label_names)

    id2label = {i: name for i, name in enumerate(class_labels.names)}
    label2id = {name: i for i, name in id2label.items()}

    def _map_label2id(example):
        v = example["label"]
        if isinstance(v, int):
            return {"label": v}
        return {"label": class_labels.str2int(v)}

    ds = ds.map(_map_label2id, batched=False)
    ds = ds.cast_column("label", class_labels)

    dsd: DatasetDict = ds.train_test_split(
        test_size=test_size, shuffle=True, stratify_by_column="label", seed=seed
    )
    train_data: Dataset = dsd["train"]
    test_data: Dataset = dsd["test"]

    return train_data, test_data, label2id, id2label, class_labels, image_key, video_key


__all__ = [
    "prepare_deepfake_dataset",
    "collate_fn",
]
