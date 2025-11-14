from __future__ import annotations
import os, hashlib
from PIL import Image

def _key(path: str) -> str:
    return hashlib.sha1(path.encode("utf-8", "ignore")).hexdigest()

def get_cached_frame_path(video_path: str, cache_dir: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, _key(video_path) + ".jpg")

def save_frame_cache(img: Image.Image, cache_path: str) -> None:
    img.save(cache_path, format="JPEG", quality=95)