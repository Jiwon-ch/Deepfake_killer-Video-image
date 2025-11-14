from __future__ import annotations
import os
from typing import List, Optional
import cv2
import numpy as np
from PIL import Image

def extract_frames_to_pil(
    video_path: str,
    num_frames: int = 12,
    strategy: str = "uniform",   # "uniform" | "middle"
    start_ratio: float = 0.0,
    end_ratio: float = 1.0,
) -> Optional[List[Image.Image]]:
    if not isinstance(video_path, str) or not os.path.exists(video_path):
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        return None

    s = int(max(0, min(total - 1, total * start_ratio)))
    e = int(max(0, min(total,     total * end_ratio)))
    if e <= s: e = total

    if strategy == "uniform":
        idxs = np.linspace(s, e - 1, num_frames, dtype=int)
    else:  # "middle"
        mid = (s + e) // 2
        half = max(1, num_frames // 2)
        lo = max(0, mid - half)
        hi = min(total - 1, mid + half)
        idxs = np.linspace(lo, hi, num_frames, dtype=int)

    frames: List[Image.Image] = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok or frame is None:
            frames.append(None)
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))

    cap.release()
    frames = [f for f in frames if f is not None]
    return frames if frames else None