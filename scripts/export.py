# scripts/export.py
import os, json, torch
from transformers import CLIPImageProcessor
from src import ClipBackbone
from src import UnifiedAdapterModel

CHECKPOINT_PATH = "./outputs/checkpoint-35690/pytorch_model.bin"
EXPORT_DIR = "./model/clip_base"
os.makedirs(EXPORT_DIR, exist_ok=True)

# (1) 같은 설정으로 모델 재구성
clip_name = "openai/clip-vit-large-patch14"
backbone = ClipBackbone(model_name=clip_name, dtype="fp32", freeze_backbone=True)
model = UnifiedAdapterModel(
    backbone=backbone,
    num_frames=12,
    adapter_type="tconv",
    temporal_pool="mean",
    head_hidden=1024,
    num_classes=2,     
    id2label={0: "real", 1: "fake"},
    label2id={"real": 0, "fake": 1}
)

# (2) 체크포인트 로드
state = torch.load(CHECKPOINT_PATH, map_location="cpu")
model.load_state_dict(state, strict=False)
print("✅ Checkpoint loaded.")

# (3) processor 저장
processor = CLIPImageProcessor.from_pretrained(clip_name)
processor.save_pretrained(EXPORT_DIR)

# (4) 모델 가중치 저장
torch.save(model.state_dict(), os.path.join(EXPORT_DIR, "pytorch_model.bin"))

# (5) config 저장
cfg = {
    "model_type": "unified_adapter",
    "clip_model_name": clip_name,
    "num_frames": 12,
    "adapter_type": "tconv",
    "temporal_pool": "mean",
    "head_hidden": 1024,
    "num_classes": 2,
    "id2label": {0: "real", 1: "fake"},
    "label2id": {"real": 0, "fake": 1}
}
with open(os.path.join(EXPORT_DIR, "custom_config.json"), "w") as f:
    json.dump(cfg, f, indent=2)

print(f"🎯 Export complete → {EXPORT_DIR}")