import os, json, torch
from transformers import CLIPImageProcessor, CLIPModel
from safetensors.torch import save_file
from src import ClipBackbone, UnifiedAdapterModel

CHECKPOINT_PATH = "./outputs/checkpoint-35690/pytorch_model.bin"
EXPORT_DIR = "./model/clip_base"
os.makedirs(EXPORT_DIR, exist_ok=True)

clip_name = "openai/clip-vit-large-patch14"

print("="*60)
print("🚀 Starting export for OFFLINE submission")
print("="*60)

# ================================================================
# (1) CLIP backbone을 로컬에 저장 (허깅페이스 다운로드 방지)
# ================================================================
print("\n[1/7] Saving CLIP backbone locally...")
clip_model = CLIPModel.from_pretrained(clip_name)
backbone_dir = os.path.join(EXPORT_DIR, "clip_backbone")
clip_model.save_pretrained(backbone_dir)
print(f"  ✅ CLIP saved to: {backbone_dir}")

# ================================================================
# (2) 수정된 ClipBackbone으로 모델 재구성
# ================================================================
print("\n[2/7] Building model with ClipBackbone...")
backbone = ClipBackbone(
    model_name=backbone_dir,  # 로컬 경로 사용
    dtype="fp32",
    freeze_backbone=True
)
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
print(f"  ✅ Model structure created")

# ================================================================
# (3) 학습된 체크포인트 로드
# ================================================================
print(f"\n[3/7] Loading checkpoint from: {CHECKPOINT_PATH}")
if not os.path.exists(CHECKPOINT_PATH):
    print(f"❌ ERROR: Checkpoint not found at {CHECKPOINT_PATH}")
    raise SystemExit(1)

state = torch.load(CHECKPOINT_PATH, map_location="cpu")
print(f"  - Checkpoint keys: {len(state.keys())}")

# strict=False로 로드 (position_ids 등 버전 차이 대응)
incompatible = model.load_state_dict(state, strict=False)
print(f"  - Missing keys: {len(incompatible.missing_keys)}")
if incompatible.missing_keys:
    position_missing = [k for k in incompatible.missing_keys if 'position' in k.lower()]
    critical_missing = [k for k in incompatible.missing_keys if 'position' not in k.lower()]
    print(f"    • Position-related (OK): {len(position_missing)}")
    if critical_missing:
        print(f"    • ⚠️ Critical missing: {critical_missing}")

print(f"  - Unexpected keys: {len(incompatible.unexpected_keys)}")
if incompatible.unexpected_keys:
    print(f"    {incompatible.unexpected_keys[:3]}...")

print(f"  ✅ Checkpoint loaded")

# ================================================================
# (4) 이미지 프로세서 저장
#    ※ 모델 저장과 별개로 전처리 설정을 로컬에 고정해둠(오프라인 추론)
# ================================================================
print(f"\n[4/7] Saving image processor...")
# (주의) processor는 model.save_pretrained에 포함되지 않으므로 model 디렉토리에서 읽지 말고 원본에서 받아 저장
processor = CLIPImageProcessor.from_pretrained(clip_name)
processor.save_pretrained(EXPORT_DIR)
print(f"  ✅ Processor saved to: {EXPORT_DIR}")

# ================================================================
# (5) Adapter+Head만 추출해서 저장 (권장: 모듈형 배포/디버깅용)
# ================================================================
print("\n[5/7] Extracting adapter+head weights...")
adapter_state = {k: v for k, v in model.state_dict().items() if not k.startswith("backbone.")}
print(f"  - Adapter+Head parameters: {len(adapter_state)}")
print(f"  - Sample keys: {list(adapter_state.keys())[:5]}")

adapter_path = os.path.join(EXPORT_DIR, "adapter_head.safetensors")
save_file(adapter_state, adapter_path)
adapter_size = os.path.getsize(adapter_path) / (1024**2)
print(f"  ✅ Adapter saved: {adapter_path} ({adapter_size:.2f} MB)")

# ================================================================
# (6) 제출용 전체 모델 저장 (필터링 금지 + position_ids 보강)
# ================================================================
print("\n[6/7] Saving FULL model for submission...")

def ensure_position_ids(sd: dict, mdl: UnifiedAdapterModel) -> dict:
    """제출(strict=True) 환경 호환을 위해 position_ids 3종을 보강."""
    import torch

    # 텍스트 길이 (CLIP text 보통 77)
    try:
        txt_conf = mdl.backbone.clip.text_model.config
        txt_len = int(getattr(txt_conf, "max_position_embeddings", 77))
    except Exception:
        txt_len = 77
    txt_pos = torch.arange(0, txt_len, dtype=torch.long).unsqueeze(0)

    # 비전 길이 (ViT-L/14: 224/14=16 → 16*16+1=257)
    try:
        vconf = mdl.backbone.clip.vision_model.config
        v_len = (vconf.image_size // vconf.patch_size) ** 2 + 1
    except Exception:
        v_len = 257
    vis_pos = torch.arange(0, v_len, dtype=torch.long).unsqueeze(0)

    # 제출 쪽이 기대하는 3개 키 모두 보장
    required = {
        "backbone.clip.text_model.embeddings.position_ids": txt_pos,
        "backbone.clip.vision_model.embeddings.position_ids": vis_pos,
        "backbone.vision.embeddings.position_ids": vis_pos,  # 일부 구현에서 이 alias를 요구
    }
    for k, v in required.items():
        if k not in sd:
            sd[k] = v
    return sd

full_state_all = model.state_dict()             # ★ 필터링 없이 전체 저장
full_state_all = ensure_position_ids(full_state_all, model)

submit_model_path = os.path.join(EXPORT_DIR, "model.bin")  # 제출 규격에 흔한 이름
torch.save(full_state_all, submit_model_path)
full_size_all = os.path.getsize(submit_model_path) / (1024**2)
print(f"  ✅ Submission model saved: {submit_model_path} ({full_size_all:.2f} MB)")

# (옵션) 디버깅/비교용: text 관련 키 제외한 필터 버전도 함께 보관 (제출용 아님)
debug_filtered = {k: v for k, v in full_state_all.items() if ('text_model' not in k and 'clip.text' not in k)}
debug_path = os.path.join(EXPORT_DIR, "pytorch_model.debug_filtered.bin")
torch.save(debug_filtered, debug_path)
debug_size = os.path.getsize(debug_path) / (1024**2)
print(f"  🧪 Debug filtered model saved: {debug_path} ({debug_size:.2f} MB)")

# 간이 검증: 필요한 키가 존재하는지 확인
for k in [
    "backbone.clip.text_model.embeddings.position_ids",
    "backbone.clip.vision_model.embeddings.position_ids",
    "backbone.vision.embeddings.position_ids",
]:
    assert k in full_state_all, f"❌ Missing required key for submission: {k}"
print("  🔎 Required position_ids keys verified.")

# ================================================================
# (7) 커스텀 설정 저장
# ================================================================
print("\n[7/7] Saving config...")
cfg = {
    "model_type": "unified_adapter",
    "clip_model_name": "clip_backbone",  # 로컬 디렉토리 참조
    "num_frames": 12,
    "adapter_type": "tconv",
    "temporal_pool": "mean",
    "head_hidden": 1024,
    "num_classes": 2,
    "id2label": {0: "real", 1: "fake"},
    "label2id": {"real": 0, "fake": 1}
}
config_path = os.path.join(EXPORT_DIR, "custom_config.json")
with open(config_path, "w") as f:
    json.dump(cfg, f, indent=2)
print(f"  ✅ Config saved: {config_path}")

# ================================================================
# 최종 파일 리스트
# ================================================================