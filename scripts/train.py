import os
import torch
from transformers import set_seed, TrainingArguments, Trainer
from transformers.trainer_utils import get_last_checkpoint
import torch.nn.functional as F
import evaluate
import numpy as np

from src import ClipBackbone, UnifiedAdapterModel
from src import prepare_deepfake_dataset
from src import attach_media_transforms
from src import collate_fn

def main():
    set_seed(42)
    use_bf16 = False
    seed = 42
    test_size = 0.2
    # 1 데이터 로드 (이미지 데이터셋이면 이대로 OK)
    train_data, test_data, label2id, id2label, class_labels, image_key, video_key = prepare_deepfake_dataset(
        data_path=None,
        split="train",
        data_files="/root/train_data/subset/metadata.tsv",
        delimiter="\t",
        seed=seed,
        test_size=test_size,
    )   

    # 2 전처리 훅 부착 (이미지는 (1,3,H,W)로, 비디오는 (T,3,H,W)로 나오게)
    attach_media_transforms(
        train_data, test_data,
        image_key=image_key,          
        video_key=video_key,           
        clip_model_name="openai/clip-vit-large-patch14",
        do_face_crop=False,
        rotation_deg=15,
        num_frames=16,
    )

    # 3 모델
    backbone = ClipBackbone(
        model_name="openai/clip-vit-large-patch14",
        dtype="fp32",
        freeze_backbone=True,
    )
    model = UnifiedAdapterModel(
        backbone=backbone,
        num_frames=12,              
        adapter_type="transformer",
        adapter_kwargs=dict(
            num_layers=2,
            num_heads=8,
            mlp_ratio=4.0,
            dropout=0.1,
            attn_dropout=0.1,
        ),
        temporal_pool="attn",
        head_hidden=(2048, 1024),
        num_classes=len(class_labels.names),
        id2label=id2label,
        label2id=label2id,
    )

    # 4) 메트릭 (이진 전제)
    acc_metric = evaluate.load("accuracy")
    auc_metric = evaluate.load("roc_auc")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.from_numpy(logits).softmax(dim=1).numpy()
        preds = probs.argmax(axis=1)
        acc = acc_metric.compute(predictions=preds, references=labels)["accuracy"]
        auc = auc_metric.compute(
            prediction_scores=probs[:, 1], references=labels, average="macro"
        )["roc_auc"]
        return {"accuracy": acc, "roc_auc": auc}


    class AdapterTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            x = inputs["pixel_values"]         # (B,T,3,H,W)
            y = inputs["labels"].long()        # (B,)
            media_type = inputs.get("media_type")
            temporal_lengths = inputs.get("temporal_lengths")
            logits = model(
                x,
                media_type=media_type,
                temporal_lengths=temporal_lengths,
            )                                   # (B,num_classes)
            loss = F.cross_entropy(logits, y, label_smoothing=0.1)
            return (loss, {"logits": logits}) if return_outputs else loss


    args = TrainingArguments(
        output_dir="./outputs_video",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=1e-4,
        num_train_epochs=10,
        weight_decay=0.02,
        warmup_ratio=0.1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_safetensors=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_roc_auc",
        greater_is_better=True,
        remove_unused_columns=False,
        dataloader_num_workers=max(4, (os.cpu_count() or 8)//2),
        dataloader_pin_memory=True,
        fp16=not use_bf16,
        bf16=use_bf16,
        logging_steps=50,
        report_to="none",
    )

    trainer = AdapterTrainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=test_data,
        data_collator=collate_fn,  
        compute_metrics=compute_metrics,
    )

    ckpt_bin = "/root/jiwon/outputs/checkpoint-35690/pytorch_model.bin"

    if os.path.exists(ckpt_bin):
        print(f">> load pretrained weights from {ckpt_bin}")
        state_dict = torch.load(ckpt_bin, map_location="cpu")

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print("   - missing keys   :", missing)
        print("   - unexpected keys:", unexpected)
        print(">> start finetuning on video dataset from loaded weights")
    else:
        print(">> checkpoint bin not found, train from scratch")

    # 👉 중요: 더 이상 resume_from_checkpoint 쓰지 않음
    trainer.train()

if __name__ == "__main__":
    main()
