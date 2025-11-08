# /scripts/train.py
import os
import torch
from transformers import set_seed
from src import ClipBackbone, UnifiedAdapterModel
from src import (
    prepare_deepfake_dataset,
    attach_clip_transforms,
    collate_fn,
)
import evaluate
import numpy as np
import torch.nn.functional as F
from transformers import TrainingArguments, Trainer
from transformers.trainer_utils import get_last_checkpoint


def main():
    # 1️⃣ 시드 설정
    set_seed(42)
    use_bf16 = False

    # 2️⃣ 데이터 로드
    train_data, test_data, label2id, id2label, class_labels = prepare_deepfake_dataset(
        hf_path="Hemg/deepfake-and-real-images",
        split="train",
        test_size=0.4,
        seed=42
    )
    attach_clip_transforms(train_data, test_data,
                           clip_model_name="openai/clip-vit-large-patch14",
                           do_face_crop=True, rotation_deg=15)

    # 3️⃣ 모델 구성
    backbone = ClipBackbone(
        model_name="openai/clip-vit-large-patch14",
        dtype="fp32",
        freeze_backbone=True,
    )
    model = UnifiedAdapterModel(
        backbone=backbone,
        num_frames=12,
        adapter_type="tconv",
        temporal_pool="mean",
        head_hidden=1024,
        num_classes=len(class_labels.names),
        id2label=id2label,
        label2id=label2id,
    )

    # 4️⃣ 메트릭 정의
    acc_metric = evaluate.load("accuracy")
    auc_metric = evaluate.load("roc_auc")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.from_numpy(logits).softmax(dim=1).numpy()
        preds = probs.argmax(axis=1)
        acc = acc_metric.compute(predictions=preds, references=labels)["accuracy"]
        auc = auc_metric.compute(prediction_scores=probs[:, 1], references=labels, average="macro")["roc_auc"]
        return {"accuracy": acc, "roc_auc": auc}

    # 5️⃣ Trainer 클래스 정의
    class AdapterTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            x = inputs["pixel_values"]
            y = inputs["labels"].long()
            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=0.1)
            return (loss, {"logits": logits}) if return_outputs else loss

    args = TrainingArguments(
        output_dir="../outputs",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=1e-4,
        num_train_epochs=5,
        weight_decay=0.02,
        warmup_ratio=0.1,
        eval_strategy="epoch",
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

    last_ckpt = get_last_checkpoint(args.output_dir)
    if last_ckpt is not None:
        print(f">> resume from {last_ckpt}")
        trainer.train(resume_from_checkpoint=last_ckpt)
    else:
        print(">> start fresh training")
        trainer.train()


if __name__ == "__main__":
    main()