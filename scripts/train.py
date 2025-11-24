import os
import math
import torch
from transformers import set_seed, TrainingArguments, Trainer
from transformers.optimization import get_cosine_with_hard_restarts_schedule_with_warmup
import evaluate
from collections import Counter

from src import ClipBackbone, LNCLIPDF
from src import prepare_deepfake_dataset
from src import attach_media_transforms
from src import collate_fn

def main():
    set_seed(42)
    seed = 42
    test_size = 0.2
    use_bf16 = True

    # 1) 데이터 로드
    train_data, test_data, label2id, id2label, class_labels, image_key, video_key = prepare_deepfake_dataset(
        data_path=None,
        split="train",
        data_files="/root/FFPP/metadata.tsv",
        delimiter="\t",
        seed=seed,
        test_size=test_size,
    )

    # 2) 전처리: 학습은 단일 프레임, 평가는 32 프레임 평균
    attach_media_transforms(
        train_data, test_data,
        image_key=image_key,
        video_key=video_key,
        clip_model_name="openai/clip-vit-large-patch14",
        do_face_crop=True,
        rotation_deg=15,
        num_frames=1,
        num_frames_val=32,
    )

    # 3) 모델: CLS 토큰만 사용, LayerNorm만 학습, 나머지 
    backbone = ClipBackbone(
        model_name="openai/clip-vit-large-patch14",
        dtype="bf16",
        freeze_backbone=True,
        use_cls_token=True,
        train_layer_norm_only=True,
    )
    model = LNCLIPDF(
        backbone=backbone,
        num_classes=len(class_labels.names),
        align_weight=0.1,
        uniform_weight=0.1,
        slerp_target=1024,
        classifier_hidden=512,
        class_weights=None,  # will be filled below
    )
    model.id2label = id2label
    model.label2id = label2id

    # Class weights to counter imbalance
    label_counts = Counter(int(l) for l in train_data["label"])
    total = sum(label_counts.values())
    weights = []
    num_classes = len(class_labels.names)
    for i in range(num_classes):
        c = label_counts.get(i, 0)
        if c == 0:
            weights.append(0.0)
        else:
            # Normalize so mean weight ~1
            weights.append(total / (num_classes * c))
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    if hasattr(model, "class_weights"):
        delattr(model, "class_weights")  # remove placeholder to allow buffer registration
    model.register_buffer("class_weights", weights_tensor)

    # 4) 메트릭 (이진 전제)
    acc_metric = evaluate.load("accuracy")
    auc_metric = evaluate.load("roc_auc")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs_t = torch.from_numpy(logits)
        # forward는 비디오 입력 시 확률을 반환하므로, 필요 시에만 softmax
        if not torch.allclose(probs_t.sum(dim=1), torch.ones_like(probs_t[:, 0]), atol=1e-3):
            probs_t = probs_t.softmax(dim=1)
        probs = probs_t.numpy()
        preds = probs.argmax(axis=1)
        acc = acc_metric.compute(predictions=preds, references=labels)["accuracy"]
        auc = auc_metric.compute(
            prediction_scores=probs[:, 1], references=labels, average="macro"
        )["roc_auc"]
        return {"accuracy": acc, "roc_auc": auc}

    class LNCLIPTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            pixel_values = inputs["pixel_values"]
            labels = inputs["labels"].long()
            outputs = model(pixel_values=pixel_values, labels=labels, apply_slerp=True, return_dict=True)
            loss = outputs["loss"]
            return (loss, {"logits": outputs["logits"]}) if return_outputs else loss

        def create_scheduler(self, num_training_steps: int, optimizer=None):
            # 두 사이클 cosine 스케줄(각 10 epoch, warmup 1 epoch) 구현
            if self.lr_scheduler is None or optimizer is not None:
                opt = optimizer if optimizer is not None else self.optimizer
                if self.args.lr_scheduler_type == "cosine_with_restarts":
                    if self.args.warmup_steps > 0:
                        num_warmup_steps = self.args.warmup_steps
                    elif self.args.warmup_ratio > 0:
                        num_warmup_steps = math.ceil(num_training_steps * self.args.warmup_ratio)
                    else:
                        num_warmup_steps = 0
                    self.lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                        opt,
                        num_warmup_steps=num_warmup_steps,
                        num_training_steps=num_training_steps,
                        num_cycles=2,
                    )
                else:
                    return super().create_scheduler(num_training_steps=num_training_steps, optimizer=opt)
            return self.lr_scheduler

    args = TrainingArguments(
        output_dir="./outputs_video",
        per_device_train_batch_size=16,
        # Eval uses 32 frames per sample, so keep the batch small to avoid OOM kills.
        per_device_eval_batch_size=1,
        learning_rate=3e-4,
        num_train_epochs=15,
        weight_decay=0.0,
        warmup_ratio=0.1,  # 1 epoch warmup (20 epochs total)
        lr_scheduler_type="cosine_with_restarts",
        evaluation_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        save_safetensors=False,
        eval_accumulation_steps=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_roc_auc",
        greater_is_better=True,
        remove_unused_columns=False,
        dataloader_num_workers=max(2, (os.cpu_count() or 8)//2),
        dataloader_pin_memory=True,
        bf16=use_bf16,
        fp16=not use_bf16,
        logging_steps=50,
        report_to="none",
    )

    trainer = LNCLIPTrainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=test_data,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    # 사전 가중치 로딩 (옵션)
    ckpt_bin = "/root/Jiwon/Deepfake_killer-Video-image/outputs_video/checkpoint-2000_1/pytorch_model.bin"
    if os.path.exists(ckpt_bin):
        print(f">> load pretrained weights from {ckpt_bin}")
        state_dict = torch.load(ckpt_bin, map_location="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print("   - missing keys   :", missing)
        print("   - unexpected keys:", unexpected)
        print(">> start finetuning on video dataset from loaded weights")
    else:
        print(">> checkpoint bin not found, train from scratch")

    trainer.train()

if __name__ == "__main__":
    main()
