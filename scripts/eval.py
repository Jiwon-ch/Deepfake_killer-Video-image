"""
Quick eval utility: load the best checkpoint in outputs_video, sample 400 examples
from the training split with the same class ratio, and log logits/preds.
"""

import argparse
import contextlib
import json
import os
import random
import sys
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import set_seed

# Ensure project root is on sys.path when running as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import (  # noqa: E402
    ClipBackbone,
    LNCLIPDF,
    prepare_deepfake_dataset,
    build_media_transforms,
    get_clip_processor,
    collate_fn,
)


def stratified_sample_indices(labels, sample_size, seed):
    rng = random.Random(seed)
    label_to_idx = defaultdict(list)
    for i, lab in enumerate(labels):
        label_to_idx[int(lab)].append(i)

    total = len(labels)
    if total == 0:
        return []
    sample_size = min(sample_size, total)

    counts = {k: len(v) for k, v in label_to_idx.items()}
    targets = {}
    remainders = []
    for k, c in counts.items():
        exact = c * sample_size / total
        take = int(exact)
        targets[k] = min(take, c)
        remainders.append((exact - take, k))

    remaining = sample_size - sum(targets.values())
    for _, k in sorted(remainders, reverse=True):
        if remaining <= 0:
            break
        if targets[k] < counts[k]:
            targets[k] += 1
            remaining -= 1

    if remaining > 0:
        # still short: top-up from any class that has room
        label_order = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        while remaining > 0:
            progressed = False
            for k, c in label_order:
                if targets[k] < c:
                    targets[k] += 1
                    remaining -= 1
                    progressed = True
                    if remaining == 0:
                        break
            if not progressed:
                break

    indices = []
    for k, idxs in label_to_idx.items():
        take = min(targets.get(k, 0), len(idxs))
        if take > 0:
            indices.extend(rng.sample(idxs, take))
    rng.shuffle(indices)
    return indices


def load_model(checkpoint_dir, num_classes, device, dtype, classifier_hidden=512, uniform_weight=0.1):
    backbone = ClipBackbone(
        model_name="openai/clip-vit-large-patch14",
        dtype=dtype,
        freeze_backbone=True,
        use_cls_token=True,
        train_layer_norm_only=True,
    )
    model = LNCLIPDF(
        backbone=backbone,
        num_classes=num_classes,
        align_weight=0.1,
        uniform_weight=uniform_weight,
        slerp_target=1024,
        classifier_hidden=classifier_hidden,
        class_weights=None,
    )

    ckpt_bin = os.path.join(checkpoint_dir, "pytorch_model.bin")
    state_dict = torch.load(ckpt_bin, map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[warn] missing keys: {missing}, unexpected keys: {unexpected}")

    # Align classifier dtype with backbone output
    if dtype == "bf16":
        model = model.to(device=device, dtype=torch.bfloat16)
    elif dtype == "fp16":
        model = model.to(device=device, dtype=torch.float16)
    else:
        model = model.to(device=device, dtype=torch.float32)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs_video/checkpoint-4000")
    parser.add_argument("--data_files", default="/root/FFPP/metadata.tsv")
    parser.add_argument("--delimiter", default="\t")
    parser.add_argument("--sample_size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_frames_val", type=int, default=32)
    parser.add_argument("--classifier_hidden", type=int, default=512)
    parser.add_argument("--output", default="outputs_video/sample_eval_400.jsonl")
    parser.add_argument(
        "--pretty_output",
        default=None,
        help="Optional TSV with ok/label/pred/probabilities for quick inspection "
        "(defaults to <output>.tsv)",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    train_data, _test_data, label2id, id2label, class_labels, image_key, video_key = prepare_deepfake_dataset(
        data_path=None,
        split="train",
        data_files=args.data_files,
        delimiter=args.delimiter,
        seed=args.seed,
        test_size=0.2,
    )

    # Stratified sample from training split
    labels = train_data["label"]
    indices = stratified_sample_indices(labels, args.sample_size, args.seed)
    sampled = train_data.select(indices)
    # preserve raw paths for logging before transform is applied (prefer video then image)
    n_rows = len(sampled)
    video_paths = sampled[video_key] if video_key is not None else [""] * n_rows
    image_paths = sampled[image_key] if image_key is not None else [""] * n_rows
    sample_paths = []
    for vp, ip in zip_longest(video_paths, image_paths, fillvalue=""):
        vp = vp or ""
        ip = ip or ""
        path = vp if (isinstance(vp, str) and vp) else (ip if isinstance(ip, str) else "")
        sample_paths.append(path)

    # Use eval/val transform (no heavy augmentation)
    processor = get_clip_processor("openai/clip-vit-large-patch14")
    _train_tfm, val_tfm = build_media_transforms(
        processor,
        image_key=image_key,
        video_key=video_key,
        do_face_crop=True,
        rotation_deg=15,
        num_frames=1,
        num_frames_val=args.num_frames_val,
        video_strategy="uniform",
    )
    sampled.set_transform(val_tfm)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = "bf16" if device.type == "cuda" else "fp32"
    model = load_model(
        args.checkpoint,
        len(class_labels.names),
        device,
        dtype,
        classifier_hidden=args.classifier_hidden,
        uniform_weight=0.1,
    )

    loader = DataLoader(
        sampled,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(2, (os.cpu_count() or 8) // 4),
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    pretty_output = args.pretty_output
    if pretty_output is None and args.output:
        stem, _ext = os.path.splitext(args.output)
        pretty_output = f"{stem}.tsv"

    os.makedirs(Path(args.output).parent, exist_ok=True)
    rows = []
    total = 0
    correct = 0
    with torch.no_grad(), contextlib.ExitStack() as stack:
        f = stack.enter_context(open(args.output, "w"))
        pretty_f = stack.enter_context(open(pretty_output, "w")) if pretty_output else None
        if pretty_f is not None:
            pretty_f.write("idx\tok\tlabel\tpred\tprob_label\tprob_pred\tpath\n")

        offset = 0
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            labels_t = batch["labels"]
            out = model(
                pixel_values=pixel_values,
                labels=None,
                apply_slerp=False,
                return_dict=True,
            )
            logits = out["logits_for_loss"].detach().cpu()
            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)

            for i in range(logits.size(0)):
                label_id = int(labels_t[i])
                pred_id = int(preds[i])
                ok = int(pred_id == label_id)
                prob_label = float(probs[i, label_id])
                prob_pred = float(probs[i, pred_id])
                row = {
                    "index": int(offset + i),
                    "path": sample_paths[offset + i],
                    "label_id": label_id,
                    "label": id2label[label_id],
                    "pred_id": pred_id,
                    "pred": id2label[pred_id],
                    "logit_0": float(logits[i, 0]),
                    "logit_1": float(logits[i, 1]),
                    "prob_0": float(probs[i, 0]),
                    "prob_1": float(probs[i, 1]),
                    "correct": bool(ok),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                if pretty_f is not None:
                    pretty_f.write(
                        f"{row['index']}\t{ok}\t{row['label']}\t{row['pred']}\t"
                        f"{prob_label:.6f}\t{prob_pred:.6f}\t{row['path']}\n"
                    )
                rows.append(row)
                total += 1
                correct += ok
            offset += logits.size(0)

    acc = correct / max(1, total)
    print(f"[done] wrote {len(rows)} rows to {args.output}")
    if pretty_output:
        print(f"[done] wrote readable summary to {pretty_output}")
    print(f"[done] accuracy on sampled set: {acc:.4f}")


if __name__ == "__main__":
    main()
