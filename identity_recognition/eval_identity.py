import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DEFAULT_DATA_ROOT, IdentityDataset, discover_subject_ids
from models import ArcMarginProduct, build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained identity classifier.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--sessions", nargs="+", default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out_dir", default="identity_recognition/runs/eval")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable, falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    embedding_dim = ckpt.get("embedding_dim", 256)
    model = build_model(
        ckpt["model_name"], ckpt["num_classes"], embedding_dim=embedding_dim
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    arcface_head = None
    if ckpt.get("arcface_state") is not None:
        arcface_config = ckpt.get("arcface_config", {})
        arcface_head = ArcMarginProduct(
            embedding_dim,
            ckpt["num_classes"],
            scale=arcface_config.get("scale", 30.0),
            margin=arcface_config.get("margin", 0.3),
        ).to(device)
        arcface_head.load_state_dict(ckpt["arcface_state"])
        arcface_head.eval()
    model.eval()

    label_to_idx = ckpt.get("label_to_idx")
    if label_to_idx is None:
        raise ValueError("Checkpoint does not contain label_to_idx; cannot build closed-set labels.")
    subject_ids = [sid for sid in discover_subject_ids(args.data_root) if sid in label_to_idx]
    if not subject_ids:
        raise ValueError(
            "The data root has no subjects present in the checkpoint label map."
        )
    sessions = args.sessions or ckpt.get("config", {}).get("val_sessions")
    if not sessions:
        raise ValueError("No evaluation sessions specified or recorded in checkpoint")
    print(f"Using held-out sessions: {sessions}")

    dataset = IdentityDataset(
        data_root=args.data_root,
        sessions=sessions,
        subject_ids=subject_ids,
        label_to_idx=label_to_idx,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    all_logits = []
    all_labels = []
    all_embeddings = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="eval", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            logits = arcface_head(outputs) if arcface_head is not None else outputs
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            if arcface_head is not None:
                all_embeddings.append(outputs.cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    preds = logits.argmax(dim=1)

    correct = (preds == labels).sum().item() / max(len(labels), 1)
    top5 = (
        torch.topk(logits, k=min(5, logits.shape[1]), dim=1)
        .indices.eq(labels.unsqueeze(1))
        .any(dim=1)
        .sum()
        .item()
        / max(len(labels), 1)
    )

    probs = torch.softmax(logits, dim=1)
    subject_correct = 0
    subject_total = 0
    per_subject = {}
    for label_idx in torch.unique(labels).tolist():
        mask = labels == label_idx
        subject_prob = probs[mask].mean(dim=0)
        subject_pred = subject_prob.argmax().item()
        subject_correct += int(subject_pred == label_idx)
        subject_total += 1
        true_subject_id = ckpt["idx_to_label"].get(str(label_idx), ckpt["idx_to_label"].get(label_idx))
        pred_subject_id = ckpt["idx_to_label"].get(str(subject_pred), ckpt["idx_to_label"].get(subject_pred, str(subject_pred)))
        per_subject[true_subject_id] = {
            "true_label_idx": label_idx,
            "pred_label_idx": subject_pred,
            "pred_subject_id": pred_subject_id,
            "correct": int(subject_pred == label_idx),
        }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "acc_sample": correct,
        "acc_top5": top5,
        "acc_subject": subject_correct / max(subject_total, 1),
        "subject_total": subject_total,
        "checkpoint_epoch": ckpt.get("epoch"),
        "sessions": sessions,
        "includes_training_sessions": bool(
            set(sessions) & set(ckpt.get("config", {}).get("train_sessions", []))
        ),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    (out_dir / "predictions.json").write_text(json.dumps(per_subject, indent=2, ensure_ascii=False))
    if all_embeddings:
        torch.save(
            {
                "embeddings": torch.cat(all_embeddings, dim=0),
                "labels": labels,
                "idx_to_label": ckpt["idx_to_label"],
                "sessions": sessions,
                "includes_training_sessions": metrics["includes_training_sessions"],
            },
            out_dir / "embeddings.pt",
        )

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Predictions saved to {out_dir / 'predictions.json'}")


if __name__ == "__main__":
    main()
