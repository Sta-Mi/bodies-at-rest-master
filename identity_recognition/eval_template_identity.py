import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DEFAULT_DATA_ROOT, IdentityDataset, discover_subject_ids
from eval_verification import binary_verification_metrics
from models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate enrollment-template identity matching.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out_dir", required=True, type=Path)
    return parser.parse_args()


def encode(model, loader, device):
    embeddings = []
    labels = []
    model.eval()
    with torch.no_grad():
        for images, batch_labels in tqdm(loader, desc="encode", leave=False):
            embeddings.append(model(images.to(device)).cpu())
            labels.append(batch_labels)
    return F.normalize(torch.cat(embeddings), dim=1), torch.cat(labels)


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if ckpt.get("arcface_state") is None:
        raise ValueError("Template evaluation requires a pressure_arcface checkpoint")

    config = ckpt.get("config", {})
    train_sessions = config.get("train_sessions")
    val_sessions = config.get("val_sessions")
    if not train_sessions or not val_sessions:
        raise ValueError("Checkpoint must record enrollment and probe sessions")
    if set(train_sessions) & set(val_sessions):
        raise ValueError("Enrollment and probe sessions overlap")

    label_to_idx = ckpt["label_to_idx"]
    subject_ids = [sid for sid in discover_subject_ids(args.data_root) if sid in label_to_idx]
    common = dict(
        data_root=args.data_root,
        subject_ids=subject_ids,
        label_to_idx=label_to_idx,
    )
    enrollment = IdentityDataset(**common, sessions=train_sessions)
    probes = IdentityDataset(**common, sessions=val_sessions)
    enrollment_loader = DataLoader(enrollment, batch_size=args.batch_size, shuffle=False)
    probe_loader = DataLoader(probes, batch_size=args.batch_size, shuffle=False)

    embedding_dim = ckpt.get("embedding_dim", 256)
    model = build_model(
        ckpt["model_name"], ckpt["num_classes"], embedding_dim=embedding_dim
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    enrollment_embeddings, enrollment_labels = encode(model, enrollment_loader, device)
    probe_embeddings, probe_labels = encode(model, probe_loader, device)

    templates = []
    for label in range(ckpt["num_classes"]):
        templates.append(enrollment_embeddings[enrollment_labels == label].mean(dim=0))
    templates = F.normalize(torch.stack(templates), dim=1)
    scores = probe_embeddings @ templates.T
    predictions = scores.argmax(dim=1)
    top5 = scores.topk(k=min(5, scores.shape[1]), dim=1).indices

    subject_correct = 0
    for label in torch.unique(probe_labels):
        mean_embedding = F.normalize(probe_embeddings[probe_labels == label].mean(dim=0), dim=0)
        subject_correct += int((mean_embedding @ templates.T).argmax() == label)

    matches = probe_labels[:, None].eq(torch.arange(scores.shape[1])[None, :])
    metrics = binary_verification_metrics(
        scores.flatten().numpy(), matches.flatten().numpy()
    )
    metrics.update(
        {
            "acc_sample": float(predictions.eq(probe_labels).float().mean()),
            "acc_top5": float(top5.eq(probe_labels[:, None]).any(dim=1).float().mean()),
            "acc_subject": subject_correct / len(torch.unique(probe_labels)),
            "num_enrollment_embeddings": len(enrollment_embeddings),
            "num_probe_embeddings": len(probe_embeddings),
            "enrollment_sessions": train_sessions,
            "probe_sessions": val_sessions,
        }
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / "template_metrics.json"
    output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    torch.save({"templates": templates, "idx_to_label": ckpt["idx_to_label"]}, args.out_dir / "templates.pt")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Template metrics saved to {output}")


if __name__ == "__main__":
    main()
