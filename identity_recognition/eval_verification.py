import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate identity verification embeddings.")
    parser.add_argument("--embeddings", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--allow_training_poses", action="store_true")
    return parser.parse_args()


def binary_verification_metrics(scores, matches):
    scores = np.asarray(scores)
    matches = np.asarray(matches, dtype=np.int64)
    order = np.argsort(-scores)
    scores = scores[order]
    matches = matches[order]
    positives = int(matches.sum())
    negatives = int(len(matches) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("Verification requires both positive and negative pairs")

    true_positive_rate = np.cumsum(matches) / positives
    false_positive_rate = np.cumsum(1 - matches) / negatives
    false_negative_rate = 1.0 - true_positive_rate
    eer_index = int(np.argmin(np.abs(false_positive_rate - false_negative_rate)))

    fpr = np.concatenate(([0.0], false_positive_rate, [1.0]))
    tpr = np.concatenate(([0.0], true_positive_rate, [1.0]))
    metrics = {
        "num_positive_pairs": positives,
        "num_negative_pairs": negatives,
        "roc_auc": float(np.sum((tpr[1:] + tpr[:-1]) * np.diff(fpr) * 0.5)),
        "eer": float(
            (false_positive_rate[eer_index] + false_negative_rate[eer_index]) / 2
        ),
        "eer_threshold": float(scores[eer_index]),
    }
    for target_far in (0.1, 0.01, 0.001):
        valid = np.flatnonzero(false_positive_rate <= target_far)
        tar = true_positive_rate[valid[-1]] if len(valid) else 0.0
        metrics[f"tar_at_far_{target_far:g}"] = float(tar)
    return metrics


def verification_metrics(embeddings, labels):
    embeddings = F.normalize(embeddings.float(), p=2, dim=1)
    labels = labels.long()
    row, col = torch.triu_indices(len(labels), len(labels), offset=1)
    scores = (embeddings[row] * embeddings[col]).sum(dim=1).cpu().numpy()
    matches = labels[row].eq(labels[col]).cpu().numpy().astype(np.int64)
    metrics = binary_verification_metrics(scores, matches)
    metrics["num_embeddings"] = int(len(labels))
    return metrics


def main():
    args = parse_args()
    data = torch.load(args.embeddings, map_location="cpu", weights_only=False)
    if data.get("includes_training_poses", False) and not args.allow_training_poses:
        raise ValueError(
            "Embeddings include training poses. Use held-out embeddings for formal "
            "verification, or pass --allow_training_poses for diagnostics only."
        )
    metrics = verification_metrics(data["embeddings"], data["labels"])
    metrics["pose_indices"] = data.get("pose_indices")
    output = args.out or args.embeddings.with_name("verification_metrics.json")
    output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Verification metrics saved to {output}")


if __name__ == "__main__":
    main()
