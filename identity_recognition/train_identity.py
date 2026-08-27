import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from dataset import IdentityDataset, BASE_PATH, DATA_PATH, MODES, load_subject_ids
from eval_verification import verification_metrics
from models import ArcMarginProduct, build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train a closed-set body identity classifier.")
    parser.add_argument("--mode", default="pressure", choices=["pressure", "depth_cover1", "depth_cover2", "depth_uncover"])
    parser.add_argument("--train_split", default="real_all.txt")
    parser.add_argument("--val_split", default="real_all.txt")
    parser.add_argument("--model", default="convnextv2_base")
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--arcface_scale", type=float, default=30.0)
    parser.add_argument("--arcface_margin", type=float, default=0.3)
    parser.add_argument("--samples_per_subject", type=int, default=4)
    parser.add_argument("--supcon_weight", type=float, default=0.05)
    parser.add_argument("--supcon_temperature", type=float, default=0.07)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4, help="Base learning rate for pretrained backbone parameters.")
    parser.add_argument("--head_lr_mult", type=float, default=10.0, help="Learning-rate multiplier for the randomly initialized classification head.")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=15,
        help="Stop after this many epochs without improving subject accuracy; 0 disables it.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default=str(Path("/home/shnh/DATA/zjy/BodyMAP_identity")))
    parser.add_argument("--limit_subjects", type=int, default=None)
    parser.add_argument("--limit_poses", type=int, default=None)
    parser.add_argument("--split_strategy", choices=["stratified", "range"], default="stratified")
    parser.add_argument("--pose_folds", type=int, default=5)
    parser.add_argument("--pose_fold", type=int, default=4)
    parser.add_argument("--train_pose_start", type=int, default=0)
    parser.add_argument("--train_pose_end", type=int, default=35)
    parser.add_argument("--val_pose_start", type=int, default=35)
    parser.add_argument("--val_pose_end", type=int, default=None)
    parser.add_argument(
        "--allow_disjoint_subjects",
        action="store_true",
        help="Allow subject-disjoint validation for representation diagnostics only; closed-set identity accuracy is not meaningful.",
    )
    return parser.parse_args()


def split_head_backbone_parameters(model):
    head_keywords = ("head", "classifier", "fc")
    head_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        target = head_params if any(part in name for part in head_keywords) else backbone_params
        target.append(param)
    return backbone_params, head_params


def stratified_pose_split(total_poses, folds, fold):
    if folds < 2:
        raise ValueError("--pose_folds must be at least 2")
    if fold < 0 or fold >= folds:
        raise ValueError(f"--pose_fold must be in [0, {folds}), got {fold}")
    val_indices = [index for index in range(total_poses) if index % folds == fold]
    train_indices = [index for index in range(total_poses) if index % folds != fold]
    return train_indices, val_indices


class PKBatchSampler(Sampler):
    """Build batches with K samples per subject for metric learning."""

    def __init__(self, dataset, batch_size, samples_per_subject, seed=42):
        if samples_per_subject < 2:
            raise ValueError("--samples_per_subject must be at least 2")
        self.subjects_per_batch = batch_size // samples_per_subject
        if self.subjects_per_batch < 2:
            raise ValueError("batch_size must contain at least two subjects")
        self.samples_per_subject = samples_per_subject
        self.num_batches = math.ceil(len(dataset) / batch_size)
        self.seed = seed
        self.epoch = 0
        self.indices_by_label = {}
        for index, sample in enumerate(dataset.samples):
            self.indices_by_label.setdefault(sample[2], []).append(index)
        self.labels = np.array(sorted(self.indices_by_label))

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        for _ in range(self.num_batches):
            labels = rng.choice(
                self.labels,
                size=self.subjects_per_batch,
                replace=len(self.labels) < self.subjects_per_batch,
            )
            batch = []
            for label in labels:
                candidates = self.indices_by_label[int(label)]
                selected = rng.choice(
                    candidates,
                    size=self.samples_per_subject,
                    replace=len(candidates) < self.samples_per_subject,
                )
                batch.extend(selected.tolist())
            rng.shuffle(batch)
            yield batch


def supervised_contrastive_loss(embeddings, labels, temperature=0.07):
    embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
    logits = embeddings @ embeddings.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
    if not positive_mask.any(dim=1).all():
        raise ValueError("Every sample needs a positive pair; use PKBatchSampler with K>=2")
    exp_logits = torch.exp(logits).masked_fill(self_mask, 0.0)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    mean_positive_log_prob = (
        (log_prob * positive_mask).sum(dim=1) / positive_mask.sum(dim=1)
    )
    return -mean_positive_log_prob.mean()


def evaluate(model, loader, criterion, device, arcface_head=None):
    model.eval()
    if arcface_head is not None:
        arcface_head.eval()
    total_loss = 0.0
    correct = 0
    top5_correct = 0
    total = 0
    all_logits = []
    all_labels = []
    all_embeddings = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="eval", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            logits = arcface_head(outputs) if arcface_head is not None else outputs
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()

            top5 = torch.topk(logits, k=min(5, logits.shape[1]), dim=1).indices
            top5_correct += top5.eq(labels.unsqueeze(1)).any(dim=1).sum().item()
            total += labels.size(0)

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            if arcface_head is not None:
                all_embeddings.append(outputs.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    probs = torch.softmax(all_logits, dim=1)
    subject_correct = 0
    subject_total = 0
    for label_idx in torch.unique(all_labels).tolist():
        mask = all_labels == label_idx
        subject_prob = probs[mask].mean(dim=0)
        subject_pred = subject_prob.argmax().item()
        subject_correct += int(subject_pred == label_idx)
        subject_total += 1

    metrics = {
        "loss": total_loss / max(total, 1),
        "acc_sample": correct / max(total, 1),
        "acc_top5": top5_correct / max(total, 1),
        "acc_subject": subject_correct / max(subject_total, 1),
    }
    if all_embeddings:
        metrics.update(verification_metrics(torch.cat(all_embeddings), all_labels))
    return metrics


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable, falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    train_subjects = load_subject_ids(args.train_split, args.limit_subjects)
    val_subjects = load_subject_ids(args.val_split, args.limit_subjects)
    shared_subjects = [sid for sid in val_subjects if sid in set(train_subjects)]
    if not shared_subjects and not args.allow_disjoint_subjects:
        raise ValueError(
            "Closed-set identity validation requires train/val subject overlap. "
            f"Got 0 shared subjects between {args.train_split} and {args.val_split}. "
            "Use the default real_all.txt pose split, provide overlapping splits, or pass "
            "--allow_disjoint_subjects only for non-closed-set diagnostics."
        )

    label_to_idx = {sid: i for i, sid in enumerate(train_subjects)}
    train_pose_indices = None
    val_pose_indices = None
    if args.split_strategy == "stratified":
        total_poses = np.load(DATA_PATH / MODES[args.mode], mmap_mode="r").shape[1]
        train_pose_indices, val_pose_indices = stratified_pose_split(
            total_poses, args.pose_folds, args.pose_fold
        )
    train_dataset = IdentityDataset(
        args.train_split,
        mode=args.mode,
        limit_poses=args.limit_poses if train_pose_indices is None else None,
        subject_ids=train_subjects,
        label_to_idx=label_to_idx,
        pose_start=args.train_pose_start if train_pose_indices is None else None,
        pose_end=args.train_pose_end if train_pose_indices is None else None,
        pose_indices=train_pose_indices,
    )
    val_dataset = IdentityDataset(
        args.val_split,
        mode=args.mode,
        limit_poses=args.limit_poses if val_pose_indices is None else None,
        subject_ids=shared_subjects if shared_subjects else val_subjects,
        label_to_idx=label_to_idx,
        pose_start=args.val_pose_start if val_pose_indices is None else None,
        pose_end=args.val_pose_end if val_pose_indices is None else None,
        pose_indices=val_pose_indices,
    )

    if args.model == "pressure_arcface":
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=PKBatchSampler(
                train_dataset,
                args.batch_size,
                args.samples_per_subject,
                seed=args.seed,
            ),
            num_workers=args.workers,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            drop_last=False,
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    model = build_model(
        args.model, train_dataset.num_classes, embedding_dim=args.embedding_dim
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    arcface_head = None
    if args.model == "pressure_arcface":
        arcface_head = ArcMarginProduct(
            args.embedding_dim,
            train_dataset.num_classes,
            scale=args.arcface_scale,
            margin=args.arcface_margin,
        ).to(device)
    backbone_params, head_params = split_head_backbone_parameters(model)
    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": args.lr})
    if head_params:
        param_groups.append({"params": head_params, "lr": args.lr * args.head_lr_mult})
    if arcface_head is not None:
        param_groups.append(
            {"params": arcface_head.parameters(), "lr": args.lr * args.head_lr_mult}
        )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "best_model.pt"
    metrics_path = out_dir / "metrics.jsonl"

    config = vars(args)
    config["base_path"] = str(BASE_PATH)
    config["train_subjects"] = train_dataset.subject_ids
    config["val_subjects"] = val_dataset.subject_ids
    config["shared_subjects"] = shared_subjects
    config["closed_set_identity"] = bool(shared_subjects)
    config["random_sample_acc"] = 1.0 / max(train_dataset.num_classes, 1)
    config["random_top5_acc"] = min(5, train_dataset.num_classes) / max(train_dataset.num_classes, 1)
    config["train_samples"] = len(train_dataset)
    config["val_samples"] = len(val_dataset)
    config["train_pose_indices"] = train_pose_indices
    config["val_pose_indices"] = val_pose_indices
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))

    print(
        "Identity protocol: "
        f"{train_dataset.num_classes} classes, "
        f"{len(train_dataset)} train samples in {len(train_loader)} batches, "
        f"{len(val_dataset)} val samples in {len(val_loader)} batches, "
        f"random top-1≈{config['random_sample_acc']:.4f}, "
        f"random top-5≈{config['random_top5_acc']:.4f}."
    )
    if train_pose_indices is not None:
        print(
            f"Stratified pose split: train={train_pose_indices}, "
            f"val={val_pose_indices}."
        )
    if args.model == "convnextv2_base":
        print(
            "ConvNeXt V2 uses an ImageNet-pretrained backbone but a new randomly "
            f"initialized identity head; head_lr={args.lr * args.head_lr_mult:g}, "
            f"backbone_lr={args.lr:g}. First-epoch accuracy near random is expected."
        )

    best_subject_acc = 0.0
    best_score = None
    best_epoch = 0
    epochs_without_improvement = 0
    warned_not_learning = False
    metrics_fp = metrics_path.open("a", encoding="utf-8")

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        if arcface_head is not None:
            arcface_head.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for images, labels in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            logits = (
                arcface_head(outputs, labels) if arcface_head is not None else outputs
            )
            loss = criterion(logits, labels)
            if arcface_head is not None and args.supcon_weight > 0:
                loss = loss + args.supcon_weight * supervised_contrastive_loss(
                    outputs, labels, temperature=args.supcon_temperature
                )
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            running_correct += (logits.argmax(dim=1) == labels).sum().item()
            running_total += labels.size(0)

        if running_total == 0:
            raise RuntimeError(
                "Training processed zero samples. Check dataset size, batch size, "
                "and DataLoader drop_last settings."
            )

        val_metrics = evaluate(
            model, val_loader, criterion, device, arcface_head=arcface_head
        )
        train_acc = running_correct / max(running_total, 1)
        record = {
            "epoch": epoch,
            "train_loss": running_loss / max(running_total, 1),
            "train_acc_sample": train_acc,
            **val_metrics,
            "seconds": round(time.time() - start, 2),
        }
        metrics_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        metrics_fp.flush()

        print(
            f"epoch {epoch:02d} train_loss={record['train_loss']:.4f} "
            f"train_acc={train_acc:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['acc_sample']:.4f} "
            f"val_top5={val_metrics['acc_top5']:.4f} "
            f"val_subject_acc={val_metrics['acc_subject']:.4f}"
        )
        if arcface_head is not None:
            print(
                f"           verification_auc={val_metrics['roc_auc']:.4f} "
                f"eer={val_metrics['eer']:.4f} "
                f"tar@far1%={val_metrics['tar_at_far_0.01']:.4f}"
            )
        random_loss = float(np.log(max(train_dataset.num_classes, 1)))
        if (
            epoch >= 5
            and not warned_not_learning
            and train_acc <= config["random_sample_acc"] * 1.5
            and record["train_loss"] >= random_loss * 0.98
        ):
            print(
                "WARNING: training is still at the random baseline after 5 epochs. "
                "This is not normal; verify input statistics and try the small_cnn "
                "overfit check before a full ConvNeXt run."
            )
            warned_not_learning = True

        if arcface_head is not None:
            score = (
                val_metrics["roc_auc"],
                -val_metrics["eer"],
                val_metrics["acc_subject"],
                val_metrics["acc_sample"],
            )
        else:
            score = (
                val_metrics["acc_subject"],
                val_metrics["acc_sample"],
                -val_metrics["loss"],
            )
        if best_score is None or score > best_score:
            best_score = score
            best_subject_acc = val_metrics["acc_subject"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "arcface_state": (
                        arcface_head.state_dict() if arcface_head is not None else None
                    ),
                    "model_name": args.model,
                    "embedding_dim": args.embedding_dim,
                    "arcface_config": {
                        "scale": args.arcface_scale,
                        "margin": args.arcface_margin,
                    },
                    "num_classes": train_dataset.num_classes,
                    "mode": args.mode,
                    "label_to_idx": train_dataset.label_to_idx,
                    "idx_to_label": train_dataset.idx_to_label,
                    "val_metrics": val_metrics,
                    "config": config,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                f"Early stopping at epoch {epoch}: no validation-objective "
                f"improvement for {args.early_stopping_patience} epochs."
            )
            break

    metrics_fp.close()
    print(
        f"Best checkpoint: epoch {best_epoch}, "
        f"subject-level accuracy={best_subject_acc:.4f}, score={best_score}"
    )
    print(f"Checkpoint saved to {checkpoint_path}")
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
