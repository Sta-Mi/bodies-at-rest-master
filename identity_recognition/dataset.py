from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


BASE_PATH = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_PATH / "data_BP" / "slp_real_cleaned"
SPLIT_PATH = BASE_PATH / "BodyMAP" / "data_files"


MODES = {
    "pressure": "pressure_recon_Pplus_gt_0to102.npy",
    "depth_cover1": "depth_cover1_cleaned_0to102.npy",
    "depth_cover2": "depth_cover2_cleaned_0to102.npy",
    "depth_uncover": "depth_uncover_cleaned_0to102.npy",
}


def load_subject_ids(split_file: str, limit_subjects: int | None = None):
    split_path = SPLIT_PATH / split_file
    lines = [
        line.strip()
        for line in split_path.read_text().splitlines()
        if line.strip()
    ]
    if limit_subjects is not None:
        lines = lines[:limit_subjects]
    return lines


def resolve_pose_range(
    total_poses: int,
    pose_start: int | None = None,
    pose_end: int | None = None,
    limit_poses: int | None = None,
):
    start = 0 if pose_start is None else pose_start
    end = total_poses if pose_end is None else pose_end
    if limit_poses is not None:
        end = min(end, start + limit_poses)
    if start < 0 or end < start or end > total_poses:
        raise ValueError(
            f"Invalid pose range [{start}, {end}) for {total_poses} poses. "
            "Use 0-based --*_pose_start/--*_pose_end values."
        )
    return range(start, end)


class IdentityDataset(Dataset):
    def __init__(
        self,
        split_file: str,
        mode: str = "pressure",
        limit_subjects: int | None = None,
        limit_poses: int | None = None,
        subject_ids: list[str] | None = None,
        label_to_idx: dict[str, int] | None = None,
        pose_start: int | None = None,
        pose_end: int | None = None,
        pose_indices: list[int] | None = None,
    ):
        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode}, choose from {list(MODES)}")

        self.mode = mode
        self.subject_ids = subject_ids or load_subject_ids(split_file, limit_subjects)
        self.label_to_idx = label_to_idx or {
            sid: i for i, sid in enumerate(self.subject_ids)
        }
        self.idx_to_label = {i: sid for sid, i in self.label_to_idx.items()}
        self.num_classes = len(self.label_to_idx)

        array_path = DATA_PATH / MODES[mode]
        self.data = np.load(array_path, mmap_mode="r")
        if pose_indices is None:
            pose_indices = list(
                resolve_pose_range(
                    self.data.shape[1],
                    pose_start=pose_start,
                    pose_end=pose_end,
                    limit_poses=limit_poses,
                )
            )
        else:
            if pose_start is not None or pose_end is not None or limit_poses is not None:
                raise ValueError("pose_indices cannot be combined with pose ranges or limit_poses")
            if not pose_indices or min(pose_indices) < 0 or max(pose_indices) >= self.data.shape[1]:
                raise ValueError(
                    f"Invalid pose_indices for {self.data.shape[1]} poses: {pose_indices}"
                )

        self.samples = []
        for subject_id in self.subject_ids:
            if subject_id not in self.label_to_idx:
                continue
            person_idx = int(subject_id) - 1
            label_idx = self.label_to_idx[subject_id]
            for pose_idx in pose_indices:
                self.samples.append((person_idx, pose_idx, label_idx, subject_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        person_idx, pose_idx, label_idx, subject_id = self.samples[index]
        image = self.data[person_idx, pose_idx].astype(np.float32)

        if self.mode == "pressure":
            image = np.clip(image, 0.0, 100.0) / 100.0
        else:
            image = np.clip(image, 0.0, 102.0) / 102.0

        image = np.expand_dims(image, axis=0)
        image = np.ascontiguousarray(image)
        return torch.from_numpy(image), label_idx
