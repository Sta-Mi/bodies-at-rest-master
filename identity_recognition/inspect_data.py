import argparse
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect an SLP cleaned pressure array.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--clip_max", type=float, default=100.0)
    return parser.parse_args()


def main():
    args = parse_args()
    data = np.load(args.path, mmap_mode="r")
    if data.ndim != 4:
        raise ValueError(f"Expected [subject, pose, height, width], got {data.shape}")

    total = data.size
    negative = 0
    positive = 0
    saturated = 0
    zero = 0
    value_sum = 0.0
    value_sq_sum = 0.0
    subject_positive = []

    for subject in data:
        values = np.asarray(subject, dtype=np.float64)
        negative += np.count_nonzero(values < 0)
        positive += np.count_nonzero(values > 0)
        saturated += np.count_nonzero(values > args.clip_max)
        zero += np.count_nonzero(values == 0)
        value_sum += values.sum()
        value_sq_sum += np.square(values).sum()
        subject_positive.append(float(np.mean(values > 0)))

    mean = value_sum / total
    std = max(value_sq_sum / total - mean * mean, 0.0) ** 0.5
    print(f"path={args.path.resolve()}")
    print(f"shape={data.shape} dtype={data.dtype}")
    print(f"min={data.min()} max={data.max()} mean={mean:.6f} std={std:.6f}")
    print(f"negative_ratio={negative / total:.6f}")
    print(f"zero_ratio={zero / total:.6f}")
    print(f"positive_ratio={positive / total:.6f}")
    print(f"above_clip_max_ratio={saturated / total:.6f}")
    print(
        "subject_positive_ratio="
        f"min:{min(subject_positive):.6f} "
        f"median:{np.median(subject_positive):.6f} "
        f"max:{max(subject_positive):.6f}"
    )


if __name__ == "__main__":
    main()
